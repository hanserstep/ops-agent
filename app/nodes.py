"""图节点定义。

整条流程：

    START
      │
    router ──chitchat/kb_qa──────────────────────┐
      │                                          │
    troubleshoot / action                        │
      ▼                                          │
    planner                                      │
      ▼                                          │
    executor ◀──────────────┐                    │
      │  （有危险工具调用）    │                    │
      ├──▶ human_gate ───────┤（批准）             │
      │        └─────────────┴──▶ answer         │
      ▼                     （拒绝）               │
    reflector ──需要更多信息──┘                    │
      │                                          │
      └──信息足够──────────────────────────────────┘
                                   ▼
                                 answer
                                   ▼
                                  END

三个循环控制点在 executor 和 reflector 上，这是 LangGraph 相比
「一条 if-else 链」的核心价值：状态机可以回退、可以重试、可以中断等待。
"""

import json
from typing import Any, Dict, List

from langgraph.types import interrupt

from .state import MAX_STEPS
from .tools import registry


def _obs_entry(result: Dict[str, Any]) -> Dict[str, Any]:
    """把工具返回整理成 observation。

    失败也要进 observations：reflector 需要知道「这个方向试过了但失败了」，
    否则它会指示下一轮重试同一个工具，白白浪费步数。
    """
    entry: Dict[str, Any] = {"tool": result.get("tool", "")}
    if result.get("ok"):
        entry["result"] = result.get("result")
    else:
        entry["error"] = result.get("error", "unknown error")
    entry["elapsed_ms"] = result.get("elapsed_ms", 0)
    return entry


def make_nodes(llm):
    """把所有节点绑到同一个 llm 实例上。

    用闭包而不是把 llm 塞进 state：state 会在 checkpoint 里被序列化落盘，
    llm 客户端不可序列化，塞进去会直接报错。
    """

    def route_node(state: Dict) -> Dict:
        user_input = state.get("user_input", "")
        intent, reason = llm.route(user_input)
        return {"intent": intent, "messages": [
            {"role": "system", "content": f"意图判定：{intent}（{reason}）"}
        ]}

    def plan_node(state: Dict) -> Dict:
        user_input = state.get("user_input", "")
        observations = state.get("observations", [])
        context = ""
        if observations:
            context = json.dumps(observations[-2:], ensure_ascii=False)[:800]
        steps = llm.plan(user_input, context)
        return {
            "plan": steps,
            "current_step": 0,
            "need_more_info": True,
            "messages": [{"role": "system", "content": f"排查计划：{steps}"}],
        }

    def execute_node(state: Dict) -> Dict:
        plan = state.get("plan", [])
        step = state.get("current_step", 0)
        observations = state.get("observations", [])

        choice = llm.choose_tool(
            user_input=state.get("user_input", ""),
            plan=plan, current_step=step,
            observations=observations,
            tools=registry.openai_tools(),
        )

        if not choice:
            # 模型认为不需要再调工具，交给 reflector 决定是否收尾
            return {"need_more_info": False, "next_tool": None,
                    "messages": [{"role": "system", "content": "模型判断无需再调工具"}]}

        spec = registry.get(choice["name"])
        if spec is not None and spec.dangerous:
            # 危险工具不在这里执行，交给 human_gate 挂起等确认
            return {
                "next_tool": choice["name"],
                "tool_args": choice["args"],
                "pending_action": {
                    "tool": choice["name"],
                    "args": choice["args"],
                    "risk": spec.risk_note,
                },
                "messages": [{"role": "system",
                              "content": f"待审批的危险操作：{choice['name']}"}],
            }

        result = registry.call(choice["name"], choice["args"])
        return {
            "observations": [_obs_entry(result)],
            "step_count": 1,
            "current_step": min(step + 1, max(len(plan) - 1, 0)),
            "next_tool": choice["name"],
            "tool_args": choice["args"],
            "messages": [{"role": "assistant", "content":
                          f"调用 {choice['name']}，结果：{json.dumps(result, ensure_ascii=False)[:400]}"}],
        }

    def human_gate_node(state: Dict) -> Dict:
        """危险操作的人工审批闸门。

        interrupt() 会让整张图在这里挂起，状态写进 checkpointer。
        进程可以退出、可以重启，之后用 Command(resume=...) 从断点继续。
        这是 Agent 能上生产的前提——没人会允许一个 LLM 半夜自己重启数据库。
        """
        pending = state.get("pending_action") or {}
        decision = interrupt({
            "type": "approval_required",
            "tool": pending.get("tool"),
            "args": pending.get("args"),
            "risk": pending.get("risk", ""),
            "question": "是否批准执行该操作？",
        })

        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)

        if not approved:
            return {
                "approved": False,
                "pending_action": None,
                "need_more_info": False,
                "observations": [{
                    "tool": pending.get("tool", ""),
                    "error": "用户拒绝执行，操作已取消",
                }],
                "messages": [{"role": "system", "content": "危险操作被人工拒绝"}],
            }

        result = registry.call(pending.get("tool", ""), pending.get("args", {}))
        return {
            "approved": True,
            "pending_action": None,
            "observations": [_obs_entry(result)],
            "step_count": 1,
            "need_more_info": False,
            "messages": [{"role": "system", "content": "危险操作已获批准并执行"}],
        }

    def reflect_node(state: Dict) -> Dict:
        steps = state.get("step_count", 0)
        if steps >= MAX_STEPS:
            return {
                "need_more_info": False,
                "messages": [{"role": "system",
                              "content": f"已达最大步数 {MAX_STEPS}，强制收尾"}],
            }

        verdict = llm.reflect(
            user_input=state.get("user_input", ""),
            plan=state.get("plan", []),
            observations=state.get("observations", []),
            steps=steps,
        )
        return {
            "need_more_info": not verdict["answer_ready"],
            "current_step": state.get("current_step", 0) + 1,
            "messages": [{"role": "system",
                          "content": f"评估：{verdict.get('reason', '')} "
                                     f"下一步：{verdict.get('next_focus', '')}"}],
        }

    def answer_node(state: Dict) -> Dict:
        text = llm.answer(
            user_input=state.get("user_input", ""),
            observations=state.get("observations", []),
        )
        return {"final_answer": text, "messages": [{"role": "assistant", "content": text}]}

    def kb_answer_node(state: Dict) -> Dict:
        """知识问答分支：直接查知识库再作答，不做多步排查。"""
        result = registry.call("search_kb", {"query": state.get("user_input", "")})
        observations = [_obs_entry(result)]
        text = llm.answer(user_input=state.get("user_input", ""), observations=observations)
        return {
            "observations": observations,
            "step_count": 1,
            "final_answer": text,
            "messages": [{"role": "assistant", "content": text}],
        }

    def chitchat_node(state: Dict) -> Dict:
        text = "我是运维助手，可以帮你查监控、搜历史故障、执行诊断脚本、创建工单。有什么问题？"
        return {"final_answer": text,
                "messages": [{"role": "assistant", "content": text}]}

    return {
        "router": route_node,
        "planner": plan_node,
        "executor": execute_node,
        "human_gate": human_gate_node,
        "reflector": reflect_node,
        "answer": answer_node,
        "kb_answer": kb_answer_node,
        "chitchat": chitchat_node,
    }


# ---- 条件边 ----

def after_router(state: Dict) -> str:
    intent = state.get("intent", "troubleshoot")
    if intent == "kb_qa":
        return "kb_answer"
    if intent == "chitchat":
        return "chitchat"
    return "planner"


def after_execute(state: Dict) -> str:
    if state.get("pending_action"):
        return "human_gate"
    if state.get("need_more_info") is False and not state.get("next_tool"):
        return "answer"
    return "reflector"


def after_gate(state: Dict) -> str:
    """审批之后直接收尾，无论批准还是拒绝。

    这里曾经走过 reflector，结果是：批准 → 执行 → reflector 认为
    「还没验证效果」→ 回到 executor → 模型再次请求同一个危险操作 →
    第二次审批 → 第二次执行。实际生产里这就是重启两次。

    Agent 系统里这类 bug 特别危险：每一步单独看都合理，
    合起来变成「用户点一次批准，系统执行两次」。
    危险操作的正确语义是「一次授权对应一次执行」，执行完就该总结汇报。
    """
    return "answer"


def after_reflect(state: Dict) -> str:
    return "executor" if state.get("need_more_info") else "answer"
