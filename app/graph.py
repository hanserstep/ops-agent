"""图组装与运行入口。

编译后的图对外只有两个动作：

    run(question)                跑一轮，遇到审批点会挂起并返回待审批信息
    resume(thread_id, approved)  对挂起的流程给出审批结果，从断点继续

这就是「可中断、可恢复」的全部接口。加了 checkpointer 之后，
同一个 thread_id 的所有中间状态都存在 SQLite 里，进程重启也能接着跑。
"""

import os
from typing import Any, Dict, Optional, Tuple

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .llm import DeepSeekLLM
from .nodes import (after_execute, after_gate, after_reflect, after_router,
                    make_nodes)
from .state import AgentState

DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "checkpoints.db"
)


def build_graph(llm=None, checkpointer=None):
    llm = llm or DeepSeekLLM()
    nodes = make_nodes(llm)

    g = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", after_router, {
        "kb_answer": "kb_answer",
        "chitchat": "chitchat",
        "planner": "planner",
    })
    g.add_edge("planner", "executor")
    g.add_conditional_edges("executor", after_execute, {
        "human_gate": "human_gate",
        "reflector": "reflector",
        "answer": "answer",
    })
    g.add_conditional_edges("human_gate", after_gate, {
        "reflector": "reflector",
        "answer": "answer",
    })
    g.add_conditional_edges("reflector", after_reflect, {
        "executor": "executor",
        "answer": "answer",
    })
    for terminal in ("answer", "kb_answer", "chitchat"):
        g.add_edge(terminal, END)

    return g.compile(checkpointer=checkpointer)


class OpsAgent:
    """对外封装的运行器。"""

    def __init__(self, llm=None, db_path: str = DEFAULT_DB):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._saver_ctx = SqliteSaver.from_conn_string(db_path)
        self.checkpointer = self._saver_ctx.__enter__()
        self.llm = llm or DeepSeekLLM()
        self.graph = build_graph(self.llm, self.checkpointer)

    def close(self):
        try:
            self._saver_ctx.__exit__(None, None, None)
        except Exception:
            pass

    # ---- 运行 ----
    def run(self, question: str, thread_id: str = "default") -> Dict[str, Any]:
        state = {
            "user_input": question,
            "messages": [{"role": "user", "content": question}],
            "observations": [],
            "step_count": 0,
            "current_step": 0,
            "plan": [],
        }
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(state, config=config)
        return self._shape(result, thread_id)

    def resume(self, thread_id: str, approved: bool,
               approved_by: str = "human") -> Dict[str, Any]:
        """给挂起的流程一个审批结论，从断点继续。

        interrupt() 挂起时，状态已经写进 checkpointer。这里用 Command(resume=)
        把审批结果送回 human_gate 节点，节点从 interrupt() 那一行之后继续执行。
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(
            Command(resume={"approved": approved, "by": approved_by}),
            config=config,
        )
        return self._shape(result, thread_id)

    @staticmethod
    def _shape(result: Dict, thread_id: str) -> Dict[str, Any]:
        pending = result.get("pending_action")
        # langgraph 把 interrupt 的内容放在 __interrupt__ 里
        interrupts = result.get("__interrupt__") or []
        if interrupts and not pending:
            payload = getattr(interrupts[0], "value", None) or {}
            pending = {
                "tool": payload.get("tool"),
                "args": payload.get("args"),
                "risk": payload.get("risk", ""),
            }

        return {
            "thread_id": thread_id,
            "intent": result.get("intent"),
            "plan": result.get("plan", []),
            "observations": result.get("observations", []),
            "answer": result.get("final_answer", ""),
            "steps": result.get("step_count", 0),
            "awaiting_approval": bool(pending),
            "pending_action": pending if pending else None,
            "tokens": getattr(result.get("_llm", None), "tokens", None),
        }
