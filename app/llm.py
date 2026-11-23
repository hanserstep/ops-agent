"""LLM 调用层。

五个角色都用同一份模型，但 prompt 和输出结构完全不同：

    route      意图分类，输出固定枚举
    plan       任务拆解，输出步骤数组
    choose     工具选择，走 function calling（有 schema 约束，最可靠）
    reflect    结果评估，输出「信息够不够 + 理由」
    answer     生成最终回答

为什么不把五个 prompt 合成一个大 prompt：
    拆开之后每个 prompt 只干一件事，输出格式能被严格约束，出错时也好定位
    是哪一步判断错了。合在一起时模型容易「顺手把下一步也做了」，
    结果是流程不可控、无法评测。

FakeLLM 用于单测：不花 token、不依赖网络，能精确控制每一步的输出，
这样 graph 的分支逻辑才能被覆盖到。
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

INTENTS = ["kb_qa", "troubleshoot", "action", "chitchat"]

_ROUTE_PROMPT = """判断用户这句话属于哪一类，只输出 JSON，不要任何解释。

类别：
- kb_qa: 询问怎么做、某个概念是什么、标准流程是什么（不需要看实时数据）
- troubleshoot: 系统出问题了，需要排查（需要查监控、看日志、搜历史故障）
- action: 要求执行操作（重启、清理、扩容）
- chitchat: 打招呼、闲聊、与运维无关

输出格式：{{"intent": "类别", "reason": "一句话理由"}}

用户输入：{user_input}"""

_PLAN_PROMPT = """你是运维排障专家。把用户的问题拆成可执行的排查步骤。

要求：
1. 每步只描述「要确认什么」，不要指定用什么工具（工具选择是下一步的事）
2. 最多 4 步，从最容易验证、最能快速定位的方向开始
3. 如果问题信息不足，第一步应该是收集现场信息

已知信息：
{context}

用户问题：{user_input}

只输出 JSON：{{"steps": ["步骤1", "步骤2"]}}"""

_REFLECT_PROMPT = """判断当前收集到的信息是否足以回答用户问题。

用户问题：{user_input}
当前计划：{plan}
已收集的信息：
{observations}

判断标准：
- 如果已经能定位原因或给出明确结论，answer_ready 为 true
- 如果关键信息缺失（比如怀疑磁盘但没查磁盘），answer_ready 为 false 并说明还缺什么
- 已经执行 {steps} 步，如果信息仍不足且没有更好的方向，也应该收尾

只输出 JSON：{{"answer_ready": true/false, "reason": "理由", "next_focus": "下一步该查什么"}}"""

_ANSWER_PROMPT = """你是运维助手。基于下面收集到的信息回答用户问题。

规则：
1. 每个结论都要有依据，指明来自哪个工具或哪条信息
2. 如果信息不足以定论，明确说明「目前只能判断到…，建议进一步…」，不要编造原因
3. 如果涉及需要执行的操作，说清楚风险，并提示需要人工确认
4. 用中文，简洁，分点

用户问题：{user_input}

收集到的信息：
{observations}"""


class DeepSeekLLM:

    def __init__(self, api_key: str = "", base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", timeout: float = 60.0):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=base_url, timeout=timeout,
        )
        self.model = model
        self.calls = 0
        self.tokens = 0

    # ---- 内部 ----
    def _chat(self, prompt: str, temperature: float = 0.1,
              json_mode: bool = False) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        self.calls += 1
        if resp.usage:
            self.tokens += resp.usage.total_tokens
        return resp.choices[0].message.content or ""

    def _json(self, prompt: str) -> Dict:
        raw = self._chat(prompt, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 模型偶尔会在 JSON 外面包一层 ```json```，兜一下
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    pass
            return {}

    @staticmethod
    def _fmt_observations(observations: List[Dict]) -> str:
        if not observations:
            return "（暂无）"
        lines = []
        for i, obs in enumerate(observations, 1):
            tool = obs.get("tool", "?")
            body = obs.get("result", obs.get("error", ""))
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False)
            lines.append(f"{i}. [{tool}] {str(body)[:600]}")
        return "\n".join(lines)

    # ---- 五个角色 ----
    def route(self, user_input: str) -> Tuple[str, str]:
        data = self._json(_ROUTE_PROMPT.format(user_input=user_input))
        intent = data.get("intent", "troubleshoot")
        if intent not in INTENTS:
            intent = "troubleshoot"
        return intent, data.get("reason", "")

    def plan(self, user_input: str, context: str = "") -> List[str]:
        data = self._json(_PLAN_PROMPT.format(user_input=user_input, context=context or "（无）"))
        steps = data.get("steps") or []
        return [str(s) for s in steps][:4] or ["收集当前系统状态信息"]

    def choose_tool(self, user_input: str, plan: List[str], current_step: int,
                    observations: List[Dict], tools: List[Dict]) -> Optional[Dict]:
        """用 function calling 让模型选工具。

        比让它输出 JSON 再解析可靠：schema 由 API 强制约束，
        参数类型和枚举值在服务端就校验过一轮。

        关于危险操作的那句 system prompt 很关键：

            如果不告诉模型「审批由系统负责」，它会因为风险高而拒绝调用
            restart_service，转而在回答里写「建议人工确认」——听起来对，
            但整个 interrupt 机制就被绕过了。

            安全不该靠模型自觉。正确分工是：**模型负责提出操作请求，
            系统负责拦下来问人**。模型过度谨慎 = 机制失效。
        """
        step_desc = plan[current_step] if current_step < len(plan) else "补充信息"
        messages = [
            {"role": "system", "content":
                "你是运维排障助手。根据当前要确认的事项，选择最合适的工具并给出参数。"
                "如果已有信息足够，不要调用工具。\n\n"
                "重要：如果用户明确要求执行某个操作（重启服务、创建工单等），"
                "应当直接调用对应工具。系统会自动挂起并要求人工审批，"
                "你不需要因为操作有风险而回避调用，也不要在回答里替用户做这个决定。"},
            {"role": "user", "content":
                f"用户问题：{user_input}\n\n"
                f"当前要确认：{step_desc}\n\n"
                f"已收集信息：\n{self._fmt_observations(observations)}"},
        ]
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools,
            tool_choice="auto", temperature=0.0,
        )
        self.calls += 1
        if resp.usage:
            self.tokens += resp.usage.total_tokens

        msg = resp.choices[0].message
        if not msg.tool_calls:
            return None
        call = msg.tool_calls[0]
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        return {"name": call.function.name, "args": args}

    def reflect(self, user_input: str, plan: List[str], observations: List[Dict],
                steps: int) -> Dict:
        data = self._json(_REFLECT_PROMPT.format(
            user_input=user_input, plan="；".join(plan) or "（无）",
            observations=self._fmt_observations(observations), steps=steps,
        ))
        return {
            "answer_ready": bool(data.get("answer_ready", True)),
            "reason": data.get("reason", ""),
            "next_focus": data.get("next_focus", ""),
        }

    def answer(self, user_input: str, observations: List[Dict]) -> str:
        return self._chat(_ANSWER_PROMPT.format(
            user_input=user_input,
            observations=self._fmt_observations(observations),
        ), temperature=0.3)


class FakeLLM:
    """测试用的确定性 LLM。

    可以脚本化每一步的返回，用来覆盖 graph 的所有分支：
    正常路径、工具失败重试、信息不足收尾、危险操作审批。
    """

    def __init__(self, intents=None, plans=None, tool_choices=None,
                 reflections=None, answers=None):
        self.intents = list(intents or [("troubleshoot", "test")])
        self.plans = list(plans or [["查看监控"]])
        self.tool_choices = list(tool_choices or [])
        self.reflections = list(reflections or [{"answer_ready": True, "reason": "够了"}])
        self.answers = list(answers or ["结论：测试答案"])
        self.calls = 0
        self.tokens = 0

    def route(self, user_input):
        self.calls += 1
        return self.intents.pop(0) if len(self.intents) > 1 else self.intents[0]

    def plan(self, user_input, context=""):
        self.calls += 1
        return self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]

    def choose_tool(self, user_input, plan, current_step, observations, tools):
        self.calls += 1
        if not self.tool_choices:
            return None
        return self.tool_choices.pop(0) if len(self.tool_choices) > 1 else self.tool_choices[0]

    def reflect(self, user_input, plan, observations, steps):
        self.calls += 1
        return self.reflections.pop(0) if len(self.reflections) > 1 else self.reflections[0]

    def answer(self, user_input, observations):
        self.calls += 1
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
