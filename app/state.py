"""Agent 状态定义。

为什么用 TypedDict + Annotated reducer 而不是普通 dataclass：

LangGraph 的核心是「状态在节点之间流转，每个节点只返回它修改的字段」。
普通 dataclass 做增量更新时，节点返回的 dict 会覆盖整个字段；而
Annotated[list, add_messages] 告诉框架「这个字段是追加语义」——
节点返回新消息时框架自动 append，而不是替换。

这个区别在 messages 上尤其关键：如果节点每次都覆盖 messages，
多轮对话历史就丢了，模型会「忘记」上一步查到什么。

其它字段用默认覆盖语义（operator 不指定），因为它们是「当前值」而非「累积值」：
  intent   每次路由都重新判定，应该覆盖
  plan     每次规划都重新生成，应该覆盖
  observations  工具结果需要累积，所以也用 add
  step_count    需要累加，用 operator.add
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

try:
    from langgraph.graph.message import add_messages
except ImportError:  # 兼容不同版本的位置
    from langgraph.graph import add_messages  # type: ignore


Intent = Literal["kb_qa", "troubleshoot", "action", "chitchat", "out_of_scope"]


class AgentState(TypedDict, total=False):
    # 对话历史，追加语义
    messages: Annotated[List[Any], add_messages]

    # 用户最初的输入，整个流程中不变
    user_input: str

    # 意图分类结果：决定走哪条分支
    intent: Intent

    # 计划步骤（由 planner 生成，executor 逐步消费）
    plan: List[str]
    current_step: int

    # 工具调用记录，追加语义
    observations: Annotated[List[Dict[str, Any]], operator.add]

    # 当前这一步选中的工具与参数
    next_tool: Optional[str]
    tool_args: Optional[Dict[str, Any]]

    # 循环控制：已执行轮数、是否还需要继续
    step_count: Annotated[int, operator.add]
    need_more_info: bool

    # 人工审批
    pending_action: Optional[Dict[str, Any]]
    approved: Optional[bool]

    # 终止状态
    final_answer: str
    error: Optional[str]


MAX_STEPS = 4          # 工具调用上限，防止 reflector 无限循环
MAX_RETRY_PER_STEP = 2  # 单步失败重试上限
