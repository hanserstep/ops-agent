"""工具注册表。

设计要点（面试必问：Agent 的工具怎么管）：

1. **工具是唯一的副作用出口**。LLM 只能输出「调哪个工具、传什么参数」，
   不能直接执行任何东西。所有动作都在这里收敛、校验、记录。

2. **dangerous 标记决定是否需要人工审批**。重启服务、清理磁盘这类
   不可逆操作必须在执行前 interrupt，等人类点确认。这是 Agent 能不能
   上生产的核心差别——demo 里 Agent 可以随便调工具，线上不行。

3. **参数用 JSON Schema 描述**，模型据此生成调用参数，我们也据此校验。
   模型生成的枚举值不可信，必须用 schema 里的 enum 约束住，
   否则它可能编出一个不存在的主机名。

4. **每个工具都要能被单独导入和测试**，不依赖 LLM。
   这样单测可以在不花 token 的情况下覆盖全部工具逻辑。
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("tools")


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]
    dangerous: bool = False
    # 危险操作的说明，审批时展示给人类看
    risk_note: str = ""
    tags: List[str] = field(default_factory=list)

    def to_openai_schema(self) -> Dict[str, Any]:
        """转成 OpenAI / DeepSeek 的 function calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_mcp_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }


REGISTRY: Dict[str, ToolSpec] = {}


def register(name: str, description: str, parameters: Dict[str, Any],
             dangerous: bool = False, risk_note: str = "", tags: Optional[List[str]] = None):
    def deco(fn: Callable) -> Callable:
        REGISTRY[name] = ToolSpec(
            name=name, description=description, parameters=parameters,
            func=fn, dangerous=dangerous, risk_note=risk_note, tags=tags or [],
        )
        return fn
    return deco


def get(name: str) -> Optional[ToolSpec]:
    return REGISTRY.get(name)


def all_specs() -> List[ToolSpec]:
    return list(REGISTRY.values())


def openai_tools() -> List[Dict]:
    return [s.to_openai_schema() for s in REGISTRY.values()]


def validate_args(spec: ToolSpec, args: Dict[str, Any]) -> Dict[str, Any]:
    """按 schema 校验并补默认值。

    只做基础校验（必填、类型、枚举）。不用 jsonschema 库是因为工具参数
    都很简单，而且错误信息想自己控制——模型看不懂标准 jsonschema 报错。
    """
    params = spec.parameters or {}
    props = params.get("properties", {})
    required = set(params.get("required", []))

    cleaned: Dict[str, Any] = {}
    for key in required:
        if key not in args or args[key] in (None, ""):
            raise ValueError(f"缺少必填参数: {key}")

    for key, value in args.items():
        if key not in props:
            continue  # 忽略模型多传的字段，不让它成为错误来源
        schema = props[key]
        expected = schema.get("type")
        if expected == "integer" and isinstance(value, str) and value.isdigit():
            value = int(value)
        if expected == "integer" and not isinstance(value, int):
            raise ValueError(f"参数 {key} 应为整数，收到 {type(value).__name__}")
        if expected == "string" and not isinstance(value, str):
            value = str(value)
        enum = schema.get("enum")
        if enum and value not in enum:
            raise ValueError(f"参数 {key} 必须是 {enum} 之一，收到 {value!r}")
        cleaned[key] = value

    # 补默认值
    for key, schema in props.items():
        if key not in cleaned and "default" in schema:
            cleaned[key] = schema["default"]

    return cleaned


def call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """统一执行入口：校验 → 执行 → 记录耗时 → 结构化返回。

    永远返回 dict 而不是抛异常给上层：Agent 循环里一个工具失败不应该
    让整个流程崩掉，而应该把错误信息交回给模型，让它换个思路。
    这是「可恢复」和「一崩到底」的区别。
    """
    spec = REGISTRY.get(name)
    if spec is None:
        return {"ok": False, "error": f"未知工具: {name}",
                "available": list(REGISTRY.keys())}

    started = time.perf_counter()
    try:
        clean = validate_args(spec, args or {})
        result = spec.func(**clean)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        log.info(json.dumps({"event": "tool_call", "tool": name,
                             "args": clean, "ok": True, "ms": elapsed},
                            ensure_ascii=False, default=str))
        return {"ok": True, "tool": name, "result": result, "elapsed_ms": elapsed}
    except Exception as e:
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        log.warning(json.dumps({"event": "tool_call", "tool": name,
                                "args": args, "ok": False, "error": str(e)},
                               ensure_ascii=False, default=str))
        return {"ok": False, "tool": name, "error": str(e), "elapsed_ms": elapsed}
