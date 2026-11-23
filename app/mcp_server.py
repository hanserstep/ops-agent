"""MCP Server：把运维工具暴露给任意 MCP 客户端。

为什么单独做一层 MCP：

    LangGraph 里的工具是「给这个 Agent 用的」，绑定在流程里。
    MCP 是「给所有 AI 客户端用的」——配好之后，Cursor、Claude Desktop、
    Windsurf 都能直接调这里的工具，不需要知道 LangGraph 的存在。

    同一批工具两种暴露方式，是当前 AI 工程的常见形态：
    内部编排走框架（LangGraph 管状态和循环），外部复用走协议（MCP 管能力分发）。
    JD 里 MCP 出现频次已经到 6/15，但候选人里真正写过 Server 的很少。

启动方式（stdio，由客户端拉起）：
    python -m app.mcp_server

在 Cursor 的 mcp.json 里配置：
    {
      "mcpServers": {
        "ops-tools": {
          "command": "/path/to/python",
          "args": ["-m", "app.mcp_server"],
          "cwd": "/path/to/ops-agent"
        }
      }
    }

设计上刻意复用 registry.call 而不是重新实现：
    工具的参数校验、耗时记录、错误处理、审计日志只应该有一份实现。
    两套实现迟早会不一致——MCP 那边能执行的命令比 Agent 多一条，
    就是安全事故。
"""

import json
import logging
import sys
from typing import Any, Dict

from mcp.server.mcpserver import MCPServer

# 导入即注册
from .tools import diagnostic, kb_search, metrics, ticket  # noqa: F401
from .tools.registry import call as call_tool
from .tools.registry import get as get_spec

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger("mcp")

server = MCPServer("ops-tools")


def _wrap(tool_name: str, **kwargs) -> str:
    """统一包装：调用注册表里的工具，把结果转成 MCP 能返回的字符串。

    MCP 工具返回文本给客户端（也就是给另一个 LLM 看），所以格式化要
    兼顾机器可读和人可读——用缩进 JSON 而不是紧凑 JSON。
    """
    result = call_tool(tool_name, kwargs)
    log.info(json.dumps({"event": "mcp_tool_call", "tool": tool_name, "ok": result.get("ok")},
                        ensure_ascii=False))
    if not result.get("ok"):
        return json.dumps({"error": result.get("error")}, ensure_ascii=False, indent=2)
    return json.dumps(result.get("result"), ensure_ascii=False, indent=2)


@server.tool(description=get_spec("search_kb").description)
def search_kb(query: str, top_k: int = 3) -> str:
    """在运维知识库中检索历史故障处理方案。"""
    return _wrap("search_kb", query=query, top_k=top_k)


@server.tool(description=get_spec("query_metrics").description)
def query_metrics(host: str, metric: str, window: str = "1h") -> str:
    """查询主机监控指标。host 可选 web-01/web-02/db-01/cache-01。"""
    return _wrap("query_metrics", host=host, metric=metric, window=window)


@server.tool(description=get_spec("run_diagnostic").description)
def run_diagnostic(script_id: str) -> str:
    """执行只读诊断脚本。script_id 必须在白名单内。"""
    return _wrap("run_diagnostic", script_id=script_id)


@server.tool(description=get_spec("create_ticket").description)
def create_ticket(title: str, priority: str = "P2", detail: str = "") -> str:
    """创建运维工单。priority 可选 P0/P1/P2/P3。"""
    return _wrap("create_ticket", title=title, priority=priority, detail=detail)


@server.tool(description="列出当前所有可用工具的清单，用于排查客户端为什么看不到某个工具")
def list_tools() -> str:
    from .tools.registry import all_specs
    return json.dumps([
        {"name": s.name, "dangerous": s.dangerous,
         "description": s.description}
        for s in all_specs()
    ], ensure_ascii=False, indent=2)


def main() -> None:
    log.info("starting ops-tools MCP server (stdio)")
    server.run()


if __name__ == "__main__":
    main()
