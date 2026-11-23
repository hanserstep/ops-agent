"""工具：知识库检索。

为什么单独抽成一个工具而不是让 Agent 直接调 RAG 的 /api/qa：

    /api/qa 会走完整链路——检索 + 重排 + LLM 生成。Agent 用它等于
    「让一个 LLM 去问另一个 LLM」，token 翻倍、延迟翻倍，而且中间层
    的答案无法被 Agent 校验。

    检索工具只返回原始片段，由 Agent 自己决定怎么用。这样 Agent 拿到的是
    事实，而不是别人加工过的结论——多 Agent 系统里这一条很重要，
    否则错误会层层传递且无法定位。

对应项目 A 的 /api/search 接口（只检索不生成）。
"""

import os
from typing import Any, Dict, List

import httpx

from .registry import register

RAG_BASE = os.getenv("RAG_BASE_URL", "http://127.0.0.1:8000")
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        # trust_env=False：环境里注入了 HTTP_PROXY，本地请求不该走代理
        _client = httpx.Client(timeout=20.0, trust_env=False)
    return _client


@register(
    name="search_kb",
    description="在运维知识库中检索历史故障处理方案和技术文档。"
                "用于查找已知问题的标准处理流程、错误码含义、配置说明。",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索问题，用自然语言描述即可，例如「磁盘满了怎么清理」",
            },
            "top_k": {
                "type": "integer",
                "description": "返回片段数量",
                "default": 3,
            },
        },
        "required": ["query"],
    },
    tags=["readonly", "knowledge"],
)
def search_kb(query: str, top_k: int = 3) -> Dict[str, Any]:
    try:
        resp = _get_client().post(
            f"{RAG_BASE}/api/search",
            json={"question": query, "top_k": max(1, min(top_k, 10)), "rerank_top_n": max(1, min(top_k, 10))},
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return {
            "found": False,
            "error": f"知识库服务不可用: {e}",
            "hint": "可以先跳过知识库，直接从监控数据排查",
            "snippets": [],
        }

    hits: List[Dict] = data.get("hits", [])
    return {
        "found": bool(hits),
        "count": len(hits),
        "snippets": [
            {
                "doc_id": h.get("doc_id", ""),
                "content": (h.get("content") or "")[:400],
                "score": round(float(h.get("score", 0)), 4),
            }
            for h in hits
        ],
    }
