"""工具：创建工单。

两个细节值得说：

**幂等**。Agent 的重试循环可能把同一个工单建好几次（executor 超时后
reflector 让它重试，但实际第一次已经写进去了）。所以按「标题 + 时间窗口」
做去重，一小时内同标题只建一张。这是 Agent 系统里特别容易踩的坑——
LLM 不知道上次调用是否成功，只会按指示再试一次。

**结构化字段**。priority 用 enum 钉死，不让模型自由发挥。
否则会出现「紧急」「critical」「P0」「特别急」这种五花八门的值，下游没法统计。
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict

from .registry import register

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "tickets.db"
)

PRIORITIES = ["P0", "P1", "P2", "P3"]
DEDUP_WINDOW_HOURS = 1


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            priority TEXT NOT NULL,
            detail TEXT,
            created_at TEXT NOT NULL,
            source TEXT DEFAULT 'agent'
        )
    """)
    conn.commit()
    return conn


@register(
    name="create_ticket",
    description="创建运维工单，记录本次排查的结论和需要后续跟进的事项。"
                "如果一小时内已经建过同标题的工单，不会重复创建。",
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "工单标题，简明描述问题，例如「db-01 磁盘使用率 93%」",
            },
            "priority": {
                "type": "string",
                "description": "优先级，P0 最高",
                "enum": PRIORITIES,
                "default": "P2",
            },
            "detail": {
                "type": "string",
                "description": "详细描述：现象、已排查的内容、建议的处理动作",
            },
        },
        "required": ["title"],
    },
    tags=["write", "ticketing"],
)
def create_ticket(title: str, priority: str = "P2", detail: str = "") -> Dict[str, Any]:
    title = title.strip()
    if not title:
        return {"ok": False, "error": "工单标题不能为空"}

    since = (datetime.now() - timedelta(hours=DEDUP_WINDOW_HOURS)).isoformat()
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT id FROM tickets WHERE title = ? AND created_at >= ?",
            (title, since),
        ).fetchone()
        if row:
            return {
                "ok": True,
                "deduplicated": True,
                "ticket_id": row[0],
                "message": f"一小时内已有同标题工单 #{row[0]}，未重复创建",
            }

        cur = conn.execute(
            "INSERT INTO tickets (title, priority, detail, created_at) VALUES (?, ?, ?, ?)",
            (title, priority, detail.strip(), datetime.now().isoformat()),
        )
        conn.commit()
        ticket_id = cur.lastrowid
    finally:
        conn.close()

    return {
        "ok": True,
        "deduplicated": False,
        "ticket_id": ticket_id,
        "priority": priority,
        "message": f"已创建工单 #{ticket_id}（{priority}）",
    }


def list_tickets(limit: int = 20) -> list:
    """给演示脚本和测试用，不注册成 Agent 工具（Agent 不需要查工单列表）。"""
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT id, title, priority, created_at FROM tickets ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {"id": r[0], "title": r[1], "priority": r[2], "created_at": r[3]}
        for r in rows
    ]
