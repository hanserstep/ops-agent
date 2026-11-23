"""工具：监控指标查询。

真实环境里这个工具对接 Prometheus / Zabbix / 云监控的查询接口。
本项目用本地生成的时序数据模拟，接口形状保持一致：

    query_metrics(host, metric, window) -> {series: [{ts, value}], summary: {...}}

这样替换成真实后端时，只需要改 _fetch 的实现，Agent 侧不用动。

设计要点：
- **host 和 metric 都用 enum 约束**。模型很容易编造不存在的主机名
  （比如把 web-01 写成 web1），用 enum 把可选值钉死，参数校验层就能拦住。
- **返回里带 summary 而不只是原始序列**。模型不擅长从 60 个数字里读出趋势，
  直接给它「均值、峰值、是否超过阈值」比给原始数据可靠得多。
"""

import json
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

from .registry import register

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

HOSTS = ["web-01", "web-02", "db-01", "cache-01"]
METRICS = ["cpu_usage", "memory_usage", "disk_usage", "error_rate", "latency_p99"]

# 阈值：超过就在 summary 里标记，省得模型自己判断
THRESHOLDS = {
    "cpu_usage": 80.0,
    "memory_usage": 85.0,
    "disk_usage": 90.0,
    "error_rate": 5.0,
    "latency_p99": 500.0,
}


def _seed_for(host: str, metric: str) -> random.Random:
    """用 host+metric 做种子，保证同一台机器的数据每次一致（可复现）。"""
    return random.Random(hash(f"{host}:{metric}") & 0xFFFFFFFF)


def _generate(host: str, metric: str, points: int = 12) -> List[Dict]:
    rng = _seed_for(host, metric)
    base = {
        "cpu_usage": 45, "memory_usage": 60, "disk_usage": 55,
        "error_rate": 1.2, "latency_p99": 180,
    }[metric]

    # 让 db-01 的磁盘和 web-01 的内存明显偏高，演示时才有「故障」可查
    if host == "db-01" and metric == "disk_usage":
        base = 93
    if host == "web-01" and metric == "memory_usage":
        base = 88
    if host == "web-01" and metric == "error_rate":
        base = 7.5

    now = datetime.now()
    series = []
    for i in range(points, 0, -1):
        ts = now - timedelta(minutes=5 * i)
        val = base + rng.uniform(-base * 0.15, base * 0.15)
        series.append({
            "ts": ts.strftime("%H:%M"),
            "value": round(max(0.0, val), 2),
        })
    return series


@register(
    name="query_metrics",
    description="查询指定主机在最近一小时内的监控指标时序数据，"
                "返回均值、峰值以及是否超过告警阈值。用于判断是否存在资源瓶颈。",
    parameters={
        "type": "object",
        "properties": {
            "host": {
                "type": "string",
                "description": "主机名",
                "enum": HOSTS,
            },
            "metric": {
                "type": "string",
                "description": "指标名",
                "enum": METRICS,
            },
            "window": {
                "type": "string",
                "description": "时间窗口",
                "enum": ["15m", "1h", "6h"],
                "default": "1h",
            },
        },
        "required": ["host", "metric"],
    },
    tags=["readonly", "monitoring"],
)
def query_metrics(host: str, metric: str, window: str = "1h") -> Dict[str, Any]:
    points = {"15m": 3, "1h": 12, "6h": 24}.get(window, 12)
    series = _generate(host, metric, points)
    values = [p["value"] for p in series]
    threshold = THRESHOLDS.get(metric, 100.0)
    peak = max(values)

    return {
        "host": host,
        "metric": metric,
        "window": window,
        "unit": "%" if metric.endswith("usage") or metric == "error_rate" else "ms",
        "series": series,
        "summary": {
            "avg": round(sum(values) / len(values), 2),
            "peak": round(peak, 2),
            "threshold": threshold,
            "exceeded": peak > threshold,
            "trend": "rising" if values[-1] > values[0] * 1.1
                     else "falling" if values[-1] < values[0] * 0.9
                     else "stable",
        },
    }
