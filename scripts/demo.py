"""端到端演示：三个场景，覆盖三类路径。

    python scripts/demo.py              # 全部跑
    python scripts/demo.py kb           # 只跑知识问答
    python scripts/demo.py approve      # 只跑需要审批的场景

依赖：项目 A 的 RAG 服务在 127.0.0.1:8000 运行，.env 里有 DEEPSEEK_API_KEY。

演示的价值在于把「流程图」变成「实际输出」：面试时直接讲
「用户说 db-01 磁盘告警，Agent 调了两次工具、拿到 93% 的数据、
生成了带依据的结论」比画图有说服力得多。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_env(path: str = ".env") -> None:
    """极简 .env 加载，避免为一行配置引入 python-dotenv。"""
    if not os.path.exists(path):
        return
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.split(" #")[0].strip().strip('"').strip("'"))


load_env()

from app.graph import OpsAgent
from app.llm import DeepSeekLLM


def banner(text: str):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def show(result: dict):
    print(f"\n意图: {result['intent']}    步数: {result['steps']}")
    if result.get("plan"):
        print("计划:")
        for i, s in enumerate(result["plan"], 1):
            print(f"  {i}. {s}")
    if result.get("observations"):
        print("工具调用:")
        for o in result["observations"]:
            status = "OK" if "result" in o else f"失败: {o.get('error')}"
            print(f"  - {o.get('tool')} ({o.get('elapsed_ms', 0)}ms) {status}")
    if result.get("awaiting_approval"):
        pa = result["pending_action"]
        print(f"\n[!] 等待人工审批: {pa['tool']}({pa['args']})")
        print(f"    风险: {pa.get('risk', '')}")
    if result.get("answer"):
        print(f"\n回答:\n{result['answer']}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    llm = DeepSeekLLM()
    agent = OpsAgent(llm=llm)

    try:
        if which in ("all", "kb"):
            banner("场景 1：知识问答（走 kb_qa 分支，不调实时数据）")
            r = agent.run("磁盘写满了应该按什么顺序清理才安全？", thread_id="demo-kb")
            show(r)

        if which in ("all", "troubleshoot"):
            banner("场景 2：故障排查（planner → executor → reflector 循环）")
            r = agent.run(
                "db-01 磁盘告警了，帮我看看是什么情况、怎么处理",
                thread_id="demo-ts",
            )
            show(r)

        if which in ("all", "approve"):
            banner("场景 3：危险操作（触发 interrupt，等人工确认）")
            r = agent.run(
                "nginx 内存泄漏的事已经排查清楚了，现在执行重启操作：重启 nginx",
                thread_id="demo-ap",
            )
            show(r)

            if r.get("awaiting_approval"):
                print("\n>>> 模拟运维人员批准执行 ...")
                r2 = agent.resume(thread_id="demo-ap", approved=True, approved_by="demo")
                show(r2)
            else:
                print("\n（本次模型判断信息不足，未直接执行；这本身是期望行为）")

        print(f"\n[统计] LLM 调用 {llm.calls} 次，累计 {llm.tokens} tokens")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
