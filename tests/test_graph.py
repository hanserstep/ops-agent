"""图流程测试。

用 FakeLLM + 内存 checkpointer + mock 过的工具，覆盖所有分支：
正常排查、知识问答、闲聊、危险操作审批（批准/拒绝）、步数上限。

这些用例的价值在于：Agent 系统的 bug 几乎都在分支边界上——
循环没退出、审批被绕过、工具失败后无限重试。用真实 LLM 测这些
既不稳定又贵，FakeLLM 能精确构造每一个边界。
"""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.graph import build_graph
from app.llm import FakeLLM
from app.state import MAX_STEPS
from app.tools import registry

# 触发工具注册
import app.tools.kb_search  # noqa: F401
import app.tools.metrics    # noqa: F401
import app.tools.diagnostic  # noqa: F401
import app.tools.ticket     # noqa: F401


@pytest.fixture
def patched_tools(monkeypatch):
    """把工具执行替换成固定返回，避免真调 HTTP 和 subprocess。"""
    def fake_call(name, args):
        if name == "query_metrics":
            return {"ok": True, "tool": name, "elapsed_ms": 1.0, "result": {
                "host": args.get("host"), "metric": args.get("metric"),
                "summary": {"peak": 93.0, "exceeded": True},
            }}
        if name == "search_kb":
            return {"ok": True, "tool": name, "elapsed_ms": 1.0, "result": {
                "found": True, "count": 1,
                "snippets": [{"content": "磁盘清理标准流程：先查大文件再删归档"}]}}
        if name == "restart_service":
            return {"ok": True, "tool": name, "elapsed_ms": 1.0,
                    "result": {"simulated": True, "service": args.get("service")}}
        if name == "run_diagnostic":
            return {"ok": True, "tool": name, "elapsed_ms": 1.0,
                    "result": {"output": "/dev/disk1s5  93%"}}
        return {"ok": False, "tool": name, "error": "not mocked"}
    monkeypatch.setattr(registry, "call", fake_call)


def make_agent(llm):
    return build_graph(llm, checkpointer=MemorySaver())


def run(graph, question, thread="t1"):
    return graph.invoke(
        {"user_input": question, "messages": [{"role": "user", "content": question}],
         "observations": [], "step_count": 0, "current_step": 0, "plan": []},
        config={"configurable": {"thread_id": thread}},
    )


class TestRouting:

    def test_chitchat_short_circuits(self):
        llm = FakeLLM(intents=[("chitchat", "打招呼")])
        out = run(make_agent(llm), "你好")
        assert out["intent"] == "chitchat"
        assert "运维助手" in out["final_answer"]
        # 闲聊不应该调任何工具
        assert out["observations"] == []

    def test_kb_qa_hits_knowledge_base(self, patched_tools):
        llm = FakeLLM(intents=[("kb_qa", "问流程")],
                      answers=["按标准流程先查大文件再删归档 [search_kb]"])
        out = run(make_agent(llm), "磁盘满了怎么清理")
        assert out["intent"] == "kb_qa"
        assert out["observations"][0]["tool"] == "search_kb"
        assert "归档" in out["final_answer"]

    def test_troubleshoot_plans_first(self, patched_tools):
        llm = FakeLLM(
            intents=[("troubleshoot", "要看实时数据")],
            plans=[["查磁盘使用率", "查大文件"]],
            tool_choices=[{"name": "query_metrics",
                           "args": {"host": "db-01", "metric": "disk_usage"}}],
            reflections=[{"answer_ready": True, "reason": "已确认磁盘超阈值"}],
            answers=["db-01 磁盘 93%，超过 90% 阈值"],
        )
        out = run(make_agent(llm), "db-01 好像有问题")
        assert out["intent"] == "troubleshoot"
        assert out["plan"] == ["查磁盘使用率", "查大文件"]
        assert out["observations"][0]["tool"] == "query_metrics"
        assert "93%" in out["final_answer"]


class TestLoopControl:

    def test_stops_when_model_says_enough(self, patched_tools):
        """reflector 判断信息足够后必须停下，不能继续循环。"""
        llm = FakeLLM(
            intents=[("troubleshoot", "")],
            plans=[["查监控"]],
            tool_choices=[{"name": "query_metrics", "args": {"host": "web-01", "metric": "cpu_usage"}}],
            reflections=[{"answer_ready": True, "reason": "够了"}],
            answers=["CPU 正常"],
        )
        out = run(make_agent(llm), "web-01 CPU 高吗")
        assert out["step_count"] == 1
        assert out["final_answer"] == "CPU 正常"

    def test_max_steps_forces_exit(self, patched_tools):
        """模型一直说'还不够'时，必须被 MAX_STEPS 强制收尾，否则无限循环。"""
        llm = FakeLLM(
            intents=[("troubleshoot", "")],
            plans=[["一直查"]],
            tool_choices=[{"name": "query_metrics",
                           "args": {"host": "web-01", "metric": "cpu_usage"}}] * (MAX_STEPS + 3),
            reflections=[{"answer_ready": False, "reason": "还不够"}] * (MAX_STEPS + 3),
            answers=["信息不足，建议人工介入"],
        )
        out = run(make_agent(llm), "查不出来")
        assert out["step_count"] == MAX_STEPS
        assert out["final_answer"] == "信息不足，建议人工介入"

    def test_no_tool_choice_ends_early(self, patched_tools):
        """模型选择不调工具时，不应进入空转的重试循环。"""
        llm = FakeLLM(
            intents=[("troubleshoot", "")],
            plans=[["看看情况"]],
            tool_choices=[None],
            reflections=[{"answer_ready": True, "reason": "无需工具"}],
            answers=["暂时没有异常"],
        )
        out = run(make_agent(llm), "现在系统怎么样")
        assert out["step_count"] == 0
        assert out["final_answer"] == "暂时没有异常"


class TestApproval:

    def _dangerous_agent(self):
        llm = FakeLLM(
            intents=[("action", "要求重启")],
            plans=[["确认服务状态", "重启服务"]],
            tool_choices=[{"name": "restart_service",
                           "args": {"service": "nginx", "reason": "内存泄漏"}}],
            reflections=[{"answer_ready": True, "reason": "已处理"}],
            answers=["已重启 nginx"],
        )
        return llm

    def test_dangerous_tool_suspends(self, patched_tools):
        """危险操作必须挂起等审批，不能直接执行。"""
        graph = make_agent(self._dangerous_agent())
        out = run(graph, "重启一下 nginx")
        assert out.get("pending_action"), "应当挂起等待审批"
        assert out["pending_action"]["tool"] == "restart_service"
        # 挂起阶段不能产生执行结果
        assert not any(o.get("tool") == "restart_service" and o.get("result")
                       for o in out.get("observations", []))

    def test_approve_executes(self, patched_tools):
        graph = make_agent(self._dangerous_agent())
        run(graph, "重启一下 nginx", thread="ap1")
        from langgraph.types import Command
        out = graph.invoke(Command(resume={"approved": True}),
                           config={"configurable": {"thread_id": "ap1"}})
        executed = [o for o in out["observations"]
                    if o.get("tool") == "restart_service" and o.get("result")]
        assert executed, "批准后应当真正执行"
        assert out.get("pending_action") is None

    def test_reject_cancels(self, patched_tools):
        graph = make_agent(self._dangerous_agent())
        run(graph, "重启一下 nginx", thread="ap2")
        from langgraph.types import Command
        out = graph.invoke(Command(resume={"approved": False}),
                           config={"configurable": {"thread_id": "ap2"}})
        assert out["approved"] is False
        assert any("拒绝" in str(o.get("error", "")) for o in out["observations"])
        # 拒绝后不应该真的有执行结果
        assert not any(o.get("tool") == "restart_service" and o.get("result")
                       for o in out["observations"])


class TestToolFailure:

    def test_tool_error_is_recorded_not_raised(self, monkeypatch):
        """工具失败要把错误交回给模型，而不是让整条流程崩掉。"""
        def boom(name, args):
            return {"ok": False, "tool": name, "error": "连接超时", "elapsed_ms": 5.0}
        monkeypatch.setattr(registry, "call", boom)

        llm = FakeLLM(
            intents=[("troubleshoot", "")],
            plans=[["查监控"]],
            tool_choices=[{"name": "query_metrics",
                           "args": {"host": "web-01", "metric": "cpu_usage"}}],
            reflections=[{"answer_ready": True, "reason": "工具不可用，先收尾"}],
            answers=["监控接口不可用，建议检查采集端"],
        )
        out = run(make_agent(llm), "查一下 web-01")
        assert out["observations"][0]["error"] == "连接超时"
        assert out["final_answer"]


class TestStateAccumulation:

    def test_observations_accumulate(self, patched_tools):
        """observations 必须是追加语义，否则多步排查会丢掉前面的结论。"""
        llm = FakeLLM(
            intents=[("troubleshoot", "")],
            plans=[["查磁盘", "查大文件"]],
            tool_choices=[
                {"name": "query_metrics", "args": {"host": "db-01", "metric": "disk_usage"}},
                {"name": "run_diagnostic", "args": {"script_id": "disk_usage"}},
            ],
            reflections=[{"answer_ready": False, "reason": "还要看大文件"},
                         {"answer_ready": True, "reason": "够了"}],
            answers=["磁盘 93%，需要清理"],
        )
        out = run(make_agent(llm), "db-01 磁盘告警")
        tools_called = [o["tool"] for o in out["observations"]]
        assert "query_metrics" in tools_called
        assert "run_diagnostic" in tools_called
        assert len(tools_called) >= 2


class TestNoDoubleExecution:
    """危险操作只能执行一次。

    曾经出现过的 bug：批准 → 执行 → reflector 认为「还没验证」→
    再次请求同一个危险工具 → 第二次审批。用户点一次批准，系统重启两次。
    """

    def test_approved_action_runs_exactly_once(self, monkeypatch):
        calls = []

        def counting_call(name, args):
            calls.append(name)
            if name == "restart_service":
                return {"ok": True, "tool": name, "elapsed_ms": 1.0,
                        "result": {"simulated": True, "service": args.get("service")}}
            if name == "query_metrics":
                return {"ok": True, "tool": name, "elapsed_ms": 1.0,
                        "result": {"summary": {"peak": 98.0, "exceeded": True}}}
            return {"ok": True, "tool": name, "elapsed_ms": 1.0, "result": {}}

        monkeypatch.setattr(registry, "call", counting_call)

        llm = FakeLLM(
            intents=[("action", "要求重启")],
            plans=[["确认状态", "重启"]],
            tool_choices=[
                {"name": "query_metrics", "args": {"host": "web-01", "metric": "memory_usage"}},
                {"name": "restart_service", "args": {"service": "nginx", "reason": "内存泄漏"}},
            ],
            reflections=[{"answer_ready": False, "reason": "还没重启"}],
            answers=["已重启 nginx，内存已回落"],
        )
        graph = build_graph(llm, checkpointer=MemorySaver())
        run(graph, "重启 nginx", thread="once1")

        from langgraph.types import Command
        out = graph.invoke(Command(resume={"approved": True}),
                           config={"configurable": {"thread_id": "once1"}})

        assert calls.count("restart_service") == 1, f"重启执行了 {calls.count('restart_service')} 次"
        assert out["final_answer"]
        assert not out.get("pending_action"), "执行完不应再次挂起"
