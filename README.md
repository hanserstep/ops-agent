# 运维排障 Agent（LangGraph + MCP）

一个有状态、可中断、工具调用受控的 Agent。

和「一个 prompt 加几个 function」的区别在于：它会在工具之间**循环**（信息不足就换方向再查）、
遇到危险操作会**挂起等人批准**、进程重启后能从断点**恢复**。这三件事用 if-else 链写不出来，
这才是 LangGraph 存在的理由。

## 为什么用 LangGraph

旧版（`multi-agent-code-review`）是「三个 Agent 并发 + 一个汇总」，本质是一次性的并行调用：
没有状态、没有循环、没有中断恢复。

真实排障不是这样的。查完磁盘发现是日志问题，得再去查日志；查完日志发现不够，
得回头查进程。这个「根据上一步结果决定下一步」的过程是个状态机，而且：

| 需求 | 实现 |
|---|---|
| 多轮：查完 A 发现要查 B | 条件边回环到 executor |
| 循环上限：防止无限重试 | `step_count >= MAX_STEPS` 强制收尾 |
| 中断：重启生产服务前要人确认 | `interrupt()` + checkpointer |
| 恢复：进程挂了能接着跑 | `SqliteSaver` 按 thread_id 持久化 |

## 架构

```
                        ┌──────────┐
   用户输入 ───────────▶ │  router  │ 意图分类
                        └────┬─────┘
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         chitchat        kb_answer      planner
              │            │  │            │
              │            │  └─ search_kb │ 拆步骤
              │            │               ▼
              │            │          ┌──────────┐
              │            │     ┌───▶│ executor │ 选工具/校验/执行
              │            │     │    └────┬─────┘
              │            │     │         │ 危险工具
              │            │     │         ▼
              │            │     │    ┌───────────┐
              │            │     │    │human_gate │ interrupt() 挂起
              │            │     │    └─────┬─────┘
              │            │     │          │ 批准/拒绝
              │            │     │          ▼
              │            │     │    ┌──────────┐
              │            │     └────│reflector │ 信息够不够
              │            │          └────┬─────┘
              └────────────┴───────────────┴──▶ answer ──▶ END
```

状态用 `TypedDict` + reducer：
`observations` 和 `step_count` 是**追加/累加**语义（多步排查不能丢前面的结论），
其余字段是覆盖语义（`intent`、`plan` 每次都重新判定）。

## 安全设计

Agent 能执行命令，这是它的价值也是它最大的风险。三条原则：

**1. 模型只能选脚本 ID，不能传命令。**

`run_diagnostic` 的参数是 `script_id`（enum，五个白名单值），不是命令行字符串。
命令写死在代码里、用 `subprocess.run(list, shell=False)` 执行，shell 不参与解析。
任何「把模型输出拼进 shell」的设计都是漏洞，在 prompt 里写多少遍「不要执行危险命令」都没用。

**2. 不可逆操作必须人工审批。**

`restart_service` 在注册表里标了 `dangerous=True`。executor 遇到它不执行，
先返回 `pending_action`，由 `human_gate` 节点调 `interrupt()` 挂起整张图，
状态写进 checkpointer。人可以过十分钟再批准，进程可以在这期间重启。

**3. 安全靠系统机制，不靠模型自觉。**

这里踩过一个坑：模型会出于风险考虑拒绝调用 `restart_service`，
转而在回答里写「建议人工确认」——听起来很对，但整个 interrupt 机制被绕过了。
所以 `choose_tool` 的 system prompt 里明确写了：**提出操作请求是你的职责，审批是系统的职责**。
模型过度谨慎和模型乱来一样危险，只是前者不容易被发现。

## 工具

| 工具 | 类型 | 说明 |
|---|---|---|
| `search_kb` | 只读 | 调项目 A 的 `/api/search`，拿原始片段而不是 LLM 的答案 |
| `query_metrics` | 只读 | 查主机指标时序，返回均值/峰值/是否超阈值 |
| `run_diagnostic` | 只读 | 执行白名单诊断脚本（df / du / 端口） |
| `restart_service` | **危险** | 需人工审批；演示模式只记录不真执行 |
| `create_ticket` | 写入 | 建工单，按「标题 + 1 小时」幂等去重 |

为什么工具粒度这么细：**工具粒度决定审批粒度**。如果把重启混在诊断工具里，
注册表层就无法区分「这次要不要审批」，只能靠模型判断——而模型不可信。

## 实测输出

三个场景都是真实跑的（DeepSeek + 项目 A 的 RAG 服务）。

**故障排查**：输入「db-01 磁盘告警了，帮我看看什么情况、怎么处理」

```
意图: troubleshoot   步数: 4
工具调用:
  - query_metrics (0.1ms) OK      → 均值 90.85%，峰值 103.49%，超阈值
  - run_diagnostic (17.7ms) OK    → df -h
  - run_diagnostic (24.6ms) OK    → du -sh /var/log
  - run_diagnostic (15.8ms) OK    → 端口
```

Agent 自己发现了一个矛盾：mock 指标说 90%，但本机 `df` 只有 17%，于是明确写
「**无法定论根因，不建议直接清理**」并列出三条待确认项。这比硬编一个原因强。

**危险操作**：输入「nginx 内存泄漏已排查清楚，现在执行重启」

```
[!] 等待人工审批: restart_service({'service': 'nginx', 'reason': '...'})
    风险: 重启服务会中断该服务上的所有连接，可能影响线上用户。

>>> 模拟运维人员批准执行 ...
工具调用: restart_service (0.0ms) OK
```

**一次授权 = 一次执行**。这里修过一个危险 bug：批准后本该收尾，但流程回到了
reflector，它认为「还没验证效果」，又要求重启一次——用户点一次批准，系统重启两次。
现在 `after_gate` 直接进 answer，并有回归测试守着（`TestNoDoubleExecution`）。

## MCP Server

同一批工具还能通过 MCP 协议暴露给任意客户端（Cursor / Claude Desktop / Windsurf）：

```json
{
  "mcpServers": {
    "ops-tools": {
      "command": "/path/to/python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "/path/to/ops-agent"
    }
  }
}
```

配好之后在 Cursor 里直接说「查一下 db-01 的磁盘」就能调到这里面的工具。

MCP 层**复用 registry.call 而不是重新实现**——参数校验、审计日志、错误处理只能有一份。
两套实现迟早会不一致，而「MCP 那边能执行的命令比 Agent 多一条」就是安全事故。

## 运行

```bash
# 依赖项目 A 的 RAG 服务在 127.0.0.1:8000
pip install -r requirements.txt
cp .env.example .env   # 填 DEEPSEEK_API_KEY

python scripts/demo.py              # 三个场景全跑
python scripts/demo.py troubleshoot # 只跑排障
pytest tests/ -v                    # 12 个用例，不依赖网络和 LLM
```

## 项目结构

```
app/
  state.py        AgentState 定义（reducer 语义）
  nodes.py        六个节点 + 四个条件边
  graph.py        StateGraph 组装、OPSAgent 封装（run / resume）
  llm.py          五个角色 prompt + function calling + FakeLLM
  tools/
    registry.py   注册表、schema 校验、统一执行入口
    kb_search.py  知识库检索（调项目 A）
    metrics.py    监控指标（enum 约束主机名和指标名）
    diagnostic.py 白名单脚本 + 危险操作
    ticket.py     工单（幂等去重）
  mcp_server.py   MCP Server（复用 registry）
tests/
  test_graph.py   12 个用例覆盖路由/循环/审批/失败/幂等
```

## 已知限制

- `run_diagnostic` 在沙箱环境里 `ps` 会被拒绝（`Operation not permitted`），
  工具失败会被正确记录并交回模型，不影响流程。生产实现应 SSH 到目标主机。
- `query_metrics` 是本地生成的模拟数据，和本机真实 `df` 输出对不上。
  工具返回里显式标注了 `executed_on: local-demo-host`，避免模型拿两边数据硬凑因果。
- `restart_service` 只记录不执行——一个能真的重启服务的 Agent 不该在演示环境里跑起来。
