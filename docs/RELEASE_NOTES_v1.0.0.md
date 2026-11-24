# Release Notes — YUZHU LAB v1.0.0

- 发布类型：`PUBLIC_SOURCE_ONLY`
- 许可证：Apache License 2.0（仅项目原创源码）
- 版权：Copyright 2026 kmld

## 1. 发布范围

YUZHU LAB 是本地 AI Coding Agent 开源研究项目。

本版本只发布经审核的原创源码、安装脚本/配置模板、测试、文档与合规示例：

```text
yuzhu_lab/         运行时、Harness、RAG、Hindsight 适配、MCP
tests/             公开回归与对抗测试
docs/              安装、模型获取、MCP、RAG/Harness、限制说明
examples/          只读示例与 MCP 配置模板
README.md / README_EN.md
LICENSE / NOTICE / THIRD_PARTY_NOTICES.md
SECURITY.md / CONTRIBUTING.md / CHANGELOG.md
pyproject.toml / requirements*.txt
```

## 2. 不包含

以下资产不随本版本分发：

- MiniCPM5-2B 模型权重；
- P04.0 LoRA Adapter（公开再分发未获授权，**不存在公开下载**）；
- P04.2 Adapter（**未训练**）；
- 训练数据、评测题目、评分标准、模型答案；
- RAG 索引、Hindsight 数据库、Claude Code 会话、审计日志、Checkpoint；
- API Token、密码、用户配置或其他凭据。

## 3. 主要变更

### 评测来源隔离（P0）

- 新增统一来源准入策略 `yuzhu_lab/source_policy.py`；
- 覆盖 RAG 摄入/检索、RagAdapter、ContextBuilder、Harness 文件/执行工具、
  Hindsight recall/retain、MCP；
- 修复相对 citation 在 workspace 中恒为真的缺陷；
- 修复 MCP `allowed_prefixes=()` 未排除评测资产的问题；
- 新增路径穿越、大小写/分隔符、junction、前缀碰撞、缺失/伪造 provenance、
  旧索引污染、Memory 泄漏与 Context 重新注入等对抗测试。

### 干净 RAG 索引

- 提供从旧索引过滤生成新索引的流程；
- 新索引不包含受保护 chunks；
- manifest 记录来源策略、排除数量与 SHA256；
- 旧索引保留为内部历史证据，不覆盖、不再用于评测。

### MCP / Harness

- 修复 MCP stdio 服务器内子进程继承 stdin 导致 `run_tests` 永久阻塞的问题；
- Windows 执行工具在子进程运行前将其加入独立 Job Object，并在任务结束时回收该 Job 中的后代。
  无法启动受控进程时会拒绝执行，无法确认清理时会停止后续执行。
- 内部链路验证：Claude Code → MCP → Harness → MiniCPM5 → `read_file`；
- 简单 Bug Fix 通过独立 Oracle 6/6，测试文件未被修改。

### 诊断健壮性

- 修复 `nvidia-smi` 输出非 UTF-8 或为空时 `full_snapshot()` 崩溃的问题；
- 覆盖 `resources.gpu_snapshot()`、`resources.gpu_processes()` 与 Runtime
  `_gpu_snapshot()`；
- 新增 2 个回归测试。

### 许可证

- 新增 `LICENSE`（Apache-2.0 官方正文，逐字未改）；
- 新增 `NOTICE`；
- 移除预发布阶段被修改过附录的旧许可证副本 `LICENSE-APACHE-2.0.txt`；
- `README.md`、`README_EN.md`、`pyproject.toml` 与 `CONTRIBUTING.md` 更新为
  Apache-2.0。

## 4. 历史内部验证摘要

```text
干净索引受保护 chunks    0
MCP 对抗读取             DENIED
C2 Oracle                6 / 6
```

这些结果来自早期内部研究环境。外部环境的验证范围见下一节。

## 5. 已知限制与明确不声称

```text
P04.2 training          NOT_AUTHORIZED / NOT_STARTED
GPU probe               FAILED
Independent heldout     NOT_RUN / NOT_EXISTS
SQLite core             0 / 10
SQLite extension        0 / 6
```

- 不声称优于 Qwen3.5-9B 或任何外部模型；
- 不声称存在独立 heldout、公开盲测或外部用户验证；
- 不声称 P04.2 已训练或存在 Adapter；
- 不声称复杂 SQLite 自主编码能力达标；
- **不提供 OS 级沙箱**；路径策略是进程内检查，不是内核级隔离；
- **不保证任意模型生成代码的安全执行**；执行工具默认关闭，只能在受控工作区
  显式启用；
- Runtime 当前要求本地模型目录与 PEFT Adapter，不能仅凭基座启动；
- 仅基座模式尚未实现，也未验证真实模型运行；
- 完整外部 MiniCPM5 与自备 Adapter 部署、全新 Windows 机器的 PyTorch/CUDA
  安装、真实 Hindsight 空数据库初始化及外部 Claude Code → MCP 端到端流程未验证；
- Linux 和 macOS 尚未完成安装与回归验证；
- 修复 RAG 污染前的历史结果不能作为干净评测证据，模型生成的变更须由用户审查。

## 6. 发布工程说明

公开 Git 历史始于 v1.0.0 的单一源码提交。源码 ZIP 与 v1.0.0 Tag 对应。

## 7. 升级注意

- 不要复用旧 RAG 索引进行评测；
- 重新构建索引会执行来源准入；
- 不要把旧评测结果当作干净成绩；
- 不要关闭 Adapter 哈希校验来“修复”启动失败；
- 不要通过修改 workspace 或索引绕过评测来源策略。