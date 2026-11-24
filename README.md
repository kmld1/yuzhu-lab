# YUZHU LAB

[English](README_EN.md) | 简体中文

**仓库**：<https://github.com/kmld1/yuzhu-lab>

YUZHU LAB 是一个面向**本地小模型工程化**的实验性运行时与 Agent Harness：

- 本地 MiniCPM5-2B 推理服务（OpenAI / Anthropic 兼容 HTTP）
- Hybrid RAG：结构感知切分、FAISS + BM25S、RRF 融合、citation
- Hindsight 长期记忆适配（可选）
- 可复现的单 Agent Harness：工具调用、权限边界、完成门禁、Checkpoint/Resume
- MCP stdio Server，可接入 Claude Code
- **评测来源隔离**：冻结评测题目、标准与答案不进入 RAG、Context、Memory 或 MCP 工具返回

> 版本：`v1.0.0`（Source-Only 公开发布版本）
> 状态：**PUBLIC_SOURCE_ONLY**。本仓库只发布源码，不包含模型权重、LoRA Adapter、
> 训练数据、评测数据或任何凭据。

## 不包含什么

以下资产**不在**本仓库中，也不得从本仓库获得：

- MiniCPM5-2B 权重；
- 内部 P04.0 LoRA Adapter（公开再分发许可未确认，**不存在公开下载**）；
- 训练数据、评测题目、评分标准、模型答案；
- Hindsight 数据库、Claude Code 会话、审计日志、Checkpoint；
- API Token、密码或用户配置。

官方 MiniCPM5-2B 权重由 OpenBMB 以 Apache-2.0 发布，请从官方渠道获取并自行核对
许可证。说明见 [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md)。

## 能力边界

| 组件 | 是否需要模型/Adapter | 说明 |
|---|---|---|
| RAG | 否 | 可独立构建与检索，强制来源准入 |
| Hindsight 适配 | 否 | 可选；无服务时自动降级 |
| Harness 工具/权限/Checkpoint | 否 | 可用脚本化模型或兼容 HTTP 端点 |
| MCP Server | 取决于任务 | 只读/文件任务需要模型端点 |
| Runtime | 是 | 当前源码要求本地模型目录与 PEFT Adapter 目录 |

**重要：** 本版本的 Runtime 加载器要求一个 PEFT Adapter 目录。仓库不提供 P04.0
Adapter，也不提供可公开下载的替代 Adapter。没有自备 Adapter 时：

- 可以运行 RAG、Hindsight 适配、Harness 单元/集成测试；
- 可以把 Harness/MCP 指向任何兼容的模型 HTTP 端点（`--runtime-url`）；
- 不能直接用本仓库的 Runtime 启动 MiniCPM5 基座模型。

仅基座模式未实现，也未独立验证：`BASE_ONLY_REAL_MODEL_VALIDATION = NOT_VERIFIED`。

## 快速开始

```bash
git clone https://github.com/kmld1/yuzhu-lab.git
cd yuzhu-lab
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 运行包含的测试（不需要模型或 GPU）
python -m unittest discover -s tests -t . -p "test_*.py"
```

构建一个干净的本地 RAG 索引（需要 embedding 模型，见下方）：

```bash
python -m yuzhu_lab.rag.cli build --root . --store .yuzhu/rag_index
python -m yuzhu_lab.rag.cli search "Hybrid retrieval" --store .yuzhu/rag_index
```

运行 Harness（示例使用本地 Runtime 8802）：

```bash
python -m yuzhu_lab.harness.agent.cli run \
  --workspace examples \
  --task "读取 demo.py，解释 add 函数。只使用 read_file。" \
  --runtime-url http://127.0.0.1:8802
```

启动 MCP Server（供 Claude Code 使用）：

```bash
python -m yuzhu_lab.harness.mcp_server \
  --workspace . \
  --runtime-url http://127.0.0.1:8802 \
  --state-dir .yuzhu/mcp_state
```

完整步骤见 [docs/QUICKSTART.md](docs/QUICKSTART.md)、
[docs/WINDOWS_INSTALL.md](docs/WINDOWS_INSTALL.md)、
[docs/CLAUDE_CODE_MCP.md](docs/CLAUDE_CODE_MCP.md)。

## 评测来源隔离

Harness 的 RAG、文件工具、ContextBuilder、Hindsight 适配与 MCP 共用
`yuzhu_lab/source_policy.py`：

- `eval/` 与明确的评测生成/评分脚本是不可准入来源；
- 路径经过规范化、workspace 包含、reparse point 与存在性检查；
- 缺失或无法确认的 citation/来源一律拒绝（fail closed）；
- 新索引只摄入准入来源，并在检索时再次执行策略；
- 正常项目源码、Markdown 与工程测试文件不受影响。

细节见 [docs/RAG_HINDSIGHT_HARNESS.md](docs/RAG_HINDSIGHT_HARNESS.md) 与
[SECURITY.md](SECURITY.md)。

## 文档

- [Quick Start](docs/QUICKSTART.md)
- [Windows 安装](docs/WINDOWS_INSTALL.md)
- [模型与 Adapter 获取](docs/MODEL_SETUP.md)
- [Claude Code MCP 配置](docs/CLAUDE_CODE_MCP.md)
- [RAG / Hindsight / Harness](docs/RAG_HINDSIGHT_HARNESS.md)
- [已知限制](docs/KNOWN_LIMITATIONS.md)
- [发布说明](docs/RELEASE_NOTES_v1.0.0.md)
- [第三方声明](THIRD_PARTY_NOTICES.md)
- [安全策略](SECURITY.md)
- [贡献指南](CONTRIBUTING.md)

## 许可证

本仓库的**原创源码**采用 [Apache License 2.0](LICENSE) 许可，归属声明见
[NOTICE](NOTICE) 与 [LICENSE_DECISION.md](LICENSE_DECISION.md)。

Copyright 2026 kmld。

该许可不适用于模型权重、LoRA Adapter、训练数据、评测数据或第三方依赖；
这些资产均不随本仓库分发。第三方依赖的许可证见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

Windows 执行工具使用独立 Job Object 管理受控的子进程树。无法确认进程清理时，当前任务失败，后续执行被阻止。

## 外部环境验证范围

真实 MiniCPM5 模型与自备 Adapter 的完整外部部署、全新 Windows 机器的 PyTorch/CUDA
安装、真实 Hindsight 空数据库初始化及外部 Claude Code → MCP 端到端流程均未独立验证。
Linux 和 macOS 尚未完成安装与回归验证。

## 明确不声称

- 不声称优于任何外部模型；
- 不声称经过独立 heldout 或公开盲测；
- 不声称具备 OS 级沙箱；
- 不声称任意模型生成的代码可以安全执行；
- 不声称外部用户已独立验证本版本；文档中的测试结果是内部验证结果；
- 不声称 P04.2 已训练（`NOT_AUTHORIZED / NOT_STARTED`）；
- 不声称复杂 SQLite 自主编码能力达标（历史结果：SQLite core 0/10、
  SQLite extension 0/6；简单 Python Bug Fix 独立 Oracle 6/6。评测集与运行条件
  不同，不能直接比较）；
- 不声称旧评测结果是干净基准（历史 RAG 索引曾包含评测资产，缺陷已修复；
  修复前结果不得作为干净成绩）；
- 不声称 P04.0 Adapter 可公开分发；
- 不声称工具执行环境是 OS 级沙箱；工具调用受授权与路径策略限制，用户必须
  审查模型生成的变更，不能保证复杂任务输出正确；
- 不声称所有第三方依赖或服务都可离线使用；部分功能需要额外组件、网络或配置。