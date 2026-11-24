# 已知限制

本文件随发布保留；任何对外描述都不应删除这些限制。

## 1. 模型与训练

```text
P04.2 training          NOT_AUTHORIZED / NOT_STARTED
GPU probe               FAILED
Independent heldout     NOT_RUN / NOT_EXISTS
```

- 本仓库不包含 P04.0 Adapter，也不存在公开下载；
- 不声称 P04.2 已训练或存在 Adapter；
- 不声称有独立 heldout、公开盲测或优于任何外部模型。

## 2. 编码能力

内部 P04.0 组合在简单代码阅读与短小 Bug Fix 上可用；复杂 SQLite 事务任务失败：

```text
SQLite core       0 / 10
SQLite extension  0 / 6
```

这些历史成绩保留，不因其他测试通过而改写。模型工具调用纠错弱、长任务容易重复
失败或超时。

## 3. 评测完整性

- 历史 RAG 索引曾包含评测资产；该缺陷已修复，并建立了新的干净索引；
- **修复前的评测结果不能视为未接触评测来源的干净成绩**；
- 当前没有真正独立的 heldout；
- 旧索引保留为历史证据，但不应继续用于模型评测；
- 来源策略是进程内策略，不是 OS 沙箱。

## 4. 安全

- 无 OS 级隔离；恶意代码仍拥有当前用户权限；
- 执行工具默认关闭，只能在受控工作区启用；
- MCP 的客户端 deny 不是 OS 权限；
- 凭据迁移、ACL 收紧、历史会话日志整改属于部署方责任；
- 本仓库不包含任何 Token、密码或会话数据。

## 5. 资源

- 内部验证为 8 GB 笔记本 GPU 单实例；
- prompt 过大时 Runtime 会拒绝（默认 `prompt_reject_tokens=6144`）；
- 不做并发压力或多小时稳定性承诺；
- 外部 GPU 占用会显著降低生成速度，甚至导致请求失败；
- 单次输出上限默认 2048 tokens。

## 6. 功能

- Hindsight 是可选外部服务；无服务时降级；
- 全局记忆 Bank 可能不存在，按缺失处理；
- Reranker 可选，默认 RRF；
- 未验证多小时连续运行、并发或多用户；
- Runtime 当前要求 PEFT Adapter，不支持“仅基座”启动；
- Linux/macOS 未做完整验证。

## 7. 许可证

- 项目原创源码采用 Apache License 2.0（见 `LICENSE` 与 `NOTICE`）；
- `NOTICE` 的版权行为 `Copyright 2026 kmld`；
- 模型权重、Adapter、训练数据、评测数据不随仓库分发；
- P04.0 Adapter 公开再分发未获授权；
- 上游数据许可证（ODC-BY/OpenRAIL 等）需单独审查。

## 8. 禁止性声明

- 不声称优于 Qwen3.5-9B 或任何外部模型；
- 不声称独立盲测；
- 不声称 OS 级沙箱；
- 不声称 P04.2 已训练；
- 不声称凭据已轮换；
- 不声称 SQLite 能力达标；
- 不把工程参考实现算作模型自主成绩。