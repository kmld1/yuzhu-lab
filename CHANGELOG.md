# Changelog

## v1.0.0（source-only public release）

### Added

- 统一评测来源准入策略 `yuzhu_lab/source_policy.py`；
- 干净 RAG 索引生成与验证流程；
- 定向对抗安全测试（23 项）；
- Windows 安装、模型获取、Claude Code MCP、RAG/Hindsight/Harness 文档；
- 第三方许可证声明与安全策略。

### Changed

- RAG 索引构建、检索与 RagAdapter 统一执行来源策略；
- ContextBuilder 对来源与 citation 再次校验；
- Harness 文件/执行工具拒绝受保护评测路径；
- Hindsight recall/retain 增加来源与内容指纹检查；
- MCP RAG/Memory/Agent Task 与 Harness 使用同一策略；
- 包 `__version__` 与 Runtime 服务版本统一为 `1.0.0`。

### Fixed

- 相对 citation 在 workspace 检查中恒为真的缺陷；
- MCP `allowed_prefixes=()` 未排除评测资产的问题；
- MCP stdio 服务器内子进程继承 stdin 导致 `run_tests` 阻塞的问题；
- `nvidia-smi` 输出非 UTF-8 或为空时资源快照崩溃的问题（新增 2 个回归测试）；
- 旧索引污染无法仅靠查询过滤消除的问题（新增干净索引）。

### Security

- 缺失/伪造 provenance fail closed；
- 路径穿越、大小写/分隔符、junction、前缀碰撞覆盖；
- 执行工具仍非 OS 沙箱，默认关闭。

### License

- 项目原创源码采用 Apache-2.0；
- 新增官方 `LICENSE` 正文与 `NOTICE`；
- 移除预发布阶段被修改过附录的 `LICENSE-APACHE-2.0.txt`；
- 版权行：`Copyright 2026 kmld`（所有者已确认）。

### Not included

- 模型权重、P04.0 Adapter、P04.2 Adapter、训练数据、评测数据、凭据。