# License Decision

- 状态：项目所有者已选择 **Apache License 2.0**。
- 适用范围：**仅项目原创源码**。
- 版权行：`Copyright 2026 kmld`（项目所有者已确认）。
- 本文件是发布决策记录，不是法律意见。

## 1. 许可范围

Apache-2.0 适用于本仓库中的原创源码，包括 `yuzhu_lab/`、公开测试、文档与示例。
完整许可证正文见 [LICENSE](LICENSE)，归属声明见 [NOTICE](NOTICE)。

## 2. 不适用范围

以下资产不随本仓库分发，也不受本仓库 Apache-2.0 许可约束：

- MiniCPM5-2B 模型权重；
- Qwen3-Embedding / Qwen3-Reranker 权重；
- P04.0 LoRA Adapter 或任何其他 Adapter；
- 训练数据、评测数据、评测题目、评分标准或模型答案；
- 第三方依赖（通过包管理器单独获取，按各自许可证授权）。

## 3. 第三方依赖

本仓库不 vendoring 第三方源码。声明依赖的许可证标识见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)；具体义务以各上游 LICENSE
原文为准。

## 4. 模型权重

模型权重以官方模型卡与官方 LICENSE 为准。MiniCPM5-2B 与 Qwen3 系列权重由各自
发布方以 Apache-2.0 发布；本仓库不提供模型再分发许可，也不提供镜像。

## 5. P04.0 Adapter

P04.0 Adapter 是内部训练产物，公开再分发许可未确认：

```text
P04_0_ADAPTER_DISTRIBUTION = NOT_AUTHORIZED
```

本仓库不包含 Adapter 权重、配置或合并模型，也不存在公开下载。

## 6. 训练与评测数据

训练数据与评测数据不包含在本仓库中。内部数据涉及的上游许可证（包括但不限于
ODC-BY、OpenRAIL）需要单独审查；本仓库不授予任何数据再分发权利。

## 7. 商标

Apache-2.0 不授予项目名称、商标或服务标记的使用许可。OpenBMB、MiniCPM、Qwen、
Claude、GitHub 等名称与商标归其各自所有者。

## 8. 发布前确认状态

项目所有者已确认：

1. 对本次公开的原创源码拥有开源授权权利；
2. `NOTICE` 中的版权行为 `Copyright 2026 kmld`；
3. 发布范围为 `SOURCE_ONLY`，不包含模型权重、Adapter、训练数据与评测数据。

第三方声明与上游许可证原文仍应按 `THIRD_PARTY_NOTICES.md` 复核。