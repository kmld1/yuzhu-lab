# 模型与 Adapter 获取

本仓库**不包含**任何模型权重、LoRA Adapter 或训练数据。以下说明区分“官方可获取
资产”与“本项目不公开发布的资产”。

本仓库的原创源码采用 Apache-2.0（见 `LICENSE`/`NOTICE`）。模型权重不适用本项目
许可证，必须按各自官方模型卡与 LICENSE 获取和使用。

## 1. MiniCPM5-2B 基座

- 模型：MiniCPM5-2B
- 发布方：OpenBMB
- 许可证：Apache-2.0（以官方模型卡与 LICENSE 为准）
- 获取：从 OpenBMB / Hugging Face / ModelScope 的**官方仓库**下载
- 本仓库不分发权重，也不提供镜像

下载后核对：

1. 官方模型卡中的文件清单与大小；
2. 官方 LICENSE 文本（Apache-2.0）；
3. 如官方提供 SHA256/校验和，逐一核对。

推荐目录：

```text
models/base/MiniCPM5-2B-BF16/
  config.json
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
  model-00000-of-00001.safetensors
  model.safetensors.index.json
```

## 2. P04.0 Adapter（不公开发布）

本项目内部使用的 P04.0 LoRA Adapter：

- **不存在公开下载**；
- 公开再分发许可未确认；
- 本仓库不包含权重、配置或合并模型；
- 不应从非官方渠道获取或分发。

没有 P04.0 Adapter 时，Runtime 不能加载内部验证的模型组合。你可以：

- 使用自备、你有权使用的 PEFT Adapter；
- 或只使用 RAG / Hindsight / Harness，并把模型端点指向外部兼容服务。

## 3. Runtime 的 Adapter 契约

Runtime 期望：

```text
<ADAPTER_DIR>/
  adapter_config.json
  adapter_model.safetensors
```

环境变量：

```powershell
$env:YUZHU_MODEL_DIR = "<MODEL_DIR>"
$env:YUZHU_ADAPTER_DIR = "<ADAPTER_DIR>"
$env:YUZHU_ADAPTER_SHA256 = "<SHA256_OF_ADAPTER_MODEL_SAFETENSORS>"
```

启动时行为：

- Adapter 目录不存在 → 明确报错并退出；
- `adapter_config.json`/`adapter_model.safetensors` 缺失 → 明确报错；
- `YUZHU_ADAPTER_SHA256` 不匹配 → fail-closed；
- 校验通过后才会加载模型。

`YUZHU_VERIFY_ADAPTER=0` 只用于受控调试；公开发布与生产使用不应关闭校验。

## 4. Embedding / Reranker

RAG 需要 embedding 模型；reranker 可选。

内部验证使用：

- `Qwen3-Embedding-0.6B`（Apache-2.0）
- `Qwen3-Reranker-0.6B`（Apache-2.0，可选）

它们同样不随本仓库分发。请从官方渠道获取并遵守其许可证。

## 5. 训练数据

训练数据不在本仓库中，也不得以本仓库名义再分发。内部数据涉及的上游许可证
（包括但不限于 ODC-BY 与 OpenRAIL 组件）需要单独法律审查；本版本不提供
数据再分发授权。

## 6. 不声称

- 不声称 P04.0 Adapter 可公开下载；
- 不声称 P04.2 已训练或存在；
- 不声称基座模型可以替代内部 Adapter 组合；
- 不声称本仓库提供模型再分发许可。