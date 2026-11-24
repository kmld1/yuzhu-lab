# Third-Party Notices

本仓库不 vendoring 第三方源码，也不分发任何第三方代码、模型权重或数据。以下
依赖通过 Python 包管理器单独安装，其许可证与归属由各自项目提供。版本列来自内部
验证锁文件，不是强制版本；许可证标识已通过已安装包的元数据核对，具体义务以各
上游 LICENSE 原文为准。

## 运行时依赖

| 包 | 验证版本 | 许可证（上游元数据） | 用途 |
|---|---|---|---|
| torch | 2.10.0+cu130 | BSD-3-Clause | 模型推理 |
| transformers | 5.17.0 | Apache-2.0 | 模型/分词器 |
| peft | 0.21.2 | Apache-2.0 | LoRA Adapter 加载 |
| fastapi | 0.163.1 | MIT | Runtime HTTP |
| uvicorn | 0.54.0 | BSD-3-Clause | ASGI 服务器 |
| starlette | 1.10.0 | BSD-3-Clause | FastAPI 依赖 |
| httpx | 0.28.1 | BSD-3-Clause | HTTP 客户端 |
| pydantic | 2.14.1 | MIT | 数据模型 |
| numpy | 2.5.2 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | 数值计算 |
| sentence-transformers | 6.1.0 | Apache-2.0 | Embedding / Reranker |
| faiss-cpu | 1.15.1 | MIT | 向量索引 |
| bm25s | 0.3.13 | MIT | 关键词检索 |

## 模型与权重（不随仓库分发）

| 资产 | 许可证（官方声明） | 分发状态 |
|---|---|---|
| MiniCPM5-2B | Apache-2.0 | 不包含；从 OpenBMB 官方渠道获取 |
| Qwen3-Embedding-0.6B | Apache-2.0 | 不包含；从官方渠道获取 |
| Qwen3-Reranker-0.6B | Apache-2.0 | 不包含；从官方渠道获取 |
| P04.0 LoRA Adapter | 未确认 | **不包含；不存在公开下载** |

模型许可证以官方模型卡与 LICENSE 原文为准；上表只是项目内记录。

## 训练与数据

训练相关依赖（如 unsloth、trl）与训练数据不包含在本仓库，也不在本公开源码的
运行依赖中。内部训练数据涉及的上游许可证（包括但不限于 ODC-BY、OpenRAIL）
需要单独法律审查；本仓库不授予任何数据再分发权利。

## 商标

OpenBMB、MiniCPM、Qwen、Claude、GitHub 等名称与商标归其各自所有者。本仓库
不使用其名称暗示官方背书。

## 免责声明

本文件基于包元数据与项目记录整理，不构成法律意见。发布范围已确认为
`SOURCE_ONLY`；上游许可证原文与分发边界应作为发布后维护的一部分持续复核。