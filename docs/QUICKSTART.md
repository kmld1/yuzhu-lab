# Quick Start

本指南不依赖任何内部路径或私有资产。

## 1. 安装

```bash
git clone https://github.com/kmld1/yuzhu-lab.git
cd yuzhu-lab
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows cmd:        .venv\Scripts\activate.bat
# Linux/macOS:        source .venv/bin/activate
pip install -r requirements.txt
```

GPU 用户请根据 PyTorch 官方说明安装与 CUDA 匹配的 `torch`；本仓库不捆绑
CUDA 运行时。

## 2. 验证源码包

```bash
python -m unittest discover -s tests -t . -p "test_*.py"
```

这些测试不需要模型权重、Adapter、Hindsight 服务或 GPU。

## 3. 运行 Harness（只读示例）

`examples/demo.py` 是安全只读示例。

```bash
python -m yuzhu_lab.harness.agent.cli run \
  --workspace examples \
  --task "读取 demo.py，解释 add 函数。只使用 read_file。" \
  --runtime-url http://127.0.0.1:8802 \
  --checkpoint-db .yuzhu/agent_state.db \
  --audit-log .yuzhu/agent_audit.jsonl
```

没有本地 Runtime 时，可把 `--runtime-url` 指向任何兼容的
Anthropic/OpenAI 风格模型端点。Harness 默认关闭命令执行；需要时显式传
`--allow-execute`，并且只应在隔离工作区使用。

## 4. 构建 RAG 索引

RAG 需要一个 embedding 模型目录。默认路径为
`models/embedding/Qwen3-Embedding-0.6B`，可用 `--embedding` 覆盖。

```bash
python -m yuzhu_lab.rag.cli build \
  --root . \
  --store .yuzhu/rag_index \
  --embedding <path-to-Qwen3-Embedding-0.6B>

python -m yuzhu_lab.rag.cli search \
  "source admission policy" \
  --store .yuzhu/rag_index \
  --embedding <path-to-Qwen3-Embedding-0.6B>
```

索引构建默认执行来源准入：`eval/` 与明确的评测脚本不会进入索引。不要用
`--force` 或自定义脚本绕过该策略来处理含评测数据的目录。

## 5. 启动 MCP Server

```bash
python -m yuzhu_lab.harness.mcp_server \
  --workspace . \
  --runtime-url http://127.0.0.1:8802 \
  --hindsight-url http://127.0.0.1:8890 \
  --rag-store .yuzhu/rag_index \
  --embedding-model <path-to-Qwen3-Embedding-0.6B> \
  --state-dir .yuzhu/mcp_state
```

默认只暴露：

- `yuzhu_service_status`
- `yuzhu_rag_search`
- `yuzhu_memory_recall`
- `yuzhu_agent_task`

执行工具默认关闭；`--allow-execute` 仅在受控工作区使用。配置 Claude Code 见
[CLAUDE_CODE_MCP.md](CLAUDE_CODE_MCP.md)。

## 6. Runtime（可选）

Runtime 需要 MiniCPM5-2B 权重与一个 PEFT Adapter 目录：

```bash
# PowerShell
$env:YUZHU_MODEL_DIR = "<path-to-MiniCPM5-2B-BF16>"
$env:YUZHU_ADAPTER_DIR = "<path-to-your-peft-adapter>"
$env:YUZHU_ADAPTER_SHA256 = "<adapter-weights-sha256>"
python -m yuzhu_lab.runtime.cli --host 127.0.0.1 --port 8802
```

官方 P04.0 Adapter **不公开发布**；没有自备 Adapter 时 Runtime 会明确报告
Adapter 缺失并退出。模型获取与许可证说明见 [MODEL_SETUP.md](MODEL_SETUP.md)。

## 7. 常见检查

```bash
python -m yuzhu_lab.runtime.cli --help
python -m yuzhu_lab.harness.mcp_server --help
python -m yuzhu_lab.harness.agent.cli --help
python -m yuzhu_lab.rag.cli --help
```

如果 Runtime 启动失败，先检查：

1. `YUZHU_MODEL_DIR` 是否存在且包含模型文件；
2. `YUZHU_ADAPTER_DIR` 是否存在 `adapter_config.json` 与
   `adapter_model.safetensors`；
3. `YUZHU_ADAPTER_SHA256` 是否与权重一致；
4. CUDA 是否可用（CPU 推理不在本版本支持范围内）。

## 8. 许可证

本仓库原创源码采用 Apache License 2.0（见 `LICENSE` 与 `NOTICE`）。模型权重、
LoRA Adapter、训练数据与评测数据不随本仓库分发，也不适用本项目的 Apache-2.0
许可；请按各自官方许可证获取和使用。