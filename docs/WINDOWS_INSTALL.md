# Windows 安装指南

本版本在 Windows 11 上开发与验证。Linux/macOS 未做完整验证。

## 1. 先决条件

- Windows 10/11 64 位
- Python 3.10–3.12（内部验证使用 3.12）
- 推荐 NVIDIA GPU（8 GB 显存为内部验证配置；CPU 推理未验证）
- 可选：Conda 或 venv
- 可选：Claude Code CLI（用于 MCP 接入）

## 2. 创建环境

```powershell
cd <EXPORT_DIR>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若使用 Conda：

```powershell
conda create -n yuzhu python=3.12 -y
conda activate yuzhu
pip install -r requirements.txt
```

## 3. 安装 PyTorch

`requirements.txt` 不强制指定 CUDA 版本。请按 PyTorch 官方命令安装与本机驱动
匹配的 `torch`。内部验证组合为：

```text
torch 2.10.0+cu130
transformers 5.17.0
peft 0.21.2
sentence-transformers 6.1.0
faiss-cpu 1.15.1
bm25s 0.3.13
```

这些是内部验证版本，不是对所有机器的硬性要求。

## 4. 目录布局

建议：

```text
<EXPORT_DIR>\
  yuzhu_lab\
  tests\
  examples\
  models\
    embedding\Qwen3-Embedding-0.6B\
    reranker\Qwen3-Reranker-0.6B\      (可选)
  .yuzhu\                              (运行时生成，不提交)
```

模型与 Adapter 均不随仓库分发。

## 5. 环境变量（PowerShell）

```powershell
$env:YUZHU_MODEL_DIR = "<path-to-MiniCPM5-2B-BF16>"
$env:YUZHU_ADAPTER_DIR = "<path-to-your-peft-adapter>"
$env:YUZHU_ADAPTER_SHA256 = "<adapter-weights-sha256>"
$env:YUZHU_DEVICE = "cuda:0"
$env:YUZHU_HOST = "127.0.0.1"
$env:YUZHU_PORT = "8802"
```

可选：

```powershell
$env:YUZHU_MAX_NEW_TOKENS = "2048"
$env:YUZHU_PROMPT_REJECT_TOKENS = "6144"
$env:YUZHU_MIN_FREE_GIB = "0.35"
```

不要在没有评估的情况下提高 `YUZHU_PROMPT_REJECT_TOKENS`；8 GB 设备上过大的
prompt 会导致显存不足。

## 6. 启动与检查

```powershell
python -m yuzhu_lab.runtime.cli --host 127.0.0.1 --port 8802
```

另一个终端：

```powershell
curl.exe http://127.0.0.1:8802/health
```

成功时返回 `status=ok`、`model_loaded=true`。如果 Adapter 缺失或哈希不匹配，
Runtime 会 fail-closed 并报告具体原因。

## 7. 运行测试

```powershell
python -m unittest discover -s tests -t . -p "test_*.py"
```

## 8. 常见问题

### `adapter directory is missing`

这是预期行为：本仓库不提供 P04.0 Adapter。请配置自备 Adapter，或只使用
RAG/Harness/MCP 对外部模型端点。

### `adapter SHA256 mismatch`

`YUZHU_ADAPTER_SHA256` 与 `adapter_model.safetensors` 不一致。核对文件来源与
哈希；不要关闭校验来“修复”。

### `faiss` / `bm25s` 导入失败

确认 `pip install -r requirements.txt` 在**当前**虚拟环境中完成，并检查
Python 位数（64 位）。

### MCP 客户端连不上

确认命令使用绝对路径、`--workspace` 存在、JSON-RPC 日志写到 stderr 而不是
stdout。示例见 [CLAUDE_CODE_MCP.md](CLAUDE_CODE_MCP.md)。