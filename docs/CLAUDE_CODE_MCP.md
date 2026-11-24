# Claude Code MCP 配置

YUZHU LAB 通过 MCP stdio 暴露 Harness。推荐为项目使用**独立配置目录**，不要修改
全局 Claude Code 设置。

## 1. 启动前提

- 已安装 Python 依赖；
- 有一个可用的模型端点（本地 Runtime 或兼容 HTTP 服务）；
- RAG 可选；如启用，先构建干净索引；
- MCP 的执行工具默认关闭。

## 2. mcp_config.json

将 `<EXPORT_DIR>`、`<PYTHON_EXE>`、`<STATE_DIR>` 替换为真实路径：

```json
{
  "mcpServers": {
    "yuzhu": {
      "type": "stdio",
      "command": "<PYTHON_EXE>",
      "args": [
        "-m",
        "yuzhu_lab.harness.mcp_server",
        "--workspace",
        "<EXPORT_DIR>",
        "--project-id",
        "local-project",
        "--runtime-url",
        "http://127.0.0.1:8802",
        "--hindsight-url",
        "http://127.0.0.1:8890",
        "--rag-store",
        "<EXPORT_DIR>/.yuzhu/rag_index",
        "--embedding-model",
        "<EXPORT_DIR>/models/embedding/Qwen3-Embedding-0.6B",
        "--state-dir",
        "<STATE_DIR>"
      ],
      "env": {
        "PYTHONPATH": "<EXPORT_DIR>",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

说明：

- `--workspace` 必须存在，MCP 的所有文件工具都被限制在该目录内；
- `--rag-store`/`--embedding-model` 仅在启用 RAG 时需要；
- `--state-dir` 存放 Checkpoint 与审计日志，不要放进版本库；
- 不传 `--allow-execute` 时，`run_python`/`run_powershell`/`run_tests` 不可用；
- 不传 `--allow-retain` 时，不写入 Hindsight。

## 3. settings.json（最小权限示例）

```json
{
  "permissions": {
    "defaultMode": "manual",
    "allow": [
      "mcp__yuzhu__yuzhu_service_status",
      "mcp__yuzhu__yuzhu_agent_task"
    ],
    "deny": [
      "Bash",
      "PowerShell",
      "Read",
      "Write",
      "Edit",
      "Glob",
      "Grep",
      "WebFetch",
      "WebSearch",
      "mcp__yuzhu__yuzhu_rag_search",
      "mcp__yuzhu__yuzhu_memory_recall"
    ]
  }
}
```

按需放开 `yuzhu_rag_search` 或 `yuzhu_memory_recall`。不要使用
`bypassPermissions`。

## 4. 启动 Claude Code

```powershell
$env:CLAUDE_CONFIG_DIR = "<CONFIG_DIR>"
claude -p "请调用 yuzhu_agent_task 工具。任务：读取 demo.py，解释 add 函数。只使用 read_file。" `
  --output-format json `
  --mcp-config "<CONFIG_DIR>\mcp_config.json" `
  --strict-mcp-config `
  --settings "<CONFIG_DIR>\settings.json" `
  --permission-mode manual `
  --permission-prompts none `
  --allowedTools "mcp__yuzhu__yuzhu_agent_task,mcp__yuzhu__yuzhu_service_status" `
  --max-turns 6
```

## 5. 安全注意

- MCP 的 `deny` 不是操作系统权限；它只约束 Claude Code 的调用面。
- Harness 的路径策略是进程内策略，不是 OS 沙箱。
- 只有受控工作区才启用 `--allow-execute`；执行器可以运行工作区内的脚本，
  脚本本身仍拥有当前用户权限。
- 评测资产目录（`eval/`）和评测生成/评分脚本被来源策略拒绝，不应通过修改
  workspace 或索引绕过。
- 不要把 Token、密码或会话 JSONL 放在 workspace 或 state 目录。

## 6. 故障排查

服务端日志写 stderr；stdout 只承载 JSON-RPC。若 Claude Code 无响应：

1. 直接运行 `python -m yuzhu_lab.harness.mcp_server --help`；
2. 检查 `--workspace` 与 `--state-dir` 是否存在；
3. 检查模型端点 `/health`；
4. 查看 MCP 日志文件（如配置了 `--log-file`）。