# RAG / Hindsight / Harness 使用指南

## 1. RAG

### 1.1 架构

```text
文件发现 → 结构感知切分 → 来源准入 → 去重 → embedding → FAISS
                                      ↘ BM25S
查询 → FAISS + BM25S → RRF 融合 → (可选 reranker) → token budget → citation
```

- 解析/切分：Python AST 与 Markdown 标题/行号；
- 向量：sentence-transformers embedding；
- 关键词：BM25S + 技术 token 处理；
- 融合：RRF（不直接相加 FAISS 与 BM25 分数）；
- reranker：可选，只重排候选集，不扫描全库；
- citation：`path:start-end (symbol)`。

### 1.2 来源准入

实现：`yuzhu_lab/source_policy.py`。

规则：

- `eval/` 目录与明确的评测生成/评分脚本是受保护来源；
- 来源必须是 workspace 内、可解析、真实存在的文件；
- 拒绝 `..`、绝对路径逃逸、UNC/设备路径、ADS、junction/symlink 逃逸；
- 拒绝缺失或伪造的 citation；
- 前缀按路径段匹配，`evaluation_notes.md` 不会被误判为 `eval/`；
- 无法确认时 fail closed（`RETRIEVAL_SOURCE_DENIED`）。

同一策略用于：

- 索引构建（`rag/store.py`）；
- 检索（`rag/search.py`、`rag/cli.py`）；
- Harness `RagAdapter`；
- ContextBuilder；
- `read_file` / `search_text` / `list_files` / `get_file_info`；
- `run_python` / `run_powershell` / `run_tests` 的脚本/目标路径；
- Hindsight recall/retain；
- MCP RAG / Memory / Agent Task。

### 1.3 构建与查询

```bash
python -m yuzhu_lab.rag.cli build --root . --store .yuzhu/rag_index \
  --embedding <path-to-embedding-model>
python -m yuzhu_lab.rag.cli search "RRF fusion" --store .yuzhu/rag_index \
  --embedding <path-to-embedding-model> --k 5
```

索引 manifest 记录来源策略、排除文件数与各文件 SHA256。不要用含评测数据的
目录重建索引；旧索引可能已被污染，不能仅靠查询过滤继续信任。

## 2. Hindsight

Hindsight 是可选的外部记忆服务。Harness 通过 HTTP 适配：

- 默认只保留 `verification=verified` 的记忆；
- 未验证/实验性记忆默认不进入上下文；
- 记忆是历史，不是当前代码事实；
- 无法连接时降级为空记忆，不伪造；
- recall/retain 都执行来源与内容指纹检查，受保护评测内容不进入模型可见记忆。

没有 Hindsight 服务时，使用 `NullMemoryAdapter`，任务仍可运行。

## 3. Harness

### 3.1 工具

默认注册的工具包括：

- 文件：`read_file`、`list_files`、`search_text`、`get_file_info`、
  `write_file`、`edit_file`、`apply_patch`；
- 执行（默认关闭）：`run_python`、`run_powershell`、`run_tests`；
- Git 只读：`git_status`、`git_diff`、`git_log`。

权限模型：

- `read_file` 等工作区限定；
- 写入受保护路径（`eval/`、`models/`、`adapters/`、`logs/` 等）被拒绝；
- 执行需要显式 `--allow-execute`；
- destructive 操作需要显式批准；
- 每次工具调用写入审计日志（参数摘要，不记录完整内容）。

### 3.2 Checkpoint / Resume

任务状态、工具调用与事件写入 SQLite。中断后可用同一 `task_id` resume；
已完成工具调用不会重复产生副作用。

### 3.3 完成门禁

Completion Gate 只依据可观察证据（工具执行、测试结果、文件修改）判断
`TASK_COMPLETED`；模型自述不算证据。测试通过不等于语义正确；需要独立 Oracle
时请在外部验证。

### 3.4 上下文预算

ContextBuilder 按预算裁剪：

```text
conversation summary → verified memory → RAG evidence → 渲染
```

优先级：当前用户指令 > 当前会话状态 > 当前文件/测试证据 > 已验证记忆 >
RAG > 模型推断。超出预算时按低排名项逐步删除，并返回可见 warning。

## 4. MCP

MCP Server 只暴露少量工具，stdout 仅承载 JSON-RPC。配置见
[CLAUDE_CODE_MCP.md](CLAUDE_CODE_MCP.md)。

安全默认值：

- 执行关闭；
- 记忆写入关闭；
- 一次只允许一个 agent task；
- 文件工具受 workspace 限制；
- 评测来源拒绝策略与 Harness 完全一致。

## 5. 评测隔离的残余风险

- 不是 OS 沙箱；恶意脚本仍可能读取当前用户可读的文件；
- 执行器只应在受控工作区启用；
- 如果使用外部模型端点，端点侧的日志与数据策略不在本项目控制范围内；
- 公开版本不包含评测数据；内部评测数据必须留在受保护环境。