# Safe read-only example task

```text
读取 demo.py，解释 add 函数的作用。只使用 read_file 工具。
```

预期：

- 只调用 `read_file`；
- 不修改文件；
- 不执行命令；
- MCP 执行权限保持关闭。

不要在没有隔离工作区时启用 `--allow-execute`。