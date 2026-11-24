# Contributing

## 基本原则

- 不提交凭据、Token、密码、会话数据或用户专属路径；
- 不提交冻结评测题目、评分标准、模型答案或训练数据；
- 不修改测试来掩盖失败；
- 不声称未验证的能力；
- 保持最小改动，不为一次性需求增加通用框架。

## 开发环境

```bash
python -m venv .venv
pip install -r requirements.txt
python -m unittest discover -s tests -t . -p "test_*.py"
```

## 提交前检查

```bash
python -m compileall -q yuzhu_lab
python -m unittest discover -s tests -t . -p "test_*.py"
```

如改动涉及来源策略、RAG、MCP 或文件工具，必须新增/更新对抗测试，并确认：

- 受保护评测来源仍被拒绝；
- 正常项目源码/Markdown/测试文件仍可读取；
- 缺失 provenance fail closed；
- 旧索引不会绕过检索时策略。

## 代码风格

- 遵循现有模块结构与命名；
- 路径处理必须使用规范化后的真实路径，不做字符串包含判断；
- 错误信息不得包含凭据或评测内容；
- 不引入未经审查的依赖；
- 不进行无关重构或全仓格式化。

## 许可证

项目原创源码采用 Apache License 2.0，见 [LICENSE](LICENSE) 与
[NOTICE](NOTICE)。提交即表示你有权提交该内容，并同意其按 Apache-2.0 发布。
第三方内容必须保留其许可证与归属，且不得提交与本项目许可证不兼容的代码。

## 安全问题

不要用公开 PR 报告安全问题；见 [SECURITY.md](SECURITY.md)。