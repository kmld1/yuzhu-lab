# YUZHU LAB

English | [简体中文](README.md)

**Repository**: <https://github.com/kmld1/yuzhu-lab>

YUZHU LAB is an experimental local small-model engineering runtime and agent
harness:

- local MiniCPM5-2B inference service (OpenAI/Anthropic-compatible HTTP);
- hybrid RAG with structure-aware chunking, FAISS + BM25S, RRF fusion and
  citations;
- optional Hindsight long-term memory adapter;
- a single-agent harness with tools, permission boundaries, completion gates and
  checkpoint/resume;
- an MCP stdio server for Claude Code;
- **evaluation-source isolation**: frozen benchmark prompts, criteria and
  answers cannot enter RAG, model context, memory or MCP tool results.

> Version: `v1.0.0` (source-only public release)
> Status: **PUBLIC_SOURCE_ONLY**. This repository contains source code only.
> Model weights, LoRA adapters, training data, evaluation data and credentials
> are not included.

## Not included

- MiniCPM5-2B weights;
- the internal P04.0 LoRA adapter (public redistribution is not cleared; there
  is **no public download**);
- training or evaluation data, benchmark questions, criteria or answers;
- Hindsight databases, Claude Code sessions, audit logs or checkpoints;
- API tokens, passwords or user configuration.

Official MiniCPM5-2B weights are published by OpenBMB under Apache-2.0. Obtain
them from the official channel and verify the license yourself. See
[docs/MODEL_SETUP.md](docs/MODEL_SETUP.md).

## Support boundary

| Component | Needs model/adapter | Notes |
|---|---|---|
| RAG | no | builds and searches independently; source admission enforced |
| Hindsight adapter | no | optional; degrades cleanly without a service |
| Harness tools/permissions/checkpoint | no | can use a scripted model or a compatible HTTP endpoint |
| MCP server | task-dependent | model endpoint required for model-driven tasks |
| Runtime | yes | this source tree requires a local model directory and a PEFT adapter directory |

**Important:** the Runtime loader in this release requires a PEFT adapter
directory. The repository does not ship the P04.0 adapter and there is no public
download. Without your own adapter you can still:

- use RAG, the Hindsight adapter and the harness tests;
- point the harness/MCP at any compatible model HTTP endpoint
  (`--runtime-url`);
- but you cannot start MiniCPM5 base-only through the bundled Runtime.

Base-only mode is not implemented or independently verified:
`BASE_ONLY_REAL_MODEL_VALIDATION = NOT_VERIFIED`.

## Quick start

```bash
git clone https://github.com/kmld1/yuzhu-lab.git
cd yuzhu-lab
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m unittest discover -s tests -t . -p "test_*.py"
```

Build a clean local RAG index:

```bash
python -m yuzhu_lab.rag.cli build --root . --store .yuzhu/rag_index
python -m yuzhu_lab.rag.cli search "Hybrid retrieval" --store .yuzhu/rag_index
```

Run the harness:

```bash
python -m yuzhu_lab.harness.agent.cli run \
  --workspace examples \
  --task "Read demo.py and explain the add function. Use read_file only." \
  --runtime-url http://127.0.0.1:8802
```

Start the MCP server:

```bash
python -m yuzhu_lab.harness.mcp_server \
  --workspace . \
  --runtime-url http://127.0.0.1:8802 \
  --state-dir .yuzhu/mcp_state
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md),
[docs/WINDOWS_INSTALL.md](docs/WINDOWS_INSTALL.md) and
[docs/CLAUDE_CODE_MCP.md](docs/CLAUDE_CODE_MCP.md).

## Evaluation-source isolation

RAG ingestion/retrieval, file tools, ContextBuilder, the Hindsight adapter and
the MCP server share `yuzhu_lab/source_policy.py`:

- `eval/` and explicit evaluation generator/scoring scripts are never
  admissible;
- paths are normalized, workspace-confined, reparse-point checked and must
  exist;
- missing or unverifiable citations are denied (fail closed);
- new indexes ingest only admitted sources and re-check at retrieval time;
- normal project source, Markdown and engineering tests remain readable.

See [docs/RAG_HINDSIGHT_HARNESS.md](docs/RAG_HINDSIGHT_HARNESS.md) and
[SECURITY.md](SECURITY.md).

## License

The **original source code** in this repository is licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) and
[LICENSE_DECISION.md](LICENSE_DECISION.md).

Copyright 2026 kmld.

The license does not apply to model weights, LoRA adapters, training data,
evaluation data or third-party dependencies; none of those assets are included
in this distribution. Third-party license information is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

On Windows, execution tools use a dedicated Job Object to manage their subprocess trees. If cleanup cannot be verified, the current task fails and further execution is blocked.

## External validation scope

Full MiniCPM5 deployment with a user-supplied adapter, a fresh Windows PyTorch/CUDA
installation, initialization against a new Hindsight database, and an external
Claude Code–to–MCP end-to-end run are unverified. Installation and regression
testing on Linux and macOS remain incomplete.

## Explicit non-claims

- no claim of superiority over any external model;
- no claim of an independent heldout or public blind evaluation;
- no claim of an OS-level sandbox;
- no claim that arbitrary model-generated code is safe to execute;
- no claim that external users have independently verified this release; the
  test results in the documentation are internal validation results;
- no claim that P04.2 was trained (`NOT_AUTHORIZED / NOT_STARTED`);
- no claim of solved complex SQLite autonomy (historical results: SQLite core
  0/10, SQLite extension 0/6; simple Python bug fixes passed an independent
  oracle 6/6. The evaluation sets and conditions differ and are not directly
  comparable);
- no claim that older evaluation results are a clean baseline (a historical RAG
  index contained evaluation assets; the defect was fixed and pre-fix results
  must not be treated as clean scores);
- no claim that the P04.0 adapter may be publicly redistributed;
- no claim that tool execution is an OS-level sandbox; tool calls are limited by
  authorization and path policies, users must review model-generated changes,
  and correct output on complex tasks is not guaranteed;
- no claim that every third-party dependency or service works offline; some
  features require additional components, network access or configuration.