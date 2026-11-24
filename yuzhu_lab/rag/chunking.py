"""Structure-aware chunking for Python, Markdown and plain text files."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

DEFAULT_INCLUDE_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".ps1",
    ".jinja",
    ".cfg",
    ".ini",
}

DEFAULT_EXCLUDE_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".cache",
    ".tmp",
    "models",
    "adapters",
    "data",
    "logs",
    "unsloth_compiled_cache",
    "node_modules",
    "site-packages",
    "results",
    "audits",
    "artifacts",
    "runs",
    "reports",
    "baseline_copy",
    "fixtures_final",
    "fixtures_final_default",
    "fixtures_run_001",
    "fixtures_run_002",
}

MAX_FILE_BYTES = 512 * 1024
MAX_CHUNK_CHARS = 1500


@dataclass
class Chunk:
    chunk_id: str
    source: str
    kind: str
    content: str
    symbol: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    heading: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["content_hash"] = self.content_hash
        return data

    def citation(self) -> str:
        location = self.source
        if self.start_line is not None:
            location += f":{self.start_line}"
            if self.end_line is not None and self.end_line != self.start_line:
                location += f"-{self.end_line}"
        if self.symbol:
            location += f" ({self.symbol})"
        return location


def _chunk_id(source: str, symbol: str | None, start: int | None, content: str) -> str:
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]
    parts = [source.replace("\\", "/"), symbol or "", str(start or 0), digest]
    return "chunk_" + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _make_chunk(
    *,
    source: str,
    kind: str,
    content: str,
    symbol: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    heading: str | None = None,
) -> Chunk:
    text = content.strip("\n")
    return Chunk(
        chunk_id=_chunk_id(source, symbol, start_line, text),
        source=source,
        kind=kind,
        content=text,
        symbol=symbol,
        start_line=start_line,
        end_line=end_line,
        heading=heading,
    )


def discover_files(
    root: Path,
    include_suffixes: set[str] | None = None,
    exclude_parts: set[str] | None = None,
) -> list[Path]:
    include_suffixes = include_suffixes or DEFAULT_INCLUDE_SUFFIXES
    exclude_parts = exclude_parts or DEFAULT_EXCLUDE_PARTS
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in include_suffixes:
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in exclude_parts for part in relative_parts):
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        files.append(path)
    return sorted(files)


def _module_header_end(tree: ast.Module, line_count: int) -> int:
    first = line_count + 1
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min(first, node.lineno)
    return first


def chunk_python(text: str, source: str) -> list[Chunk]:
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_text(text, source, kind="python")

    chunks: list[Chunk] = []
    header_end = _module_header_end(tree, len(lines))
    header = "\n".join(lines[: header_end - 1])
    if header.strip():
        chunks.append(
            _make_chunk(
                source=source,
                kind="python",
                content=header,
                symbol="<module>",
                start_line=1,
                end_line=header_end - 1,
            )
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            content = "\n".join(lines[node.lineno - 1 : node.end_lineno or node.lineno])
            chunks.append(
                _make_chunk(
                    source=source,
                    kind="python",
                    content=content,
                    symbol=node.name,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                )
            )
        elif isinstance(node, ast.ClassDef):
            methods = [
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            class_end = (methods[0].lineno - 1) if methods else (node.end_lineno or node.lineno)
            header_text = "\n".join(lines[node.lineno - 1 : class_end])
            if header_text.strip():
                chunks.append(
                    _make_chunk(
                        source=source,
                        kind="python",
                        content=header_text,
                        symbol=node.name,
                        start_line=node.lineno,
                        end_line=class_end,
                    )
                )
            for method in methods:
                content = "\n".join(
                    lines[method.lineno - 1 : method.end_lineno or method.lineno]
                )
                chunks.append(
                    _make_chunk(
                        source=source,
                        kind="python",
                        content=content,
                        symbol=f"{node.name}.{method.name}",
                        start_line=method.lineno,
                        end_line=method.end_lineno,
                    )
                )
    return chunks


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk_markdown(text: str, source: str, max_chars: int = 1800) -> list[Chunk]:
    lines = text.splitlines()
    sections: list[tuple[list[str], int, str]] = []
    heading_path: list[str] = []
    current: list[str] = []
    start_line = 1
    current_heading = source

    def flush(end_line: int) -> None:
        nonlocal current
        content = "\n".join(current).strip("\n")
        if content:
            sections.append((current, start_line, current_heading))
        current = []

    for index, line in enumerate(lines, start=1):
        match = _HEADING_RE.match(line)
        if not match:
            current.append(line)
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        heading_path = heading_path[: level - 1]
        heading_path.append(title)
        flush(index - 1)
        current_heading = " > ".join(heading_path)
        start_line = index
        current = [line]

    flush(len(lines))

    chunks: list[Chunk] = []
    for content_lines, section_start, heading in sections:
        content = "\n".join(content_lines)
        if len(content) <= max_chars:
            chunks.append(
                _make_chunk(
                    source=source,
                    kind="markdown",
                    content=content,
                    heading=heading,
                    start_line=section_start,
                    end_line=section_start + len(content_lines) - 1,
                )
            )
            continue
        for offset in range(0, len(content_lines), 40):
            window = content_lines[offset : offset + 40]
            window_text = "\n".join(window)
            if not window_text.strip():
                continue
            chunks.append(
                _make_chunk(
                    source=source,
                    kind="markdown",
                    content=window_text,
                    heading=heading,
                    start_line=section_start + offset,
                    end_line=section_start + offset + len(window) - 1,
                )
            )
    return chunks


def chunk_text(
    text: str, source: str, kind: str = "text", max_lines: int = 60, overlap: int = 8
) -> list[Chunk]:
    lines = text.splitlines()
    chunks: list[Chunk] = []
    step = max(1, max_lines - overlap)
    for offset in range(0, max(1, len(lines)), step):
        window = lines[offset : offset + max_lines]
        content = "\n".join(window)
        if not content.strip():
            continue
        chunks.append(
            _make_chunk(
                source=source,
                kind=kind,
                content=content,
                start_line=offset + 1,
                end_line=offset + len(window),
            )
        )
        if offset + max_lines >= len(lines):
            break
    return chunks


def _split_oversized(chunks: Iterable[Chunk], max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split any chunk that would create an oversized embedding sequence."""
    result: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.content) <= max_chars:
            result.append(chunk)
            continue
        lines = chunk.content.splitlines() or [chunk.content]
        base_line = chunk.start_line or 1

        def flush(buffer: list[str], start_line: int, end_line: int) -> None:
            if not buffer:
                return
            result.append(
                _make_chunk(
                    source=chunk.source,
                    kind=chunk.kind,
                    content="\n".join(buffer),
                    symbol=chunk.symbol,
                    start_line=start_line,
                    end_line=end_line,
                    heading=chunk.heading,
                )
            )

        buffer: list[str] = []
        buffer_start = base_line
        for offset, line in enumerate(lines):
            line_number = base_line + offset
            if len(line) > max_chars:
                flush(buffer, buffer_start, line_number - 1)
                buffer = []
                for position in range(0, len(line), max_chars):
                    result.append(
                        _make_chunk(
                            source=chunk.source,
                            kind=chunk.kind,
                            content=line[position : position + max_chars],
                            symbol=chunk.symbol,
                            start_line=line_number,
                            end_line=line_number,
                            heading=chunk.heading,
                        )
                    )
                buffer_start = line_number + 1
                continue
            pending = sum(len(item) + 1 for item in buffer) + len(line) + 1
            if buffer and pending > max_chars:
                flush(buffer, buffer_start, line_number - 1)
                buffer = []
                buffer_start = line_number
            buffer.append(line)
        flush(buffer, buffer_start, buffer_start + len(buffer) - 1)
    return result


def chunk_file(path: Path, root: Path) -> list[Chunk]:
    source = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    if suffix == ".py":
        chunks = chunk_python(text, source)
    elif suffix == ".md":
        chunks = chunk_markdown(text, source)
    else:
        chunks = chunk_text(text, source)
    return _split_oversized(chunks)


def deduplicate(chunks: Iterable[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    result: list[Chunk] = []
    for chunk in chunks:
        digest = chunk.content_hash
        if digest in seen:
            continue
        seen.add(digest)
        result.append(chunk)
    return result