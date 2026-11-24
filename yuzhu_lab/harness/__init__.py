"""Minimal Agent Harness context layer.

Only the context boundary is implemented here.  The execution loop and coding
tools are deliberately out of scope for H1.
"""

from .context_builder import ContextBuilder, ContextBuilderConfig
from .context_models import (
    ContextBundle,
    ContextRequest,
    ContextWarning,
    MemoryContextItem,
    RagEvidenceItem,
    SourceReference,
    TokenEstimate,
)
from .memory_adapter import HindsightMemoryAdapter, MemorySearchOutcome, NullMemoryAdapter
from .rag_adapter import NullRagAdapter, RagAdapter, RagSearchOutcome

__all__ = [
    "ContextBuilder",
    "ContextBuilderConfig",
    "ContextBundle",
    "ContextRequest",
    "ContextWarning",
    "HindsightMemoryAdapter",
    "MemoryContextItem",
    "MemorySearchOutcome",
    "NullMemoryAdapter",
    "NullRagAdapter",
    "RagAdapter",
    "RagEvidenceItem",
    "RagSearchOutcome",
    "SourceReference",
    "TokenEstimate",
]