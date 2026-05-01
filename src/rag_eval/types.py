from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str


@dataclass(frozen=True)
class GoldSpan:
    document_id: str
    start_char: int
    end_char: int
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Example:
    example_id: str
    benchmark: str
    query: str
    gold_spans: list[GoldSpan]
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "benchmark": self.benchmark,
            "query": self.query,
            "gold_spans": [span.to_dict() for span in self.gold_spans],
            "tags": self.tags,
        }


@dataclass(frozen=True)
class Chunk:
    document_id: str
    start_char: int
    end_char: int
    text: str
    title: str = ""
    level: int = 0


@dataclass
class RetrievedSpan:
    document_id: str
    start_char: int
    end_char: int
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.estimated_cost_usd += other.estimated_cost_usd

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LLMResponse:
    text: str
    usage: Usage


@dataclass
class RetrievalOutput:
    spans: list[RetrievedSpan]
    usage: Usage = field(default_factory=Usage)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ""

