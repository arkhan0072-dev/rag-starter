"""Two chunkers: a fixed-size baseline and the structure-aware chunker used by default."""

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
BLANK_LINE = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    heading: str
    content: str


def fixed_size(text: str, size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Cut the text every `size` characters. Kept as the baseline for the evaluation."""
    text = text.strip()
    step = size - overlap
    pieces = (text[i : i + size] for i in range(0, len(text), step))
    return [Chunk("", p) for p in pieces if p.strip()]


def by_structure(text: str, max_chars: int = 1200) -> list[Chunk]:
    """Split on markdown headings, then pack whole paragraphs into chunks of up to max_chars.

    Each chunk keeps its heading path (e.g. "Returns > Refund timeline"), which is
    embedded with the text and shown in citations. Paragraphs are never cut in half,
    so a paragraph longer than max_chars becomes a chunk on its own.
    """
    chunks: list[Chunk] = []
    path: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            chunks.append(Chunk(" > ".join(path), "\n\n".join(buf)))
            buf.clear()

    for block in BLANK_LINE.split(text):
        block = block.strip()
        if not block:
            continue
        first, _, rest = block.partition("\n")
        match = HEADING.match(first)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = path[: level - 1] + [match.group(2).strip()]
            block = rest.strip()
            if not block:
                continue
        if buf and sum(len(b) for b in buf) + len(block) > max_chars:
            flush()
        buf.append(block)
    flush()
    return chunks
