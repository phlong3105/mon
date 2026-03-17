"""Data models for literature search results.

Paper and Author are frozen dataclasses — immutable after creation.
``Paper.to_bibtex()`` generates a valid BibTeX entry from metadata,
and ``Paper.cite_key`` returns a normalised citation key.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Author:
    """A paper author."""

    name: str
    affiliation: str = ""

    def last_name(self) -> str:
        """Return ASCII-folded last name for citation keys."""
        parts = self.name.strip().split()
        raw = parts[-1] if parts else "unknown"
        # Fold accented characters to ASCII
        nfkd = unicodedata.normalize("NFKD", raw)
        ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-zA-Z]", "", ascii_name).lower() or "unknown"


@dataclass(frozen=True)
class Paper:
    """A single paper from Semantic Scholar, arXiv, or similar sources.

    Fields are designed to hold the union of metadata available from both
    Semantic Scholar and arXiv APIs.
    """

    paper_id: str
    title: str
    authors: tuple[Author, ...] = ()
    year: int = 0
    abstract: str = ""
    venue: str = ""
    citation_count: int = 0
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    source: str = ""  # "semantic_scholar" | "arxiv"
    _bibtex_override: str = field(default="", repr=False)

    # ------------------------------------------------------------------
    # Citation key
    # ------------------------------------------------------------------

    @property
    def cite_key(self) -> str:
        """Normalised citation key: ``lastname<year><keyword>``.

        Example: ``smith2024transformer``
        """
        last = self.authors[0].last_name() if self.authors else "anon"
        yr = str(self.year) if self.year else "0000"
        # First meaningful noun-ish word from title (>3 chars, alpha only)
        kw = ""
        for word in self.title.split():
            cleaned = re.sub(r"[^a-zA-Z]", "", word).lower()
            if len(cleaned) > 3 and cleaned not in _STOPWORDS:
                kw = cleaned
                break
        return f"{last}{yr}{kw}"

    # ------------------------------------------------------------------
    # BibTeX generation
    # ------------------------------------------------------------------

    def to_bibtex(self) -> str:
        """Generate a BibTeX entry string.

        If ``_bibtex_override`` was populated (e.g. from CrossRef), return
        that directly.  Otherwise construct from metadata.
        """
        if self._bibtex_override:
            return self._bibtex_override.strip()

        key = self.cite_key
        authors_str = " and ".join(a.name for a in self.authors) or "Unknown"

        # Decide entry type
        if self.venue and any(
            kw in self.venue.lower()
            for kw in (
                "conference",
                "proc",
                "workshop",
                "neurips",
                "icml",
                "iclr",
                "aaai",
                "cvpr",
                "acl",
            )
        ):
            entry_type = "inproceedings"
            venue_field = f"  booktitle = {{{self.venue}}},"
        elif self.arxiv_id and not self.venue:
            entry_type = "article"
            venue_field = "  journal = {arXiv preprint},"
        else:
            entry_type = "article"
            venue_field = (
                f"  journal = {{{self.venue or 'Unknown'}}}," if self.venue else ""
            )

        lines = [f"@{entry_type}{{{key},"]
        lines.append(f"  title = {{{self.title}}},")
        lines.append(f"  author = {{{authors_str}}},")
        if self.year:
            lines.append(f"  year = {{{self.year}}},")
        if venue_field:
            lines.append(venue_field)
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.arxiv_id:
            lines.append(f"  eprint = {{{self.arxiv_id}}},")
            lines.append("  archiveprefix = {arXiv},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation helpers (for JSONL output)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict for JSON/JSONL output."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": [
                {"name": a.name, "affiliation": a.affiliation} for a in self.authors
            ],
            "year": self.year,
            "abstract": self.abstract,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "source": self.source,
            "cite_key": self.cite_key,
        }


# Common English stopwords to skip when picking a keyword for cite_key
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "upon",
        "about",
        "through",
        "using",
        "based",
        "towards",
        "toward",
        "between",
        "under",
        "more",
        "than",
        "when",
        "what",
        "which",
        "where",
        "does",
        "have",
        "been",
        "some",
        "each",
        "also",
        "much",
        "very",
        "learning",  # too generic for ML papers
    }
)
