"""Guard against benchmark-specific hardcoding in the memory engine.

A memory system must answer from the user's stored memories. Returning
hand-written answers keyed to known evaluation questions inflates local tests,
does not generalise to unseen tasks, and violates evaluation integrity.

This test fails if app/ contains entity strings taken from official evaluation
items, or synthetic memory items injected with fixed ids. Run: python3 -m pytest tests/
"""
import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"

# Entity strings lifted from official AML smoke items that were previously hardcoded.
FORBIDDEN = [
    "sarah garcia", "tarsem singh", "mukesh", "latvia", "dune messiah", "joaquim molins",
    "ethos water", "city autocomplete", "christina baker", "mil-53", "5, 5, 5, 8",
    "ntsb", "0816 local", "13033", "13009", "as_scalar_or_list_str", "pour-over kit",
    "订婚纪念夜", "王景川", "空白页", "棘手项目", "扶他林", "晨僵",
    "obrenović", "private island miami", "retriever hip dysplasia",
]

# Synthetic items returned with a fixed, hand-chosen id.
INJECTED_ID = re.compile(r"[\"']id[\"']\s*:\s*[\"'](?:prop|eng|fix|synth)_[a-z0-9_]+[\"']")


def _sources():
    for p in APP.rglob("*.py"):
        yield p, p.read_text(encoding="utf-8")


def test_no_benchmark_entity_strings():
    hits = []
    for p, src in _sources():
        low = src.lower()
        for term in FORBIDDEN:
            if term in low:
                hits.append(f"{p.relative_to(APP.parent)}: {term!r}")
    assert not hits, "benchmark-specific strings found in app/:\n" + "\n".join(hits)


def test_no_injected_fixed_id_items():
    hits = []
    for p, src in _sources():
        for m in INJECTED_ID.finditer(src):
            line = src.count("\n", 0, m.start()) + 1
            hits.append(f"{p.relative_to(APP.parent)}:{line}: {m.group(0)}")
    assert not hits, "hand-written memory items with fixed ids found:\n" + "\n".join(hits)


# A literal full calendar date inside engine code (e.g. "January 20, 2022") almost always
# means a patch keyed to one specific evaluation item rather than a general rule.
LITERAL_DATE = re.compile(
    r"[\"'][^\"'\n]*\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},\s+(?:19|20)\d{2}[^\"'\n]*[\"']"
)


def test_no_literal_calendar_dates_in_engine():
    hits = []
    for p, src in _sources():
        for line_no, line in enumerate(src.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if LITERAL_DATE.search(line):
                hits.append(f"{p.relative_to(APP.parent)}:{line_no}: {line.strip()[:120]}")
    assert not hits, "literal calendar dates found in app/ (item-specific patch?):\n" + "\n".join(hits)
