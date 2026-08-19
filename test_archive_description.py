"""Unit test for _archive_description. Run: python test_archive_description.py

The archive skill's description is the only text the picker sees for it, and it
ships in the catalog of every pick call — so it must carry topic AND stay
bounded. These assert both halves.
"""
from agent import (
    ARCHIVE_DESC_ITEM_CAP,
    ARCHIVE_DESC_TOTAL_CAP,
    Skill,
    _archive_description,
)


def _s(desc):
    return Skill(tier="active", name="session-x", description=desc, body="b")


def test():
    # Topic text survives; the count + merge date stay for provenance.
    d = _archive_description([_s("Diesel crack spread"), _s("Treasury yields")],
                             "2026-08-19")
    assert d == "2 sessions merged 2026-08-19 — Diesel crack spread; Treasury yields."

    # A single-session archive reads as "1 session", not "1 sessions".
    assert _archive_description([_s("Guitars")], "2026-08-19") == (
        "1 session merged 2026-08-19 — Guitars."
    )

    # The words a picker would match on are actually present.
    assert "Diesel" in d and "Treasury" in d

    # Trailing periods are stripped so the join does not double up.
    assert ".;" not in _archive_description([_s("One."), _s("Two.")], "2026-08-19")

    # Whitespace/newlines in a source description are flattened — the result is
    # one catalog line.
    assert chr(10) not in _archive_description([_s("a\nb")], "2026-08-19")

    # No usable topic anywhere: fall back to the bare provenance line rather
    # than emitting a dangling dash.
    assert _archive_description([_s(""), _s("   ")], "2026-08-19") == (
        "2 sessions merged 2026-08-19."
    )
    assert _archive_description([], "2026-08-19") == "0 sessions merged 2026-08-19."

    # One verbose session cannot crowd the others out.
    long = _archive_description([_s("word " * 60), _s("Guitars")], "2026-08-19")
    assert "Guitars" in long
    assert chr(0x2026) in long

    # Total stays bounded no matter how many sessions are folded, and the
    # omitted ones are counted, not hidden.
    many = _archive_description([_s(f"topic number {i} about something") for i in range(40)],
                                "2026-08-19")
    assert len(many) < ARCHIVE_DESC_TOTAL_CAP + 120, len(many)
    assert "more)" in many

    # Per-item cap is honoured on the truncated entry.
    one = _archive_description([_s("z" * 300)], "2026-08-19")
    assert len(one) < ARCHIVE_DESC_ITEM_CAP + 60

    print("ok")


if __name__ == "__main__":
    test()
