"""Self-check for _build_graph — the join between the live skill catalog and
historical pick stats has to drop skills that consolidation deleted, and must
not report system skills as never-retrieved. Run: python test_graph.py"""
from datetime import datetime, timezone

from store import _build_graph

TS = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)

CATALOG = [
    {"name": "user-profile", "tier": "system", "description": "who", "chars": 10,
     "updated_at": TS},
    {"name": "tech-stack", "tier": "active", "description": "stack", "chars": 20,
     "updated_at": TS},
    {"name": "user-pets", "tier": "active", "description": "pets", "chars": 30,
     "updated_at": TS},
]


def test():
    g = _build_graph(
        CATALOG,
        [
            {"name": "tech-stack", "picks": 3, "avg_score": 0.876, "last_picked": TS},
            # folded into archive and deleted by consolidate() — must not appear
            {"name": "session-20260525-150947", "picks": 9, "avg_score": 0.5,
             "last_picked": TS},
        ],
        [
            {"src": "tech-stack", "dst": "user-pets", "weight": 2},
            # endpoint no longer in the catalog — dropped
            {"src": "session-20260525-150947", "dst": "tech-stack", "weight": 4},
        ],
    )
    by_name = {n["name"]: n for n in g["nodes"]}

    # catalog is the source of truth: no ghosts, nothing missing
    assert set(by_name) == {"user-profile", "tech-stack", "user-pets"}

    # picked skill carries its stats, score rounded to 2dp
    assert by_name["tech-stack"]["picks"] == 3
    assert by_name["tech-stack"]["avg_score"] == 0.88
    assert by_name["tech-stack"]["last_picked"] == TS.isoformat()

    # never retrieved: survives as a node, flagged as dead memory
    assert by_name["user-pets"]["picks"] == 0
    assert by_name["user-pets"]["avg_score"] is None
    assert by_name["user-pets"]["last_picked"] is None

    # system tier bypasses the picker — flagged, and never wired into edges
    assert by_name["user-profile"]["always"] is True
    assert by_name["tech-stack"]["always"] is False

    assert g["edges"] == [
        {"source": "tech-stack", "target": "user-pets", "weight": 2}
    ]

    # a system skill can never be an edge endpoint even if picked data claims it
    g2 = _build_graph(
        CATALOG, [], [{"src": "tech-stack", "dst": "user-profile", "weight": 5}]
    )
    assert g2["edges"] == []

    # empty everything — never raises
    assert _build_graph([], [], []) == {"nodes": [], "edges": []}

    print("ok")


if __name__ == "__main__":
    test()
