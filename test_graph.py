"""Self-check for _build_graph — the join between the live skill catalog and
historical pick stats has to drop skills that consolidation deleted, must not
report system skills as never-retrieved, and must only draw an edge it can
justify. Run: python test_graph.py"""
from datetime import datetime, timezone

from store import _build_graph, _content_edges, _sim_tokens

TS = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)

CATALOG = [
    {"name": "user-profile", "tier": "system", "description": "who", "chars": 10,
     "updated_at": TS},
    {"name": "tech-stack", "tier": "active", "description": "stack", "chars": 20,
     "updated_at": TS},
    {"name": "user-pets", "tier": "active", "description": "pets", "chars": 30,
     "updated_at": TS},
]


def test_nodes():
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

    # 3 turns used tech-stack, 2 of them also used user-pets, which has no pick
    # row of its own — union falls back to the co-count, so 2 / 3.
    assert g["edges"] == [
        {"source": "tech-stack", "target": "user-pets", "kind": "coactivation",
         "weight": 2, "strength": 0.667}
    ]

    # a system skill can never be an edge endpoint even if picked data claims it
    g2 = _build_graph(
        CATALOG, [], [{"src": "tech-stack", "dst": "user-profile", "weight": 5}]
    )
    assert g2["edges"] == []

    # empty everything — never raises
    assert _build_graph([], [], []) == {"nodes": [], "edges": [], "turns": 0}


def test_coactivation_gate():
    """One shared turn is a coincidence. Two thirds of the edges on a real
    store were weight 1, which is what made the map look arbitrary."""
    picks = [
        {"name": "tech-stack", "picks": 4, "avg_score": 0.8, "last_picked": TS},
        {"name": "user-pets", "picks": 4, "avg_score": 0.8, "last_picked": TS},
    ]
    once = _build_graph(
        CATALOG, picks, [{"src": "tech-stack", "dst": "user-pets", "weight": 1}]
    )
    assert once["edges"] == []

    twice = _build_graph(
        CATALOG, picks, [{"src": "tech-stack", "dst": "user-pets", "weight": 2}]
    )
    # 2 shared of the (4 + 4 - 2) turns that used either
    assert twice["edges"][0]["strength"] == 0.333

    # strength, not the raw count, separates these: the busier pair co-occurs
    # more often in absolute terms but is together a smaller share of the time.
    busy = _build_graph(
        CATALOG,
        [
            {"name": "tech-stack", "picks": 40, "avg_score": 0.8, "last_picked": TS},
            {"name": "user-pets", "picks": 40, "avg_score": 0.8, "last_picked": TS},
        ],
        [{"src": "tech-stack", "dst": "user-pets", "weight": 5}],
    )
    assert busy["edges"][0]["weight"] > twice["edges"][0]["weight"]
    assert busy["edges"][0]["strength"] < twice["edges"][0]["strength"]


def test_containers_never_link():
    """Rollups and archives hold whole conversations, so they relate to
    everything and turn into hubs joining unrelated subjects."""
    catalog = CATALOG + [
        {"name": "sessions-archive-20260819-015153", "tier": "archive",
         "description": "many topics", "chars": 9000, "updated_at": TS},
        {"name": "session-20260820-072731", "tier": "active",
         "description": "one chat", "chars": 900, "updated_at": TS},
    ]
    g = _build_graph(
        catalog,
        [{"name": "sessions-archive-20260819-015153", "picks": 6,
          "avg_score": 0.4, "last_picked": TS}],
        [
            {"src": "sessions-archive-20260819-015153", "dst": "tech-stack",
             "weight": 3},
            {"src": "session-20260820-072731", "dst": "user-pets", "weight": 4},
        ],
        # bodies that would otherwise score a strong content match
        bodies={
            "tech-stack": "gpu cuda inference quantisation llm",
            "sessions-archive-20260819-015153": "gpu cuda inference quantisation llm",
            "session-20260820-072731": "pets cat vet vaccination",
            "user-pets": "pets cat vet vaccination",
        },
    )
    assert g["edges"] == []
    # they are still nodes — only the lines are withheld
    assert "sessions-archive-20260819-015153" in {n["name"] for n in g["nodes"]}


def test_content_edges():
    """Body overlap needs no pick history, which is the point: on a young store
    the picker has touched a quarter of the catalog and the rest would float
    unconnected."""
    docs = {
        "interest-sports-rules-fifa":
            _sim_tokens("fifa offside referee match football laws"),
        "interest-sports-rules-fifa-subs":
            _sim_tokens("fifa substitution referee match football laws"),
        "interest-baking-sourdough":
            _sim_tokens("sourdough starter hydration levain crumb"),
    }
    edges = _content_edges(docs)
    assert len(edges) == 1
    e = edges[0]
    assert {e["source"], e["target"]} == {
        "interest-sports-rules-fifa", "interest-sports-rules-fifa-subs"
    }
    assert e["kind"] == "content"
    assert e["strength"] >= 0.08

    # degenerate inputs never raise
    assert _content_edges({}) == []
    assert _content_edges({"a": {"x"}}) == []


def test_sim_tokens():
    t = _sim_tokens("Tech-Stack: the local LLM is on a RTX-4090")
    assert {"tech", "stack", "local", "llm"} <= t
    assert "4090" in t          # hyphen split keeps the model number
    assert "the" not in t       # stopword
    assert "is" not in t        # too short to be distinctive
    assert "on" not in t


def test_kinds_coexist():
    """A pair can be both about the same thing and retrieved together; the
    co-activation edge wins so the panel states the stronger claim once."""
    g = _build_graph(
        CATALOG,
        [
            {"name": "tech-stack", "picks": 3, "avg_score": 0.8, "last_picked": TS},
            {"name": "user-pets", "picks": 3, "avg_score": 0.8, "last_picked": TS},
        ],
        [{"src": "tech-stack", "dst": "user-pets", "weight": 3}],
        bodies={
            "tech-stack": "gpu cuda inference quantisation llm ollama",
            "user-pets": "gpu cuda inference quantisation llm ollama",
        },
    )
    assert len(g["edges"]) == 1
    assert g["edges"][0]["kind"] == "coactivation"


if __name__ == "__main__":
    for fn in (test_nodes, test_coactivation_gate, test_containers_never_link,
               test_content_edges, test_sim_tokens, test_kinds_coexist):
        fn()
        print(f"  {fn.__name__} ok")
    print("ok")
