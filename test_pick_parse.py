"""Self-check for _parse_picks — the picker's JSON is model-authored, so the
parse has to survive both shapes and junk. Run: python test_pick_parse.py"""
from agent import _parse_picks

VALID = {"user-profile", "tech-stack", "project-feedback"}


def test():
    # scored objects: sorted best-first, clamped to 0..1
    assert _parse_picks(
        [
            {"name": "tech-stack", "score": 0.4},
            {"name": "user-profile", "score": 0.9},
            {"name": "project-feedback", "score": 3},
        ],
        VALID,
    ) == [("project-feedback", 1.0), ("user-profile", 0.9), ("tech-stack", 0.4)]

    # bare strings (model ignored the format) still load, scored 1.0
    assert _parse_picks(["tech-stack"], VALID) == [("tech-stack", 1.0)]

    # unknown names dropped, dupes collapsed, non-numeric score defaults to 1.0
    assert _parse_picks(
        [
            {"name": "nope", "score": 0.9},
            {"name": "tech-stack", "score": "high"},
            {"name": "tech-stack", "score": 0.1},
            42,
        ],
        VALID,
    ) == [("tech-stack", 1.0)]

    # garbage in, empty out — never raises
    assert _parse_picks(None, VALID) == []
    assert _parse_picks("tech-stack", VALID) == []
    assert _parse_picks([{"score": 0.5}], VALID) == []

    print("ok")


if __name__ == "__main__":
    test()
