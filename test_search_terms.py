"""Unit test for _search_terms. Run: python test_search_terms.py

The output is interpolated into a to_tsquery OR-query, so a term carrying a
tsquery operator (& | ! : *) or a paren would turn a miss into a SQL error.
"""
from agent import _search_terms


def test():
    # Distinctive tokens survive, lowercased, in order.
    assert _search_terms("What is Zephyr's deploy target?") == ["zephyr", "deploy", "target"]

    # Stopwords dropped (incl. recall verbs like "say"); dupes dropped.
    assert _search_terms("did you say the dog and the dog") == ["dog"]
    assert _search_terms("dog cat dog cat bird") == ["dog", "cat", "bird"]

    # Sub-3-char tokens dropped (too common to narrow anything).
    assert _search_terms("my ai is up") == []

    # Nothing that to_tsquery would parse as an operator can survive.
    hostile = "a|b & c:d !e (f) *g 'h' \\i"
    for term in _search_terms(hostile):
        assert term.isalnum(), term

    # Empty in, empty out — the caller skips the query entirely.
    assert _search_terms("") == []
    assert _search_terms("!!! ???") == []

    # Cap holds.
    assert len(_search_terms(" ".join(f"tok{i}" for i in range(50)))) == 8

    print("_search_terms unit tests ok")


if __name__ == "__main__":
    test()
