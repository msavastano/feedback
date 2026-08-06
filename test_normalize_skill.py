"""Unit test for normalize_skill_name. Run: python test_normalize_skill.py"""
from store import normalize_skill_name


def test():
    # Basic slugification
    assert normalize_skill_name("User-Profile") == "user-profile"
    assert normalize_skill_name("user_profile") == "user_profile"
    assert normalize_skill_name("Tech Stack & Tools!") == "tech-stack-tools"
    assert normalize_skill_name("  Project: AI Assistant  ") == "project-ai-assistant"
    assert normalize_skill_name("rule-topic") == "rule-topic"
    assert normalize_skill_name("P1: critical_rules") == "p1-critical_rules"

    # Edge cases
    assert normalize_skill_name("") == ""
    assert normalize_skill_name("   ") == "   "
    assert normalize_skill_name("---") == "---"
    assert normalize_skill_name("!!!") == "!!!"

    print("normalize_skill_name unit tests ok")


if __name__ == "__main__":
    test()
