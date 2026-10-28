"""Unit test for selectable personas. Run: python test_personas.py

A persona is a repo-shipped skill (personas/<name>/SKILL.md) the user picks per
request. These cover loading, name resolution, that respond() puts the body in
the system prompt only while selected, and the run_turn_events wiring from
UserCtx.persona through to the done event and turns.picked.
"""
import os

os.environ["AGENT_SWEEP"] = "0"
os.environ["AGENT_REFLECT"] = "0"
os.environ["AGENT_IMAGE_GEN"] = "0"

import agent  # noqa: E402


def _capture_respond():
    seen = {}

    def fake(clients, model, sys_prompt, history, prompt, attachments, **kw):
        seen["sys_prompt"] = sys_prompt
        return "ok", {"in": 1, "out": 1, "total": 2}

    return seen, fake


def test_load_and_resolve():
    personas = agent.load_personas()
    assert "elements-of-style" in personas, personas.keys()
    p = personas["elements-of-style"]
    assert p.description and "Omit needless words" in p.body
    assert not p.body.startswith("---"), "frontmatter leaked into body"
    # The book transcript is not shipped, so the persona must not point at it.
    assert "Strunk_White.pdf" not in p.body

    assert agent.UserCtx(user_id="alice").persona is None
    assert agent.UserCtx(user_id="alice", persona="elements-of-style").persona == "elements-of-style"
    assert agent.UserCtx(user_id="alice", persona=" Elements-Of-Style ").persona == "elements-of-style"
    assert agent.UserCtx(user_id="alice", persona="nope").persona is None
    assert agent.UserCtx(user_id="alice", persona="").persona is None


def test_respond_prompt():
    real = agent._chat_respond
    seen, fake = _capture_respond()
    persona = agent.load_personas()["elements-of-style"]
    try:
        agent._chat_respond = fake
        agent.respond(None, "hi", [], [], [], [], persona=persona)
        assert "PERSONA: elements-of-style" in seen["sys_prompt"]
        assert "Omit needless words" in seen["sys_prompt"]
        # Persona comes after the loaded skills so it has the last word on style.
        assert seen["sys_prompt"].index("LOADED SKILLS") < seen["sys_prompt"].index("PERSONA:")

        agent.respond(None, "hi", [], [], [], [])
        assert "PERSONA" not in seen["sys_prompt"]
    finally:
        agent._chat_respond = real


def test_run_turn_wiring():
    patched = {
        "load_skills": lambda ctx: [],
        "pick_skills": lambda *a, **k: [],
        "fallback_body_picks": lambda *a, **k: [],
        "load_session": lambda ctx, sid: [],
    }
    turns = []
    patched["append_turn"] = lambda ctx, sid, role, text, **kw: turns.append((role, kw))
    seen, fake = _capture_respond()
    patched["_chat_respond"] = fake
    real = {k: getattr(agent, k) for k in patched}
    try:
        for k, v in patched.items():
            setattr(agent, k, v)
        clients = agent.Clients(gemini=None, anthropic=None)

        ctx = agent.UserCtx(user_id="alice", persona="elements-of-style")
        done = list(agent.run_turn_events(clients, ctx, "s1", "hello"))[-1]
        assert done["stage"] == "done"
        assert done["loaded"]["persona"] == "elements-of-style"
        assert "PERSONA: elements-of-style" in seen["sys_prompt"]
        model_row = [kw for role, kw in turns if role == "model"][-1]
        assert model_row["picked"]["persona"] == "elements-of-style"

        turns.clear()
        ctx = agent.UserCtx(user_id="alice")
        done = list(agent.run_turn_events(clients, ctx, "s1", "hello"))[-1]
        assert done["loaded"]["persona"] is None
        assert "PERSONA" not in seen["sys_prompt"]
        model_row = [kw for role, kw in turns if role == "model"][-1]
        assert "persona" not in model_row["picked"]
    finally:
        for k, v in real.items():
            setattr(agent, k, v)


if __name__ == "__main__":
    test_load_and_resolve()
    test_respond_prompt()
    test_run_turn_wiring()
    print("persona unit tests ok")
