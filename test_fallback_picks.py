"""Unit test for fallback_body_picks. Run: python test_fallback_picks.py

The body search runs on every turn now, so what it declines to return matters as
much as what it returns: an unranked OR-query surfaces a skill about an
unrelated subject on one incidental word. The rank floor lives in SQL (sampled
against a real store — see FALLBACK_MIN_RANK); these cover the wiring around it.
"""
import agent


def test():
    real = agent.SkillStore.search_bodies
    calls = []

    def fake(user_id, terms, limit=3, min_rank=0.0):
        calls.append({'terms': terms, 'min_rank': min_rank})
        return [('interest-diesel', 0.072), ('ghost-skill', 0.031)]

    ctx = agent.UserCtx(user_id='alice')
    skills = [
        agent.Skill(name='interest-diesel', description='d', body='b'),
        agent.Skill(name='sys', description='s', body='b', tier='system'),
    ]
    try:
        agent.SkillStore.search_bodies = staticmethod(fake)  # type: ignore[assignment]
        got = agent.fallback_body_picks(ctx, 'diesel crack spread', skills)

        # The floor is actually passed down — the whole point of the change.
        assert calls[0]['min_rank'] == agent.FALLBACK_MIN_RANK, calls
        assert calls[0]['terms'] == ['diesel', 'crack', 'spread'], calls

        # Real ranks are logged but never ride downstream: hits carry the flat
        # FALLBACK_SCORE so they stay distinguishable from picker confidence.
        assert got == [('interest-diesel', agent.FALLBACK_SCORE)], got

        # A name the store returned but the catalog no longer has is dropped
        # rather than crashing the lookup that follows.
        assert 'ghost-skill' not in dict(got)

        # System tier never comes back through this path; it loads every turn.
        agent.SkillStore.search_bodies = staticmethod(  # type: ignore[assignment]
            lambda u, t, limit=3, min_rank=0.0: [('sys', 0.9)]
        )
        assert agent.fallback_body_picks(ctx, 'anything at all', skills) == []

        # Nothing clears the floor: empty, not an error.
        agent.SkillStore.search_bodies = staticmethod(  # type: ignore[assignment]
            lambda u, t, limit=3, min_rank=0.0: []
        )
        assert agent.fallback_body_picks(ctx, 'photosynthesis explained', skills) == []

        # A search failure must never cost the user their turn.
        def boom(u, t, limit=3, min_rank=0.0):
            raise RuntimeError('tsquery syntax error')

        agent.SkillStore.search_bodies = staticmethod(boom)  # type: ignore[assignment]
        assert agent.fallback_body_picks(ctx, 'diesel', skills) == []
    finally:
        agent.SkillStore.search_bodies = real  # type: ignore[assignment]

    print('ok')


if __name__ == '__main__':
    test()
