"""Unit test for sweep_stale_sessions. Run: python test_sweep.py

The sweep is what makes a session become memory even when the browser never
fired its end-of-session beacon. It runs at the top of every turn, so its
failure modes matter more than its happy path: it must never raise, and one bad
session must not stop the rest.
"""
import agent


class _Recorder:
    def __init__(self, ids, fail_on=(), raise_lookup=False):
        self.ids = ids
        self.fail_on = set(fail_on)
        self.raise_lookup = raise_lookup
        self.seen = []
        self.kwargs = None

    def stale_unrolled(self, user_id, **kw):
        if self.raise_lookup:
            raise RuntimeError("db down")
        self.kwargs = kw
        return list(self.ids)


def _install(monkey_store, summarize):
    agent.SessionStore.stale_unrolled = staticmethod(  # type: ignore[assignment]
        monkey_store.stale_unrolled
    )
    agent.summarize_session_to_skill = summarize  # type: ignore[assignment]


def test():
    real_store = agent.SessionStore.stale_unrolled
    real_sum = agent.summarize_session_to_skill
    ctx = agent.UserCtx(user_id='alice')
    try:
        # Happy path: a session that summarizes to nothing (trivial chat) is
        # simply not reported; the others are.
        rec = _Recorder(['s1', 's2', 's3'])
        seen = []

        def summarize(clients, c, sid):
            seen.append(sid)
            return None if sid == 's2' else 'session-' + sid

        _install(rec, summarize)
        out = agent.sweep_stale_sessions(None, ctx, exclude_session_id='live')
        assert out == ['session-s1', 'session-s3'], out
        assert seen == ['s1', 's2', 's3']

        # The live session is handed to the store so it can exclude itself, and
        # the bounds are passed through rather than silently defaulted.
        assert rec.kwargs['exclude_session_id'] == 'live'
        assert rec.kwargs['idle_minutes'] == agent.SWEEP_IDLE_MINUTES
        assert rec.kwargs['limit'] == agent.SWEEP_MAX_PER_RUN

        # One session blowing up (bad key, model error) must not cost the
        # others — they are independent conversations.
        rec = _Recorder(['a', 'bad', 'b'])

        def summarize_raises(clients, c, sid):
            if sid == 'bad':
                raise RuntimeError("model 500")
            return 'session-' + sid

        _install(rec, summarize_raises)
        assert agent.sweep_stale_sessions(None, ctx) == ['session-a', 'session-b']

        # A lookup failure returns empty instead of raising: the sweep runs at
        # the top of every turn and must never cost the user their turn.
        _install(_Recorder([], raise_lookup=True), summarize)
        assert agent.sweep_stale_sessions(None, ctx) == []

        # Nothing stale is the overwhelmingly common case.
        _install(_Recorder([]), summarize)
        assert agent.sweep_stale_sessions(None, ctx) == []
    finally:
        agent.SessionStore.stale_unrolled = real_store  # type: ignore[assignment]
        agent.summarize_session_to_skill = real_sum  # type: ignore[assignment]

    print('ok')


if __name__ == '__main__':
    test()
