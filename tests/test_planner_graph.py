"""Deterministic, offline tests for the Mr. Bounce graph runner.

No DB, no network.  Uses ``InMemorySink`` and a fake registry of trivial
node functions.  Run via:

    cd /Users/gel/worktrees/ai-playground-v2-travel-planner && .venv/bin/python -m pytest tests/test_planner_graph.py -q
"""
from __future__ import annotations

import os
from datetime import datetime

import pytest

from services.planner_graph import (
    GRAPH_PATH,
    InMemorySink,
    PostgresSink,
    advance,
    load_graph,
    run_status,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_ctx(**overrides) -> dict:
    """Return a blank session context dict seeded with the full key set."""
    ctx: dict = {
        "session_id": "test-000",
        "destination": "Singapore",
        "start_date": "2026-09-10",
        "num_days": 2,
        "payload": {},
        "pins": [],
        "research": [],
        "legs": [],
        "schedule": None,
        "critic": None,
        "alternatives": None,
        "itinerary": None,
        "critic_round": 0,
        "current_node": None,
        "errors": [],
    }
    ctx.update(overrides)
    return ctx


def _make_registry(critic_fn=None) -> dict:
    """Build a fake 7-node registry.  ``critic_fn`` overrides the critic."""
    def _noop(ctx):
        return {}

    def _ingest(ctx):
        ctx["pins"] = [{"pin_id": "p0", "name": "Marina Bay"}]
        return {"pins": len(ctx["pins"])}

    def _scout(ctx):
        ctx["research"] = [{"pin_id": "p0"}]
        return {"research": len(ctx["research"])}

    def _logistics(ctx):
        ctx["legs"] = [{"from_name": "A", "to_name": "B"}]
        return {"legs": len(ctx["legs"])}

    def _scheduler(ctx):
        ctx["schedule"] = {"days": []}
        return {"schedule": True}

    def _default_critic(ctx):
        ctx["critic"] = {"verdict": "PASS"}
        return {"verdict": "PASS"}

    def _alternatives(ctx):
        ctx["alternatives"] = {"days": []}
        return {"alternatives": 2}

    def _compiler(ctx):
        ctx["itinerary"] = {"days": []}
        return {"itinerary": "compiled"}

    return {
        "ingest": _ingest,
        "scout": _scout,
        "logistics": _logistics,
        "scheduler": _scheduler,
        "critic": critic_fn if critic_fn is not None else _default_critic,
        "alternatives": _alternatives,
        "compiler": _compiler,
    }


# ---------------------------------------------------------------------------
# load_graph — validation
# ---------------------------------------------------------------------------


class TestLoadGraph:
    def test_real_yaml_validates(self):
        graph = load_graph()
        names = [n["name"] for n in graph["nodes"]]
        assert names == [
            "ingest",
            "scout",
            "logistics",
            "scheduler",
            "critic",
            "alternatives",
            "compiler",
        ]
        types = {n["name"]: n["type"] for n in graph["nodes"]}
        assert types == {
            "ingest": "tool",
            "scout": "agent",
            "logistics": "tool",
            "scheduler": "tool",
            "critic": "agent",
            "alternatives": "agent",
            "compiler": "agent",
        }

    def test_graph_path_points_to_repo_root(self):
        assert GRAPH_PATH.endswith("planner_graph.yaml")
        assert os.path.isfile(GRAPH_PATH)

    def test_rejects_edge_to_unknown_node(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(
            """
name: bad
entry: a
max_critic_rounds: 1
nodes:
  - name: a
    type: tool
  - name: b
    type: tool
    terminal: true
edges:
  - {from: a, to: zzz}
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unknown node"):
            load_graph(str(p))

    def test_rejects_missing_terminal(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(
            """
name: bad
entry: a
max_critic_rounds: 1
nodes:
  - name: a
    type: tool
  - name: b
    type: tool
edges:
  - {from: a, to: b}
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="terminal"):
            load_graph(str(p))

    def test_rejects_conditional_missing_pass_next(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(
            """
name: bad
entry: a
max_critic_rounds: 3
nodes:
  - name: a
    type: tool
  - name: b
    type: agent
    conditional:
      "on": verdict
      pass_value: PASS
      fail_next: a
  - name: c
    type: agent
    terminal: true
edges:
  - {from: a, to: b}
  - {from: b, to: c, when: pass}
  - {from: b, to: a, when: fail}
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="pass_next"):
            load_graph(str(p))

    def test_rejects_unknown_node_type(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(
            """
name: bad
entry: a
max_critic_rounds: 1
nodes:
  - name: a
    type: robot
  - name: b
    type: tool
    terminal: true
edges:
  - {from: a, to: b}
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unknown type"):
            load_graph(str(p))

    def test_rejects_max_critic_rounds_below_one(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(
            """
name: bad
entry: a
max_critic_rounds: 0
nodes:
  - name: a
    type: tool
  - name: b
    type: tool
    terminal: true
edges:
  - {from: a, to: b}
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="max_critic_rounds"):
            load_graph(str(p))

    def test_rejects_entry_not_a_node(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(
            """
name: bad
entry: nonexistent
max_critic_rounds: 1
nodes:
  - name: a
    type: tool
  - name: b
    type: tool
    terminal: true
edges:
  - {from: a, to: b}
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="entry.*not a known node"):
            load_graph(str(p))


# ---------------------------------------------------------------------------
# Happy path — straight through the 7-node backbone
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_straight_through(self):
        ctx = _fresh_ctx()
        sink = InMemorySink()
        graph = load_graph()
        registry = _make_registry()  # critic returns PASS on first run

        rows: list[dict] = []
        for _ in range(20):  # generous upper bound
            row = advance(ctx, registry, sink, graph)
            rows.append(row)
            if run_status(ctx)["is_completed"]:
                break

        assert run_status(ctx)["is_completed"] is True
        assert ctx["current_node"] is None

        # Exactly 7 nodes ran (ingest -> scout -> logistics -> scheduler ->
        # critic -> alternatives -> compiler).
        assert len(rows) == 7
        assert [r["node_name"] for r in rows] == [
            "ingest",
            "scout",
            "logistics",
            "scheduler",
            "critic",
            "alternatives",
            "compiler",
        ]
        # Every row is ok, has ISO timestamps, and non-negative duration.
        for r in rows:
            assert r["status"] == "ok"
            assert r["error"] is None
            assert r["duration_ms"] >= 0
            # ISO-format timestamps parse cleanly.
            datetime.fromisoformat(r["started_at"])
            datetime.fromisoformat(r["finished_at"])
        # Round numbers are all 0 on the happy path.
        assert all(r["round"] == 0 for r in rows)
        # next_node chain is correct.
        assert rows[-1]["output"]["next_node"] is None


# ---------------------------------------------------------------------------
# Loop-back — critic fails twice then passes
# ---------------------------------------------------------------------------


class TestLoopBack:
    def test_critic_loops_then_passes(self):
        ctx = _fresh_ctx()
        sink = InMemorySink()
        graph = load_graph()

        call_count = {"n": 0}

        def critic_fn(ctx):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                ctx["critic"] = {"verdict": "ISSUES", "issues": ["too many stops"]}
                return {"verdict": "ISSUES", "issues": ["too many stops"]}
            ctx["critic"] = {"verdict": "PASS"}
            return {"verdict": "PASS"}

        registry = _make_registry(critic_fn=critic_fn)

        rows: list[dict] = []
        for _ in range(30):
            row = advance(ctx, registry, sink, graph)
            rows.append(row)
            if run_status(ctx)["is_completed"]:
                break

        node_seq = [r["node_name"] for r in rows]

        # scheduler ran 3 times (initial + 2 loops).
        assert node_seq.count("scheduler") == 3
        # critic ran 3 times (2 fails + 1 pass).
        assert node_seq.count("critic") == 3
        # Pipeline completed.
        assert ctx["current_node"] is None
        assert run_status(ctx)["is_completed"] is True
        # critic_round is 2 at end (incremented after 2 fail routes).
        assert ctx["critic_round"] == 2

        # Trace rows for critic show rounds 0, 1, 2.
        critic_rows = [r for r in rows if r["node_name"] == "critic"]
        assert [r["round"] for r in critic_rows] == [0, 1, 2]

        # Alternatives and compiler each ran once (after the pass).
        assert node_seq.count("alternatives") == 1
        assert node_seq.count("compiler") == 1


# ---------------------------------------------------------------------------
# Bound — critic ALWAYS fails; must force-proceed after 3 rounds
# ---------------------------------------------------------------------------


class TestBound:
    def test_critic_always_fails_force_proceeds(self):
        ctx = _fresh_ctx()
        sink = InMemorySink()
        graph = load_graph()

        def critic_fn(ctx):
            ctx["critic"] = {"verdict": "ISSUES"}
            return {"verdict": "ISSUES", "issues": ["nope"]}

        registry = _make_registry(critic_fn=critic_fn)

        rows: list[dict] = []
        for _ in range(30):
            row = advance(ctx, registry, sink, graph)
            rows.append(row)
            if run_status(ctx)["is_completed"]:
                break

        node_seq = [r["node_name"] for r in rows]

        # scheduler ran exactly 3 times (rounds 0, 1, 2).
        assert node_seq.count("scheduler") == 3
        # critic ran exactly 3 times (always fails, but bounded).
        assert node_seq.count("critic") == 3
        # No infinite loop — pipeline terminated.
        assert ctx["current_node"] is None
        assert run_status(ctx)["is_completed"] is True
        # The error message appears.
        assert any("critic: max rounds reached" in e for e in ctx["errors"])
        # Alternatives + compiler still ran.
        assert "alternatives" in node_seq
        assert "compiler" in node_seq


# ---------------------------------------------------------------------------
# Failed node — logistics raises, pipeline continues
# ---------------------------------------------------------------------------


class TestFailedNode:
    def test_logistics_fails_pipeline_continues(self):
        ctx = _fresh_ctx()
        sink = InMemorySink()
        graph = load_graph()

        registry = _make_registry()

        def boom(ctx):
            raise RuntimeError("logistics exploded")

        registry["logistics"] = boom

        rows: list[dict] = []
        for _ in range(20):
            row = advance(ctx, registry, sink, graph)
            rows.append(row)
            if run_status(ctx)["is_completed"]:
                break

        node_seq = [r["node_name"] for r in rows]

        # Logistics ran but failed.
        log_row = next(r for r in rows if r["node_name"] == "logistics")
        assert log_row["status"] == "failed"
        assert "logistics exploded" in log_row["error"]
        # Pipeline still reached terminal.
        assert ctx["current_node"] is None
        assert "compiler" in node_seq
        # The error is in ctx["errors"].
        assert any("logistics exploded" in e for e in ctx["errors"])

    def test_failed_critic_routes_forward(self):
        """A failed critic (exception, not a verdict) routes to pass_next."""
        ctx = _fresh_ctx()
        sink = InMemorySink()
        graph = load_graph()

        registry = _make_registry()

        def boom(ctx):
            raise RuntimeError("critic crashed")

        registry["critic"] = boom

        rows: list[dict] = []
        for _ in range(20):
            row = advance(ctx, registry, sink, graph)
            rows.append(row)
            if run_status(ctx)["is_completed"]:
                break

        node_seq = [r["node_name"] for r in rows]

        # Critic failed but routed forward to alternatives (not back to scheduler).
        assert "critic" in node_seq
        assert "alternatives" in node_seq
        # Scheduler ran only once (no loop-back on a failed critic).
        assert node_seq.count("scheduler") == 1
        assert ctx["current_node"] is None


# ---------------------------------------------------------------------------
# run_status helper
# ---------------------------------------------------------------------------


class TestRunStatus:
    def test_initial_status(self):
        ctx = _fresh_ctx()
        ctx["current_node"] = "ingest"
        status = run_status(ctx)
        assert status["current_node"] == "ingest"
        assert status["is_completed"] is False
        assert status["critic_round"] == 0
        assert status["errors"] == []

    def test_completed_status(self):
        ctx = _fresh_ctx()
        ctx["current_node"] = None
        status = run_status(ctx)
        assert status["is_completed"] is True


# ---------------------------------------------------------------------------
# PostgresSink — importable + does not raise on broken DSN
# ---------------------------------------------------------------------------


class TestPostgresSink:
    def test_importable(self):
        assert PostgresSink is not None
        sink = PostgresSink("fake-session-id")
        assert sink.session_id == "fake-session-id"

    def test_append_does_not_raise_on_db_failure(self, monkeypatch):
        # services.database reads DATABASE_URL at import time — provide a
        # dummy so the module is importable, then monkeypatch get_conn to
        # raise (simulates a broken DSN / connection refused).
        monkeypatch.setenv("DATABASE_URL", "postgres://dummy:dummy@localhost/dummy")
        import services.database as db_module

        def broken_get_conn():
            raise RuntimeError("connection refused (bad DSN)")

        monkeypatch.setattr(db_module, "get_conn", broken_get_conn)

        sink = PostgresSink("bad-session")
        # Must not raise — trace failures must never kill the run.
        row = {
            "seq": 0,
            "node_name": "ingest",
            "node_type": "tool",
            "round": 0,
            "status": "ok",
            "input": {"pins": 0},
            "output": {"next_node": "scout"},
            "error": None,
            "started_at": "2026-09-03T00:00:00+00:00",
            "finished_at": "2026-09-03T00:00:01+00:00",
            "duration_ms": 1000,
        }
        sink.append(row)  # should not propagate the exception
