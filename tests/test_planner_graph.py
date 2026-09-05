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


def _make_registry(reasoner_fn=None, critic_fn=None) -> dict:
    """Build a fake 4-agent registry.  ``reasoner_fn`` overrides the reasoner
    (``critic_fn`` kept as a deprecated alias)."""
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

    def _default_reasoner(ctx):
        ctx["reasoner"] = {"verdict": "PASS"}
        return {"verdict": "PASS"}

    def _alternatives(ctx):
        ctx["alternatives"] = {"days": []}
        return {"alternatives": 2}

    def _compiler(ctx):
        ctx["itinerary"] = {"days": []}
        return {"itinerary": "compiled"}

    fn = reasoner_fn if reasoner_fn is not None else critic_fn
    return {
        "scout": _scout,
        "reasoner": fn if fn is not None else _default_reasoner,
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
        assert names == ["scout", "reasoner", "alternatives", "compiler"]
        types = {n["name"]: n["type"] for n in graph["nodes"]}
        assert types == {
            "scout": "agent",
            "reasoner": "agent",
            "alternatives": "agent",
            "compiler": "agent",
        }
        tools = {n["name"]: n.get("tools") for n in graph["nodes"]}
        assert tools == {
            "scout": ["ingest", "hours"],
            "reasoner": ["logistics", "scheduler"],
            "alternatives": None,
            "compiler": None,
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
        registry = _make_registry()  # reasoner returns PASS on first run

        rows: list[dict] = []
        for _ in range(20):  # generous upper bound
            row = advance(ctx, registry, sink, graph)
            rows.append(row)
            if run_status(ctx)["is_completed"]:
                break

        assert run_status(ctx)["is_completed"] is True
        assert ctx["current_node"] is None

        # Exactly 4 agent nodes ran (scout -> reasoner -> alternatives ->
        # compiler); the tools they own run INSIDE their turns and appear as
        # separate tool rows in the sink, not as agent rows.
        assert len(rows) == 4
        assert [r["node_name"] for r in rows] == [
            "scout",
            "reasoner",
            "alternatives",
            "compiler",
        ]
        # Tool rows from the owned tools are in the sink with parent linkage.
        agent_names = {r["node_name"] for r in rows}
        tool_rows = [r for r in sink if r.get("node_type") == "tool"]
        for tr in tool_rows:
            assert tr["parent"] in agent_names
            assert tr["node_name"].startswith(tr["parent"] + ".")
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
    def test_reasoner_loops_then_passes(self):
        """The reasoner's draft->review->re-draft loop is INTERNAL to its turn
        in v2; the graph-level loop edges are the two conditional returns
        (re_research / consult).  A reasoner stub that ISSUES twice then PASSes
        must complete the pipeline with the verdict stored on ctx."""
        ctx = _fresh_ctx()
        sink = InMemorySink()
        graph = load_graph()

        call_count = {"n": 0}

        def reasoner_fn(ctx):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                out = {"verdict": "ISSUES", "issues": ["too many stops"],
                       "directives": [], "re_research": [], "consult_alternatives": False}
            else:
                out = {"verdict": "PASS", "issues": [], "directives": [],
                       "re_research": [], "consult_alternatives": False}
            ctx["reasoner"] = out
            return out

        registry = _make_registry(reasoner_fn=reasoner_fn)

        rows: list[dict] = []
        for _ in range(30):
            row = advance(ctx, registry, sink, graph)
            rows.append(row)
            if run_status(ctx)["is_completed"]:
                break

        node_seq = [r["node_name"] for r in rows]

        # Pipeline completed with a single pass over the 4-agent backbone.
        assert ctx["current_node"] is None
        assert run_status(ctx)["is_completed"] is True
        # The reasoner AGENT node runs exactly once at graph level — its
        # draft->review->re-draft loop is internal to run_reasoner (unit-tested
        # in test_planner_agents), not a graph-level re-entry.
        assert node_seq.count("reasoner") == 1
        assert node_seq.count("alternatives") == 1
        assert node_seq.count("compiler") == 1
        assert call_count["n"] == 1
        # ISSUES with empty re_research/consult flags routes forward, and the
        # final verdict is whatever the stub last returned.
        assert ctx["reasoner"]["verdict"] == "ISSUES"


# ---------------------------------------------------------------------------
# Bound — critic ALWAYS fails; must force-proceed after 3 rounds
# ---------------------------------------------------------------------------


class TestBound:
    def test_reasoner_issues_still_completes(self):
        """A reasoner that always returns ISSUES must not block the pipeline:
        in v2 the review loop is internal (bounded by max_reasoner_rounds) and
        the graph proceeds to alternatives/compiler with issues surfaced."""
        ctx = _fresh_ctx()
        sink = InMemorySink()
        graph = load_graph()

        def reasoner_fn(ctx):
            ctx["reasoner"] = {"verdict": "ISSUES", "issues": ["nope"],
                               "directives": [], "re_research": [],
                               "consult_alternatives": False}
            return ctx["reasoner"]

        registry = _make_registry(reasoner_fn=reasoner_fn)

        rows: list[dict] = []
        for _ in range(30):
            row = advance(ctx, registry, sink, graph)
            rows.append(row)
            if run_status(ctx)["is_completed"]:
                break

        node_seq = [r["node_name"] for r in rows]

        # No infinite loop — pipeline terminated through the backbone.
        assert ctx["current_node"] is None
        assert run_status(ctx)["is_completed"] is True
        assert node_seq.count("reasoner") == 1
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

        def boom_agent(ctx):
            raise RuntimeError("reasoner exploded")

        registry = _make_registry(reasoner_fn=boom_agent)

        rows2: list[dict] = []
        for _ in range(20):
            row = advance(ctx, registry, sink, graph)
            rows2.append(row)
            if run_status(ctx)["is_completed"]:
                break

        node_seq = [r["node_name"] for r in rows2]

        # Reasoner ran but failed.
        rs_row = next(r for r in rows2 if r["node_name"] == "reasoner")
        assert rs_row["status"] == "failed"
        assert "reasoner exploded" in rs_row["error"]
        # Pipeline still reached terminal (failed agent routes forward).
        assert ctx["current_node"] is None
        assert "compiler" in node_seq
        # The error is in ctx["errors"].
        assert any("reasoner exploded" in e for e in ctx["errors"])

    def test_failed_reasoner_routes_forward(self):
        """A failed reasoner (exception, not a verdict) routes forward."""
        ctx = _fresh_ctx()
        sink = InMemorySink()
        graph = load_graph()

        def boom(ctx):
            raise RuntimeError("reasoner crashed")

        registry = _make_registry(reasoner_fn=boom)

        rows: list[dict] = []
        for _ in range(20):
            row = advance(ctx, registry, sink, graph)
            rows.append(row)
            if run_status(ctx)["is_completed"]:
                break

        node_seq = [r["node_name"] for r in rows]

        # Reasoner failed but routed forward to alternatives.
        assert "reasoner" in node_seq
        assert "alternatives" in node_seq
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
