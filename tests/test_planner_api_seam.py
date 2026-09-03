"""Integration: the PRODUCTION registry + tools dict must run the real graph.

Guards the api/planner.py seam (build_registry wiring, tools dict, start
node) — the class of wiring bug where every component passes its own tests
but the assembled pipeline is dead. LLM and SerpApi are stubbed; the
scheduler, graph runner, and adapters run for real.
Run: .venv/bin/python -m pytest tests/test_planner_api_seam.py -q
"""
import os
from unittest.mock import patch

import pytest

if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://placeholder-for-import"

import api.planner as planner_api  # noqa: E402
from services import planner_graph  # noqa: E402


@pytest.fixture(autouse=True)
def stubbed_db(monkeypatch):
    """run_ingest persists via save_pins; tests run without a database."""
    import services.planner_ingest as pi
    monkeypatch.setattr(pi, "save_pins", lambda session_id, pins: None)


@pytest.fixture()
def stubbed_llm(monkeypatch):
    """Reasoner/Scout/Alternatives/Compiler LLM calls return canned JSON."""
    import services.planner_agents as pa

    def fake_llm(prompt, system_prompt=None, schema=None, model=None):
        if system_prompt is pa.REASONER_SYSTEM:
            return {
                "verdict": "PASS",
                "issues": [],
                "directives": [],
                "re_research": "",
                "consult_alternatives": False,
            }
        if system_prompt is pa.SCOUT_SYSTEM:
            return {
                "category": "attraction",
                "dwell_minutes": 60,
                "booking_required": False,
                "tip": "Go early.",
                "meal_fit": None,
                "best_time": None,
            }
        if system_prompt is pa.ALTERNATIVES_SYSTEM:
            return {"days": []}
        if system_prompt is pa.COMPILER_SYSTEM:
            return {"themes": ["A day out"], "intro": "Have fun."}
        raise AssertionError(f"unexpected LLM call: {system_prompt!r}")

    monkeypatch.setattr(pa, "call_openrouter_json", fake_llm)


@pytest.fixture()
def stubbed_serp(monkeypatch):
    """Geocode + hours + directions come back as plausible fixtures."""
    import services.planner_serp as ps

    def fake_geocode(name, city):
        return {
            "name": name,
            "lat": 1.28 + 0.001 * (len(name) % 5),
            "lng": 103.85 + 0.001 * (len(name) % 3),
            "address": f"1 Test Walk, {city}",
            "place_id": f"pid-{abs(hash(name)) % 9999}",
            "raw_hours": {"monday": "9 AM-5 PM", "tuesday": "9 AM-5 PM",
                          "wednesday": "9 AM-5 PM", "thursday": "9 AM-5 PM",
                          "friday": "9 AM-5 PM", "saturday": "9 AM-5 PM",
                          "sunday": "9 AM-5 PM"},
        }

    def fake_hours(name, city, place_id):
        return {"monday": "9 AM-5 PM", "tuesday": "9 AM-5 PM",
                "wednesday": "9 AM-5 PM", "thursday": "9 AM-5 PM",
                "friday": "9 AM-5 PM", "saturday": "9 AM-5 PM",
                "sunday": "9 AM-5 PM"}

    def fake_directions(start, end, mode, city):
        return {"distance_km": 2.0, "minutes": 10.0}

    monkeypatch.setattr(ps, "geocode_place", fake_geocode)
    monkeypatch.setattr(ps, "place_hours", fake_hours)
    monkeypatch.setattr(ps, "directions", fake_directions)


def _run_full_pipeline():
    """Advance the real graph through all nodes with the production wiring."""
    registry, tools = planner_api.build_registry()

    # The registry must expose the 4-agent graph's node names.
    for name in ("scout", "reasoner", "alternatives", "compiler"):
        assert name in registry, f"registry missing node {name!r}"

    # The tools dict must contain every tool the YAML grants the agents.
    graph = planner_graph.load_graph()
    for node in graph["nodes"]:
        for tool in node.get("tools") or []:
            assert tool in tools, (
                f"tool {tool!r} (owned by {node['name']}) missing from tools dict"
            )

    ctx = {
        "session_id": "seam-test",
        "destination": "Singapore",
        "start_date": "2026-09-10",
        "num_days": 1,
        "payload": {"pins": "Marina Bay Sands\nGardens by the Bay"},
        "pins": [],
        "research": [],
        "legs": [],
        "schedule": None,
        "reasoner": None,
        "alternatives": None,
        "itinerary": None,
        "errors": [],
        "current_node": None,
    }

    sink = planner_graph.InMemorySink()
    rows = []
    for _ in range(40):
        row = planner_graph.advance(ctx, registry, sink, graph, tools=tools)
        rows.append(row)
        if planner_graph.run_status(ctx)["is_completed"]:
            break

    return ctx, rows, sink


def test_production_registry_runs_full_pipeline(stubbed_llm, stubbed_serp):
    ctx, rows, sink = _run_full_pipeline()

    assert planner_graph.run_status(ctx)["is_completed"] is True
    agent_seq = [r["node_name"] for r in rows]
    assert agent_seq == ["scout", "reasoner", "alternatives", "compiler"]

    # The agents' owned tools actually ran (tool rows with dotted names).
    tool_names = {r["node_name"] for r in sink if r.get("node_type") == "tool"}
    assert "scout.ingest" in tool_names
    assert "scout.hours" in tool_names
    assert "reasoner.logistics" in tool_names
    assert "reasoner.scheduler" in tool_names
    for r in sink:
        if r.get("node_type") == "tool":
            assert r["parent"] == r["node_name"].split(".")[0]

    # The pipeline produced a real schedule and itinerary end to end.
    assert ctx["schedule"] and ctx["schedule"].get("days")
    assert ctx["itinerary"]


def test_reasoner_directives_reach_scheduler(stubbed_llm, stubbed_serp):
    """A reasoner directive flows through the tools dict into schedule_trip."""
    import services.planner_agents as pa

    seen_directives = []

    real_llm = pa.call_openrouter_json

    def directive_llm(prompt, system_prompt=None, schema=None, model=None):
        if system_prompt is pa.REASONER_SYSTEM:
            # One compress_dwell directive on the first review, then PASS.
            if not seen_directives:
                return {
                    "verdict": "ISSUES",
                    "issues": [{"day_index": 0, "severity": "low",
                                "message": "tight"}],
                    "directives": [{"action": "compress_dwell",
                                    "stop": "Marina Bay Sands",
                                    "reference": None, "day": None,
                                    "dwell_minutes": 30,
                                    "reason": "squeeze"}],
                    "re_research": "",
                    "consult_alternatives": False,
                }
            return {"verdict": "PASS", "issues": [], "directives": [],
                    "re_research": "", "consult_alternatives": False}
        return real_llm(prompt, system_prompt=system_prompt,
                        schema=schema, model=model)

    with patch.object(pa, "call_openrouter_json", directive_llm):
        ctx, rows, sink = _run_full_pipeline()

    assert planner_graph.run_status(ctx)["is_completed"] is True
    # The scheduler tool was invoked more than once, and a re-draft carried a
    # NON-EMPTY directives list (visible in the tool row's tool_args).
    sched_tool_rows = [r for r in sink
                       if r.get("node_name") == "reasoner.scheduler"]
    assert len(sched_tool_rows) >= 2  # initial draft + re-draft
    directive_lists = [
        (r["input"].get("tool_args") or {}).get("directives")
        for r in sched_tool_rows
        if isinstance(r.get("input"), dict)
    ]
    assert any(d and "n=0" not in d for d in directive_lists), (
        f"no re-draft carried directives; tool_args: {directive_lists}"
    )
    # The schedule result records the directive processing outcome.
    assert "applied_directives" in (ctx.get("schedule") or {})
