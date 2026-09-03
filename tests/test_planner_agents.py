"""Deterministic tests for the planner agent nodes.

All tests use small hand-built fixtures — no network, no DB, no LLM.
Run: .venv/bin/python -m pytest tests/test_planner_agents.py -q
"""

import pytest

from services.planner_agents import (
    parse_raw_hours,
    _parse_12h_to_24h,
    _parse_time_range,
    render_itinerary_md,
    _assemble_itinerary_json,
    run_scout,
    run_reasoner,
    run_critic,
    run_alternatives,
    run_compiler,
    SCOUT_DEFAULTS,
)


def _inject_scout_tools(ctx: dict) -> None:
    """Give a scout ctx a runner-style ``_call_tool`` dispatcher (offline).

    ``ingest`` is a no-op passthrough (pins are pre-seeded in these fixtures);
    ``hours`` delegates to the real ``hours_tool`` against the mocked
    SerpApi/LLM so the full research path is exercised.
    """
    from services.planner_agents import hours_tool

    def fake_tool(name, *args, **kwargs):
        if name == "ingest":
            return {"ingested": len(ctx.get("pins") or [])}
        if name == "hours":
            return hours_tool(ctx, kwargs.get("pin"))
        raise KeyError(name)

    ctx["_call_tool"] = fake_tool


def _inject_reasoner_tools(ctx: dict) -> None:
    """Give a reasoner ctx a runner-style ``_call_tool`` dispatcher (offline).

    ``logistics`` is a no-op; ``scheduler`` is a pass-through that returns the
    fixture's schedule unchanged so the LLM review drives the assertions.
    """

    def fake_tool(name, *args, **kwargs):
        if name == "logistics":
            ctx.setdefault("legs", [])
            return {"legs_count": len(ctx["legs"])}
        if name == "scheduler":
            return ctx.get("schedule") or {}
        raise KeyError(name)

    ctx["_call_tool"] = fake_tool


# ---------------------------------------------------------------------------
# parse_raw_hours tests
# ---------------------------------------------------------------------------
class TestParseRawHours:
    """parse_raw_hours: SerpApi-style hours passthrough -> canonical shape."""

    def test_single_window_en_dash(self):
        """Single window with en-dash."""
        raw = {"monday": "9 AM–9 PM"}
        result = parse_raw_hours(raw)
        assert result == {"days": {"0": [{"open": "09:00", "close": "21:00"}]}}

    def test_hyphen_variant(self):
        """Hyphen instead of en-dash."""
        raw = {"monday": "9 AM-9 PM"}
        result = parse_raw_hours(raw)
        assert result == {"days": {"0": [{"open": "09:00", "close": "21:00"}]}}

    def test_serp_list_of_single_key_day_dicts_string_values(self):
        """SerpApi place_results list form observed live: list[7] of {day: 'window'}."""
        raw = [{"monday": "9 AM-9 PM"}, {"tuesday": "Closed"}]
        result = parse_raw_hours(raw)
        assert result == {"days": {"0": [{"open": "09:00", "close": "21:00"}], "1": []}}

    def test_serp_list_of_single_key_day_dicts_dict_values(self):
        """SerpApi list form where each day maps to an {open, close} dict."""
        raw = [{"friday": {"open": "10:00", "close": "18:00"}}, {"saturday": "Closed"}]
        result = parse_raw_hours(raw)
        assert result == {"days": {"4": [{"open": "10:00", "close": "18:00"}], "5": []}}

    def test_serp_list_skips_non_dict_items(self):
        """Non-dict items in the list form are skipped, not fatal."""
        raw = ["garbage", {"sunday": "Open 24 hours"}]
        result = parse_raw_hours(raw)
        assert result == {"days": {"6": [{"open": "00:00", "close": "23:59"}]}}

    def test_closed(self):
        """Closed day -> empty list."""
        raw = {"monday": "Closed"}
        result = parse_raw_hours(raw)
        assert result == {"days": {"0": []}}

    def test_closed_with_period(self):
        """'Closed.' -> empty list."""
        raw = {"monday": "Closed."}
        result = parse_raw_hours(raw)
        assert result == {"days": {"0": []}}

    def test_open_24_hours(self):
        """'Open 24 hours' -> 00:00-23:59."""
        raw = {"monday": "Open 24 hours"}
        result = parse_raw_hours(raw)
        assert result == {"days": {"0": [{"open": "00:00", "close": "23:59"}]}}

    def test_24_hours_short(self):
        """'24 hours' -> 00:00-23:59."""
        raw = {"monday": "24 hours"}
        result = parse_raw_hours(raw)
        assert result == {"days": {"0": [{"open": "00:00", "close": "23:59"}]}}

    def test_split_windows_sorted(self):
        """Comma-separated windows -> multiple intervals sorted by open time."""
        raw = {"monday": "10 AM-12 PM, 2 PM-4:30 PM"}
        result = parse_raw_hours(raw)
        days = result["days"]
        assert "0" in days
        assert len(days["0"]) == 2
        assert days["0"][0] == {"open": "10:00", "close": "12:00"}
        assert days["0"][1] == {"open": "14:00", "close": "16:30"}

    def test_split_windows_unsorted_input(self):
        """Windows given out of order get sorted by open time."""
        raw = {"monday": "2 PM-4 PM, 9 AM-11 AM"}
        result = parse_raw_hours(raw)
        days = result["days"]["0"]
        assert len(days) == 2
        assert days[0] == {"open": "09:00", "close": "11:00"}
        assert days[1] == {"open": "14:00", "close": "16:00"}

    def test_12_am_midnight(self):
        """'12 AM' -> 00:00."""
        raw = {"monday": "12 AM-5 AM"}
        result = parse_raw_hours(raw)
        assert result["days"]["0"] == [{"open": "00:00", "close": "05:00"}]

    def test_12_pm_noon(self):
        """'12 PM' -> 12:00."""
        raw = {"monday": "12 PM-9 PM"}
        result = parse_raw_hours(raw)
        assert result["days"]["0"] == [{"open": "12:00", "close": "21:00"}]

    def test_12_30_pm(self):
        """'12:30 PM' -> 12:30."""
        raw = {"monday": "12:30 PM-9 PM"}
        result = parse_raw_hours(raw)
        assert result["days"]["0"][0]["open"] == "12:30"

    def test_none_input(self):
        """None -> empty days."""
        result = parse_raw_hours(None)
        assert result == {"days": {}}

    def test_empty_dict(self):
        """{} -> empty days."""
        result = parse_raw_hours({})
        assert result == {"days": {}}

    def test_missing_day_keys(self):
        """Only some days present."""
        raw = {"monday": "9 AM-5 PM", "sunday": "Closed"}
        result = parse_raw_hours(raw)
        assert "0" in result["days"]
        assert "6" in result["days"]
        assert result["days"]["0"] == [{"open": "09:00", "close": "17:00"}]
        assert result["days"]["6"] == []

    def test_unparseable_window_skipped(self):
        """Unparseable individual windows are skipped, not raised."""
        raw = {"monday": "gibberish", "tuesday": "9 AM-5 PM"}
        result = parse_raw_hours(raw)
        # monday is unparseable -> skipped (not present)
        assert "0" not in result["days"]
        # tuesday is valid -> present
        assert "1" in result["days"]

    def test_none_value_for_day(self):
        """None value for a day key is skipped."""
        raw = {"monday": None, "tuesday": "9 AM-5 PM"}
        result = parse_raw_hours(raw)
        assert "0" not in result["days"]
        assert "1" in result["days"]

    def test_all_seven_days(self):
        """Full week of values, verify day-key mapping 0=Mon..6=Sun."""
        raw = {
            "monday": "9 AM-5 PM",
            "tuesday": "9 AM-5 PM",
            "wednesday": "9 AM-5 PM",
            "thursday": "9 AM-5 PM",
            "friday": "9 AM-9 PM",
            "saturday": "9 AM-9 PM",
            "sunday": "Closed",
        }
        result = parse_raw_hours(raw)
        assert set(result["days"].keys()) == {"0", "1", "2", "3", "4", "5", "6"}
        assert result["days"]["6"] == []
        assert result["days"]["5"] == [{"open": "09:00", "close": "21:00"}]


# ---------------------------------------------------------------------------
# _parse_12h_to_24h helper tests
# ---------------------------------------------------------------------------
class TestParse12h:
    def test_basic_am(self):
        assert _parse_12h_to_24h("9 AM") == "09:00"
        assert _parse_12h_to_24h("9:30 AM") == "09:30"

    def test_basic_pm(self):
        assert _parse_12h_to_24h("3 PM") == "15:00"
        assert _parse_12h_to_24h("3:30 PM") == "15:30"

    def test_12_am(self):
        assert _parse_12h_to_24h("12 AM") == "00:00"

    def test_12_pm(self):
        assert _parse_12h_to_24h("12 PM") == "12:00"

    def test_12_30_pm(self):
        assert _parse_12h_to_24h("12:30 PM") == "12:30"

    def test_12_30_am(self):
        assert _parse_12h_to_24h("12:30 AM") == "00:30"

    def test_lowercase(self):
        assert _parse_12h_to_24h("9 am") == "09:00"
        assert _parse_12h_to_24h("3 pm") == "15:00"

    def test_invalid(self):
        assert _parse_12h_to_24h("abc") is None
        assert _parse_12h_to_24h("") is None
        assert _parse_12h_to_24h("25 PM") is None


# ---------------------------------------------------------------------------
# render_itinerary_md tests
# ---------------------------------------------------------------------------
class TestRenderItineraryMd:
    def test_basic_render(self):
        """Tiny itinerary_json -> markdown with day headings, times, flags."""
        itin = {
            "trip": {
                "city": "Singapore",
                "start_date": "2026-09-10",
                "num_days": 1,
                "intro": "A fun trip!",
                "days": [
                    {
                        "day_index": 0,
                        "date": "2026-09-10",
                        "theme": "City Highlights",
                        "stops": [
                            {
                                "name": "Marina Bay Sands",
                                "category": "attraction",
                                "start": "09:00",
                                "end": "11:00",
                                "dwell_minutes": 120,
                                "address": "Marina Bay",
                                "lat": 1.28,
                                "lng": 103.86,
                                "tip": "Visit the SkyPark",
                                "booking_required": True,
                                "hours_verified": True,
                                "hours_flag": "",
                                "kind": "stop",
                            },
                            {
                                "name": "Lunch",
                                "category": "meal",
                                "start": "12:00",
                                "end": "13:00",
                                "dwell_minutes": 60,
                                "address": "",
                                "lat": None,
                                "lng": None,
                                "tip": "",
                                "booking_required": False,
                                "hours_verified": True,
                                "hours_flag": "",
                                "kind": "meal",
                            },
                            {
                                "name": "Gardens by the Bay",
                                "category": "attraction",
                                "start": "14:00",
                                "end": "17:00",
                                "dwell_minutes": 180,
                                "address": "",
                                "lat": 1.28,
                                "lng": 103.86,
                                "tip": "",
                                "booking_required": False,
                                "hours_verified": False,
                                "hours_flag": "[unverified]",
                                "kind": "stop",
                            },
                        ],
                        "legs_between": [
                            {"from": "Marina Bay Sands", "to": "Lunch",
                             "mode": "walk", "minutes": 10},
                        ],
                        "total_travel_minutes": 480,
                        "load_minutes": 300,
                    },
                ],
                "alternatives": {
                    "days": [
                        {
                            "day_index": 0,
                            "swaps": [
                                {
                                    "remove_name": "Gardens by the Bay",
                                    "add_name": "Sentosa",
                                    "trade_off": "+20 min travel, but beach vibes",
                                },
                            ],
                        },
                    ],
                },
                "notes": [],
                "sources": {
                    "hours_source": "serpapi",
                    "travel_source": "serpapi_directions_or_estimated",
                    "research_source": "openrouter",
                },
            }
        }
        md = render_itinerary_md(itin)
        # Day heading
        assert "## Day 1 — 2026-09-10 — City Highlights" in md
        # HH:MM times
        assert "09:00" in md
        assert "11:00" in md
        # [unverified] flag for stop with hours_verified False
        assert "[unverified]" in md
        # meal/rest lines present
        assert "meal" in md.lower()
        # Alternatives section
        assert "**Alternatives:**" in md
        assert "Sentosa" in md
        # Never raises
        assert isinstance(md, str)

    def test_empty_itinerary(self):
        """Empty itinerary never raises."""
        md = render_itinerary_md({"trip": {}})
        assert isinstance(md, str)
        assert "Trip" in md

    def test_no_alternatives(self):
        """Itinerary without alternatives section."""
        itin = {
            "trip": {
                "city": "Test City",
                "start_date": "2026-01-01",
                "num_days": 1,
                "intro": "",
                "days": [
                    {
                        "day_index": 0,
                        "date": "2026-01-01",
                        "theme": "Day 1",
                        "stops": [],
                        "legs_between": [],
                        "total_travel_minutes": 0,
                        "load_minutes": 0,
                    },
                ],
                "alternatives": {"days": []},
                "notes": [],
                "sources": {},
            }
        }
        md = render_itinerary_md(itin)
        assert isinstance(md, str)
        assert "Day 1" in md


# ---------------------------------------------------------------------------
# run_compiler assembly tests
# ---------------------------------------------------------------------------
def _make_ctx_for_compiler():
    """Build a minimal ctx fixture for compiler tests."""
    return {
        "session_id": "test-session",
        "destination": "Singapore",
        "start_date": "2026-09-10",
        "num_days": 1,
        "payload": {},
        "pins": [],
        "research": [],
        "legs": [],
        "schedule": {
            "days": [
                {
                    "day_index": 0,
                    "date": "2026-09-10",
                    "slots": [
                        {
                            "pin_id": "p1",
                            "name": "Marina Bay Sands",
                            "kind": "stop",
                            "start_min": 540,
                            "end_min": 660,
                            "dwell_minutes": 120,
                        },
                        {
                            "pin_id": None,
                            "name": "Lunch",
                            "kind": "meal",
                            "start_min": 720,
                            "end_min": 780,
                            "dwell_minutes": 60,
                        },
                    ],
                    "legs": [
                        {"from_name": "Marina Bay Sands", "to_name": "Lunch",
                         "mode": "walk", "minutes": 10},
                    ],
                    "total_scheduled_minutes": 120,
                },
            ],
            "unplaced": [],
            "repairs": [],
            "stats": {"total_travel_minutes": 10, "load_ratio": 1.0},
        },
        "critic": None,
        "alternatives": {"days": []},
        "itinerary": None,
        "critic_round": 0,
        "current_node": "compiler",
        "errors": [],
    }


class TestCompilerAssembly:
    def test_llm_raises_default_themes(self, monkeypatch):
        """When call_openrouter_json raises, themes default to 'Day N'."""
        from services import planner_agents

        def raise_fn(prompt, system_prompt, schema, model):
            raise ValueError("LLM unavailable")

        monkeypatch.setattr(planner_agents, "call_openrouter_json", raise_fn)
        ctx = _make_ctx_for_compiler()
        result = run_compiler(ctx)

        itin = result["itinerary_json"]
        day = itin["trip"]["days"][0]
        assert day["theme"] == "Day 1"
        assert itin["trip"]["intro"] == ""
        assert isinstance(result["markdown"], str)

    def test_llm_returns_themes_intro(self, monkeypatch):
        """When call_openrouter_json returns themes + intro, they are used."""
        from services import planner_agents

        def mock_fn(prompt, system_prompt, schema, model):
            return {"themes": ["Cultural Adventure"], "intro": "A wonderful trip awaits!"}

        monkeypatch.setattr(planner_agents, "call_openrouter_json", mock_fn)
        ctx = _make_ctx_for_compiler()
        result = run_compiler(ctx)

        itin = result["itinerary_json"]
        day = itin["trip"]["days"][0]
        assert day["theme"] == "Cultural Adventure"
        assert itin["trip"]["intro"] == "A wonderful trip awaits!"
        assert "A wonderful trip awaits!" in result["markdown"]

    def test_stops_carry_research_data(self, monkeypatch):
        """Stops in itinerary_json carry tip/category/coords from research."""
        from services import planner_agents

        ctx = _make_ctx_for_compiler()
        ctx["research"] = [
            {
                "pin_id": "p1",
                "name": "Marina Bay Sands",
                "neighborhood": "Marina Bay",
                "lat": 1.28,
                "lng": 103.86,
                "rating": None,
                "opening_hours": {"days": {"0": [{"open": "09:00", "close": "17:00"}]}},
                "hours_verified": False,
                "category": "attraction",
                "dwell_minutes": 120,
                "booking_required": True,
                "tip": "Visit the SkyPark",
                "address": "10 Bayfront Ave",
            },
        ]

        monkeypatch.setattr(
            planner_agents,
            "call_openrouter_json",
            lambda *a, **kw: {"themes": ["City Day"], "intro": ""},
        )
        result = run_compiler(ctx)
        itin = result["itinerary_json"]

        stop = itin["trip"]["days"][0]["stops"][0]
        assert stop["name"] == "Marina Bay Sands"
        assert stop["category"] == "attraction"
        assert stop["tip"] == "Visit the SkyPark"
        assert stop["lat"] == 1.28
        assert stop["lng"] == 103.86
        assert stop["booking_required"] is True
        # hours_flag set iff hours_verified is False
        assert stop["hours_flag"] == "[unverified]"
        assert stop["hours_verified"] is False

    def test_hours_flag_empty_when_verified(self, monkeypatch):
        """hours_flag is empty when hours_verified is True."""
        from services import planner_agents

        ctx = _make_ctx_for_compiler()
        ctx["research"] = [
            {
                "pin_id": "p1",
                "name": "Marina Bay Sands",
                "neighborhood": "",
                "lat": 1.28,
                "lng": 103.86,
                "rating": None,
                "opening_hours": {"days": {"0": [{"open": "09:00", "close": "17:00"}]}},
                "hours_verified": True,
                "category": "attraction",
                "dwell_minutes": 120,
                "booking_required": False,
                "tip": "",
                "address": "",
            },
        ]

        monkeypatch.setattr(
            planner_agents,
            "call_openrouter_json",
            lambda *a, **kw: {"themes": ["Day 1"], "intro": ""},
        )
        result = run_compiler(ctx)
        itin = result["itinerary_json"]
        stop = itin["trip"]["days"][0]["stops"][0]
        assert stop["hours_flag"] == ""
        assert stop["hours_verified"] is True

    def test_legs_copied_from_schedule(self, monkeypatch):
        """legs_between in itinerary_json come from schedule legs."""
        from services import planner_agents

        monkeypatch.setattr(
            planner_agents,
            "call_openrouter_json",
            lambda *a, **kw: {"themes": ["Day 1"], "intro": ""},
        )
        ctx = _make_ctx_for_compiler()
        result = run_compiler(ctx)
        itin = result["itinerary_json"]
        legs = itin["trip"]["days"][0]["legs_between"]
        assert len(legs) == 1
        assert legs[0]["from"] == "Marina Bay Sands"
        assert legs[0]["to"] == "Lunch"
        assert legs[0]["mode"] == "walk"
        assert legs[0]["minutes"] == 10


# ---------------------------------------------------------------------------
# run_critic deterministic pre-check tests
# ---------------------------------------------------------------------------
class TestCriticDeterministic:
    def test_long_day_no_meal_forces_issues(self, monkeypatch):
        """A 7h day lacking a meal slot -> ISSUES even when LLM says PASS."""
        from services import planner_agents

        ctx = {
            "session_id": "test",
            "destination": "Singapore",
            "start_date": "2026-09-10",
            "num_days": 1,
            "payload": {},
            "pins": [],
            "research": [],
            "legs": [],
            "schedule": {
                "days": [
                    {
                        "day_index": 0,
                        "date": "2026-09-10",
                        "slots": [
                            {
                                "pin_id": "p1",
                                "name": "Museum A",
                                "kind": "stop",
                                "start_min": 540,
                                "end_min": 960,
                                "dwell_minutes": 420,
                            },
                        ],
                        "legs": [],
                        "total_scheduled_minutes": 420,
                    },
                ],
                "unplaced": [],
                "repairs": [],
                "stats": {},
            },
            "critic": None,
            "alternatives": None,
            "itinerary": None,
            "critic_round": 0,
            "current_node": "critic",
            "errors": [],
        }

        # LLM says PASS
        monkeypatch.setattr(
            planner_agents,
            "call_openrouter_json",
            lambda *a, **kw: {"verdict": "PASS"},
        )
        _inject_reasoner_tools(ctx)
        output = run_critic(ctx)
        assert output["verdict"] == "ISSUES"
        # The issue should mention the meal
        assert any("meal" in i["message"].lower() for i in output["issues"])
        assert ctx["reasoner"] == output

    def test_llm_failure_defaults_pass_with_det_issues(self, monkeypatch):
        """LLM failure -> PASS with critic_error, unless det finds issues."""
        from services import planner_agents

        ctx = {
            "session_id": "test",
            "destination": "Singapore",
            "start_date": "2026-09-10",
            "num_days": 1,
            "payload": {},
            "pins": [],
            "research": [],
            "legs": [],
            "schedule": {
                "days": [
                    {
                        "day_index": 0,
                        "date": "2026-09-10",
                        "slots": [
                            {
                                "pin_id": "p1",
                                "name": "Museum A",
                                "kind": "stop",
                                "start_min": 540,
                                "end_min": 600,
                                "dwell_minutes": 60,
                            },
                        ],
                        "legs": [],
                        "total_scheduled_minutes": 60,
                    },
                ],
                "unplaced": [],
                "repairs": [],
                "stats": {},
            },
            "critic": None,
            "alternatives": None,
            "itinerary": None,
            "critic_round": 0,
            "current_node": "critic",
            "errors": [],
        }

        def raise_fn(prompt, system_prompt, schema, model):
            raise ValueError("LLM unavailable")

        monkeypatch.setattr(planner_agents, "call_openrouter_json", raise_fn)
        _inject_reasoner_tools(ctx)
        output = run_critic(ctx)
        assert output["verdict"] == "PASS"
        assert any("reasoner: LLM review failed" in e for e in ctx["errors"])
        assert any("reasoner" in e for e in ctx["errors"])

    def test_pass_when_no_issues(self, monkeypatch):
        """Normal short day with a meal -> PASS."""
        from services import planner_agents

        ctx = {
            "session_id": "test",
            "destination": "Singapore",
            "start_date": "2026-09-10",
            "num_days": 1,
            "payload": {},
            "pins": [],
            "research": [],
            "legs": [],
            "schedule": {
                "days": [
                    {
                        "day_index": 0,
                        "date": "2026-09-10",
                        "slots": [
                            {
                                "pin_id": "p1",
                                "name": "Museum A",
                                "kind": "stop",
                                "start_min": 540,
                                "end_min": 660,
                                "dwell_minutes": 120,
                            },
                            {
                                "pin_id": None,
                                "name": "Lunch",
                                "kind": "meal",
                                "start_min": 720,
                                "end_min": 780,
                                "dwell_minutes": 60,
                            },
                        ],
                        "legs": [],
                        "total_scheduled_minutes": 120,
                    },
                ],
                "unplaced": [],
                "repairs": [],
                "stats": {},
            },
            "critic": None,
            "alternatives": None,
            "itinerary": None,
            "critic_round": 0,
            "current_node": "critic",
            "errors": [],
        }

        monkeypatch.setattr(
            planner_agents,
            "call_openrouter_json",
            lambda *a, **kw: {"verdict": "PASS"},
        )
        _inject_reasoner_tools(ctx)
        output = run_critic(ctx)
        assert output["verdict"] == "PASS"
        assert output["issues"] == []


# ---------------------------------------------------------------------------
# run_scout tests
# ---------------------------------------------------------------------------
class TestRunScout:
    def test_serp_none_hours_unverified_llm_defaults(self, monkeypatch):
        """Pin whose SerpApi lookup returns None -> hours_verified False,
        LLM defaults used, node still returns for all pins."""
        from services import planner_agents

        # Mock planner_serp functions
        class MockSerp:
            @staticmethod
            def geocode_place(name, city):
                return None

            @staticmethod
            def place_hours(name, city, place_id):
                return None

        # Mock call_openrouter_json to return defaults
        def mock_llm(prompt, system_prompt, schema, model):
            return {
                "category": "restaurant",
                "dwell_minutes": 60,
                "booking_required": True,
                "tip": "Try the lunch special",
            }

        monkeypatch.setattr(planner_agents, "call_openrouter_json", mock_llm)

        ctx = {
            "session_id": "test",
            "destination": "Singapore",
            "start_date": "2026-09-10",
            "num_days": 2,
            "payload": {},
            "pins": [
                {
                    "pin_id": "p1",
                    "name": "Some Restaurant",
                    "resolved": True,
                    "lat": 1.3,
                    "lng": 103.8,
                    "address": "123 Main St",
                },
            ],
            "research": [],
            "legs": [],
            "schedule": None,
            "critic": None,
            "alternatives": None,
            "itinerary": None,
            "critic_round": 0,
            "current_node": "scout",
            "errors": [],
        }

        # Patch the lazy import path
        import sys
        original = sys.modules.get("services.planner_serp")
        sys.modules["services.planner_serp"] = MockSerp
        try:
            _inject_scout_tools(ctx)
            result = run_scout(ctx)
        finally:
            if original is not None:
                sys.modules["services.planner_serp"] = original
            else:
                del sys.modules["services.planner_serp"]

        assert result["researched"] == 1
        assert result["hours_verified"] == 0
        assert result["hours_unverified"] == 1
        assert len(ctx["research"]) == 1
        r = ctx["research"][0]
        assert r["hours_verified"] is False
        assert r["category"] == "restaurant"
        assert r["dwell_minutes"] == 60
        assert r["booking_required"] is True
        assert r["tip"] == "Try the lunch special"

    def test_multiple_pins_one_fails(self, monkeypatch):
        """One pin's SerpApi/LLM fails -> per-pin resilience, all pins returned."""
        from services import planner_agents

        call_count = [0]

        def flaky_llm(prompt, system_prompt, schema, model):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("LLM failed for pin 2")
            return {
                "category": "attraction",
                "dwell_minutes": 90,
                "booking_required": False,
                "tip": "",
            }

        monkeypatch.setattr(planner_agents, "call_openrouter_json", flaky_llm)

        class MockSerp:
            @staticmethod
            def geocode_place(name, city):
                return {
                    "name": name,
                    "lat": 1.3,
                    "lng": 103.8,
                    "address": "123 Main St, Singapore",
                    "place_id": "abc123",
                    "raw_hours": None,
                }

            @staticmethod
            def place_hours(name, city, place_id):
                return None

        import sys
        original = sys.modules.get("services.planner_serp")
        sys.modules["services.planner_serp"] = MockSerp
        try:
            ctx = {
                "session_id": "test",
                "destination": "Singapore",
                "start_date": "2026-09-10",
                "num_days": 2,
                "payload": {},
                "pins": [
                    {"pin_id": "p1", "name": "Place A", "resolved": True,
                     "lat": None, "lng": None, "address": None},
                    {"pin_id": "p2", "name": "Place B", "resolved": True,
                     "lat": None, "lng": None, "address": None},
                    {"pin_id": "p3", "name": "Place C", "resolved": True,
                     "lat": None, "lng": None, "address": None},
                ],
                "research": [],
                "legs": [],
                "schedule": None,
                "critic": None,
                "alternatives": None,
                "itinerary": None,
                "critic_round": 0,
                "current_node": "scout",
                "errors": [],
            }
            _inject_scout_tools(ctx)
            result = run_scout(ctx)
        finally:
            if original is not None:
                sys.modules["services.planner_serp"] = original
            else:
                del sys.modules["services.planner_serp"]

        # All 3 pins in research
        assert result["researched"] == 3
        assert len(ctx["research"]) == 3
        # Pin 2 had LLM failure -> defaults used
        r2 = ctx["research"][1]
        assert r2["dwell_minutes"] == SCOUT_DEFAULTS["dwell_minutes"]
        # Error logged
        assert any("LLM fallback" in e for e in ctx["errors"])

    def test_unresolved_pins_skipped(self):
        """Unresolved pins are not researched."""
        from services import planner_agents
        import sys

        class MockSerp:
            @staticmethod
            def geocode_place(name, city):
                return None

            @staticmethod
            def place_hours(name, city, place_id):
                return None

        original = sys.modules.get("services.planner_serp")
        sys.modules["services.planner_serp"] = MockSerp
        try:
            ctx = {
                "session_id": "test",
                "destination": "Singapore",
                "start_date": "2026-09-10",
                "num_days": 1,
                "payload": {},
                "pins": [
                    {"pin_id": "p1", "name": "Place A", "resolved": False},
                    {"pin_id": "p2", "name": "Place B", "resolved": True,
                     "lat": 1.0, "lng": 103.0, "address": "Addr"},
                ],
                "research": [],
                "legs": [],
                "schedule": None,
                "critic": None,
                "alternatives": None,
                "itinerary": None,
                "critic_round": 0,
                "current_node": "scout",
                "errors": [],
            }

            # Need to also mock the LLM
            planner_agents.call_openrouter_json = lambda *a, **kw: {
                "category": "museum",
                "dwell_minutes": 60,
                "booking_required": False,
                "tip": "Book ahead",
            }
            _inject_scout_tools(ctx)
            result = run_scout(ctx)
        finally:
            if original is not None:
                sys.modules["services.planner_serp"] = original
            else:
                del sys.modules["services.planner_serp"]

        assert result["researched"] == 1
        assert len(ctx["research"]) == 1
        assert ctx["research"][0]["name"] == "Place B"


# ---------------------------------------------------------------------------
# Fixture-driven regression tests (parse_raw_hours vs the REAL fixture data)
# ---------------------------------------------------------------------------
from services.planner_fixtures import FIXTURE_PLACES


class TestParseRawHoursFixtures:
    """parse_raw_hours must handle the exact raw_hours strings in FIXTURE_PLACES,
    including bare-hour ranges ("9-9 PM"), split windows ("2-4:30 PM"), and
    "12 AM" closes ("5 AM-12 AM" -> 05:00-23:59)."""

    def test_all_fixture_places_parse(self):
        for name, place in FIXTURE_PLACES.items():
            days = parse_raw_hours(place["raw_hours"])["days"]
            assert len(days) == 7, f"{name}: expected 7 day keys, got {len(days)}"
            for day_idx, intervals in days.items():
                for iv in intervals:
                    assert iv["open"] < iv["close"], (
                        f"{name} day {day_idx}: invalid interval {iv}"
                    )

    def test_national_gallery_closed_monday(self):
        days = parse_raw_hours(FIXTURE_PLACES["National Gallery Singapore"]["raw_hours"])["days"]
        assert days["0"] == []
        assert len(days["1"]) == 1

    def test_sultan_mosque_two_sorted_intervals(self):
        days = parse_raw_hours(FIXTURE_PLACES["Sultan Mosque"]["raw_hours"])["days"]
        assert len(days["0"]) == 2, days["0"]
        assert days["0"][0]["open"] <= days["0"][0]["close"]
        assert days["0"][0]["close"] <= days["0"][1]["open"]

    def test_gardens_by_the_bay_bare_hour_range(self):
        days = parse_raw_hours(FIXTURE_PLACES["Gardens by the Bay"]["raw_hours"])["days"]
        assert days["0"] == [{"open": "09:00", "close": "21:00"}], days["0"]

    def test_botanic_gardens_12am_close(self):
        days = parse_raw_hours(FIXTURE_PLACES["Singapore Botanic Gardens"]["raw_hours"])["days"]
        assert days["0"] == [{"open": "05:00", "close": "23:59"}], days["0"]

    def test_maxwell_food_centre(self):
        days = parse_raw_hours(FIXTURE_PLACES["Maxwell Food Centre"]["raw_hours"])["days"]
        assert days["0"] == [{"open": "08:00", "close": "22:00"}], days["0"]
