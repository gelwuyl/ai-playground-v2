"""Deterministic tests for scheduler directive semantics + graded remediation.

Covers: each directive action (move_before/move_after/move_to_day/
compress_dwell/drop), every rejection path (unknown stop/reference, out-of-range
day, hard-constraint violation reverting a directive), the compress_dwell floor
and time recompute, advisory_note arithmetic on a closing-squeeze fixture,
drop-to-unplaced, determinism, and meal_fit anchoring.

No network, no DB, no LLM.
Run: .venv/bin/python -m pytest tests/test_planner_directives.py -q
"""

import pytest

from services.planner_scheduler import (
    schedule_trip,
    DIRECTIVE_COMPRESS_FLOOR,
)


# ---------------------------------------------------------------------------
# Fixture builders (mirror test_planner_scheduler.py)
# ---------------------------------------------------------------------------
def make_pin(pin_id, name, seq, lat=1.0, lng=103.0, neighborhood="",
             category="museum", dwell=60, hours=None, meal_fit=None):
    """Build a PlaceResearch-shaped pin compactly."""
    return {
        "pin_id": pin_id,
        "name": name,
        "seq": seq,
        "lat": lat,
        "lng": lng,
        "neighborhood": neighborhood,
        "category": category,
        "dwell_minutes": dwell,
        "opening_hours": hours,
        "meal_fit": meal_fit,
    }


def make_leg(a, b, mode="drive", minutes=15):
    """Build a Leg dict."""
    return {
        "from_name": a,
        "to_name": b,
        "chosen_mode": mode,
        "chosen_minutes": minutes,
        "walk_minutes": minutes if mode == "walk" else None,
        "transit_minutes": minutes if mode == "transit" else None,
        "drive_minutes": minutes if mode == "drive" else None,
        "distance_km": 1.0,
        "estimated": False,
    }


def hours_9to5():
    """Standard 09:00-17:00 every day."""
    return {
        "days": {str(d): [{"open": "09:00", "close": "17:00"}] for d in range(7)}
    }


def hours_closed_on(dow):
    """Open 09:00-17:00 every day except `dow` (which is closed)."""
    days = {}
    for d in range(7):
        if d == dow:
            days[str(d)] = []
        else:
            days[str(d)] = [{"open": "09:00", "close": "17:00"}]
    return {"days": days}


def hours_open_close(open_s, close_s):
    """Same open->close interval on every day-of-week."""
    return {
        "days": {str(d): [{"open": open_s, "close": close_s}] for d in range(7)}
    }


def day_slots(result, day_index):
    """Return the named stop slots (kind stop or anchored meal) for a day."""
    return [
        s for s in result["days"][day_index]["slots"]
        if s["kind"] in ("stop", "meal") and s.get("pin_id")
    ]


def slot_by_name(result, day_index, name):
    for s in day_slots(result, day_index):
        if s["name"] == name:
            return s
    return None


# ---------------------------------------------------------------------------
# Directive actions: each applied
# ---------------------------------------------------------------------------
class TestDirectiveActions:
    def test_move_before_applied(self):
        pins = [
            make_pin("pA", "Place A", 0, 1.28, 103.85, dwell=120, hours=hours_9to5()),
            make_pin("pB", "Place B", 1, 1.29, 103.86, dwell=120, hours=hours_9to5()),
        ]
        legs = [make_leg("Place A", "Place B", "walk", 60)]
        # Natural travel order starts at min-seq (Place A) => A before B.
        directives = [
            {"action": "move_before", "stop": "Place B", "reference": "Place A", "reason": "want B early"},
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 1, directives=directives)
        stops = day_slots(result, 0)
        names = [s["name"] for s in stops]
        assert names == ["Place B", "Place A"], names
        assert result["applied_directives"][0]["status"] == "applied"
        # Times recomputed for the new order: B first at 09:00.
        b = slot_by_name(result, 0, "Place B")
        assert b["start_min"] == 540

    def test_move_after_applied(self):
        pins = [
            make_pin("pA", "Place A", 0, 1.28, 103.85, dwell=120, hours=hours_9to5()),
            make_pin("pB", "Place B", 1, 1.29, 103.86, dwell=120, hours=hours_9to5()),
        ]
        legs = [make_leg("Place A", "Place B", "walk", 60)]
        directives = [
            {"action": "move_after", "stop": "Place A", "reference": "Place B", "reason": "want A later"},
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 1, directives=directives)
        stops = day_slots(result, 0)
        names = [s["name"] for s in stops]
        assert names == ["Place B", "Place A"], names
        assert result["applied_directives"][0]["status"] == "applied"

    def test_move_to_day_applied(self):
        # Different neighborhoods -> natural day split (North day 0, South day 1).
        pins = [
            make_pin("pA", "Place A", 0, 1.28, 103.85, neighborhood="North", dwell=60, hours=hours_9to5()),
            make_pin("pB", "Place B", 1, 1.29, 103.86, neighborhood="South", dwell=60, hours=hours_9to5()),
        ]
        legs = [make_leg("Place A", "Place B", "walk", 15)]
        directives = [
            {"action": "move_to_day", "stop": "Place B", "day": 0, "reason": "pair with A"},
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 2, directives=directives)
        assert result["applied_directives"][0]["status"] == "applied"
        assert slot_by_name(result, 0, "Place B") is not None
        assert slot_by_name(result, 1, "Place B") is None

    def test_compress_dwell_applied_and_times_recompute(self):
        pins = [
            make_pin("pA", "Museum A", 0, 1.28, 103.85, dwell=300, hours=hours_9to5()),
        ]
        directives = [
            {"action": "compress_dwell", "stop": "Museum A", "dwell_minutes": 120, "reason": "shorten"},
        ]
        result = schedule_trip(pins, [], "2026-09-07", 1, directives=directives)
        assert result["applied_directives"][0]["status"] == "applied"
        a = slot_by_name(result, 0, "Museum A")
        assert a is not None
        assert a["dwell_minutes"] == 120
        # Times recomputed from the NEW dwell: 09:00 + 120 = 11:00 (was 14:00).
        assert a["start_min"] == 540
        assert a["end_min"] == 660

    def test_drop_applied_lands_in_unplaced(self):
        pins = [
            make_pin("pA", "Place A", 0, 1.28, 103.85, dwell=60, hours=hours_9to5()),
            make_pin("pB", "Place B", 1, 1.29, 103.86, dwell=60, hours=hours_9to5()),
        ]
        legs = [make_leg("Place A", "Place B", "walk", 15)]
        reason = "skip: closed for renovation"
        directives = [
            {"action": "drop", "stop": "Place B", "reason": reason},
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 1, directives=directives)
        assert result["applied_directives"][0]["status"] == "applied"
        # Place B excluded from scheduling.
        assert slot_by_name(result, 0, "Place B") is None
        assert slot_by_name(result, 0, "Place A") is not None
        # ... and lands in unplaced with the directive's reason.
        unplaced = [u for u in result["unplaced"] if u["name"] == "Place B"]
        assert len(unplaced) == 1
        assert unplaced[0]["reason"] == reason


# ---------------------------------------------------------------------------
# Rejection paths
# ---------------------------------------------------------------------------
class TestDirectiveRejections:
    def test_unknown_stop_rejected(self):
        pins = [make_pin("pA", "Place A", 0, dwell=60, hours=hours_9to5())]
        directives = [
            {"action": "drop", "stop": "Ghost Town", "reason": "nope"},
        ]
        result = schedule_trip(pins, [], "2026-09-07", 1, directives=directives)
        entry = result["applied_directives"][0]
        assert entry["status"] == "rejected"
        assert entry["reject_reason"] == "unknown stop"

    def test_unknown_reference_rejected(self):
        pins = [
            make_pin("pA", "Place A", 0, dwell=60, hours=hours_9to5()),
            make_pin("pB", "Place B", 1, dwell=60, hours=hours_9to5()),
        ]
        directives = [
            {"action": "move_before", "stop": "Place A", "reference": "Ghost Town"},
        ]
        result = schedule_trip(pins, [], "2026-09-07", 1, directives=directives)
        entry = result["applied_directives"][0]
        assert entry["status"] == "rejected"
        assert entry["reject_reason"] == "unknown reference stop"

    def test_move_to_day_out_of_range_rejected(self):
        pins = [
            make_pin("pA", "Place A", 0, dwell=60, hours=hours_9to5()),
            make_pin("pB", "Place B", 1, dwell=60, hours=hours_9to5()),
        ]
        directives = [
            {"action": "move_to_day", "stop": "Place B", "day": 5, "reason": "oops"},
            {"action": "move_to_day", "stop": "Place B", "day": -1, "reason": "oops"},
        ]
        result = schedule_trip(pins, [], "2026-09-07", 2, directives=directives)
        for entry in result["applied_directives"]:
            assert entry["status"] == "rejected"
            assert entry["reject_reason"] == "day out of range"

    def test_compress_dwell_below_floor_rejected_not_clamped(self):
        pins = [make_pin("pA", "Place A", 0, dwell=60, hours=hours_9to5())]
        directives = [
            {"action": "compress_dwell", "stop": "Place A", "dwell_minutes": DIRECTIVE_COMPRESS_FLOOR - 5},
        ]
        result = schedule_trip(pins, [], "2026-09-07", 1, directives=directives)
        entry = result["applied_directives"][0]
        assert entry["status"] == "rejected"
        assert f"dwell_minutes must be >= {DIRECTIVE_COMPRESS_FLOOR}" in entry["reject_reason"]

    def test_unknown_action_rejected(self):
        pins = [make_pin("pA", "Place A", 0, dwell=60, hours=hours_9to5())]
        directives = [
            {"action": "frobnicate", "stop": "Place A"},
        ]
        result = schedule_trip(pins, [], "2026-09-07", 1, directives=directives)
        entry = result["applied_directives"][0]
        assert entry["status"] == "rejected"
        assert "unknown action" in entry["reject_reason"]

    def test_self_reference_rejected(self):
        pins = [make_pin("pA", "Place A", 0, dwell=60, hours=hours_9to5())]
        directives = [
            {"action": "move_before", "stop": "Place A", "reference": "Place A"},
        ]
        result = schedule_trip(pins, [], "2026-09-07", 1, directives=directives)
        entry = result["applied_directives"][0]
        assert entry["status"] == "rejected"
        assert entry["reject_reason"] == "reference must differ from stop"

    def test_malformed_directive_rejected_no_crash(self):
        """A non-dict entry in the directives list is rejected, never a crash."""
        pins = [make_pin("pA", "Place A", 0, dwell=60, hours=hours_9to5())]
        directives = ["not a dict"]
        result = schedule_trip(pins, [], "2026-09-07", 1, directives=directives)
        entry = result["applied_directives"][0]
        assert entry["status"] == "rejected"
        assert entry["reject_reason"] == "malformed directive"
        # The pin is still scheduled normally.
        assert slot_by_name(result, 0, "Place A") is not None

    def test_duplicate_ordering_directive_second_rejected(self):
        """Two ordering directives on the same stop: first wins honestly, the
        duplicate is rejected rather than silently ignored by setdefault."""
        pins = [
            make_pin("pA", "Place A", 0, 1.28, 103.85, dwell=120, hours=hours_9to5()),
            make_pin("pB", "Place B", 1, 1.29, 103.86, dwell=120, hours=hours_9to5()),
        ]
        legs = [make_leg("Place A", "Place B", "walk", 60)]
        directives = [
            {"action": "move_before", "stop": "Place B", "reference": "Place A"},
            {"action": "move_after", "stop": "Place B", "reference": "Place A"},
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 1, directives=directives)
        assert result["applied_directives"][0]["status"] == "applied"
        assert result["applied_directives"][1]["status"] == "rejected"
        assert result["applied_directives"][1]["reject_reason"] == "duplicate ordering directive for stop"
        # First (move_before) still honored.
        names = [s["name"] for s in day_slots(result, 0)]
        assert names == ["Place B", "Place A"], names

    def test_hard_constraint_violation_reverts_directive(self):
        """move_to_day onto a day the stop is closed on is REJECTED and the
        pin is reverted to its feasible placement (never dropped silently)."""
        # 2026-09-07 = Mon (dow 0), 2026-09-08 = Tue (dow 1).
        # Place B is closed Tue (dow 1), open Mon. move_to_day day 1 must fail.
        pins = [
            make_pin("pA", "Place A", 0, 1.28, 103.85, neighborhood="X", dwell=90, hours=hours_9to5()),
            make_pin("pB", "Place B", 1, 1.29, 103.86, neighborhood="X", dwell=90, hours=hours_closed_on(1)),
        ]
        legs = [make_leg("Place A", "Place B", "walk", 15)]
        directives = [
            {"action": "move_to_day", "stop": "Place B", "day": 1, "reason": "want it Tue"},
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 2, directives=directives)
        entry = result["applied_directives"][0]
        assert entry["status"] == "rejected"
        assert "could not place stop on day 1" == entry["reject_reason"]
        # Place B reverted and still scheduled on its feasible day (Mon/day 0).
        assert slot_by_name(result, 0, "Place B") is not None, result["unplaced"]
        assert not any(u["name"] == "Place B" for u in result["unplaced"])

    def test_ordering_reference_on_different_day_rejected(self):
        pins = [
            make_pin("pA", "Place A", 0, 1.28, 103.85, neighborhood="North", dwell=60, hours=hours_9to5()),
            make_pin("pB", "Place B", 1, 1.29, 103.86, neighborhood="South", dwell=60, hours=hours_9to5()),
        ]
        legs = [make_leg("Place A", "Place B", "walk", 15)]
        directives = [
            {"action": "move_before", "stop": "Place B", "reference": "Place A"},
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 2, directives=directives)
        entry = result["applied_directives"][0]
        assert entry["status"] == "rejected"
        assert entry["reject_reason"] == "reference on a different day"
        # Both still scheduled (one per day) — nothing dropped.
        assert slot_by_name(result, 0, "Place A") is not None
        assert slot_by_name(result, 1, "Place B") is not None


# ---------------------------------------------------------------------------
# Graded remediation: squeeze fixture -> reorder then compress
# ---------------------------------------------------------------------------
class TestGradedRemediation:
    def _squeeze_pins(self):
        """Museum A (09:00-18:00, 300 min) then a 120-min leg to Gallery B
        (11:00-17:30, 120 min) leaves B arriving at 16:00 and ending 18:00,
        past B's 17:30 close — B is unplaced in the natural schedule."""
        h_a = hours_open_close("09:00", "18:00")
        h_b = hours_open_close("11:00", "17:30")
        pins = [
            make_pin("pA", "Museum A", 0, 1.28, 103.85, dwell=300, hours=h_a),
            make_pin("pB", "Gallery B", 1, 1.29, 103.86, dwell=120, hours=h_b),
        ]
        legs = [make_leg("Museum A", "Gallery B", "walk", 120)]
        return pins, legs

    def test_base_squeeze_leaves_b_unplaced(self):
        pins, legs = self._squeeze_pins()
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        unplaced_names = [u["name"] for u in result["unplaced"]]
        assert "Gallery B" in unplaced_names

    def test_reorder_then_compress_remediates(self):
        """Least-destructive-first: reorder (move B before A), then compress A's
        dwell so both fit — Gallery B placed AND Museum A kept (not dropped)."""
        pins, legs = self._squeeze_pins()
        directives = [
            {"action": "move_before", "stop": "Gallery B", "reference": "Museum A",
             "reason": "B closes 17:30; visit it first"},
            {"action": "compress_dwell", "stop": "Museum A", "dwell_minutes": 120,
             "reason": "free the afternoon slot"},
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 1, directives=directives)
        assert result["unplaced"] == [], result["unplaced"]
        assert result["applied_directives"][0]["status"] == "applied"
        assert result["applied_directives"][1]["status"] == "applied"

        stops = day_slots(result, 0)
        names = [s["name"] for s in stops]
        assert names == ["Gallery B", "Museum A"], names
        # Gallery B first inside its 11:00-17:30 window; A compressed to 120.
        b = slot_by_name(result, 0, "Gallery B")
        a = slot_by_name(result, 0, "Museum A")
        assert b["start_min"] == 660 and b["end_min"] == 780
        assert a["dwell_minutes"] == 120
        # Times recomputed: A 15:00-17:00 (inside 18:00 close).
        assert a["start_min"] == 900 and a["end_min"] == 1020

    def test_advisory_note_tight_closing_arithmetic(self):
        """A visit ending within 15 min of closing carries an advisory_note
        computed from real times (contract 4), <= 140 chars, advisory only."""
        # Museum A 09:00-20:00 dwell 400 (ends 15:40); leg 60 to Gallery B
        # (11:00-17:30) dwell 40 => arrive 16:40, end 17:20 = 10 min before close.
        h_a = hours_open_close("09:00", "20:00")
        h_b = hours_open_close("11:00", "17:30")
        pins = [
            make_pin("pA", "Museum A", 0, 1.28, 103.85, dwell=400, hours=h_a),
            make_pin("pB", "Gallery B", 1, 1.29, 103.86, dwell=40, hours=h_b),
        ]
        legs = [make_leg("Museum A", "Gallery B", "walk", 60)]
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        b = slot_by_name(result, 0, "Gallery B")
        assert b is not None
        assert b["start_min"] == 1000 and b["end_min"] == 1040  # 16:40-17:20
        note = b["advisory_note"]
        assert note == "Tight: closes 17:30, arrive 16:40 — keep visit to ~30 min", note
        assert len(note) <= 140
        # Museum A ends at 15:40 with a 20:00 close — not tight, no note.
        a = slot_by_name(result, 0, "Museum A")
        assert "advisory_note" not in a

    def test_advisory_note_compressed_dwell(self):
        pins, legs = self._squeeze_pins()
        directives = [
            {"action": "compress_dwell", "stop": "Museum A", "dwell_minutes": 120},
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 1, directives=directives)
        # A compressed but still scheduled (B stays unplaced here — directive
        # only about A). The slot carries the compressed-dwell advisory.
        a = slot_by_name(result, 0, "Museum A")
        assert a is not None
        assert a["dwell_minutes"] == 120
        assert a["advisory_note"] == "Dwell compressed 300->120 min by directive"

    def test_advisory_only_advisory_stop_stays_scheduled(self):
        """The tight stop is NEVER dropped for being tight — note only."""
        pins, legs = self._squeeze_pins()
        # Compress alone does not fix B (still unplaced) — but a directive that
        # reorders B first DOES, and B stays scheduled with no note required.
        directives = [
            {"action": "move_before", "stop": "Gallery B", "reference": "Museum A"},
            {"action": "compress_dwell", "stop": "Museum A", "dwell_minutes": 120},
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 1, directives=directives)
        assert slot_by_name(result, 0, "Gallery B") is not None
        assert result["applied_directives"][0]["status"] == "applied"

    def test_advisory_note_squeezed_meal(self):
        """A packed day where _insert_meal_by_shift succeeds carries the
        squeezed-meal note on the meal slot with correct times."""
        # Long Visit A 09:00-12:00, 15-min leg, Long Visit B 12:15-15:15.
        # Span 375 >= 360. No free 60-min gap -> meal shift-inserted 12:00-13:00,
        # B pushed +60 to 13:15-16:15 (still inside 09:00-17:00 hours).
        pins = [
            make_pin("p1", "Long Visit A", 0, 1.28, 103.85, dwell=180, hours=hours_9to5()),
            make_pin("p2", "Long Visit B", 1, 1.29, 103.86, dwell=180, hours=hours_9to5()),
        ]
        legs = [make_leg("Long Visit A", "Long Visit B", "walk", 15)]
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        meals = [s for s in result["days"][0]["slots"] if s["kind"] == "meal"]
        assert len(meals) == 1, result["days"][0]["slots"]
        meal = meals[0]
        assert meal["start_min"] == 720 and meal["end_min"] == 780  # 12:00-13:00
        note = meal["advisory_note"]
        assert note == "Meal squeezed between visits (12:00-13:00) to fit a 60-min window", note
        assert len(note) <= 140


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDirectiveDeterminism:
    def test_same_input_directives_same_output(self):
        pins = [
            make_pin("pA", "Place A", 0, 1.28, 103.85, neighborhood="North", dwell=120, hours=hours_9to5()),
            make_pin("pB", "Place B", 1, 1.29, 103.86, neighborhood="North", dwell=60, hours=hours_9to5()),
            make_pin("pC", "Place C", 2, 1.40, 103.90, neighborhood="South", dwell=90, hours=hours_9to5()),
        ]
        legs = [make_leg("Place A", "Place B", "walk", 10), make_leg("Place B", "Place C", "walk", 20)]
        directives = [
            {"action": "compress_dwell", "stop": "Place B", "dwell_minutes": 30},
            {"action": "move_to_day", "stop": "Place C", "day": 0},
        ]
        r1 = schedule_trip(pins, legs, "2026-09-07", 2, directives=directives)
        r2 = schedule_trip(pins, legs, "2026-09-07", 2, directives=directives)
        assert r1 == r2

    def test_no_directives_unchanged_shape(self):
        """Calling without directives behaves as before (empty list reported)."""
        pins = [make_pin("pA", "Place A", 0, dwell=60, hours=hours_9to5())]
        result = schedule_trip(pins, [], "2026-09-07", 1)
        assert result["applied_directives"] == []
        assert "days" in result and "unplaced" in result


# ---------------------------------------------------------------------------
# meal_fit anchoring (contract 5)
# ---------------------------------------------------------------------------
def _lunch_anchor_pins(include_meal_fit):
    """Two food stops on a >= 6h day.  'Cafe Near Noon' sits nearer midday;
    'Lunch Spot' is later (13:45).  With meal_fit present, Lunch Spot carries
    meal_fit='lunch' and Cafe carries 'dinner' -> only Lunch Spot matches a
    lunch meal, so the meal anchors there despite being farther from noon.
    Without meal_fit, the legacy nearest-midday pick (Cafe) anchors."""
    h = hours_open_close("09:00", "18:00")
    pins = [
        make_pin("pC", "Cafe Near Noon", 0, 1.28, 103.85, category="cafe", dwell=180, hours=h,
                 meal_fit="dinner" if include_meal_fit else None),
        make_pin("pL", "Lunch Spot", 1, 1.29, 103.86, category="restaurant", dwell=180, hours=h,
                 meal_fit="lunch" if include_meal_fit else None),
    ]
    legs = [make_leg("Cafe Near Noon", "Lunch Spot", "walk", 15)]
    return pins, legs


class TestMealFitAnchoring:
    def test_lunch_fit_venue_anchors_meal(self):
        pins, legs = _lunch_anchor_pins(include_meal_fit=True)
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        day = result["days"][0]
        # Day spans >= 360 => a meal slot exists.
        meals = [s for s in day["slots"] if s["kind"] == "meal"]
        assert len(meals) == 1, day["slots"]
        # Anchored at the meal_fit='lunch' venue, not the nearer-midday cafe.
        assert meals[0]["pin_id"] == "pL"
        assert meals[0]["name"] == "Lunch Spot"
        assert not any(s["name"] == "Meal" for s in day["slots"])

    def test_missing_meal_fit_keeps_old_behavior(self):
        pins, legs = _lunch_anchor_pins(include_meal_fit=False)
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        day = result["days"][0]
        meals = [s for s in day["slots"] if s["kind"] == "meal"]
        assert len(meals) == 1, day["slots"]
        # Legacy behavior: nearest-midday food stop anchors (Cafe, at 10:30).
        assert meals[0]["pin_id"] == "pC"

    def test_no_food_pin_generic_meal_fallback(self):
        """No food stop => generic Meal slot stays the fallback."""
        h = hours_open_close("09:00", "18:00")
        pins = [
            make_pin("pA", "Museum A", 0, 1.28, 103.85, dwell=240, hours=h),
            make_pin("pB", "Museum B", 1, 1.29, 103.86, dwell=180, hours=h),
        ]
        legs = [make_leg("Museum A", "Museum B", "walk", 30)]
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        meals = [s for s in result["days"][0]["slots"] if s["kind"] == "meal"]
        assert len(meals) == 1, result["days"][0]["slots"]
        assert meals[0]["pin_id"] is None

    def test_dinner_fit_venue_anchors_meal(self):
        """A food pin with meal_fit='dinner' is anchored at a dinner-time slot."""
        h = hours_open_close("08:00", "22:00")
        pins = [
            make_pin("pA", "Museum M", 0, 1.28, 103.85, category="museum", dwell=300, hours=h),
            make_pin("pD", "Dinner Spot", 1, 1.29, 103.86, category="restaurant",
                     dwell=120, hours=h, meal_fit="dinner"),
        ]
        legs = [make_leg("Museum M", "Dinner Spot", "walk", 30)]
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        day = result["days"][0]
        # Museum 09:00-14:00, Dinner Spot 14:30-16:30; span >= 360 => meal exists.
        meals = [s for s in day["slots"] if s["kind"] == "meal"]
        assert len(meals) == 1, day["slots"]
        assert meals[0]["pin_id"] == "pD"
        assert meals[0]["name"] == "Dinner Spot"
        # Dinner-time meal slot (mid-visit >= 15:00).
        mid = (meals[0]["start_min"] + meals[0]["end_min"]) // 2
        assert mid >= 15 * 60
        assert not any(s["name"] == "Meal" for s in day["slots"])
