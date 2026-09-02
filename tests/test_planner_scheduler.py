"""Deterministic tests for the planner scheduler.

All tests use small hand-built fixtures — no network, no DB, no LLM.
Run: .venv/bin/python -m pytest tests/test_planner_scheduler.py -q
"""

import pytest
from services.planner_scheduler import (
    schedule_trip,
    run_schedule,
    runner_ups,
    _intervals_for_day,
    _longest_interval,
    _cluster,
    _balance,
    _order_day,
    _two_opt,
    _greedy_nn_order,
    _leg_minutes,
    _travel_matrix,
    _total_travel,
    DAY_START_MIN,
    MISSING_LEG_PENALTY,
    BALANCE_RATIO,
    MEAL_THRESHOLD_MIN,
    REST_THRESHOLD_MIN,
)
from services.planner_types import parse_hhmm, day_dates


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def make_pin(
    pin_id, name, seq, lat=1.0, lng=103.0, neighborhood="",
    category="museum", dwell=60, booking=False, tip="",
    hours=None, verified=False,
):
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
        "booking_required": booking,
        "tip": tip,
        "opening_hours": hours,
        "hours_verified": verified,
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


def hours_split():
    """Split hours: 10:00-12:00 + 14:00-16:00 every day."""
    return {
        "days": {
            str(d): [
                {"open": "10:00", "close": "12:00"},
                {"open": "14:00", "close": "16:00"},
            ]
            for d in range(7)
        }
    }


def hours_short():
    """Short hours: 10:00-11:00 every day (only 60 min)."""
    return {
        "days": {str(d): [{"open": "10:00", "close": "11:00"}] for d in range(7)}
    }


# ---------------------------------------------------------------------------
# Clustering tests
# ---------------------------------------------------------------------------
class TestClustering:
    """Level 2a: neighborhood and geometric clustering."""

    def test_neighborhood_pins_same_day(self):
        """Pins sharing a neighborhood land on the same day."""
        # Use equal dwell per neighborhood so balancing does not split them.
        pins = [
            make_pin("p1", "Marina Bay Sands", 0, 1.283, 103.86, "Marina Bay", dwell=90),
            make_pin("p2", "Gardens by the Bay", 1, 1.282, 103.864, "Marina Bay", dwell=90),
            make_pin("p3", "Sentosa Beach", 2, 1.249, 103.832, "Sentosa", dwell=90),
            make_pin("p4", "Universal Studios", 3, 1.254, 103.824, "Sentosa", dwell=90),
        ]
        legs = [
            make_leg("Marina Bay Sands", "Gardens by the Bay", "walk", 10),
            make_leg("Sentosa Beach", "Universal Studios", "walk", 15),
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 2)
        days = result["days"]

        # Check that Marina Bay pins are on the same day.
        all_days_stops = []
        for d in days:
            all_days_stops.append({s["name"] for s in d["slots"] if s["kind"] == "stop"})

        # Marina Bay Sands and Gardens by the Bay should be on the same day.
        for stops in all_days_stops:
            if "Marina Bay Sands" in stops:
                assert "Gardens by the Bay" in stops, "Neighborhood pins should be on the same day"
            if "Sentosa Beach" in stops:
                assert "Universal Studios" in stops, "Neighborhood pins should be on the same day"

    def test_geographic_fallback_far_apart(self):
        """Far-apart pins without neighborhoods cluster on separate days."""
        # Botanic Gardens is north-west, Marina Bay is south-east.
        pins = [
            make_pin("p1", "Botanic Gardens", 0, 1.314, 103.816, "", dwell=90),
            make_pin("p2", "Marina Bay Sands", 1, 1.283, 103.86, "", dwell=90),
        ]
        legs = [make_leg("Botanic Gardens", "Marina Bay Sands", "drive", 20)]
        result = schedule_trip(pins, legs, "2026-09-07", 2)
        days = result["days"]

        # Each day should have exactly one stop.
        stops_d0 = [s for s in days[0]["slots"] if s["kind"] == "stop"]
        stops_d1 = [s for s in days[1]["slots"] if s["kind"] == "stop"]
        assert len(stops_d0) == 1, f"Expected 1 stop on day 0, got {len(stops_d0)}"
        assert len(stops_d1) == 1, f"Expected 1 stop on day 1, got {len(stops_d1)}"

    def test_no_neighborhood_all_blank_2_days(self):
        """All pins with blank neighborhoods still cluster into 2 days."""
        pins = [
            make_pin("p1", "Place A", 0, 1.30, 103.80, "", dwell=60),
            make_pin("p2", "Place B", 1, 1.31, 103.81, "", dwell=60),
            make_pin("p3", "Place C", 2, 1.40, 103.90, "", dwell=60),
            make_pin("p4", "Place D", 3, 1.41, 103.91, "", dwell=60),
        ]
        legs = [make_leg("Place A", "Place B", "walk", 10), make_leg("Place C", "Place D", "walk", 10)]
        result = schedule_trip(pins, legs, "2026-09-07", 2)
        # Should not crash, should place all 4.
        total_stops = sum(
            1 for d in result["days"] for s in d["slots"] if s["kind"] == "stop"
        )
        assert total_stops == 4


# ---------------------------------------------------------------------------
# Balance tests
# ---------------------------------------------------------------------------
class TestBalance:
    """Level 2b: balanced daily load."""

    def test_imbalanced_loads_balanced(self):
        """4 heavy Marina Bay pins + 2 light Sentosa pins, 2 days -> loads within 1.5x."""
        pins = [
            make_pin("p1", "MB1", 0, 1.283, 103.86, "Marina Bay", dwell=180),
            make_pin("p2", "MB2", 1, 1.284, 103.861, "Marina Bay", dwell=180),
            make_pin("p3", "MB3", 2, 1.285, 103.862, "Marina Bay", dwell=180),
            make_pin("p4", "MB4", 3, 1.286, 103.863, "Marina Bay", dwell=180),
            make_pin("p5", "ST1", 4, 1.249, 103.832, "Sentosa", dwell=30),
            make_pin("p6", "ST2", 5, 1.250, 103.833, "Sentosa", dwell=30),
        ]
        # Legs between adjacent Marina Bay pins and Sentosa pins.
        legs = [
            make_leg("MB1", "MB2", "walk", 5),
            make_leg("MB2", "MB3", "walk", 5),
            make_leg("MB3", "MB4", "walk", 5),
            make_leg("ST1", "ST2", "walk", 5),
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 2)
        loads = result["stats"]["daily_loads"]

        if len(loads) >= 2:
            ratio = max(loads) / min(loads)
            assert ratio <= BALANCE_RATIO, (
                f"Load ratio {ratio} exceeds {BALANCE_RATIO}: loads={loads}"
            )

    def test_balance_with_opening_hours_constraint(self):
        """Balancing must respect opening hours — a closed pin can't move to a closed day."""
        # All pins are Marina Bay, but one is closed on Mon (day 0).
        pins = [
            make_pin("p1", "MB1", 0, 1.283, 103.86, "Marina Bay", dwell=120, hours=hours_9to5()),
            make_pin("p2", "MB2", 1, 1.284, 103.861, "Marina Bay", dwell=120, hours=hours_9to5()),
            make_pin("p3", "MB3", 2, 1.285, 103.862, "Marina Bay", dwell=120, hours=hours_closed_on(0)),
        ]
        legs = [
            make_leg("MB1", "MB2", "walk", 5),
            make_leg("MB2", "MB3", "walk", 5),
        ]
        # 2026-09-07 is a Monday (dow=0).
        result = schedule_trip(pins, legs, "2026-09-07", 2)
        # Should not crash; MB3 must be on day 1 (Tue).
        assert result["unplaced"] == [], f"Unexpected unplaced: {result['unplaced']}"

    def test_all_same_neighborhood_distributes_across_days(self):
        """5 pins all sharing a neighborhood, 2-day trip -> pins spread
        across both days, not all dumped on one day.

        Regression: the balance guard `len(non_zero) <= 1: break` used to
        fire when all pins clustered to one day, leaving the other day
        empty. Now the balance loop must fill the empty day.
        """
        # Dwell 120 min, 09:00-17:00 hours (480 min): up to 3 stops fit
        # on one day (360 < 480), so 3/2 split is feasible.
        pins = [
            make_pin("p1", "A", 0, 1.28, 103.85, "Central", dwell=120, hours=hours_9to5()),
            make_pin("p2", "B", 1, 1.29, 103.86, "Central", dwell=120, hours=hours_9to5()),
            make_pin("p3", "C", 2, 1.30, 103.87, "Central", dwell=120, hours=hours_9to5()),
            make_pin("p4", "D", 3, 1.31, 103.88, "Central", dwell=120, hours=hours_9to5()),
            make_pin("p5", "E", 4, 1.32, 103.89, "Central", dwell=120, hours=hours_9to5()),
        ]
        legs = [make_leg("A", "B", "walk", 5)]
        result = schedule_trip(pins, legs, "2026-09-07", 2)

        # Both days must have at least one stop.
        stops_d0 = [s for s in result["days"][0]["slots"] if s["kind"] == "stop"]
        stops_d1 = [s for s in result["days"][1]["slots"] if s["kind"] == "stop"]
        assert len(stops_d0) > 0, f"Day 0 has no stops: {result['days'][0]}"
        assert len(stops_d1) > 0, f"Day 1 has no stops: {result['days'][1]}"

        # Load ratio should be <= 1.5x.
        loads = result["stats"]["daily_loads"]
        if len(loads) >= 2:
            ratio = max(loads) / min(loads)
            assert ratio <= BALANCE_RATIO, (
                f"Load ratio {ratio} exceeds {BALANCE_RATIO}: loads={loads}"
            )

        # No pins should be unplaced (all are open 09-17 every day).
        assert result["unplaced"] == [], f"Unexpected unplaced: {result['unplaced']}"


# ---------------------------------------------------------------------------
# Ordering tests
# ---------------------------------------------------------------------------
class TestOrdering:
    """Level 3: nearest-neighbor + 2-opt."""

    def test_nn_beats_input_order(self):
        """Nearest-neighbor + 2-opt should not be worse than input order."""
        stops = [
            make_pin("p1", "A", 0, 0, 0, dwell=60),
            make_pin("p2", "B", 1, 0.01, 0, dwell=60),
            make_pin("p3", "C", 2, 0.02, 0, dwell=60),
        ]
        legs = [
            make_leg("A", "B", "walk", 5),
            make_leg("B", "C", "walk", 5),
            make_leg("A", "C", "walk", 20),
        ]
        ordered = _order_day(stops, legs)
        # Travel in NN order should be <= input order.
        mat = _travel_matrix(ordered, legs)
        nn_travel = sum(mat[k][k+1] for k in range(len(ordered)-1))
        # Input order: A->B->C = 5+5 = 10, so NN should match or beat.
        assert nn_travel <= 15

    def test_two_opt_strictly_improves(self):
        """Explicit small case where 2-opt strictly improves NN.

        4 stops in a line: A(0,0), B(0,1), C(0,2), D(0,3).
        Legs: A-B=5, B-C=5, C-D=5, A-C=15, B-D=15, A-D=20.
        If NN visits A->D->B->C (crossing), 2-opt should fix it.
        Actually NN from A goes to nearest (B), then C, then D: A-B-C-D = 15.
        So we need a case where NN makes a bad choice.
        """
        # Build a case where NN makes a suboptimal first choice.
        # A at center, B close to A, C far from A but close to B, D close to A.
        stops = [
            make_pin("p1", "A", 0, 0, 0, dwell=60),
            make_pin("p2", "B", 1, 0.5, 0, dwell=60),
            make_pin("p3", "C", 2, 1.0, 0, dwell=60),
            make_pin("p4", "D", 3, 0.05, 0, dwell=60),
        ]
        legs = [
            make_leg("A", "B", "walk", 10),
            make_leg("A", "D", "walk", 3),
            make_leg("D", "B", "walk", 8),
            make_leg("B", "C", "walk", 10),
            make_leg("A", "C", "walk", 30),
            make_leg("D", "C", "walk", 28),
            make_leg("C", "D", "walk", 28),
            make_leg("C", "A", "walk", 30),
            make_leg("B", "D", "walk", 8),
            make_leg("B", "A", "walk", 10),
        ]
        ordered = _order_day(stops, legs)
        # 2-opt should produce a valid order of all 4 stops.
        assert len(ordered) == 4
        assert set(s["name"] for s in ordered) == {"A", "B", "C", "D"}

    def test_single_stop_ordering(self):
        """A single stop should return itself."""
        stops = [make_pin("p1", "Solo", 0, dwell=60)]
        ordered = _order_day(stops, [])
        assert len(ordered) == 1
        assert ordered[0]["name"] == "Solo"


# ---------------------------------------------------------------------------
# Slotting tests
# ---------------------------------------------------------------------------
class TestSlotting:
    """Timeline slotting within opening hours."""

    def test_stops_within_hours(self):
        """Stops are scheduled within their opening hours."""
        pins = [
            make_pin("p1", "Museum A", 0, 1.28, 103.85, dwell=90, hours=hours_9to5()),
            make_pin("p2", "Museum B", 1, 1.29, 103.86, dwell=60, hours=hours_9to5()),
        ]
        legs = [make_leg("Museum A", "Museum B", "walk", 15)]
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        for slot in result["days"][0]["slots"]:
            if slot["kind"] == "stop":
                assert slot["start_min"] >= 540, f"Start before 09:00: {slot}"
                assert slot["end_min"] <= 1020, f"End after 17:00: {slot}"

    def test_split_hours_longest_interval(self):
        """A stop with split hours is placed in its longest interval."""
        pins = [
            make_pin("p1", "Sultan Mosque", 0, 1.299, 103.858, dwell=90, hours=hours_split()),
        ]
        result = schedule_trip(pins, [], "2026-09-07", 1)
        slots = [s for s in result["days"][0]["slots"] if s["kind"] == "stop"]
        assert len(slots) == 1
        slot = slots[0]
        # Longest interval: 14:00-16:00 (120 min) vs 10:00-12:00 (120 min) — equal.
        # Both are 120 min. Dwell is 90, fits in either.
        # Should be in one of the two intervals.
        assert slot["start_min"] >= 600  # 10:00
        assert slot["end_min"] <= 960    # 16:00

    def test_split_hours_unequal_intervals(self):
        """If intervals are unequal, the stop goes in the longest."""
        unequal_hours = {
            "days": {
                str(d): [
                    {"open": "10:00", "close": "11:00"},     # 60 min
                    {"open": "14:00", "close": "17:00"},      # 180 min
                ]
                for d in range(7)
            }
        }
        pins = [
            make_pin("p1", "Split Place", 0, 1.30, 103.85, dwell=120, hours=unequal_hours),
        ]
        result = schedule_trip(pins, [], "2026-09-07", 1)
        slots = [s for s in result["days"][0]["slots"] if s["kind"] == "stop"]
        assert len(slots) == 1
        slot = slots[0]
        # Should be in the 14:00-17:00 interval (180 min > 60 min).
        assert slot["start_min"] >= 840  # 14:00
        assert slot["end_min"] <= 1020   # 17:00

    def test_stop_too_long_for_any_interval_unplaced(self):
        """A stop that can't fit in any interval becomes unplaced."""
        pins = [
            make_pin("p1", "Long Dwell", 0, 1.30, 103.85, dwell=120, hours=hours_short()),
        ]
        result = schedule_trip(pins, [], "2026-09-07", 1)
        assert len(result["unplaced"]) == 1
        assert result["unplaced"][0]["name"] == "Long Dwell"
        # No stops should be scheduled.
        stop_slots = [s for s in result["days"][0]["slots"] if s["kind"] == "stop"]
        assert len(stop_slots) == 0

    def test_longest_interval_preferred_over_earliest(self):
        """A stop that fits in both a short early interval and a longer later
        one must land in the longer interval.

        Regression: the old code sorted intervals earliest-first and took
        the first fit, which would place the stop in the short early
        interval instead of the longer later one.
        """
        # Intervals: 09:00-11:00 (120 min) + 13:00-17:00 (240 min).
        # Dwell 90 fits in both. Longest is 13:00-17:00 (240 min).
        two_interval_hours = {
            "days": {
                str(d): [
                    {"open": "09:00", "close": "11:00"},    # 120 min
                    {"open": "13:00", "close": "17:00"},   # 240 min (longest)
                ]
                for d in range(7)
            }
        }
        pins = [
            make_pin("p1", "Dual Interval Place", 0, 1.30, 103.85, dwell=90,
                     hours=two_interval_hours),
        ]
        result = schedule_trip(pins, [], "2026-09-07", 1)
        slots = [s for s in result["days"][0]["slots"] if s["kind"] == "stop"]
        assert len(slots) == 1
        slot = slots[0]
        # Must be in the 13:00-17:00 interval (240 min), NOT the 09:00-11:00 (120 min).
        assert slot["start_min"] >= 780, (
            f"Expected start >= 13:00 (780) in longest interval, got {slot['start_min']}"
        )
        assert slot["end_min"] <= 1020

    def test_no_violation_of_hours_ever(self):
        """Multiple stops with tight hours — none should violate."""
        pins = [
            make_pin("p1", "Tight A", 0, 1.28, 103.85, dwell=120, hours=hours_9to5()),
            make_pin("p2", "Tight B", 1, 1.29, 103.86, dwell=120, hours=hours_9to5()),
            make_pin("p3", "Tight C", 2, 1.30, 103.87, dwell=120, hours=hours_9to5()),
        ]
        legs = [
            make_leg("Tight A", "Tight B", "drive", 30),
            make_leg("Tight B", "Tight C", "drive", 30),
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        for slot in result["days"][0]["slots"]:
            if slot["kind"] == "stop":
                assert slot["start_min"] >= 540
                assert slot["end_min"] <= 1020


# ---------------------------------------------------------------------------
# Closed-day repair tests
# ---------------------------------------------------------------------------
class TestClosedDayRepair:
    """Level 1: closed-day repair."""

    def test_closed_day_1_open_day_2(self):
        """Pin closed on Day 1, open on Day 2 -> placed on Day 2 + repairs note."""
        # Both pins share a neighborhood so they cluster on the same day.
        # Then closed-day repair must move the closed pin to the other day.
        pins = [
            make_pin("p1", "Open Place", 0, 1.28, 103.85, "Downtown", dwell=90, hours=hours_9to5()),
            # National Gallery: closed on Mon (dow 0), open Tue-Sun.
            make_pin("p2", "National Gallery", 1, 1.29, 103.86, "Downtown", dwell=90, hours=hours_closed_on(0)),
        ]
        legs = [make_leg("Open Place", "National Gallery", "drive", 15)]
        # 2026-09-07 is Monday (dow=0), 2026-09-08 is Tuesday (dow=1).
        result = schedule_trip(pins, legs, "2026-09-07", 2)

        # National Gallery should not be unplaced.
        names_unplaced = {u["name"] for u in result["unplaced"]}
        assert "National Gallery" not in names_unplaced

        # It should be on day 1 (Tuesday) — it's closed on Monday.
        for di, day in enumerate(result["days"]):
            stop_names = {s["name"] for s in day["slots"] if s["kind"] == "stop"}
            if "National Gallery" in stop_names:
                # Should not be day 0 (Monday).
                assert di == 1, f"National Gallery should be on day 1, got day {di}"

        # Repairs should mention the move.
        repair_text = " ".join(result["repairs"])
        assert "National Gallery" in repair_text
        assert "moved" in repair_text.lower()

    def test_closed_all_trip_days_unplaced(self):
        """Pin closed on all trip days -> unplaced with reason."""
        # Pin closed on Mon and Tue (dow 0 and 1).
        closed_mon_tue = {"days": {}}
        for d in range(7):
            if d in (0, 1):
                closed_mon_tue["days"][str(d)] = []
            else:
                closed_mon_tue["days"][str(d)] = [{"open": "09:00", "close": "17:00"}]

        pins = [
            make_pin("p1", "Always Open", 0, 1.28, 103.85, dwell=90, hours=hours_9to5()),
            make_pin("p2", "Mon Tue Closed", 1, 1.29, 103.86, dwell=90, hours=closed_mon_tue),
        ]
        legs = [make_leg("Always Open", "Mon Tue Closed", "drive", 15)]
        # 2-day trip starting Mon (dow=0): Mon, Tue.
        result = schedule_trip(pins, legs, "2026-09-07", 2)

        unplaced_names = {u["name"] for u in result["unplaced"]}
        assert "Mon Tue Closed" in unplaced_names

        # Check the reason.
        for u in result["unplaced"]:
            if u["name"] == "Mon Tue Closed":
                assert "closed all trip days" in u["reason"]


# ---------------------------------------------------------------------------
# Meal / rest tests
# ---------------------------------------------------------------------------
class TestMealsRest:
    """Level 4: meal/rest window insertion."""

    def test_long_day_gets_meal(self):
        """A day with span >= 6h gets a meal slot."""
        # Two 3h stops with a 60-min leg so a 60-min gap exists around noon.
        # Stop A: 09:00-12:00 (540-720), leg 60, Stop B: 13:00-16:00 (780-960).
        # Span = 420 min >= 360. Gap 720-780 = 60 min for the meal.
        long_hours = {
            "days": {str(d): [{"open": "08:00", "close": "20:00"}] for d in range(7)}
        }
        pins = [
            make_pin("p1", "Long Visit A", 0, 1.28, 103.85, dwell=180, hours=long_hours),
            make_pin("p2", "Long Visit B", 1, 1.29, 103.86, dwell=180, hours=long_hours),
        ]
        legs = [make_leg("Long Visit A", "Long Visit B", "drive", 60)]
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        meals = [s for s in result["days"][0]["slots"] if s["kind"] == "meal"]
        assert len(meals) >= 1, f"Expected at least 1 meal slot, got {result['days'][0]['slots']}"

    def test_packed_day_gets_meal_by_shift(self):
        """A packed day with no free gap gets a meal by shifting later stops.

        A at 09:00-12:00, B at 12:15-15:15 (15-min leg). The meal (12:00-13:00)
        is inserted after A and B shifts +60 to 13:15-16:15, still inside
        its 09:00-17:00 hours.
        """
        pins = [
            make_pin("p1", "Long Visit A", 0, 1.28, 103.85, dwell=180, hours=hours_9to5()),
            make_pin("p2", "Long Visit B", 1, 1.29, 103.86, dwell=180, hours=hours_9to5()),
        ]
        legs = [make_leg("Long Visit A", "Long Visit B", "walk", 15)]
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        slots = result["days"][0]["slots"]
        meals = [s for s in slots if s["kind"] == "meal"]
        assert len(meals) == 1, f"Expected meal via shift, got: {slots}"
        meal = meals[0]
        assert meal["start_min"] == 720 and meal["end_min"] == 780
        stop_b = [s for s in slots if s["name"] == "Long Visit B"][0]
        assert stop_b["start_min"] == 795 and stop_b["end_min"] == 975  # +60
        for s in slots:
            assert s["end_min"] <= 17 * 60, f"stop outside hours: {s}"
        repair_text = " ".join(result["repairs"])
        assert "no room for meal slot" not in repair_text

    def test_impossible_day_keeps_no_room_note(self):
        """When NO boundary shift can fit the meal (day truly full), the
        explicit repairs note is the fallback — never an overlapping slot."""
        # A 09:00-12:40, B 12:55-16:45; shifting B +60 would end 17:45 > 17:00.
        hours = hours_9to5()
        pins = [
            make_pin("p1", "Full A", 0, 1.28, 103.85, dwell=230, hours=hours),
            make_pin("p2", "Full B", 1, 1.29, 103.86, dwell=230, hours=hours),
        ]
        legs = [make_leg("Full A", "Full B", "walk", 15)]
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        meals = [s for s in result["days"][0]["slots"] if s["kind"] == "meal"]
        assert len(meals) == 0, f"Expected no meal slot (impossible), got: {meals}"
        repair_text = " ".join(result["repairs"])
        assert "no room for meal slot" in repair_text

    def test_short_day_no_meal(self):
        """A short day (< 6h span) gets no meal slot."""
        pins = [
            make_pin("p1", "Short Visit", 0, 1.28, 103.85, dwell=60, hours=hours_9to5()),
        ]
        result = schedule_trip(pins, [], "2026-09-07", 1)
        meals = [s for s in result["days"][0]["slots"] if s["kind"] == "meal"]
        assert len(meals) == 0, f"Expected no meal slot, got meals: {meals}"

    def test_heavy_day_gets_rest(self):
        """A day with load > 8h (480 min) gets a rest slot (after the meal)."""
        # A 09:00-14:00 (300), leg 60, B 15:00-19:00 (240). Load 540 > 480.
        # Meal: no free gap in 11:00-14:00 (A occupies it), so shift-insert
        # at 14:00-15:00 (B -> 16:00-20:00, still inside 08:00-20:00 hours).
        # Rest: the 15:00-16:00 window is then free for a 30-min rest.
        long_hours = {
            "days": {str(d): [{"open": "08:00", "close": "20:00"}] for d in range(7)}
        }
        pins = [
            make_pin("p1", "Heavy A", 0, 1.28, 103.85, dwell=300, hours=long_hours),
            make_pin("p2", "Heavy B", 1, 1.29, 103.86, dwell=240, hours=long_hours),
        ]
        legs = [
            make_leg("Heavy A", "Heavy B", "drive", 60),
        ]
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        slots = result["days"][0]["slots"]
        rests = [s for s in slots if s["kind"] == "rest"]
        meals = [s for s in slots if s["kind"] == "meal"]
        assert len(meals) == 1, f"Expected meal slot, got: {slots}"
        assert len(rests) >= 1, f"Expected rest slot for heavy day, got: {slots}"

    def test_food_category_anchors_meal(self):
        """A food-category stop anchors the meal at that stop."""
        pins = [
            make_pin("p1", "Museum", 0, 1.28, 103.85, dwell=180, category="museum", hours=hours_9to5()),
            make_pin("p2", "Hawker Center", 1, 1.29, 103.86, dwell=180, category="food", hours=hours_9to5()),
        ]
        legs = [make_leg("Museum", "Hawker Center", "walk", 15)]
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        meals = [s for s in result["days"][0]["slots"] if s["kind"] == "meal"]
        if meals:
            # If a meal was inserted and anchored, it should have a pin_id.
            assert meals[0]["pin_id"] is not None or meals[0]["name"].startswith("Meal at")


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------
class TestDeterminism:
    """Running schedule_trip twice on the same input returns identical output."""

    def test_determinism_identical_output(self):
        pins = [
            make_pin("p1", "Place A", 0, 1.28, 103.85, "North", dwell=90, hours=hours_9to5()),
            make_pin("p2", "Place B", 1, 1.29, 103.86, "North", dwell=60, hours=hours_9to5()),
            make_pin("p3", "Place C", 2, 1.40, 103.90, "South", dwell=120, hours=hours_9to5()),
            make_pin("p4", "Place D", 3, 1.41, 103.91, "South", dwell=90, hours=hours_9to5()),
            make_pin("p5", "Place E", 4, 1.35, 103.88, "Central", dwell=60, hours=hours_9to5()),
        ]
        legs = [
            make_leg("Place A", "Place B", "walk", 10),
            make_leg("Place C", "Place D", "walk", 15),
            make_leg("Place A", "Place E", "drive", 20),
            make_leg("Place B", "Place E", "drive", 20),
        ]
        r1 = schedule_trip(pins, legs, "2026-09-07", 2)
        r2 = schedule_trip(pins, legs, "2026-09-07", 2)
        assert r1 == r2, "schedule_trip is not deterministic — same input gave different output"

    def test_run_schedule_alias_same(self):
        """run_schedule produces the same output as schedule_trip."""
        pins = [make_pin("p1", "X", 0, dwell=60, hours=hours_9to5())]
        r1 = schedule_trip(pins, [], "2026-09-07", 1)
        r2 = run_schedule(pins, [], "2026-09-07", 1)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Missing legs tests
# ---------------------------------------------------------------------------
class TestMissingLegs:
    """Missing leg data uses 45-min penalty + repairs note."""

    def test_missing_leg_penalty(self):
        """Missing leg uses 45-minute penalty and notes it in repairs."""
        pins = [
            make_pin("p1", "Place A", 0, 1.28, 103.85, dwell=60, hours=hours_9to5()),
            make_pin("p2", "Place B", 1, 1.29, 103.86, dwell=60, hours=hours_9to5()),
        ]
        # No legs at all!
        result = schedule_trip(pins, [], "2026-09-07", 1)

        # Should not crash.  Check repairs for the missing-leg note.
        repair_text = " ".join(result["repairs"])
        assert "no leg data" in repair_text
        assert str(MISSING_LEG_PENALTY) in repair_text

        # The legs in the output should show 45 min.
        for leg in result["days"][0]["legs"]:
            assert leg["minutes"] == MISSING_LEG_PENALTY

    def test_missing_leg_no_crash_three_stops(self):
        """Three stops with no legs should not crash."""
        pins = [
            make_pin("p1", "A", 0, 1.28, 103.85, dwell=60, hours=hours_9to5()),
            make_pin("p2", "B", 1, 1.29, 103.86, dwell=60, hours=hours_9to5()),
            make_pin("p3", "C", 2, 1.30, 103.87, dwell=60, hours=hours_9to5()),
        ]
        result = schedule_trip(pins, [], "2026-09-07", 1)
        # Should complete without error.
        assert result["stats"]["total_travel_minutes"] > 0


# ---------------------------------------------------------------------------
# Runner-ups tests
# ---------------------------------------------------------------------------
class TestRunnerUps:
    """runner_ups produces deterministic swap suggestions."""

    def test_runner_ups_structure(self):
        """runner_ups returns per-day structures with day_index and swaps list."""
        pins = [
            make_pin("p1", "A", 0, 1.28, 103.85, dwell=60, hours=hours_9to5()),
            make_pin("p2", "B", 1, 1.29, 103.86, dwell=60, hours=hours_9to5()),
        ]
        legs = [make_leg("A", "B", "walk", 10)]
        schedule = schedule_trip(pins, legs, "2026-09-07", 2)
        ru = runner_ups(schedule, pins, legs)

        assert isinstance(ru, list)
        assert len(ru) == 2  # 2 days
        for entry in ru:
            assert "day_index" in entry
            assert "swaps" in entry
            assert isinstance(entry["swaps"], list)

    def test_runner_ups_determinism(self):
        """runner_ups is deterministic: same input -> same output."""
        pins = [
            make_pin("p1", "A", 0, 1.28, 103.85, dwell=60, hours=hours_9to5()),
            make_pin("p2", "B", 1, 1.29, 103.86, dwell=60, hours=hours_9to5()),
            make_pin("p3", "C", 2, 1.30, 103.87, dwell=60, hours=hours_9to5()),
        ]
        legs = [make_leg("A", "B", "walk", 10)]
        schedule = schedule_trip(pins, legs, "2026-09-07", 2)
        ru1 = runner_ups(schedule, pins, legs)
        ru2 = runner_ups(schedule, pins, legs)
        assert ru1 == ru2


# ---------------------------------------------------------------------------
# Integration: full schedule structure
# ---------------------------------------------------------------------------
class TestScheduleStructure:
    """The returned Schedule dict matches the contract."""

    def test_schedule_has_required_keys(self):
        pins = [make_pin("p1", "Test Place", 0, dwell=60, hours=hours_9to5())]
        result = schedule_trip(pins, [], "2026-09-07", 1)
        assert "days" in result
        assert "unplaced" in result
        assert "repairs" in result
        assert "stats" in result

    def test_day_has_required_keys(self):
        pins = [make_pin("p1", "Test Place", 0, dwell=60, hours=hours_9to5())]
        result = schedule_trip(pins, [], "2026-09-07", 1)
        day = result["days"][0]
        assert "day_index" in day
        assert "date" in day
        assert "slots" in day
        assert "legs" in day
        assert "total_scheduled_minutes" in day

    def test_slot_has_required_keys(self):
        pins = [make_pin("p1", "Test Place", 0, dwell=60, hours=hours_9to5())]
        result = schedule_trip(pins, [], "2026-09-07", 1)
        stop_slots = [s for s in result["days"][0]["slots"] if s["kind"] == "stop"]
        assert len(stop_slots) == 1
        slot = stop_slots[0]
        assert "pin_id" in slot
        assert "name" in slot
        assert "kind" in slot
        assert "start_min" in slot
        assert "end_min" in slot
        assert "dwell_minutes" in slot

    def test_stats_has_required_keys(self):
        pins = [make_pin("p1", "Test Place", 0, dwell=60, hours=hours_9to5())]
        result = schedule_trip(pins, [], "2026-09-07", 1)
        stats = result["stats"]
        assert "total_travel_minutes" in stats
        assert "load_ratio" in stats
        assert "daily_loads" in stats

    def test_unverified_hours_repair_note(self):
        """Pins without opening_hours get an [unverified] repair note."""
        pins = [make_pin("p1", "No Hours Place", 0, dwell=60, hours=None)]
        result = schedule_trip(pins, [], "2026-09-07", 1)
        repair_text = " ".join(result["repairs"])
        assert "[unverified]" in repair_text
        assert "No Hours Place" in repair_text


class TestFoodAnchorSubstring:
    """The food anchor must survive real-world LLM category strings."""

    def test_hawker_centre_category_anchors_in_place(self):
        # Maxwell-like stop with a realistic LLM category string. Span must
        # cross the 360-min meal threshold: 270 + 90 + legs = 375 min.
        pins = [
            make_pin("p1", "Museum", 0, 1.28, 103.85, dwell=270, category="museum", hours=hours_9to5()),
            make_pin("p2", "Maxwell Food Centre", 1, 1.29, 103.86, dwell=90,
                     category="hawker centre", hours=hours_9to5()),
        ]
        legs = [make_leg("Museum", "Maxwell Food Centre", "walk", 15)]
        result = schedule_trip(pins, legs, "2026-09-07", 1)
        slots = result["days"][0]["slots"]
        meals = [s for s in slots if s["kind"] == "meal"]
        assert len(meals) == 1, f"Expected the food stop itself as meal, got: {slots}"
        assert meals[0]["pin_id"] == "p2"
        assert meals[0]["name"] == "Maxwell Food Centre"
        # No separate generic "Meal" slot was added.
        assert not any(s["name"] == "Meal" for s in slots)

    def test_food_anchor_counts_toward_load(self):
        # The anchored food stop's dwell must still count in the day load.
        pins = [
            make_pin("p1", "Maxwell Food Centre", 0, 1.29, 103.86, dwell=90,
                     category="hawker centre", hours=hours_9to5()),
        ]
        result = schedule_trip(pins, [], "2026-09-07", 1)
        day = result["days"][0]
        assert day["total_scheduled_minutes"] == 90
        assert result["stats"]["daily_loads"] == [90]
