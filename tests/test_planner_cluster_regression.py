"""Regression: blank-neighborhood pins must be placed exactly once.

The neighborhood branch of _cluster used to bucket blank-neighborhood pins
into a "_no_neighborhood_" group AND geometric-assign them as leftovers,
double-placing every one of them. Seen live on the preview deployment:
a URL pin appeared on both days and National Gallery Singapore twice.
"""
from collections import Counter

from services.planner_scheduler import run_schedule


def _pin(seq, name, lat, lng, neighborhood=""):
    return {
        "pin_id": f"p{seq}",
        "seq": seq,
        "name": name,
        "source": "text",
        "raw_input": name,
        "lat": lat,
        "lng": lng,
        "address": f"{name}, Singapore" if neighborhood else name,
        "resolved": lat is not None,
        "neighborhood": neighborhood,
        "rating": None,
        "opening_hours": None,
        "hours_verified": False,
        "category": "attraction",
        "dwell_minutes": 60,
        "booking_required": False,
        "tip": "",
    }


def test_blank_neighborhood_pins_placed_exactly_once():
    pins = [
        _pin(1, "Museum A", 1.2816, 103.8594, "Downtown"),
        _pin(2, "Temple B", 1.2815, 103.8447, "Chinatown"),
        _pin(3, "Garden C", 1.3150, 103.8158, "Tanglin"),
        _pin(4, "No-Neighborhood D", 1.2877, 103.8529),
        _pin(5, "No-Neighborhood E", None, None),
    ]
    sched = run_schedule(pins, [], "2026-09-07", 2)

    placed = [
        s.get("name")
        for d in sched.get("days", [])
        for s in d.get("slots", [])
        if s.get("kind") == "stop"
    ]
    counts = Counter(placed)
    dups = {k: v for k, v in counts.items() if v > 1}
    assert not dups, f"duplicate placements: {dups}"

    unplaced_names = {u.get("name") for u in sched.get("unplaced", [])}
    for p in pins:
        if p["name"] not in unplaced_names:
            assert counts.get(p["name"], 0) == 1, (
                f"{p['name']} placed {counts.get(p['name'], 0)} times"
            )
