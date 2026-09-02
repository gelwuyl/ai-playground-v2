"""Mr. Bounce — Trip Scheduler (deterministic, no LLM, no network).

Implements the approved objective cascade in strict priority order:

  1. OPENING-HOURS HARD FEASIBILITY
     - No stop scheduled outside its opening hours for the assigned day.
     - Closed-day repair: if a pin is closed on a candidate day, move it to
       another day where it is open and note the move in repairs.
     - If a pin is closed on all trip days, mark it unplaced.
     - Split hours (multiple intervals per day): schedule the stop in the
       LONGEST interval of that day; if it cannot fit in any single interval,
       treat that day as closed for the pin.

  2. NEIGHBORHOOD DAY-CLUSTERING WITH BALANCED DAILY LOAD
     - Primary signal: neighborhood string — pins sharing a neighborhood
       land on the same day when they fit.
     - Geometric fallback: deterministic farthest-point k-means-ish seeding
       on lat/lng, then nearest-seed assignment.
     - Balance: move pins between days so that no day's total scheduled load
       exceeds 1.5x another day's load, subject to opening-hours feasibility.
     - Load = sum of dwell_minutes of stops assigned (meals/rest excluded).

  3. LEAST TOTAL TRAVEL WITHIN A DAY
     - Greedy nearest-neighbor ordering from the stop with min seq, then
       2-opt improvement (capped at 100 iterations for determinism).
     - Missing legs use a 45-minute penalty.

  4. MEAL/REST WINDOW INSERTION
     - If a day's span >= 360 min (6 h): insert a 60-min meal slot.
       If a food-category stop exists that day, anchor the meal there;
       otherwise insert a generic meal at 12:00-13:00.
     - If the day's load > 480 min (8 h): additionally insert a 30-min
       rest slot in the 13:30-16:00 zone.

All operations are deterministic: sorted() everywhere, no set iteration order
dependence, no randomness.  Python 3.12 stdlib only.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from services.planner_types import (
    parse_hhmm,
    day_dates,
    haversine_km,
    estimated_minutes,
    MINUTES_PER_DAY,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DAY_START_MIN = 540          # 09:00 — earliest start for any day
MISSING_LEG_PENALTY = 45     # minutes assumed when no leg data exists
TWO_OPT_CAP = 100            # max 2-opt iterations for determinism
BALANCE_RATIO = 1.5         # max allowed ratio between any two days' loads
MEAL_THRESHOLD_MIN = 360     # 6 h — days at or above get a meal slot
REST_THRESHOLD_MIN = 480     # 8 h — days above get a rest slot
MEAL_DURATION = 60
REST_DURATION = 30
CLOSED_ALL_TRIP_REASON = "closed all trip days"


# ---------------------------------------------------------------------------
# Opening-hours helpers
# ---------------------------------------------------------------------------
def _day_of_week(d: str) -> int:
    """Return 0=Mon .. 6=Sun for a YYYY-MM-DD date string."""
    return date.fromisoformat(d).weekday()


def _intervals_for_day(opening_hours: dict | None, dow: int) -> list[tuple[int, int]]:
    """Return list of (open_min, close_min) intervals for the given day-of-week.

    Missing/empty hours => open all day, represented as [(0, MINUTES_PER_DAY)].
    A day key present with empty list => closed that day.
    """
    if opening_hours is None:
        return [(0, MINUTES_PER_DAY)]
    days = opening_hours.get("days")
    if not days:
        return [(0, MINUTES_PER_DAY)]
    day_entry = days.get(str(dow))
    if day_entry is None:
        # Day key missing => closed
        return []
    if day_entry == []:
        return []
    intervals = []
    for slot in day_entry:
        o = parse_hhmm(slot.get("open", ""))
        c = parse_hhmm(slot.get("close", ""))
        if o is None or c is None:
            continue
        intervals.append((o, c))
    return intervals if intervals else [(0, MINUTES_PER_DAY)]


def _longest_interval(intervals: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Return the (open, close) pair with the largest width, or None if empty."""
    if not intervals:
        return None
    return max(intervals, key=lambda iv: iv[1] - iv[0])


def _dwell_fits_in_any_interval(
    dwell: int, intervals: list[tuple[int, int]]
) -> tuple[int, int] | None:
    """Return the interval in which the dwell fits, preferring the longest."""
    viable = [iv for iv in intervals if (iv[1] - iv[0]) >= dwell]
    if not viable:
        return None
    return _longest_interval(viable)


# ---------------------------------------------------------------------------
# Leg lookup helpers
# ---------------------------------------------------------------------------
def _leg_minutes(legs: list[dict], name_a: str, name_b: str) -> float:
    """Look up chosen_minutes for the unordered pair (a, b).

    Tries both a->b and b->a.  Returns MISSING_LEG_PENALTY if not found.
    """
    for leg in legs:
        if leg.get("from_name") == name_a and leg.get("to_name") == name_b:
            return float(leg.get("chosen_minutes", MISSING_LEG_PENALTY))
        if leg.get("from_name") == name_b and leg.get("to_name") == name_a:
            return float(leg.get("chosen_minutes", MISSING_LEG_PENALTY))
    return float(MISSING_LEG_PENALTY)


def _missing_leg(legs: list[dict], name_a: str, name_b: str) -> str | None:
    """Return a repair note if the leg is missing, else None."""
    for leg in legs:
        fn, tn = leg.get("from_name"), leg.get("to_name")
        if (fn == name_a and tn == name_b) or (fn == name_b and tn == name_a):
            return None
    return f"no leg data {name_a} -> {name_b}; assumed {MISSING_LEG_PENALTY} min"


# ---------------------------------------------------------------------------
# Clustering (Level 2a: neighborhood + geometric fallback)
# ---------------------------------------------------------------------------
def _cluster(
    pins: list[dict], num_days: int
) -> list[list[dict]]:
    """Assign pins to *num_days* day-groups.

    Strategy:
      1. If pins have neighborhood strings, group by neighborhood then
         assign groups to days deterministically (sorted by name).
      2. For pins without a neighborhood (or all-blank), use a deterministic
         farthest-point k-means-ish seeding on lat/lng.

    Returns a list of length *num_days*; each element is the list of pins
    assigned to that day index.
    """
    if num_days <= 0:
        return []

    # --- Neighborhood-based clustering ---
    has_neighborhood = any(p.get("neighborhood", "") for p in pins)
    if has_neighborhood:
        # Group pins by neighborhood (sorted), then distribute groups to days.
        groups: dict[str, list[dict]] = {}
        for p in pins:
            nb = p.get("neighborhood", "") or "_no_neighborhood_"
            groups.setdefault(nb, []).append(p)
        # Sort groups by name for determinism; sort pins within by seq then name.
        sorted_group_names = sorted(groups.keys())
        days: list[list[dict]] = [[] for _ in range(num_days)]
        for i, gn in enumerate(sorted_group_names):
            day_idx = i % num_days
            group_pins = sorted(groups[gn], key=lambda p: (p.get("seq", 0), p.get("name", "")))
            days[day_idx].extend(group_pins)
        # Pins without a neighborhood: assign via geometric fallback.
        no_nb = [p for p in pins if not p.get("neighborhood", "")]
        if no_nb and len(sorted_group_names) <= 1:
            # All pins lack neighborhood -> full geometric fallback
            return _geometric_cluster(pins, num_days)
        elif no_nb:
            # Some pins lack neighborhood -> geometric-assign the leftovers
            geo_days = _geometric_cluster(no_nb, num_days)
            for di, gp in enumerate(geo_days):
                days[di].extend(gp)
        return days

    # --- Geometric fallback ---
    return _geometric_cluster(pins, num_days)


def _geometric_cluster(pins: list[dict], num_days: int) -> list[list[dict]]:
    """Deterministic farthest-point k-means-ish clustering on lat/lng.

    Seeds: first = southernmost (min lat, then min lng as tiebreaker).
    Subsequent seeds = the point maximizing min-distance to existing seeds.
    Assign each pin to nearest seed.  Deterministic: sorted() everywhere.
    """
    if not pins or num_days <= 0:
        return [[] for _ in range(max(num_days, 0))]
    if num_days == 1:
        return [sorted(pins, key=lambda p: (p.get("seq", 0), p.get("name", "")))]

    usable = [p for p in pins if p.get("lat") is not None and p.get("lng") is not None]
    if len(usable) < num_days:
        # Not enough geo-located pins to seed; fall back to round-robin by seq.
        sorted_pins = sorted(pins, key=lambda p: (p.get("seq", 0), p.get("name", "")))
        days = [[] for _ in range(num_days)]
        for i, p in enumerate(sorted_pins):
            days[i % num_days].append(p)
        return days

    # Sort for determinism before seeding.
    usable_sorted = sorted(usable, key=lambda p: (p["lat"], p["lng"], p.get("name", "")))

    # First seed: southernmost (min lat, then min lng).
    seeds = [usable_sorted[0]]
    while len(seeds) < num_days:
        best_p = None
        best_dist = -1.0
        for p in usable_sorted:
            if p in seeds:
                continue
            min_d = min(
                haversine_km(p["lat"], p["lng"], s["lat"], s["lng"])
                for s in seeds
            )
            if min_d > best_dist:
                best_d = min_d
                best_dist = min_d
                best_p = p
            elif min_d == best_dist and best_p is not None:
                # Tiebreak by seq then name
                if (p.get("seq", 0), p.get("name", "")) < (best_p.get("seq", 0), best_p.get("name", "")):
                    best_p = p
        if best_p is None:
            break
        seeds.append(best_p)

    # Assign each pin to nearest seed.
    days: list[list[dict]] = [[] for _ in range(num_days)]
    for p in pins:
        if p.get("lat") is None or p.get("lng") is None:
            # Non-geolocated pins -> assign by seq round-robin
            idx = p.get("seq", 0) % num_days
            days[idx].append(p)
            continue
        best_idx = 0
        best_d = float("inf")
        for i, s in enumerate(seeds):
            d = haversine_km(p["lat"], p["lng"], s["lat"], s["lng"])
            if d < best_d:
                best_d = d
                best_idx = i
            elif d == best_d:
                # Tiebreak by seed seq
                if (s.get("seq", 0), s.get("name", "")) < (
                    seeds[best_idx].get("seq", 0),
                    seeds[best_idx].get("name", ""),
                ):
                    best_idx = i
        days[best_idx].append(p)

    # Sort each day by (seq, name) for determinism.
    for i in range(len(days)):
        days[i] = sorted(days[i], key=lambda p: (p.get("seq", 0), p.get("name", "")))
    return days


# ---------------------------------------------------------------------------
# Opening-hours feasibility + closed-day repair (Level 1)
# ---------------------------------------------------------------------------
def _is_open_on_day(pin: dict, dow: int) -> bool:
    """True if the pin has at least one open interval on the given day-of-week."""
    intervals = _intervals_for_day(pin.get("opening_hours"), dow)
    return len(intervals) > 0


def _feasible_days(pin: dict, day_dows: list[int]) -> list[int]:
    """Return day-indices (0-based among trip days) where pin is open."""
    return [i for i, dow in enumerate(day_dows) if _is_open_on_day(pin, dow)]


def _apply_closed_day_repair(
    assignments: list[list[dict]],
    day_dows: list[int],
    repairs: list[str],
) -> tuple[list[list[dict]], list[dict]]:
    """Move pins that are closed on their assigned day to a day where they are open.

    Returns (new_assignments, unplaced).
    Pins closed on all trip days are marked unplaced.
    """
    num_days = len(assignments)
    unplaced: list[dict] = []
    new_assignments: list[list[dict]] = [[] for _ in range(num_days)]

    for day_idx in range(num_days):
        for pin in assignments[day_idx]:
            feasible = _feasible_days(pin, day_dows)
            if not feasible:
                unplaced.append({
                    "pin_id": pin.get("pin_id", ""),
                    "name": pin.get("name", ""),
                    "reason": CLOSED_ALL_TRIP_REASON,
                })
                continue
            if day_idx in feasible:
                # Already on a feasible day
                new_assignments[day_idx].append(pin)
            else:
                # Move to the first feasible day (sorted by day index)
                target = min(feasible)
                dow_name = _dow_name(day_dows[day_idx])
                target_dow_name = _dow_name(day_dows[target])
                repairs.append(
                    f"{pin.get('name', '')}: closed on Day {day_idx + 1} ({dow_name}); moved to Day {target + 1}"
                )
                new_assignments[target].append(pin)

    return new_assignments, unplaced


def _dow_name(dow: int) -> str:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return names[dow] if 0 <= dow < 7 else "?"


# ---------------------------------------------------------------------------
# Balancing (Level 2b: redistribute for load_ratio <= BALANCE_RATIO)
# ---------------------------------------------------------------------------
def _day_load(day_pins: list[dict]) -> int:
    """Sum of dwell_minutes for the pins on a day."""
    return sum(p.get("dwell_minutes", 0) for p in day_pins)


def _balance(
    assignments: list[list[dict]],
    day_dows: list[int],
) -> list[list[dict]]:
    """Move pins from high-load days to lower-load days to satisfy BALANCE_RATIO.

    Constraints:
      - A pin can only move to a day where it is open.
      - Never violate the 1.5x ratio target after all moves.

    This is a greedy pass: find the heaviest and lightest days, move the
    cheapest-to-move pin from heavy to light (if it fits the light day),
    repeat until balanced or no more moves possible.
    """
    num_days = len(assignments)
    if num_days <= 1:
        return assignments

    # Work on a copy.
    days = [list(d) for d in assignments]
    max_passes = num_days * len(pins_all(days)) + 10  # safety cap

    for _ in range(max_passes):
        loads = [_day_load(d) for d in days]
        non_zero = [l for l in loads if l > 0]
        if len(non_zero) <= 1:
            break
        max_load = max(loads)
        min_load = min(l for l in loads if l > 0)
        if max_load <= 0:
            break
        if max_load <= min_load * BALANCE_RATIO:
            break

        heavy_idx = loads.index(max_load)
        light_indices = [i for i in range(num_days) if loads[i] > 0 and i != heavy_idx]
        # If all non-heavy days have 0 load, target the first one.
        if not light_indices:
            light_indices = [i for i in range(num_days) if loads[i] == 0 and i != heavy_idx]
            if not light_indices:
                break
        light_idx = min(light_indices, key=lambda i: (loads[i], i))

        # Find a pin on the heavy day that is open on the light day.
        candidates = sorted(days[heavy_idx], key=lambda p: (p.get("dwell_minutes", 0), p.get("seq", 0), p.get("name", "")))
        moved = False
        for pin in candidates:
            if _is_open_on_day(pin, day_dows[light_idx]):
                days[heavy_idx].remove(pin)
                days[light_idx].append(pin)
                days[light_idx] = sorted(
                    days[light_idx], key=lambda p: (p.get("seq", 0), p.get("name", ""))
                )
                moved = True
                break
        if not moved:
            # Try moving to any other lighter day
            other_light = [i for i in range(num_days) if i != heavy_idx and loads[i] < max_load]
            for target_idx in sorted(other_light, key=lambda i: (loads[i], i)):
                for pin in candidates:
                    if _is_open_on_day(pin, day_dows[target_idx]):
                        days[heavy_idx].remove(pin)
                        days[target_idx].append(pin)
                        days[target_idx] = sorted(
                            days[target_idx],
                            key=lambda p: (p.get("seq", 0), p.get("name", "")),
                        )
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break  # cannot balance further

    return days


def pins_all(days: list[list[dict]]) -> list[dict]:
    """Flatten all day lists into a single list."""
    return [p for d in days for p in d]


# ---------------------------------------------------------------------------
# Ordering (Level 3: greedy NN + 2-opt)
# ---------------------------------------------------------------------------
def _travel_matrix(stops: list[dict], legs: list[dict]) -> list[list[float]]:
    """Build a len(stops) x len(stops) travel-time matrix from legs."""
    n = len(stops)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mat[i][j] = _leg_minutes(legs, stops[i]["name"], stops[j]["name"])
    return mat


def _total_travel(order: list[int], mat: list[list[float]]) -> float:
    """Sum of travel between consecutive stops in the given index order."""
    total = 0.0
    for k in range(len(order) - 1):
        total += mat[order[k]][order[k + 1]]
    return total


def _greedy_nn_order(stops: list[dict], mat: list[list[float]]) -> list[int]:
    """Greedy nearest-neighbor ordering starting from min-seq stop."""
    n = len(stops)
    if n <= 1:
        return list(range(n))
    # Start from the stop with min seq.
    start = min(range(n), key=lambda i: (stops[i].get("seq", 0), stops[i].get("name", "")))
    visited = [False] * n
    order = [start]
    visited[start] = True
    current = start
    while len(order) < n:
        best = -1
        best_d = float("inf")
        for j in range(n):
            if visited[j]:
                continue
            d = mat[current][j]
            if d < best_d:
                best_d = d
                best = j
            elif d == best_d and best >= 0:
                # Tiebreak by seq then name
                if (stops[j].get("seq", 0), stops[j].get("name", "")) < (
                    stops[best].get("seq", 0),
                    stops[best].get("name", ""),
                ):
                    best = j
        if best < 0:
            break
        order.append(best)
        visited[best] = True
        current = best
    return order


def _two_opt(order: list[int], mat: list[list[float]]) -> list[int]:
    """2-opt improvement: reverse segments to reduce total travel.

    Capped at TWO_OPT_CAP iterations for determinism.
    Returns the improved order (may be same as input if no improvement).
    """
    n = len(order)
    if n <= 2:
        return list(order)
    best = list(order)
    best_cost = _total_travel(best, mat)
    improved = True
    iters = 0
    while improved and iters < TWO_OPT_CAP:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j == i:
                    continue
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cost = _total_travel(candidate, mat)
                if cost < best_cost - 1e-9:
                    best = candidate
                    best_cost = cost
                    improved = True
                    break
            if improved:
                break
    return best


def _order_day(stops: list[dict], legs: list[dict]) -> list[dict]:
    """Order stops within a day using greedy NN + 2-opt. Returns reordered list."""
    if len(stops) <= 1:
        return list(stops)
    mat = _travel_matrix(stops, legs)
    nn_order = _greedy_nn_order(stops, mat)
    improved = _two_opt(nn_order, mat)
    return [stops[i] for i in improved]


# ---------------------------------------------------------------------------
# Timeline slotting
# ---------------------------------------------------------------------------
def _slot_day(
    stops: list[dict],
    legs: list[dict],
    day_dow: int,
    repairs: list[str],
) -> tuple[list[dict], list[dict]]:
    """Slot stops into a timeline for the given day.

    Walk stops in travel order, scheduling each at the earliest feasible time:
      - start >= DAY_START_MIN (540)
      - start >= previous_end + leg_minutes(prev, stop)
      - stop must fit fully inside one open interval for that day-of-week

    Returns (slots, unplaced).
    Each slot dict has: pin_id, name, kind, start_min, end_min, dwell_minutes.
    """
    if not stops:
        return [], []

    # Order by travel.
    ordered = _order_day(stops, legs)

    slots: list[dict] = []
    unplaced: list[dict] = []
    cursor = DAY_START_MIN

    for k, pin in enumerate(ordered):
        dwell = pin.get("dwell_minutes", 60)
        intervals = _intervals_for_day(pin.get("opening_hours"), day_dow)

        # Leg time from previous stop.
        if k == 0:
            leg_min = 0
        else:
            prev_pin = ordered[k - 1]
            leg_min = int(_leg_minutes(legs, prev_pin["name"], pin["name"]))
            # Note missing legs
            miss = _missing_leg(legs, prev_pin["name"], pin["name"])
            if miss:
                repairs.append(miss)

        earliest = max(cursor + leg_min, DAY_START_MIN)

        # Try to fit in an interval: find the earliest interval where
        # [earliest, earliest+dwell] fits entirely.
        placed = False
        for iv_open, iv_close in sorted(intervals, key=lambda iv: (iv[0], iv[1])):
            start = max(earliest, iv_open)
            end = start + dwell
            if end <= iv_close:
                slots.append({
                    "pin_id": pin.get("pin_id", ""),
                    "name": pin.get("name", ""),
                    "kind": "stop",
                    "start_min": start,
                    "end_min": end,
                    "dwell_minutes": dwell,
                })
                cursor = end
                placed = True
                break

        if not placed:
            unplaced.append({
                "pin_id": pin.get("pin_id", ""),
                "name": pin.get("name", ""),
                "reason": f"cannot fit on Day {day_dow + 1} within opening hours",
            })

    return slots, unplaced


# ---------------------------------------------------------------------------
# Meal / rest insertion (Level 4)
# ---------------------------------------------------------------------------
def _find_gap(
    slots: list[dict], duration: int, search_start: int, search_end: int
) -> tuple[int, int] | None:
    """Find the earliest (start, end) window of `duration` minutes within
    [search_start, search_end] that does not overlap any existing stop/meal slot.

    Returns (start, end) or None if no gap exists.
    """
    stop_slots = sorted(
        [s for s in slots if s["kind"] in ("stop", "meal")],
        key=lambda s: s["start_min"],
    )
    cursor = search_start
    for s in stop_slots:
        if cursor + duration <= s["start_min"]:
            return (cursor, cursor + duration)
        cursor = max(cursor, s["end_min"])
        if cursor + duration > search_end:
            break
    if cursor + duration <= search_end:
        return (cursor, cursor + duration)
    return None


def _insert_meal_generic(slots: list[dict]) -> list[dict]:
    """Insert a generic 60-min meal slot at 12:00 or the nearest free window.

    Tries 12:00 first, then searches 11:00-14:00 in 15-min increments.
    If no gap exists in that range, inserts at 12:00 anyway (overlaps are
    acceptable for meals — they are advisory, not hard constraints).
    """
    meal_start = 12 * 60  # 720
    meal_end = meal_start + MEAL_DURATION

    # Check if 12:00-13:00 is free.
    gap = _find_gap(slots, MEAL_DURATION, 11 * 60, 14 * 60)
    if gap is not None:
        meal_start, meal_end = gap
    # If no gap, just insert at 12:00 (advisory overlap is acceptable).

    new_slots = list(slots)
    new_slots.append({
        "pin_id": None,
        "name": "Meal",
        "kind": "meal",
        "start_min": meal_start,
        "end_min": meal_end,
        "dwell_minutes": MEAL_DURATION,
    })
    new_slots.sort(key=lambda s: (s["start_min"], s.get("pin_id") or ""))
    return new_slots


def _insert_rest(slots: list[dict]) -> list[dict]:
    """Insert a 30-min rest slot in the 13:30-16:00 zone."""
    gap = _find_gap(slots, REST_DURATION, 13 * 60 + 30, 16 * 60)
    if gap is not None:
        rest_start, rest_end = gap
    else:
        # Fallback: insert at 14:00 (advisory).
        rest_start = 14 * 60
        rest_end = rest_start + REST_DURATION
    new_slots = list(slots)
    new_slots.append({
        "pin_id": None,
        "name": "Rest",
        "kind": "rest",
        "start_min": rest_start,
        "end_min": rest_end,
        "dwell_minutes": REST_DURATION,
    })
    new_slots.sort(key=lambda s: (s["start_min"], s.get("pin_id") or ""))
    return new_slots


def _anchor_meal_at_food_stop(
    slots: list[dict],
    day_pins: list[dict],
) -> dict | None:
    """If a food-category pin is among the day's stops, return a meal slot
    anchored at that stop's time window. Returns None if no food stop found."""
    slot_map = {s["pin_id"]: s for s in slots if s["kind"] == "stop" and s.get("pin_id")}
    for pin in sorted(day_pins, key=lambda p: (p.get("seq", 0), p.get("name", ""))):
        if pin.get("category", "").lower() == "food" and pin.get("pin_id") in slot_map:
            stop_slot = slot_map[pin["pin_id"]]
            # Anchor the meal at the stop's time.
            return {
                "pin_id": pin.get("pin_id"),
                "name": f"Meal at {pin.get('name', '')}",
                "kind": "meal",
                "start_min": stop_slot["start_min"],
                "end_min": stop_slot["end_min"],
                "dwell_minutes": stop_slot["dwell_minutes"],
            }
    return None


# ---------------------------------------------------------------------------
# Per-day scheduling
# ---------------------------------------------------------------------------
def _schedule_day(
    day_pins: list[dict],
    day_dow: int,
    day_index: int,
    legs: list[dict],
    repairs: list[str],
) -> tuple[list[dict], list[dict]]:
    """Schedule one day: slot stops, insert meals/rest.

    Returns (slots, unplaced).
    """
    # Check for pins with missing opening_hours -> treat as open but note it.
    for pin in day_pins:
        oh = pin.get("opening_hours")
        if oh is None or not oh.get("days"):
            repairs.append(
                f"[unverified] {pin.get('name', '')} — no structured hours; treated as open"
            )

    # Slot the stops (Level 3 ordering + timeline slotting with Level 1 feasibility).
    slots, unplaced = _slot_day(day_pins, legs, day_dow, repairs)

    if not slots:
        return [], unplaced

    # Compute span to decide meal insertion (Level 4).
    first_start = min(s["start_min"] for s in slots)
    last_end = max(s["end_min"] for s in slots)
    span = last_end - first_start

    # Total stop load (meals/rest excluded).
    total_load = sum(s["dwell_minutes"] for s in slots if s["kind"] == "stop")

    if span >= MEAL_THRESHOLD_MIN:
        # Try anchoring at a food stop first.
        anchored = _anchor_meal_at_food_stop(slots, day_pins)
        if anchored is not None:
            slots.append(anchored)
            slots.sort(key=lambda s: (s["start_min"], s.get("pin_id") or ""))
        else:
            # Generic meal insertion at 12:00 or nearest free window.
            slots = _insert_meal_generic(slots)

        # Rest insertion for heavy days.
        if total_load > REST_THRESHOLD_MIN:
            slots = _insert_rest(slots)

    return slots, unplaced


# ---------------------------------------------------------------------------
# Leg list construction for output
# ---------------------------------------------------------------------------
def _build_day_legs(slots: list[dict], legs: list[dict]) -> list[dict]:
    """Build the per-day legs list from the slot order."""
    stop_slots = [s for s in slots if s["kind"] == "stop"]
    result = []
    for k in range(len(stop_slots) - 1):
        a_name = stop_slots[k]["name"]
        b_name = stop_slots[k + 1]["name"]
        minutes = _leg_minutes(legs, a_name, b_name)
        # Determine mode from legs
        mode = "drive"
        for leg in legs:
            fn, tn = leg.get("from_name"), leg.get("to_name")
            if (fn == a_name and tn == b_name) or (fn == b_name and tn == a_name):
                mode = leg.get("chosen_mode", "drive")
                break
        result.append({
            "from_name": a_name,
            "to_name": b_name,
            "mode": mode,
            "minutes": minutes,
        })
    return result


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def _compute_stats(days: list[dict]) -> dict:
    """Compute total_travel_minutes and load_ratio."""
    total_travel = 0.0
    daily_loads = []
    for day in days:
        day_legs = day.get("legs", [])
        day_travel = sum(l["minutes"] for l in day_legs)
        total_travel += day_travel
        load = sum(s["dwell_minutes"] for s in day["slots"] if s["kind"] == "stop")
        if load > 0:
            daily_loads.append(load)

    if len(daily_loads) <= 1:
        load_ratio = 0.0
    else:
        max_load = max(daily_loads)
        min_load = min(daily_loads)
        if min_load == 0:
            load_ratio = 0.0
        else:
            load_ratio = round(max_load / min_load, 4)

    return {
        "total_travel_minutes": round(total_travel, 1),
        "load_ratio": load_ratio,
        "daily_loads": daily_loads,
    }


# ---------------------------------------------------------------------------
# Runner-ups (for the Alternatives stage)
# ---------------------------------------------------------------------------
def runner_ups(
    schedule: dict,
    pins: list[dict],
    legs: list[dict],
) -> list[dict]:
    """Per-day swap suggestions for the Alternatives LLM stage.

    For each day, produce:
      - Load-balancing swaps: move one stop from the highest-loaded day to
        a lighter day where it is feasible.
      - Unplaced pin candidates: pins that are unplaced but could have fit
        that day.

    Output shape:
      [{"day_index": int, "swaps": [
          {"remove_name": str, "add_name": str, "add_pin_id": str,
           "reason_hint": str, "travel_delta_minutes": float}
      ]}]
    """
    results: list[dict] = []
    days = schedule.get("days", [])
    unplaced = schedule.get("unplaced", [])
    num_days = len(days)

    # Compute per-day loads.
    day_loads = []
    for day in days:
        load = sum(s["dwell_minutes"] for s in day["slots"] if s["kind"] == "stop")
        day_loads.append(load)

    for day_idx in range(num_days):
        swaps: list[dict] = []

        # (a) Unplaced pins that could fit this day.
        unplaced_names = {u["name"] for u in unplaced}
        pin_lookup = {p["name"]: p for p in pins}
        for up in unplaced:
            pin = pin_lookup.get(up["name"])
            if pin is None:
                continue
            # Check if pin is feasible on this day.
            dow = _day_of_week(days[day_idx]["date"])
            intervals = _intervals_for_day(pin.get("opening_hours"), dow)
            if intervals:
                swaps.append({
                    "remove_name": "",
                    "add_name": up["name"],
                    "add_pin_id": up.get("pin_id", ""),
                    "reason_hint": "unplaced pin feasible this day",
                    "travel_delta_minutes": 0.0,
                })

        # (b) Load-balancing swap: move one stop from the highest-loaded day
        #     to this day (if this day is not the highest).
        if day_loads and max(day_loads) > 0:
            heavy_idx = day_loads.index(max(day_loads))
            if heavy_idx != day_idx and day_loads[day_idx] < day_loads[heavy_idx]:
                # Find a stop on the heavy day that is open on this day.
                heavy_stops = [s for s in days[heavy_idx]["slots"] if s["kind"] == "stop"]
                dow_heavy = _day_of_week(days[heavy_idx]["date"])
                dow_this = _day_of_week(days[day_idx]["date"])
                for stop in sorted(heavy_stops, key=lambda s: (s.get("start_min", 0), s.get("name", ""))):
                    pin = pin_lookup.get(stop["name"])
                    if pin is None:
                        continue
                    if _is_open_on_day(pin, dow_this):
                        # Compute approximate travel delta.
                        travel_delta = _compute_swap_delta(
                            days[heavy_idx]["slots"],
                            days[day_idx]["slots"],
                            stop,
                            legs,
                            pin,
                        )
                        swaps.append({
                            "remove_name": stop["name"],
                            "add_name": stop["name"],
                            "add_pin_id": stop.get("pin_id", ""),
                            "reason_hint": f"load balance: move from Day {heavy_idx + 1} to Day {day_idx + 1}",
                            "travel_delta_minutes": travel_delta,
                        })
                        break  # one swap per day

        results.append({"day_index": day_idx, "swaps": swaps})

    return results


def _compute_swap_delta(
    heavy_slots: list[dict],
    light_slots: list[dict],
    stop_to_move: dict,
    legs: list[dict],
    pin: dict,
) -> float:
    """Approximate travel delta from moving a stop from heavy to light day.

    Rough: recompute greedy NN travel for the light day with the added pin,
    minus current light day travel.  Capped to keep it fast.
    """
    # Current light day travel.
    light_stops = [s for s in light_slots if s["kind"] == "stop"]
    if not light_stops:
        return 0.0
    current_mat = _travel_matrix(light_stops, legs)
    current_order = _greedy_nn_order(light_stops, current_mat)
    current_travel = _total_travel(current_order, current_mat)

    # Light day with added pin.
    new_stops = light_stops + [pin]
    new_mat = _travel_matrix(new_stops, legs)
    new_order = _greedy_nn_order(new_stops, new_mat)
    new_travel = _total_travel(new_order, new_mat)

    return round(new_travel - current_travel, 1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def schedule_trip(
    pins: list[dict],
    legs: list[dict],
    start_date: str,
    num_days: int,
) -> dict:
    """Schedule pins across num_days days following the objective cascade.

    Parameters:
        pins: list of PlaceResearch dicts (see planner_types.py).
        legs: list of Leg dicts with chosen_mode and chosen_minutes.
        start_date: "YYYY-MM-DD".
        num_days: 1..7.

    Returns: Schedule dict (see planner_types.py).
    """
    repairs: list[str] = []
    unplaced_all: list[dict] = []

    # Sort pins deterministically for stable processing.
    pins_sorted = sorted(pins, key=lambda p: (p.get("seq", 0), p.get("name", "")))

    # Compute day-of-week for each trip day.
    dates = day_dates(start_date, num_days)
    day_dows = [_day_of_week(d) for d in dates]

    # --- Level 2a: Cluster pins into days ---
    assignments = _cluster(pins_sorted, num_days)

    # --- Level 1: Closed-day repair ---
    assignments, closed_unplaced = _apply_closed_day_repair(assignments, day_dows, repairs)
    unplaced_all.extend(closed_unplaced)

    # --- Level 2b: Balance loads ---
    assignments = _balance(assignments, day_dows)

    # --- Level 1 (continued): Slot each day, enforcing hours ---
    days_output: list[dict] = []
    for day_idx in range(num_days):
        day_pins = assignments[day_idx]
        dow = day_dows[day_idx]

        slots, day_unplaced = _schedule_day(day_pins, dow, day_idx, legs, repairs)
        unplaced_all.extend(day_unplaced)

        day_legs = _build_day_legs(slots, legs)
        total_scheduled = sum(s["dwell_minutes"] for s in slots if s["kind"] == "stop")

        days_output.append({
            "day_index": day_idx,
            "date": dates[day_idx],
            "slots": slots,
            "legs": day_legs,
            "total_scheduled_minutes": total_scheduled,
        })

    # --- Stats ---
    stats = _compute_stats(days_output)

    return {
        "days": days_output,
        "unplaced": unplaced_all,
        "repairs": sorted(set(repairs)),  # deduplicate + sort for determinism
        "stats": stats,
    }


# Thin alias — the graph runner may call either name.
run_schedule = schedule_trip
