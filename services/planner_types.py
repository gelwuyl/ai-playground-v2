"""Mr. Bounce — shared pure helpers and the canonical data shapes.

Only stdlib here: every planner module (ingest, logistics, scheduler, agents,
graph) imports from this file so the JSON contracts stay in one place.
"""
import math
import re
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Canonical dict shapes (documented here, duck-typed everywhere)
# ---------------------------------------------------------------------------
# Pin (one row of planner_pins, as a dict):
# {
#     "pin_id": str,            # UUID string
#     "session_id": str,
#     "seq": int,               # user-given order, 0-based
#     "name": str,              # resolved place name (short)
#     "source": "short_link" | "text",
#     "raw_input": str,         # what the user pasted
#     "lat": float | None,
#     "lng": float | None,
#     "address": str | None,
#     "resolved": bool,
#     "resolve_error": str | None,
# }
#
# PlaceResearch (Scout output per pin, merged from SerpApi structure + LLM):
# {
#     "pin_id": str,
#     "name": str,
#     "neighborhood": str,          # from SerpApi address, e.g. "Marina District"
#     "lat": float, "lng": float,
#     "rating": float | None,       # SerpApi, never the LLM
#     "opening_hours": {            # SerpApi Google Maps extended_hours or hours
#         "days": {                 # 0=Mon .. 6=Sun
#             "0": [{"open": "09:00", "close": "17:00"}],   # intervals; [] = closed
#         },
#     },
#     "hours_verified": bool,       # True = SerpApi hours present; False => [unverified]
#     "category": str,              # LLM: e.g. "museum"
#     "dwell_minutes": int,         # LLM: estimated visit duration
#     "booking_required": bool,     # LLM
#     "tip": str,                   # LLM one-liner
# }
#
# Leg (one leg-time result, planner_leg_cache row as a dict):
# {
#     "from_name": str, "to_name": str,
#     "walk_minutes": float | None,
#     "transit_minutes": float | None,
#     "drive_minutes": float | None,
#     "distance_km": float | None,
#     "estimated": bool,            # TRUE = haversine fallback (no live directions)
#     "chosen_mode": "walk" | "transit" | "drive",
#     "chosen_minutes": float,
# }
#
# Schedule (Scheduler output):
# {
#     "days": [
#         {
#             "day_index": int,         # 0-based
#             "date": "YYYY-MM-DD",
#             "slots": [
#                 {
#                     "pin_id": str | None,   # None for meal/rest slots
#                     "name": str,
#                     "kind": "stop" | "meal" | "rest",
#                     "start_min": int,       # minutes from midnight
#                     "end_min": int,
#                     "dwell_minutes": int,
#                 },
#             ],
#             "legs": [ {"from_name": str, "to_name": str, "mode": str, "minutes": float} ],
#             "total_scheduled_minutes": int,   # stops only, excludes meals/rest
#         },
#     ],
#     "unplaced": [ {"pin_id": str, "name": str, "reason": str} ],
#     "repairs": [str],                 # human-readable closed-day repair notes
#     "stats": {"total_travel_minutes": float, "load_ratio": float},
# }
#
# Trace row (one graph node run):
# {
#     "seq": int, "node_name": str, "node_type": "agent" | "tool",
#     "round": int, "status": "ok" | "failed",
#     "input": dict | None, "output": dict | None, "error": str | None,
#     "started_at": str ISO, "finished_at": str ISO, "duration_ms": int,
# }

MINUTES_PER_DAY = 24 * 60
WALK_MAX_MINUTES = 20     # a leg walkable in <= 20 min is walked, no transit check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_place_name(name: str) -> str:
    """Lowercase, trim, collapse whitespace/punctuation — used for cache keys."""
    s = (name or "").strip().lower()
    s = re.sub(r"[\s,]+", " ", s)
    s = s.strip(" .,-")
    return s


def leg_cache_key(name_a: str, name_b: str) -> str:
    """Unordered pair key so A->B and B->A share one cached row."""
    a, b = normalize_place_name(name_a), normalize_place_name(name_b)
    lo, hi = sorted([a, b])
    return f"{lo}|{hi}"


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in km (for estimated walking/driving fallbacks)."""
    if None in (lat1, lng1, lat2, lng2):
        return 0.0
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# Rough city-speed constants for the estimated fallback ONLY (never real data).
WALK_KMH = 4.5
DRIVE_KMH = 22.0        # congested city driving
TRANSIT_KMH = 15.0      # door-to-door city transit
TRANSIT_FIXED_MIN = 7   # wait + walk-to-stop overhead per transit leg
PENALTY_KM = 1.4        # road distance vs straight line


def estimated_minutes(mode: str, km: float) -> float:
    """Estimated leg minutes from straight-line km. Flagged 'estimated' upstream."""
    km = km * PENALTY_KM
    if mode == "walk":
        return round(km / WALK_KMH * 60, 1)
    if mode == "drive":
        return round(km / DRIVE_KMH * 60, 1)
    return round(km / TRANSIT_KMH * 60 + TRANSIT_FIXED_MIN, 1)


def parse_hhmm(s: str) -> int | None:
    """'09:30' -> 570 minutes from midnight; None if unparseable."""
    m = re.match(r"^(\d{1,2}):(\d{2})$", (s or "").strip())
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 24 or mi > 59:
        return None
    return h * 60 + mi


def day_dates(start_date: str, num_days: int) -> list[str]:
    """['2026-09-10', ...] for num_days consecutive dates."""
    d0 = date.fromisoformat(start_date)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(num_days)]
