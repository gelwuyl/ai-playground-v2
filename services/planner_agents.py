"""Mr. Bounce — four LLM agent node functions for the trip-orchestrator crew.

Each function takes the session-context dict ``ctx`` and mutates it in place,
returning an output dict for the trace.  The four agents own their tools —
tools are invoked inside an agent's turn (``ctx["_call_tool"]``), NOT as
standalone pipeline steps:

  * ``run_scout``        — owns the ingest + hours tools: resolves the user's
                           pins (ingest), then researches each pin via SerpApi
                           (hours) plus one LLM call per pin for
                           category/dwell/booking/tip/meal_fit/best_time.
  * ``run_reasoner``     — owns the logistics + scheduler tools; runs a
                           draft -> review -> re-draft loop (<= 3 rounds),
                           applying GRADED REMEDIATION directives
                           deterministically and force-proceeding at the cap.
  * ``run_alternatives`` — advisor (no tools): 2-3 swaps per day with honest
                           trade-offs, read from the Scout research table.
  * ``run_compiler``     — assembles the final itinerary JSON + markdown
                           (no tools); passes scheduler advisory notes through
                           verbatim into itinerary slots and the markdown.

Structured data (hours, coords, ratings, travel times) NEVER comes from the
LLM — only from SerpApi / the deterministic scheduler via tools.  The LLM
supplies ONLY category, dwell, booking, tip, meal_fit, best_time (Scout);
judgment + directives (Reasoner); trade-off prose (Alternatives); theme +
intro (Compiler).  Every deterministic guarantee is preserved: agents never
compute travel times or hours themselves.
"""

import os
import re
import urllib.parse
from typing import Any, Callable

from services.openrouter_service import (
    OPENROUTER_MODEL,
    call_openrouter,
    call_openrouter_json,
)
from services.planner_types import parse_hhmm

# ==============================================================================
# MODEL OVERRIDES (repo pattern: shared OPENROUTER_MODEL, per-agent override)
# ==============================================================================
SCOUT_MODEL = os.environ.get("PLANNER_SCOUT_MODEL", OPENROUTER_MODEL)
REASONER_MODEL = os.environ.get(
    "PLANNER_REASONER_MODEL",
    os.environ.get("PLANNER_CRITIC_MODEL", OPENROUTER_MODEL),  # legacy env name
)
ALTERNATIVES_MODEL = os.environ.get("PLANNER_ALTERNATIVES_MODEL", OPENROUTER_MODEL)
COMPILER_MODEL = os.environ.get("PLANNER_COMPILER_MODEL", OPENROUTER_MODEL)

# ==============================================================================
# SYSTEM PROMPTS (personas, like interview_service.py)
# ==============================================================================
SCOUT_SYSTEM = (
    "You are Scout — a travel researcher who knows typical visit durations for "
    "attractions, restaurants, and landmarks worldwide. Given a place name, "
    "city, and address, return a short JSON object with the category, typical "
    "dwell time in minutes, whether booking is required, a one-line tip, "
    "which meal the place fits (when it is food), and the best time window to "
    "visit based on typical crowding."
)

REASONER_SYSTEM = (
    "You are Reasoner — a causal logistics auditor for a day-by-day trip plan. "
    "You review a deterministic schedule in which every stop has real SerpApi "
    "opening hours and real directions travel times. You never invent hours, "
    "travel times, or dwell times — you reason causally about the numbers in "
    "the schedule and research tables only.\n"
    "\n"
    "DO THE ARITHMETIC. For each stop compute arrival = previous stop end + leg "
    "minutes and squeeze = closing - arrival. When a visit is tight but still "
    "feasible, do NOT drop it — keep it, write the arithmetic into the issue "
    "message (e.g. 'closes 17:30, arrive 16:40 — keep visit to 30 min'), and, "
    "if a shorter stay would close the gap, emit a compress_dwell directive "
    "(dwell_minutes floor 20). When the schedule keeps a tight visit, surface "
    "an advisory note with the same arithmetic so the user sees it.\n"
    "\n"
    "GRADED REMEDIATION — apply the LEAST destructive remedy first:\n"
    "  1. reorder — move_before / move_after / move_to_day (same or another "
    "     day, only when opening hours allow).\n"
    "  2. compress_dwell — shorten the stay (floor 20 min) and keep the visit.\n"
    "  3. consult_alternatives=true — no reorder or compression fixes the day; "
    "     ask the Alternatives advisor for a swap.\n"
    "  4. drop — ONLY when no feasible repair exists (e.g. closed all trip "
    "     days, or hours too short to ever fit). Dropping is a last resort, "
    "     never the default for a tight-but-kept visit.\n"
    "\n"
    "MEAL LOGIC. A day spanning >= 6 h needs a meal. When a lunch-venue "
    "category stop has meal_fit == 'lunch', prefer it as the anchored lunch; "
    "sequence morning stops to arrive there at a sensible lunch time. Never "
    "schedule a stop during its closed hours.\n"
    "\n"
    "INSUFFICIENT DATA. If a stop's hours are unverified or a leg is missing "
    "and you cannot judge feasibility, list that pin's name in re_research; "
    "the Scout will re-research it (bounded to one throw-back). If you cannot "
    "reorder or compress to fix a day, set consult_alternatives to true."
)

ALTERNATIVES_SYSTEM = (
    "You are Alternatives — the Reasoner's options advisor. You read the "
    "Scout research table for ALL pins (placed and unplaced) and propose 2-3 "
    "swap options per day, each with a concrete, honest trade-off (travel "
    "impact, timing, crowding, meal fit). You only use candidate names from "
    "the user's own pins — never invented places. Trade-offs are stated in "
    "plain language with the numbers you are given; you never invent travel "
    "times or opening hours."
)

COMPILER_SYSTEM = (
    "You are Compiler — a travel editor who writes a short theme label "
    "(3-6 words) for each day and a 2-sentence trip intro. You do not invent "
    "places, times, or logistics; you only label. Advisory notes already "
    "carried on the schedule's stops pass through verbatim."
)

# ==============================================================================
# Constants
# ==============================================================================
DWELL_FLOOR_MIN = 20            # compress_dwell floor (never compress below 20 min)
MEAL_THRESHOLD_MIN = 360        # days spanning >= 6h need a meal slot
_MEAL_FITS = ("breakfast", "lunch", "dinner", "any")

# ==============================================================================
# Day-key map (SerpApi lowercase -> canonical "0".."6")
# ==============================================================================
_DAY_KEYS = {
    "monday": "0",
    "tuesday": "1",
    "wednesday": "2",
    "thursday": "3",
    "friday": "4",
    "saturday": "5",
    "sunday": "6",
}

# OSM two-letter day selectors (Overpass opening_hours) -> Monday-based index.
_OSM_DAY_IDX = {
    "Mo": 0, "Tu": 1, "We": 2, "Th": 3, "Fr": 4, "Sa": 5, "Su": 6,
}
_OSM_DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday",
                  "friday", "saturday", "sunday")


# ==============================================================================
# parse_raw_hours — pure, deterministic, unit-testable
# ==============================================================================
def _parse_12h_to_24h(token: str) -> str | None:
    """Convert a 12-hour time like '9 AM', '12 PM', '4:30 PM' to 'HH:MM' 24h.

    Returns None if the token is not a valid 12-hour time.
    Handles: "12 AM" -> "00:00", "12 PM" -> "12:00", "12:30 PM" -> "12:30".
    """
    token = token.strip()
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)$", token)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) is not None else 0
    suffix = m.group(3).upper()

    if suffix == "AM":
        if hour == 12:
            hour = 0
    else:  # PM
        if hour != 12:
            hour += 12

    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _normalize_dash(value: str) -> str:
    """Replace en-dash, em-dash, and various hyphens with a plain hyphen."""
    return (
        value
        .replace("–", "-")  # en dash
        .replace("—", "-")  # em dash
        .replace("−", "-")  # minus sign
    )


def _bare_hour_to_24h(token: str) -> str | None:
    """Bare 24h hour token like '9' or '17' -> '09:00'/'17:00'; None otherwise."""
    t = token.strip()
    if re.fullmatch(r"\d{1,2}", t):
        h = int(t)
        if 0 <= h <= 23:
            return f"{h:02d}:00"
    return None


def _parse_token_24h(token: str, meridiem_hint: str | None) -> str | None:
    """Parse one time token with an optional AM/PM hint from its pair token.

    Order: explicit 12h ("9 PM") -> bare hour with the pair's meridiem
    ("9" in "9-9 PM") -> bare 24h hour ("17") -> explicit 24h HH:MM.
    """
    token = token.strip()
    v = _parse_12h_to_24h(token)
    if v is not None:
        return v
    if meridiem_hint:
        v = _parse_12h_to_24h(f"{token} {meridiem_hint}")
        if v is not None:
            return v
    v = _bare_hour_to_24h(token)
    if v is not None:
        return v
    if parse_hhmm(token) is not None:
        return token
    return None


def _subtract_12h(hhmm: str) -> str | None:
    """'21:00' -> '09:00'; None when the result would go negative."""
    m = parse_hhmm(hhmm)
    if m is None or m - 720 < 0:
        return None
    return f"{(m - 720) // 60:02d}:{(m - 720) % 60:02d}"


def _parse_time_range(window: str) -> list[dict] | None:
    """Parse a single time window like '9 AM-9 PM' or '10 AM-12 PM, 2-4:30 PM'.

    Returns a list of interval dicts ``[{"open": "09:00", "close": "21:00"}]``
    or None if the window is unparseable.  Handles 12h AM/PM times, bare-hour
    ranges sharing the pair token's meridiem ("9-9 PM" -> 09:00-21:00), bare
    24h hours ("9-17"), and a "12 AM" close meaning end of day (23:59).
    Unparseable individual windows are skipped.
    """
    normalized = _normalize_dash(window)
    parts = [p.strip() for p in normalized.split(",") if p.strip()]
    intervals: list[dict] = []
    for part in parts:
        if "-" not in part:
            continue
        open_str, close_str = part.split("-", 1)
        open_str = open_str.strip()
        close_str = close_str.strip()

        open_meridiem = re.search(r"(AM|PM)", open_str, re.IGNORECASE)
        close_meridiem = re.search(r"(AM|PM)", close_str, re.IGNORECASE)

        close_24 = _parse_token_24h(
            close_str, open_meridiem.group(1).upper() if open_meridiem else None
        )
        open_24 = _parse_token_24h(
            open_str, close_meridiem.group(1).upper() if close_meridiem else None
        )
        if open_24 is None or close_24 is None:
            continue

        # "9-9 PM": applying the close token's PM to the bare open gives
        # 21:00 == close, so flip the open token back 12h (21:00 -> 09:00).
        open_min = parse_hhmm(open_24) or 0
        close_min = parse_hhmm(close_24) or 0
        if open_min >= close_min:
            flipped = _subtract_12h(open_24)
            if flipped is not None and (parse_hhmm(flipped) or 1440) < close_min:
                open_24 = flipped
                open_min = parse_hhmm(open_24) or 0

        # A close at 00:00 ("5 AM-12 AM") means end of day, not start of it.
        if close_24 == "00:00":
            close_24 = "23:59"
            close_min = 1439

        if open_min >= close_min:
            continue

        intervals.append({"open": open_24, "close": close_24})

    if not intervals:
        return None
    intervals.sort(key=lambda iv: parse_hhmm(iv["open"]) or 0)
    return intervals


def parse_raw_hours(raw: dict | None) -> dict:
    """Convert a SerpApi-style operating_hours passthrough to canonical shape.

    Input: ``{"monday": "9 AM-9 PM", "tuesday": "Closed", ...}`` (7 lowercase
    day keys; values may use en-dash or hyphen, may be None/absent).  Also
    accepts the list form SerpApi returns on place_results: a list of 7
    single-key day dicts ``[{"monday": {...} | "9 AM-9 PM"}, ...]``.

    Output: ``{"days": {"0": [{"open": "09:00", "close": "21:00"}], ...}}``
    where "0"=Monday .. "6"=Sunday.  Closed days map to ``[]``.  Missing/None
    raw -> ``{"days": {}}``.  "Open 24 hours" -> ``[{"open": "00:00",
    "close": "23:59"}]``.  Unparseable windows are skipped (never raise).
    """
    if isinstance(raw, list):
        merged: dict = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            for day, value in item.items():
                if isinstance(value, dict) and value:
                    o, c = value.get("open"), value.get("close")
                    value = f"{o}-{c}" if o and c else None
                if value is not None:
                    merged[str(day).strip().lower()] = value
        raw = merged or None
    if isinstance(raw, str):
        # OSM opening_hours string (Overpass source): "Mo-Su 07:00-22:00",
        # "Mo-Fr 09:00-18:00; Sa 10:00-14:00", "24/7", or a bare daily
        # window "07:00-22:00". Expand into per-day entries.
        raw = {"osm": raw}
    if not raw or not isinstance(raw, dict):
        return {"days": {}}

    # OSM opening_hours syntax (Overpass source): "Mo-Su 07:00-22:00",
    # "Mo-Fr 09:00-18:00; Sa 10:00-14:00", "24/7". Expand each selector
    # into per-day entries so the day-loop below can parse them.
    if not any(k in raw for k in _DAY_KEYS):
        expanded: dict = {}
        osm_str = str(next(iter(raw.values()))) if len(raw) == 1 else ""
        for part in osm_str.split(";"):
            part = part.strip()
            m = re.match(
                r"^(?:(Mo|Tu|We|Th|Fr|Sa|Su)(?:-(Mo|Tu|We|Th|Fr|Sa|Su))?)\s+(.*)$",
                part, re.IGNORECASE,
            )
            if m:
                start_i = _OSM_DAY_IDX[m.group(1).title()]
                end_i = _OSM_DAY_IDX[m.group(2).title()] if m.group(2) else start_i
                if end_i < start_i:
                    end_i += 7  # wraps the week (e.g. Fr-Mo)
                for i in range(start_i, end_i + 1):
                    # Key by day NAME (the day-loop below looks up raw[day_name]).
                    expanded[_OSM_DAY_NAMES[i % 7]] = m.group(3)
            elif part.lower() in ("24/7", "24/7 opening"):
                for day in _OSM_DAY_NAMES:
                    expanded[day] = "Open 24 hours"
            else:
                # Bare time range "07:00-22:00" or "09:00-18:00": same window
                # every day (OSM omits the selector when it applies daily).
                if re.match(r"^\d{1,2}:\d{2}\s*-\s*", part):
                    for day in _OSM_DAY_NAMES:
                        expanded[day] = part
        if expanded:
            raw = expanded
        else:
            return {"days": {}}

    days: dict[str, list[dict]] = {}
    for day_name, day_idx in _DAY_KEYS.items():
        value = raw.get(day_name)
        if value is None:
            continue
        if not isinstance(value, str):
            continue
        value = value.strip()
        if not value:
            continue

        normalized = _normalize_dash(value)

        if normalized.lower() in ("closed", "closed."):
            days[day_idx] = []
            continue

        if "open 24 hours" in normalized.lower() or "24 hours" in normalized.lower():
            days[day_idx] = [{"open": "00:00", "close": "23:59"}]
            continue

        intervals = _parse_time_range(normalized)
        if intervals:
            days[day_idx] = intervals
        # Unparseable: skip this day (don't mark as closed)

    return {"days": days}


# ==============================================================================
# Helpers
# ==============================================================================
def _format_min(mins: int) -> str:
    """Minutes-from-midnight -> 'HH:MM'."""
    mins = int(mins)
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _clamp_dwell(minutes: Any) -> int:
    """Clamp dwell_minutes to 15..480, defaulting to 90 if invalid."""
    try:
        val = int(minutes)
    except (TypeError, ValueError):
        return 90
    return max(15, min(480, val))


# ==============================================================================
# run_scout — agent node that OWNS the ingest + hours tools
# ==============================================================================
SCOUT_SCHEMA = {
    "category": str,
    "dwell_minutes": int,
    "booking_required": bool,
    "tip": str,
    "meal_fit": str,   # "breakfast" | "lunch" | "dinner" | "any" | null (non-food)
    "best_time": str,  # "HH:MM-HH:MM" best window to visit, or null
}

SCOUT_DEFAULTS = {
    "category": "attraction",
    "dwell_minutes": 90,
    "booking_required": False,
    "tip": "",
    "meal_fit": None,
    "best_time": None,
}


def _clean_meal_fit(value: Any) -> str | None:
    """Normalize meal_fit to the contract vocabulary (None for non-food)."""
    if value is None:
        return None
    v = str(value).strip().lower()
    return v if v in _MEAL_FITS else None


def _clean_best_time(value: Any) -> str | None:
    """Normalize best_time to 'HH:MM-HH:MM'; None when malformed."""
    if value is None:
        return None
    v = str(value).strip().replace("–", "-").replace("—", "-")
    m = re.match(r"^(\d{1,2}:\d{2})-(\d{1,2}:\d{2})$", v)
    if not m:
        return None
    a, b = parse_hhmm(m.group(1)), parse_hhmm(m.group(2))
    if a is None or b is None or a >= b:
        return None
    return f"{_format_min(a)}-{_format_min(b)}"


def _scout_one_pin(pin: dict, city: str) -> tuple[dict, str | None]:
    """Research a single pin: SerpApi geocode + hours, one LLM call.

    Returns (PlaceResearch dict, llm_error or None).  Never raises.
    """
    from services import planner_serp

    name = pin.get("name", "Unknown")
    pin_id = pin.get("pin_id", "")

    # --- SerpApi structured data (geocode falls back to Photon/Nominatim) ---
    geo = planner_serp.geocode_place(name, city)
    lat = None
    lng = None
    address = None
    place_id = None
    geocode_source = None

    if geo:
        lat = geo.get("lat") or pin.get("lat")
        lng = geo.get("lng") or pin.get("lng")
        address = geo.get("address") or pin.get("address")
        place_id = geo.get("place_id")
        geocode_source = geo.get("geocode_source") or ("serpapi" if geo.get("place_id") else None)
    else:
        lat = pin.get("lat")
        lng = pin.get("lng")
        address = pin.get("address")

    # Hours: SerpApi first; if that yields nothing, best-effort Overpass (OSM).
    raw_hours = None
    hours_source = "serpapi"
    if place_id or name:
        raw_hours = planner_serp.place_hours(name, city, place_id)
    if not raw_hours and lat is not None and lng is not None:
        from services.planner_free_geo import overpass_hours
        osm = overpass_hours(lat, lng, name)
        if osm:
            raw_hours = osm["hours"]
            hours_source = "osm"

    opening_hours = parse_raw_hours(raw_hours)
    hours_verified = bool(opening_hours["days"])
    # Diagnostics: why hours are missing when they are. LAST_ERROR carries the
    # SerpApi-side reason; the shape shows when hours exist but parsing failed.
    hours_error = getattr(planner_serp, "LAST_ERROR", None)
    raw_hours_shape = None
    if raw_hours:
        if isinstance(raw_hours, dict):
            raw_hours_shape = sorted(raw_hours.keys())
        elif isinstance(raw_hours, list):
            first = raw_hours[0] if raw_hours else None
            if isinstance(first, dict):
                raw_hours_shape = f"list[{len(raw_hours)}] item_keys={sorted(first.keys())}"
            elif first is not None:
                raw_hours_shape = f"list[{len(raw_hours)}] first={str(first)[:80]}"
            else:
                raw_hours_shape = "list[0]"

    # --- LLM call for category, dwell, booking, tip, meal_fit, best_time ---
    user_prompt = (
        f"Place name: {name}\n"
        f"City: {city}\n"
        f"Address: {address or 'unknown'}\n"
        f"Category hint: {pin.get('category', 'unknown')}\n\n"
        "Return JSON with keys: category (string), dwell_minutes (integer), "
        "booking_required (boolean), tip (string), "
        "meal_fit (null unless this is a food place; else one of "
        "breakfast|lunch|dinner|any), "
        "best_time (null, or 'HH:MM-HH:MM' the best window to visit)."
    )

    category = SCOUT_DEFAULTS["category"]
    dwell_minutes = SCOUT_DEFAULTS["dwell_minutes"]
    booking_required = SCOUT_DEFAULTS["booking_required"]
    tip = SCOUT_DEFAULTS["tip"]
    meal_fit = SCOUT_DEFAULTS["meal_fit"]
    best_time = SCOUT_DEFAULTS["best_time"]
    llm_error: str | None = None

    try:
        data = call_openrouter_json(
            user_prompt,
            SCOUT_SYSTEM,
            schema=SCOUT_SCHEMA,
            model=SCOUT_MODEL,
        )
        if isinstance(data, dict):
            cat = data.get("category")
            category = str(cat) if cat else SCOUT_DEFAULTS["category"]
            dwell_minutes = _clamp_dwell(data.get("dwell_minutes"))
            booking_required = bool(data.get("booking_required", False))
            tip_val = data.get("tip")
            tip = str(tip_val) if tip_val else ""
            meal_fit = _clean_meal_fit(data.get("meal_fit"))
            best_time = _clean_best_time(data.get("best_time"))
    except (ValueError, Exception) as e:
        llm_error = str(e)

    # Extract neighborhood from address if available
    neighborhood = ""
    if address:
        parts = [p.strip() for p in address.split(",")]
        if len(parts) >= 2:
            neighborhood = parts[-2]

    result = {
        "pin_id": pin_id,
        "name": name,
        "neighborhood": neighborhood,
        "lat": lat,
        "lng": lng,
        "address": address,
        "rating": None,
        "geocode_source": geocode_source,
        "opening_hours": opening_hours,
        "hours_verified": hours_verified,
        "hours_source": hours_source,
        "hours_error": hours_error,
        "raw_hours_shape": raw_hours_shape,
        "category": category,
        "dwell_minutes": dwell_minutes,
        "booking_required": booking_required,
        "tip": tip,
        "meal_fit": meal_fit,
        "best_time": best_time,
    }
    return result, llm_error


def _require_tool_caller(ctx: dict) -> Callable:
    """Return the runner-injected tool dispatcher ``ctx["_call_tool"]``.

    The runner injects ``_call_tool`` into any agent node that declares
    ``tools`` in the graph; invoking an agent directly without it (outside the
    runner) is a wiring error.
    """
    caller = ctx.get("_call_tool")
    if caller is None or not callable(caller):
        raise RuntimeError(
            "agent node invoked outside the runner: ctx has no '_call_tool' "
            "dispatcher (run through services.planner_graph.advance with a "
            "tools dict wired in)"
        )
    return caller


def hours_tool(ctx: dict, pin: dict | None = None) -> dict:
    """Scout's hours tool: SerpApi geocode + hours + LLM judgment for one pin.

    Emits its own trace row when invoked through the runner's ``_call_tool``.
    Returns the PlaceResearch dict for the pin.
    """
    if pin is None:
        raise ValueError("hours tool requires a pin")
    city = ctx.get("destination", "")
    result, llm_error = _scout_one_pin(pin, city)
    if llm_error:
        ctx.setdefault("errors", []).append(
            f"scout.hours: LLM fallback used for {pin.get('name', 'unknown')}"
        )
    return result


def _scout_fallback_result(pin: dict) -> dict:
    """A minimal PlaceResearch dict for a pin that could not be researched."""
    return {
        "pin_id": pin.get("pin_id", ""),
        "name": pin.get("name", "Unknown"),
        "neighborhood": "",
        "lat": pin.get("lat"),
        "lng": pin.get("lng"),
        "address": pin.get("address"),
        "rating": None,
        "geocode_source": None,
        "opening_hours": {"days": {}},
        "hours_verified": False,
        "hours_source": None,
        "category": SCOUT_DEFAULTS["category"],
        "dwell_minutes": SCOUT_DEFAULTS["dwell_minutes"],
        "booking_required": SCOUT_DEFAULTS["booking_required"],
        "tip": SCOUT_DEFAULTS["tip"],
        "meal_fit": SCOUT_DEFAULTS["meal_fit"],
        "best_time": SCOUT_DEFAULTS["best_time"],
    }


def run_scout(ctx: dict) -> dict:
    """Scout agent node: owns the ingest + hours tools.

    Internally calls its ingest tool to resolve the user's pins from
    ``ctx["payload"]`` into ``ctx["pins"]``, then researches every resolved pin
    via its hours tool (SerpApi geocode + hours + LLM judgment).  Each tool
    invocation emits its own trace row (node_type "tool", parent "scout").
    """
    errors = ctx.setdefault("errors", [])
    call_tool = _require_tool_caller(ctx)

    # Tool: ingest — parse + resolve + persist the pins (owned by Scout).
    call_tool("ingest")

    city = ctx.get("destination", "")
    pins = ctx.get("pins", [])
    resolved_pins = [p for p in pins if p.get("resolved", False)]

    research: list[dict] = []
    hours_verified_count = 0

    for pin in resolved_pins:
        try:
            result = call_tool("hours", pin=pin)
            research.append(result)
            if result.get("hours_verified"):
                hours_verified_count += 1
        except Exception as e:
            errors.append(f"scout: pin '{pin.get('name', 'unknown')}' failed: {e}")
            research.append(_scout_fallback_result(pin))

    ctx["research"] = research
    n = len(research)
    return {
        "researched": n,
        "hours_verified": hours_verified_count,
        "hours_unverified": n - hours_verified_count,
        "hours_diagnostics": [
            {
                "name": r.get("name"),
                "hours_source": r.get("hours_source"),
                "geocode_source": r.get("geocode_source"),
                "hours_error": r.get("hours_error"),
                "raw_hours_shape": r.get("raw_hours_shape"),
            }
            for r in research
        ],
    }


# ==============================================================================
# run_reasoner — causal logistics auditor (owns logistics + scheduler tools)
# ==============================================================================
REASONER_SCHEMA = {
    "verdict": str,
    "issues": list,
    "directives": list,
    "re_research": str,
    "consult_alternatives": bool,
}


def _deterministic_reasoner_checks(ctx: dict) -> list[dict]:
    """Run deterministic checks on the schedule. Returns a list of issue dicts.

    Checks:
      (a) any slot's end > its day's closing (schedule violates hours)
      (b) any day span >= 360 min has no meal slot
    """
    issues: list[dict] = []
    schedule = ctx.get("schedule")
    if not schedule or not isinstance(schedule, dict):
        return issues

    research_by_name: dict[str, dict] = {}
    for r in ctx.get("research", []):
        if isinstance(r, dict) and r.get("name"):
            research_by_name[r["name"]] = r

    for day in schedule.get("days", []):
        if not isinstance(day, dict):
            continue
        day_index = day.get("day_index", 0)
        slots = day.get("slots", [])
        day_start: int | None = None
        day_end: int | None = None
        has_meal = False

        for slot in slots:
            if not isinstance(slot, dict):
                continue
            s = slot.get("start_min")
            e = slot.get("end_min")
            if s is not None:
                if day_start is None or s < day_start:
                    day_start = s
            if e is not None:
                if day_end is None or e > day_end:
                    day_end = e
            if slot.get("kind") == "meal":
                has_meal = True

            # Check (a): slot end > closing hours for this place
            name = slot.get("name")
            if name and name in research_by_name:
                r = research_by_name[name]
                opening_hours = r.get("opening_hours", {})
                days_dict = opening_hours.get("days", {})
                day_key = str(day_index)
                intervals = days_dict.get(day_key)
                if intervals is not None and e is not None:
                    max_close = 0
                    for iv in intervals:
                        close_min = parse_hhmm(iv.get("close", "23:59"))
                        if close_min is not None and close_min > max_close:
                            max_close = close_min
                    if e > max_close:
                        issues.append({
                            "day_index": day_index,
                            "severity": "high",
                            "message": (
                                f"Stop '{name}' ends at {_format_min(e)} "
                                f"but closes at {_format_min(max_close)}"
                            ),
                        })

        # Check (b): day span >= 360 min with no meal
        if day_start is not None and day_end is not None:
            span = day_end - day_start
            if span >= MEAL_THRESHOLD_MIN and not has_meal:
                issues.append({
                    "day_index": day_index,
                    "severity": "medium",
                    "message": (
                        f"Day {day_index} spans {span} minutes "
                        f"({_format_min(day_start)}-{_format_min(day_end)}) "
                        f"with no meal slot"
                    ),
                })

    return issues


def _build_reasoner_prompt(ctx: dict, applied_directives: list[dict]) -> str:
    """Build a compact causal-audit prompt for the Reasoner LLM."""
    schedule = ctx.get("schedule", {})
    research = ctx.get("research", [])
    lines: list[str] = [
        f"Destination: {ctx.get('destination', 'unknown')}",
        f"Start date: {ctx.get('start_date', 'unknown')}",
        f"Num days: {ctx.get('num_days', 0)}",
        "",
        "Researched places:",
    ]
    for r in research:
        if not isinstance(r, dict):
            continue
        lines.append(
            f"  - {r.get('name', '?')}: category={r.get('category', '?')}, "
            f"dwell={r.get('dwell_minutes', '?')}min, "
            f"hours_verified={r.get('hours_verified', False)}, "
            f"meal_fit={r.get('meal_fit', '?')}, "
            f"best_time={r.get('best_time', '?')}"
        )

    lines.append("")
    lines.append("Schedule (all times are real SerpApi hours / directions):")
    for day in schedule.get("days", []):
        if not isinstance(day, dict):
            continue
        lines.append(f"  Day {day.get('day_index', '?')} ({day.get('date', '?')}):")
        for slot in day.get("slots", []):
            if not isinstance(slot, dict):
                continue
            s = _format_min(slot.get("start_min", 0))
            e = _format_min(slot.get("end_min", 0))
            kind = slot.get("kind", "stop")
            name = slot.get("name", "?")
            line = f"    {s}-{e} [{kind}] {name}"
            note = slot.get("advisory_note")
            if note:
                line += f"   (advisory: {note})"
            lines.append(line)
        for leg in day.get("legs", []):
            if not isinstance(leg, dict):
                continue
            lines.append(
                f"    leg: {leg.get('from_name', '?')} -> "
                f"{leg.get('to_name', '?')} ({leg.get('mode', '?')}, "
                f"{leg.get('minutes', '?')}min)"
            )
        lines.append(
            f"    total_scheduled: {day.get('total_scheduled_minutes', 0)}min"
        )

    if applied_directives:
        lines.append("")
        lines.append("Directives ALREADY applied by earlier drafts — do not repeat:")
        for d in applied_directives:
            lines.append(f"  - {d}")

    alts = ctx.get("alternatives")
    if isinstance(alts, dict) and alts.get("days"):
        lines.append("")
        lines.append("Alternatives advisor suggestions (from a prior consult):")
        for day in alts.get("days", []):
            if not isinstance(day, dict):
                continue
            for swap in day.get("swaps", []):
                if isinstance(swap, dict):
                    lines.append(
                        f"  - Day {day.get('day_index', '?')}: swap "
                        f"{swap.get('remove_name', '?')} for "
                        f"{swap.get('add_name', '?')} — {swap.get('trade_off', '')}"
                    )

    lines.append("")
    lines.append(
        "Audit the schedule CAUSALLY using only the numbers above — never invent "
        "hours, travel times, or dwell times. GRADED REMEDIATION, least "
        "destructive first: (1) reorder (move_before/move_after/move_to_day), "
        "(2) compress_dwell (dwell_minutes floor 20), "
        "(3) consult_alternatives=true, (4) drop — only when no feasible repair "
        "exists. A tight-but-kept visit is an advisory note, never a drop. "
        'Return JSON: {"verdict": "PASS" or "ISSUES", "issues": '
        '[{"day_index": int, "severity": "high|medium|low", "message": str}], '
        '"directives": [{"action": '
        '"move_before|move_after|move_to_day|compress_dwell|drop", '
        '"stop": str, "reference": str|null, "day": int|null, '
        '"dwell_minutes": int|null, "reason": str}], '
        '"re_research": str (comma-separated pin names; empty string if none), '
        '"consult_alternatives": bool}'
    )
    return "\n".join(lines)


def _clean_reasoner_issues(value: Any) -> list[dict]:
    """Normalize LLM issues to the {day_index, severity, message} contract."""
    if not isinstance(value, list):
        return []
    cleaned: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            day_index = int(item.get("day_index", 0))
        except (TypeError, ValueError):
            day_index = 0
        severity = str(item.get("severity", "low")).strip().lower()
        if severity not in ("high", "medium", "low"):
            severity = "low"
        message = str(item.get("message", "") or "").strip()
        if message:
            cleaned.append({
                "day_index": day_index,
                "severity": severity,
                "message": message,
            })
    return cleaned


def _clean_directive(value: Any) -> dict | None:
    """Normalize one LLM directive to the scheduler's contract.

    Returns None for anything unusable (unknown action or missing stop) so it
    can be skipped rather than letting a malformed directive poison the draft.
    """
    if not isinstance(value, dict):
        return None
    action = str(value.get("action", "") or "").strip()
    stop = str(value.get("stop", "") or "").strip()
    if action not in ("move_before", "move_after", "move_to_day",
                      "compress_dwell", "drop"):
        return None
    if not stop:
        return None

    reference_raw = value.get("reference")
    reference = str(reference_raw).strip() if reference_raw else None

    day_raw = value.get("day")
    day: int | None = None
    if isinstance(day_raw, bool):
        day = None
    elif isinstance(day_raw, int):
        day = day_raw
    elif isinstance(day_raw, str) and day_raw.strip().lstrip("-").isdigit():
        day = int(day_raw.strip())

    dwell_raw = value.get("dwell_minutes")
    dwell: int | None = None
    if isinstance(dwell_raw, bool):
        dwell = None
    elif isinstance(dwell_raw, int):
        dwell = dwell_raw
    elif isinstance(dwell_raw, str) and dwell_raw.strip().isdigit():
        dwell = int(dwell_raw.strip())

    reason = str(value.get("reason", "") or "").strip() or None

    return {
        "action": action,
        "stop": stop,
        "reference": reference,
        "day": day,
        "dwell_minutes": dwell,
        "reason": reason,
    }


def _alternatives_swap_directives(ctx: dict) -> list[dict]:
    """Translate a prior consult's Alternatives swaps into drop directives.

    A swap says \"remove X, add Y from the unplaced pool\".  Deterministically
    turning it into a ``drop`` on X frees X's slot so the scheduler cascade can
    place Y; the reason for the drop names the swap so the trace stays honest.
    Applied once per run (guarded by ``_swap_directives_applied``).
    """
    if ctx.get("_swap_directives_applied"):
        return []
    alts = ctx.get("alternatives")
    if not isinstance(alts, dict):
        return []
    out: list[dict] = []
    for day in alts.get("days", []):
        if not isinstance(day, dict):
            continue
        for swap in day.get("swaps", []):
            if not isinstance(swap, dict):
                continue
            remove = (swap.get("remove_name") or "").strip()
            add = (swap.get("add_name") or "").strip()
            trade = (swap.get("trade_off") or "").strip()
            if not remove:
                continue
            reason = "consult alternatives"
            if add:
                reason = f"consult: swap {remove} for {add}"
            if trade:
                reason += f" ({trade})"
            out.append({"action": "drop", "stop": remove, "reason": reason})
    return out


def _review_schedule(ctx: dict, applied_directives: list[dict]) -> dict:
    """One Reasoner LLM review of the current schedule. Never raises."""
    errors = ctx.setdefault("errors", [])
    try:
        data = call_openrouter_json(
            _build_reasoner_prompt(ctx, applied_directives),
            REASONER_SYSTEM,
            schema=REASONER_SCHEMA,
            model=REASONER_MODEL,
        )
        if isinstance(data, dict):
            return data
        raise ValueError("reasoner: LLM returned non-dict JSON")
    except (ValueError, Exception) as e:
        # Default to a safe PASS shape; deterministic checks still apply.
        errors.append(f"reasoner: LLM review failed, using defaults — {e}")
        return {
            "verdict": "PASS",
            "issues": [],
            "directives": [],
            "re_research": "",
            "consult_alternatives": False,
        }


def run_reasoner(ctx: dict) -> dict:
    """Reasoner agent node: owns the logistics + scheduler tools.

    Runs a draft -> review -> re-draft loop bounded by ``max_reasoner_rounds``
    (graph key; injected into ctx as ``_max_reasoner_rounds``).  Each draft
    calls the scheduler tool with the directives accumulated so far; each
    review asks the LLM for directives (GRADED REMEDIATION, least destructive
    first).  At the cap it force-proceeds with issues surfaced — never loops
    forever.  The final output dict drives the runner's conditional edges:
    ``re_research`` (non-empty string) throws back to Scout (max 1),
    ``consult_alternatives`` (bool) throws back via Alternatives (max 1).
    """
    errors = ctx.setdefault("errors", [])
    call_tool = _require_tool_caller(ctx)

    max_rounds = int(ctx.get("_max_reasoner_rounds", 3) or 3)
    if max_rounds < 1:
        max_rounds = 1
    start_round = int(ctx.get("reasoner_round", 0) or 0)
    start_round = min(start_round, max_rounds - 1)

    # Tool: logistics once per turn — the travel matrix does not change
    # between drafts (drop directives leave the unused legs in place but the
    # scheduler only consumes legs between scheduled stops). Long trips
    # batch: the tool fetches a limited number of legs per call; if work
    # remains, return a "working" result so the runner re-enters this node
    # (bounded by the re-research budget) instead of drafting on a partial
    # travel matrix.
    logistics_out = call_tool("logistics")
    pending = int((logistics_out or {}).get("pending_legs") or 0)
    if pending > 0:
        working = {
            "verdict": "ISSUES",
            "issues": [{"day_index": 0, "severity": "low",
                        "message": f"Collecting travel times: {pending} leg(s) remaining."}],
            "directives": [],
            "re_research": "",
            "consult_alternatives": False,
            "_logistics_pending": pending,
        }
        ctx["reasoner"] = working
        return working

    # If a prior consult already suggested swaps, honour them on the first
    # draft (drop frees a slot; the cascade re-places the added pin).
    directives_so_far: list[dict] = _alternatives_swap_directives(ctx)
    if directives_so_far:
        ctx["_swap_directives_applied"] = True

    consult_done = bool(ctx.get("consult_round", 0))  # runner-level consult budget

    final = {
        "verdict": "PASS",
        "issues": [],
        "directives": [],
        "re_research": "",
        "consult_alternatives": False,
    }

    round_no = start_round
    while round_no < max_rounds:
        # --- Draft: run the deterministic scheduler with accumulated directives.
        draft = call_tool("scheduler", directives=list(directives_so_far))
        if isinstance(draft, dict) and draft:
            ctx["schedule"] = draft
        round_no += 1
        ctx["reasoner_round"] = round_no  # a draft was consumed

        # Deterministic pre-check (belt and braces, never from the LLM).
        det_issues = _deterministic_reasoner_checks(ctx)

        # --- Review: LLM causal audit of the current draft.
        review = _review_schedule(ctx, list(directives_so_far))
        llm_issues = _clean_reasoner_issues(review.get("issues"))

        merged: list[dict] = []
        seen: set[tuple] = set()
        for issue in det_issues + llm_issues:
            key = (
                issue.get("day_index"),
                issue.get("severity"),
                issue.get("message"),
            )
            if key not in seen:
                seen.add(key)
                merged.append(issue)
        issues = merged

        re_research = str(review.get("re_research", "") or "").strip()
        consult = bool(review.get("consult_alternatives", False))

        for directive in review.get("directives", []):
            clean = _clean_directive(directive)
            if clean is not None:
                directives_so_far.append(clean)

        final = {
            "verdict": "ISSUES" if issues else "PASS",
            "issues": issues,
            "directives": list(directives_so_far),
            "re_research": re_research,
            "consult_alternatives": consult,
        }

        # --- Termination: stop refining when there is nothing left to fix.
        if not issues and not re_research and not consult:
            break
        if re_research:
            # Ask Scout to re-research; the runner throws back (max 1).
            break
        if consult and consult_done:
            errors.append(
                "reasoner: consult budget already consumed (max 1); "
                "force-proceeding with the last draft"
            )
            break
        if not directives_so_far:
            # Issues remain but no actionable remedy -> proceed, issues surfaced.
            break
    else:
        errors.append(
            f"reasoner: max rounds reached ({max_rounds}); proceeding with issues"
        )

    ctx["reasoner"] = final
    return final


# Legacy aliases — Critic was renamed Reasoner; kept so stale imports
# (e.g. api/planner.py before its registry update) keep working.
CRITIC_SYSTEM = REASONER_SYSTEM
CRITIC_MODEL = REASONER_MODEL
CRITIC_SCHEMA = REASONER_SCHEMA


def run_critic(ctx: dict) -> dict:
    """Legacy alias for ``run_reasoner`` (Critic was renamed Reasoner)."""
    return run_reasoner(ctx)


# ==============================================================================
# run_alternatives — 2-3 swaps per day with trade-offs
# ==============================================================================
ALTERNATIVES_SCHEMA = {
    "days": list,
}


def _build_alternatives_prompt(ctx: dict) -> str:
    """Build the prompt for the Alternatives LLM."""
    schedule = ctx.get("schedule", {})
    research = ctx.get("research", [])

    # Per-day placed names
    placed_by_day: dict[int, list[str]] = {}
    for day in schedule.get("days", []):
        if not isinstance(day, dict):
            continue
        di = day.get("day_index", 0)
        names = [
            s.get("name", "")
            for s in day.get("slots", [])
            if isinstance(s, dict) and s.get("kind") == "stop"
        ]
        placed_by_day[di] = names

    # Unplaced pins
    unplaced = schedule.get("unplaced", [])
    unplaced_names = [
        u.get("name", "") for u in unplaced if isinstance(u, dict)
    ]

    # Alternatives hints from the scheduler (tolerate absence)
    hints = schedule.get("alternatives_hints", {})

    lines: list[str] = []
    lines.append(f"Destination: {ctx.get('destination', 'unknown')}")
    lines.append("")

    for day in schedule.get("days", []):
        if not isinstance(day, dict):
            continue
        di = day.get("day_index", 0)
        placed = placed_by_day.get(di, [])

        # Candidate names: unplaced + names from other days
        candidates = list(unplaced_names)
        for other_di, other_names in placed_by_day.items():
            if other_di != di:
                for n in other_names:
                    if n not in candidates and n not in placed:
                        candidates.append(n)

        # Also use hints if present
        day_hints = hints.get(str(di)) or hints.get(di) or []
        if isinstance(day_hints, list):
            for h in day_hints:
                if isinstance(h, dict) and h.get("name"):
                    if h["name"] not in candidates:
                        candidates.append(h["name"])

        lines.append(f"Day {di} ({day.get('date', '?')}):")
        lines.append(
            f"  Currently placed: {', '.join(placed) if placed else 'none'}"
        )
        lines.append(
            f"  Candidate swaps: {', '.join(candidates) if candidates else 'none'}"
        )
        lines.append("")

    lines.append(
        "For each day, propose 2-3 swaps. Each swap: remove a currently placed "
        "name, add a candidate name, and state the concrete trade-off (e.g. "
        "'+20 min travel, but quieter mornings'). Only use the candidate names "
        'provided. Return JSON: {"days": [{"day_index": int, "swaps": '
        '[{"remove_name": str, "add_name": str, "trade_off": str}]}]}'
    )

    return "\n".join(lines)


def run_alternatives(ctx: dict) -> dict:
    """Alternatives node: 2-3 swaps per day with trade-offs."""
    errors = ctx.setdefault("errors", [])

    output: dict[str, Any]
    try:
        data = call_openrouter_json(
            _build_alternatives_prompt(ctx),
            ALTERNATIVES_SYSTEM,
            schema=ALTERNATIVES_SCHEMA,
            model=ALTERNATIVES_MODEL,
        )
        if isinstance(data, dict) and isinstance(data.get("days"), list):
            output = {"days": data["days"]}
        else:
            output = {"days": []}
            errors.append(
                "alternatives: LLM returned unexpected shape, defaulting to empty"
            )
    except (ValueError, Exception) as e:
        output = {"days": []}
        errors.append(f"alternatives: LLM failed, defaulting to empty — {e}")

    ctx["alternatives"] = output
    return output


# ==============================================================================
# run_compiler — final assembly (deterministic structure + LLM theme/intro)
# ==============================================================================
COMPILER_SCHEMA = {
    "themes": list,
    "intro": str,
}


def _research_lookup(research: list[dict]) -> dict[str, dict]:
    """Build a name -> research dict lookup."""
    lookup: dict[str, dict] = {}
    for r in research:
        if isinstance(r, dict) and r.get("name"):
            lookup[r["name"]] = r
    return lookup


def _build_compiler_prompt(ctx: dict) -> str:
    """Build the prompt for the Compiler LLM (themes + intro only)."""
    schedule = ctx.get("schedule", {})
    days = schedule.get("days", [])
    lines: list[str] = [
        f"Destination: {ctx.get('destination', 'unknown')}",
        f"Start date: {ctx.get('start_date', 'unknown')}",
        f"Number of days: {ctx.get('num_days', 0)}",
        "",
        "Day summaries:",
    ]
    for day in days:
        if not isinstance(day, dict):
            continue
        di = day.get("day_index", 0)
        stop_names = [
            s.get("name", "")
            for s in day.get("slots", [])
            if isinstance(s, dict) and s.get("kind") == "stop"
        ]
        lines.append(
            f"  Day {di} ({day.get('date', '?')}): "
            f"{', '.join(stop_names) if stop_names else 'rest day'}"
        )

    n_days = len(days)
    lines.append("")
    lines.append(
        f'Return JSON: {{"themes": ["3-6 word theme for each of {n_days} days"], '
        '"intro": "2-sentence trip introduction"}}'
    )
    return "\n".join(lines)


def _directions_url(day_stops: list[dict]) -> str:
    """Google Maps directions URL following a day's stops in order.

    Uses the Maps URLs api=1 form: origin, destination, and the middle stops
    as waypoints (coordinates, pipe-separated). Only stops with coordinates
    participate; names would be ambiguous and cost geocode calls in Maps.
    Returns "" when fewer than two stops have coordinates.
    """
    pts: list[str] = []
    for stop in day_stops:
        if not isinstance(stop, dict):
            continue
        lat, lng = stop.get("lat"), stop.get("lng")
        if lat is None or lng is None:
            continue
        pts.append(f"{float(lat):.6f},{float(lng):.6f}")
    if len(pts) < 2:
        return ""
    url = "https://www.google.com/maps/dir/?api=1"
    url += "&origin=" + urllib.parse.quote(pts[0])
    url += "&destination=" + urllib.parse.quote(pts[-1])
    if len(pts) > 2:
        url += "&waypoints=" + urllib.parse.quote("|".join(pts[1:-1]))
    return url


def _assemble_itinerary_json(
    ctx: dict, themes: list[str], intro: str
) -> dict:
    """Deterministically assemble the full itinerary_json from ctx data."""
    schedule = ctx.get("schedule", {})
    research = ctx.get("research", [])
    alternatives = ctx.get("alternatives", {"days": []})
    errors = ctx.get("errors", [])
    repairs = schedule.get("repairs", [])

    rlookup = _research_lookup(research)
    days_out: list[dict] = []

    for day in schedule.get("days", []):
        if not isinstance(day, dict):
            continue
        di = day.get("day_index", 0)
        date_str = day.get("date", "")

        # Theme
        if di < len(themes) and themes[di]:
            theme = str(themes[di])
        else:
            theme = f"Day {di + 1}"

        # Stops
        stops_out: list[dict] = []
        for slot in day.get("slots", []):
            if not isinstance(slot, dict):
                continue
            name = slot.get("name", "Unknown")
            kind = slot.get("kind", "stop")
            r = rlookup.get(name, {})

            start_min = slot.get("start_min", 0)
            end_min = slot.get("end_min", 0)

            hours_verified = (
                r.get("hours_verified", False) if kind == "stop" else True
            )
            stop_out = {
                "name": name,
                "category": (
                    r.get("category", "attraction")
                    if kind == "stop"
                    else kind
                ),
                "start": _format_min(start_min),
                "end": _format_min(end_min),
                "dwell_minutes": slot.get("dwell_minutes", 0),
                "address": r.get("address") or "",
                "lat": r.get("lat"),
                "lng": r.get("lng"),
                "tip": r.get("tip", ""),
                "booking_required": (
                    r.get("booking_required", False)
                    if kind == "stop"
                    else False
                ),
                "hours_verified": hours_verified,
                "kind": kind,
            }

            # hours_flag: only for stop kind stops
            if kind == "stop" and not hours_verified:
                stop_out["hours_flag"] = "[unverified]"
            else:
                stop_out["hours_flag"] = ""
            # Advisory notes from the scheduler pass through verbatim.
            stop_out["advisory_note"] = str(slot.get("advisory_note", "") or "")

            stops_out.append(stop_out)

        # Legs
        legs_out: list[dict] = []
        for leg in day.get("legs", []):
            if not isinstance(leg, dict):
                continue
            legs_out.append({
                "from": leg.get("from_name", ""),
                "to": leg.get("to_name", ""),
                "mode": leg.get("mode", ""),
                "minutes": leg.get("minutes", 0),
            })

        # Per-day travel = sum of the legs actually used between stops that day
        # (the schedule's total_scheduled_minutes counts stop time, not travel).
        total_travel = sum(
            leg.get("minutes", 0)
            for leg in day.get("legs", [])
            if isinstance(leg, dict)
        )
        load_minutes = sum(
            s.get("dwell_minutes", 0)
            for s in day.get("slots", [])
            if isinstance(s, dict) and s.get("kind") == "stop"
        )

        days_out.append({
            "day_index": di,
            "date": date_str,
            "theme": theme,
            "stops": stops_out,
            "legs_between": legs_out,
            "total_travel_minutes": total_travel,
            "load_minutes": load_minutes,
            "directions_url": _directions_url(stops_out),
        })

    # Notes: repairs + errors summary
    notes: list[str] = []
    for r in repairs:
        if isinstance(r, str):
            notes.append(r)
    if errors:
        notes.append(f"Errors encountered: {len(errors)}")
        for e in errors:
            notes.append(f"  - {e}")

    return {
        "trip": {
            "city": ctx.get("destination", ""),
            "start_date": ctx.get("start_date", ""),
            "num_days": ctx.get("num_days", 0),
            "days": days_out,
            "alternatives": alternatives,
            "intro": intro,
            "notes": notes,
            "sources": {
                "hours_source": "serpapi",
                "travel_source": "serpapi_directions_or_estimated",
                "research_source": "openrouter",
            },
        }
    }


def render_itinerary_md(itinerary_json: dict) -> str:
    """Render the itinerary_json as markdown. Pure, deterministic, no LLM.

    Produces a title, intro, per-day headings, stop tables, legs, and
    alternatives sections.
    """
    trip = itinerary_json.get("trip", {})
    city = trip.get("city", "Trip")
    start_date = trip.get("start_date", "")
    num_days = trip.get("num_days", 0)
    intro = trip.get("intro", "")
    days = trip.get("days", [])
    alternatives = trip.get("alternatives", {})
    alt_days = (
        alternatives.get("days", [])
        if isinstance(alternatives, dict)
        else []
    )

    lines: list[str] = []
    lines.append(f"# {city} Trip")
    if intro:
        lines.append("")
        lines.append(intro)
    if start_date:
        lines.append("")
        lines.append(f"*Starting {start_date}, {num_days} day(s)*")

    for day in days:
        if not isinstance(day, dict):
            continue
        di = day.get("day_index", 0)
        date_str = day.get("date", "")
        theme = day.get("theme", f"Day {di + 1}")

        lines.append("")
        lines.append(f"## Day {di + 1} — {date_str} — {theme}")
        dir_url = day.get("directions_url", "")
        if dir_url:
            lines.append("")
            lines.append(f"[Open this day's route in Google Maps]({dir_url})")
        lines.append("")

        # Stops
        stops = day.get("stops", [])
        if stops:
            lines.append("| Time | Name | Type | Tip | Booking | Hours |")
            lines.append("|------|------|------|-----|---------|-------|")
            for stop in stops:
                if not isinstance(stop, dict):
                    continue
                start = stop.get("start", "")
                end = stop.get("end", "")
                name = stop.get("name", "")
                kind = stop.get("kind", "stop")
                tip = stop.get("tip", "")
                booking = "Yes" if stop.get("booking_required") else ""
                hours_flag = stop.get("hours_flag", "")
                if (
                    not hours_flag
                    and not stop.get("hours_verified", True)
                    and stop.get("kind") == "stop"
                ):
                    hours_flag = "[unverified]"
                lines.append(
                    f"| {start}–{end} | {name} | {kind} | {tip} | "
                    f"{booking} | {hours_flag} |"
                )
            lines.append("")

        # Advisory notes (verbatim from the scheduler) — one bullet per note.
        advisory_note_lines = [
            f"- {stop.get('name', '?')}: {stop['advisory_note']}"
            for stop in stops
            if isinstance(stop, dict) and stop.get("advisory_note")
        ]
        if advisory_note_lines:
            lines.append("**Advisory notes:**")
            lines.extend(advisory_note_lines)
            lines.append("")

        # Legs
        legs = day.get("legs_between", [])
        if legs:
            lines.append("**Travel legs:**")
            for leg in legs:
                if not isinstance(leg, dict):
                    continue
                lines.append(
                    f"- {leg.get('from', '')} → {leg.get('to', '')}: "
                    f"{leg.get('mode', '')}, {leg.get('minutes', 0)} min"
                )
            lines.append("")

        # Alternatives for this day
        for alt_day in alt_days:
            if not isinstance(alt_day, dict):
                continue
            if alt_day.get("day_index") == di:
                swaps = alt_day.get("swaps", [])
                if swaps:
                    lines.append("**Alternatives:**")
                    for swap in swaps:
                        if not isinstance(swap, dict):
                            continue
                        lines.append(
                            f"- Swap **{swap.get('remove_name', '')}** for "
                            f"**{swap.get('add_name', '')}** — "
                            f"{swap.get('trade_off', '')}"
                        )
                    lines.append("")
                break

    return "\n".join(lines)


def run_compiler(ctx: dict) -> dict:
    """Compiler node: LLM themes + intro, then deterministic assembly."""
    schedule = ctx.get("schedule", {})
    days = schedule.get("days", [])
    n_days = len(days)

    # LLM for themes + intro
    themes: list[str] = []
    intro = ""

    try:
        data = call_openrouter_json(
            _build_compiler_prompt(ctx),
            COMPILER_SYSTEM,
            schema=COMPILER_SCHEMA,
            model=COMPILER_MODEL,
        )
        if isinstance(data, dict):
            raw_themes = data.get("themes", [])
            if isinstance(raw_themes, list):
                themes = [str(t) for t in raw_themes if t]
            intro = str(data.get("intro", "") or "")
    except (ValueError, Exception):
        pass  # defaults: themes=[], intro="" -> "Day N" fallbacks

    # Fill missing themes with "Day N" defaults
    while len(themes) < n_days:
        themes.append(f"Day {len(themes) + 1}")

    # Deterministic assembly
    itinerary_json = _assemble_itinerary_json(ctx, themes, intro)
    markdown = render_itinerary_md(itinerary_json)

    ctx["itinerary"] = itinerary_json
    return {"itinerary_json": itinerary_json, "markdown": markdown}
