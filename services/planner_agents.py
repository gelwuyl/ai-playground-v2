"""Mr. Bounce — four LLM agent node functions for the trip-orchestrator crew.

Each function takes the session-context dict ``ctx`` and mutates it in place,
returning an output dict for the trace.  The four nodes are:

  * ``run_scout``        — fan out per pin: SerpApi geocode + hours, one LLM call
                          for category/dwell/booking/tip.
  * ``run_critic``       — audit the schedule for tightness, missing meals,
                          out-of-hours stops, lopsided days.
  * ``run_alternatives`` — 2-3 swaps per day with trade-off prose.
  * ``run_compiler``     — assemble the final itinerary JSON + markdown.

Structured data (hours, coords, ratings) NEVER comes from the LLM — only from
SerpApi.  The LLM supplies ONLY category, dwell, booking, tip (Scout);
judgment (Critic); trade-off prose (Alternatives); theme + intro (Compiler).
"""

import os
import re
from typing import Any

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
CRITIC_MODEL = os.environ.get("PLANNER_CRITIC_MODEL", OPENROUTER_MODEL)
ALTERNATIVES_MODEL = os.environ.get("PLANNER_ALTERNATIVES_MODEL", OPENROUTER_MODEL)
COMPILER_MODEL = os.environ.get("PLANNER_COMPILER_MODEL", OPENROUTER_MODEL)

# ==============================================================================
# SYSTEM PROMPTS (personas, like interview_service.py)
# ==============================================================================
SCOUT_SYSTEM = (
    "You are Scout — a travel researcher who knows typical visit durations for "
    "attractions, restaurants, and landmarks worldwide. Given a place name, "
    "city, and address, return a short JSON object with the category, typical "
    "dwell time in minutes, whether booking is required, and a one-line tip."
)

CRITIC_SYSTEM = (
    "You are Critic — a skeptical itinerary auditor. You examine schedules "
    "for unrealistic tightness, missing meals on long days, stops outside "
    "opening hours, and lopsided or empty days. You are precise and concise."
)

ALTERNATIVES_SYSTEM = (
    "You are Alternatives — a travel planner who proposes 2-3 swap options per "
    "day, each with a concrete trade-off. You only use candidate names provided "
    "to you and you state the trade-off in plain language."
)

COMPILER_SYSTEM = (
    "You are Compiler — a travel editor who writes a short theme label "
    "(3-6 words) for each day and a 2-sentence trip intro. You do not invent "
    "places, times, or logistics; you only label."
)

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

MEAL_THRESHOLD_MIN = 360  # days spanning >= 6h need a meal slot


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
    day keys; values may use en-dash or hyphen, may be None/absent).

    Output: ``{"days": {"0": [{"open": "09:00", "close": "21:00"}], ...}}``
    where "0"=Monday .. "6"=Sunday.  Closed days map to ``[]``.  Missing/None
    raw -> ``{"days": {}}``.  "Open 24 hours" -> ``[{"open": "00:00",
    "close": "23:59"}]``.  Unparseable windows are skipped (never raise).
    """
    if not raw or not isinstance(raw, dict):
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
# run_scout — fan-out node
# ==============================================================================
SCOUT_SCHEMA = {
    "category": str,
    "dwell_minutes": int,
    "booking_required": bool,
    "tip": str,
}

SCOUT_DEFAULTS = {
    "category": "attraction",
    "dwell_minutes": 90,
    "booking_required": False,
    "tip": "",
}


def _scout_one_pin(pin: dict, city: str) -> tuple[dict, str | None]:
    """Research a single pin: SerpApi geocode + hours, one LLM call.

    Returns (PlaceResearch dict, llm_error or None).  Never raises.
    """
    from services import planner_serp

    name = pin.get("name", "Unknown")
    pin_id = pin.get("pin_id", "")

    # --- SerpApi structured data ---
    geo = planner_serp.geocode_place(name, city)
    lat = None
    lng = None
    address = None
    place_id = None

    if geo:
        lat = geo.get("lat") or pin.get("lat")
        lng = geo.get("lng") or pin.get("lng")
        address = geo.get("address") or pin.get("address")
        place_id = geo.get("place_id")
    else:
        lat = pin.get("lat")
        lng = pin.get("lng")
        address = pin.get("address")

    # Hours
    raw_hours = None
    if place_id or name:
        raw_hours = planner_serp.place_hours(name, city, place_id)

    opening_hours = parse_raw_hours(raw_hours)
    hours_verified = bool(opening_hours["days"])

    # --- LLM call for category, dwell, booking, tip ---
    user_prompt = (
        f"Place name: {name}\n"
        f"City: {city}\n"
        f"Address: {address or 'unknown'}\n"
        f"Category hint: {pin.get('category', 'unknown')}\n\n"
        "Return JSON with keys: category (string), dwell_minutes (integer), "
        "booking_required (boolean), tip (string)."
    )

    category = SCOUT_DEFAULTS["category"]
    dwell_minutes = SCOUT_DEFAULTS["dwell_minutes"]
    booking_required = SCOUT_DEFAULTS["booking_required"]
    tip = SCOUT_DEFAULTS["tip"]
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
        "opening_hours": opening_hours,
        "hours_verified": hours_verified,
        "category": category,
        "dwell_minutes": dwell_minutes,
        "booking_required": booking_required,
        "tip": tip,
    }
    return result, llm_error


def run_scout(ctx: dict) -> dict:
    """Scout node: fan out per pin, merge SerpApi + LLM into PlaceResearch list."""
    city = ctx.get("destination", "")
    pins = ctx.get("pins", [])
    resolved_pins = [p for p in pins if p.get("resolved", False)]

    research: list[dict] = []
    hours_verified_count = 0
    errors = ctx.setdefault("errors", [])

    for pin in resolved_pins:
        try:
            result, llm_error = _scout_one_pin(pin, city)
            research.append(result)
            if result["hours_verified"]:
                hours_verified_count += 1
            if llm_error:
                errors.append(
                    f"scout: LLM fallback used for {pin.get('name', 'unknown')}"
                )
        except Exception as e:
            errors.append(f"scout: pin '{pin.get('name', 'unknown')}' failed: {e}")
            research.append({
                "pin_id": pin.get("pin_id", ""),
                "name": pin.get("name", "Unknown"),
                "neighborhood": "",
                "lat": pin.get("lat"),
                "lng": pin.get("lng"),
                "rating": None,
                "opening_hours": {"days": {}},
                "hours_verified": False,
                "category": SCOUT_DEFAULTS["category"],
                "dwell_minutes": SCOUT_DEFAULTS["dwell_minutes"],
                "booking_required": SCOUT_DEFAULTS["booking_required"],
                "tip": SCOUT_DEFAULTS["tip"],
            })

    ctx["research"] = research
    n = len(research)
    return {
        "researched": n,
        "hours_verified": hours_verified_count,
        "hours_unverified": n - hours_verified_count,
    }


# ==============================================================================
# run_critic — audit node (LLM + deterministic pre-check)
# ==============================================================================
CRITIC_SCHEMA = {
    "verdict": str,
    "issues": list,
}


def _deterministic_critic_checks(ctx: dict) -> list[dict]:
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


def _build_critic_prompt(ctx: dict) -> str:
    """Build a compact prompt for the Critic LLM."""
    schedule = ctx.get("schedule", {})
    research = ctx.get("research", [])
    lines: list[str] = []

    lines.append(f"Destination: {ctx.get('destination', 'unknown')}")
    lines.append(f"Start date: {ctx.get('start_date', 'unknown')}")
    lines.append(f"Num days: {ctx.get('num_days', 0)}")
    lines.append("")

    lines.append("Researched places:")
    for r in research:
        if not isinstance(r, dict):
            continue
        lines.append(
            f"  - {r.get('name', '?')}: category={r.get('category', '?')}, "
            f"dwell={r.get('dwell_minutes', '?')}min, "
            f"hours_verified={r.get('hours_verified', False)}"
        )
    lines.append("")

    lines.append("Schedule:")
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
            lines.append(f"    {s}-{e} [{kind}] {name}")
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

    lines.append("")
    lines.append(
        "Check for: (a) unrealistic tightness — back-to-back stops without "
        "slack, long legs after late stops; (b) missing meals on days spanning "
        ">= 6 hours; (c) stops outside opening hours or on closed days; "
        "(d) empty or lopsided days. "
        'Return JSON: {"verdict": "PASS"} or '
        '{"verdict": "ISSUES", "issues": [{"day_index": int, '
        '"severity": "high|medium|low", "message": str}]}'
    )

    return "\n".join(lines)


def run_critic(ctx: dict) -> dict:
    """Critic node: LLM audit + deterministic pre-check (belt and braces)."""
    errors = ctx.setdefault("errors", [])

    # Deterministic pre-check
    det_issues = _deterministic_critic_checks(ctx)

    # LLM audit
    output: dict[str, Any]
    try:
        data = call_openrouter_json(
            _build_critic_prompt(ctx),
            CRITIC_SYSTEM,
            schema=CRITIC_SCHEMA,
            model=CRITIC_MODEL,
        )
        if isinstance(data, dict) and data.get("verdict") in ("PASS", "ISSUES"):
            output = data
            if data.get("verdict") == "ISSUES":
                llm_issues = data.get("issues", [])
                if not isinstance(llm_issues, list):
                    llm_issues = []
                output["issues"] = det_issues + [
                    i for i in llm_issues if isinstance(i, dict)
                ]
            else:
                # LLM says PASS but deterministic found issues
                output["issues"] = det_issues
                if det_issues:
                    output["verdict"] = "ISSUES"
        else:
            output = {"verdict": "PASS", "issues": det_issues}
            if det_issues:
                output["verdict"] = "ISSUES"
    except (ValueError, Exception) as e:
        output = {
            "verdict": "PASS",
            "critic_error": str(e),
            "issues": det_issues,
        }
        if det_issues:
            output["verdict"] = "ISSUES"
        errors.append(
            "critic: LLM failed, defaulting PASS — do not silently hide; "
            "the trace shows it"
        )

    ctx["critic"] = output
    return output


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
