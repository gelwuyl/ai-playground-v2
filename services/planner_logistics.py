"""Mr. Bounce — Logistics node: leg-time computation with Postgres cache.

Walk vs transit vs drive decision (settled in HANDOFF.md):
  1. If walking the leg takes <= WALK_MAX_MINUTES (20 min), walk — skip
     transit/drive SerpApi calls to save quota.
  2. Otherwise, fetch both transit and driving durations; pick the practical
     winner (shorter minutes; ties prefer transit — city transit avoids
     parking pain). If only one is available, use that.
  3. If all modes are None but coords exist, fall back to haversine estimates
     (estimated=True). If coords are also missing, return a 45-min penalty
     leg with estimated=True so the scheduler can still place the stop.

Cache contract (planner_leg_cache):
  - Keyed by leg_cache_key(a, b) — an unordered normalized pair, so A->B and
    B->A share one row. Drive times are treated as symmetric in v1.
  - Each row stores per-mode minutes (walk/transit/drive) + distance_km +
    estimated flag. NULL for a mode means "not fetched" (honest cache).
  - A row with all three per-mode values NULL or missing a freshly-needed mode
    is a partial miss — we refill from SerpApi/haversine and update the row.
  - Cache reads/writes are wrapped in try/except so logistics works cache-less
    when the DB is unreachable (graceful degradation).

No LLM is used anywhere in this module. Real durations come from
services.planner_serp.directions (SerpApi google_directions); when SerpApi
is unavailable or returns None, we fall back to planner_types.estimated_minutes
from haversine distance and mark estimated=True.
"""
import os
import socket
from urllib.parse import urlparse

import services.planner_types as pt

# Observability counter: incremented on each real (non-fixture) SerpApi call.
_serp_calls_made = 0
_serp_errors: list[str] = []   # deduped reasons for failed directions calls

# DB availability flag: checked once; if False, skip all cache I/O.
_db_checked = False
_db_ok = False


def _db_available() -> bool:
    """Quick socket probe so we never block on an unreachable DB pool.

    The ConnectionPool in services.database retries indefinitely in the
    background; calling get_conn() with a bad URL would hang. We probe the
    host/port with a 2-second socket timeout once, then cache the result.
    """
    global _db_checked, _db_ok
    if _db_checked:
        return _db_ok
    _db_checked = True
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        return False
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        sock = socket.create_connection((host, port), timeout=2)
        sock.close()
        _db_ok = True
    except Exception:
        _db_ok = False
        print(f"[planner_logistics] DB unreachable at {host}:{port} — running cache-less")
    return _db_ok


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------
def choose_mode(walk: float | None, transit: float | None, drive: float | None) -> tuple[str, float, bool]:
    """Apply the walk-vs-transit-vs-drive decision to per-mode minutes.

    Returns (chosen_mode, chosen_minutes, estimated).
    Walk if walk is not None and walk <= WALK_MAX_MINUTES.
    Otherwise pick the shorter of transit/drive (tie -> transit).
    If transit and drive are both None but walk is available, use walk.
    If everything is None, return a 45-min penalty drive leg (estimated).
    """
    if walk is not None and walk <= pt.WALK_MAX_MINUTES:
        return "walk", walk, False

    candidates = []
    if transit is not None:
        candidates.append(("transit", transit))
    if drive is not None:
        candidates.append(("drive", drive))

    if candidates:
        # Sort by minutes; tie -> transit (city transit preferred over parking).
        candidates.sort(key=lambda c: (c[1], 0 if c[0] == "transit" else 1))
        mode, minutes = candidates[0]
        return mode, minutes, False

    if walk is not None:
        # Walk is the only thing we have — use it even though it's long.
        return "walk", walk, False

    # Everything is None — penalty leg.
    return "drive", 45.0, True


# ---------------------------------------------------------------------------
# Degenerate pair (same normalized name)
# ---------------------------------------------------------------------------
def _degenerate_leg(a: str, b: str) -> dict:
    """Two pins with the same normalized name -> zero-minute walk leg."""
    return {
        "from_name": a,
        "to_name": b,
        "walk_minutes": 0.0,
        "transit_minutes": None,
        "drive_minutes": None,
        "distance_km": 0.0,
        "estimated": False,
        "chosen_mode": "walk",
        "chosen_minutes": 0.0,
    }


# ---------------------------------------------------------------------------
# SerpApi fetch (lazy import so the module imports without planner_serp)
# ---------------------------------------------------------------------------
def _serp_directions(start: str, end: str, mode: str, city: str) -> dict | None:
    """Call planner_serp.directions, incrementing the real-call counter.

    Returns {"distance_km": float|None, "minutes": float|None} | None.
    On any exception, prints a warning and returns None (caller falls back to
    haversine estimate).
    """
    global _serp_calls_made
    try:
        from services import planner_serp
        result = planner_serp.directions(start=start, end=end, mode=mode, city=city)
        # Only count real (non-fixture) calls.
        if not getattr(planner_serp, "USE_FIXTURES", False):
            _serp_calls_made += 1
        if result is None:
            reason = getattr(planner_serp, "LAST_ERROR", None)
            if reason and reason not in _serp_errors:
                _serp_errors.append(reason)
        return result
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if msg not in _serp_errors:
            _serp_errors.append(msg)
        print(f"[planner_logistics] SerpApi directions failed ({start} -> {end}, {mode}): {msg}")
        return None


# ---------------------------------------------------------------------------
# Haversine fallbacks
# ---------------------------------------------------------------------------
def _haversine_leg(mode: str, lat_a, lng_a, lat_b, lng_b) -> tuple[float | None, float | None, bool]:
    """Compute estimated minutes + distance from haversine.

    Returns (minutes, distance_km, estimated).
    If coords are missing, returns (None, None, False).
    """
    if None in (lat_a, lng_a, lat_b, lng_b):
        return None, None, False
    km = pt.haversine_km(lat_a, lng_a, lat_b, lng_b)
    minutes = pt.estimated_minutes(mode, km)
    return minutes, km, True


# ---------------------------------------------------------------------------
# Cache read/write (lazy DB, wrapped in try/except)
# ---------------------------------------------------------------------------
def _cache_read(cache_key: str) -> dict | None:
    """Read a leg row from planner_leg_cache. Returns None on miss or DB error."""
    if not _db_available():
        return None
    try:
        from services.database import get_conn
        from services.planner_db import _ensure_tables
        _ensure_tables()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT walk_minutes, transit_minutes, drive_minutes, "
                    "distance_km, estimated FROM planner_leg_cache WHERE cache_key = %s",
                    (cache_key,),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "walk_minutes": row[0],
                        "transit_minutes": row[1],
                        "drive_minutes": row[2],
                        "distance_km": row[3],
                        "estimated": row[4],
                    }
    except Exception as e:
        print(f"[planner_logistics] Cache read failed for {cache_key}: {type(e).__name__}: {e}")
    return None


def _cache_write(cache_key: str, from_name: str, to_name: str,
                 walk_minutes, transit_minutes, drive_minutes,
                 distance_km, estimated: bool) -> None:
    """Upsert a leg row into planner_leg_cache. Swallows DB errors.

    Estimated (haversine-fallback) rows are NOT cached - caching them would
    permanently mask a directions failure behind a fake-precision number.
    """
    if estimated:
        return
    if not _db_available():
        return
    try:
        from services.database import get_conn
        from services.planner_db import _ensure_tables
        _ensure_tables()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO planner_leg_cache "
                    "(cache_key, from_name, to_name, walk_minutes, transit_minutes, "
                    " drive_minutes, distance_km, estimated) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (cache_key) DO UPDATE SET "
                    " from_name = EXCLUDED.from_name,"
                    " to_name = EXCLUDED.to_name,"
                    " walk_minutes = EXCLUDED.walk_minutes,"
                    " transit_minutes = EXCLUDED.transit_minutes,"
                    " drive_minutes = EXCLUDED.drive_minutes,"
                    " distance_km = EXCLUDED.distance_km,"
                    " estimated = EXCLUDED.estimated",
                    (cache_key, from_name, to_name, walk_minutes,
                     transit_minutes, drive_minutes, distance_km, estimated),
                )
            conn.commit()
    except Exception as e:
        print(f"[planner_logistics] Cache write failed for {cache_key}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Core: get_leg
# ---------------------------------------------------------------------------
def get_leg(a: str, b: str, city: str,
            lat_a=None, lng_a=None, lat_b=None, lng_b=None,
            use_cache: bool = True) -> dict:
    """Compute one Leg dict for the unordered pair (a, b).

    a and b are the display names in the caller's deterministic order
    (pin i -> pin j where i < j). The cache key is leg_cache_key(a, b)
    which normalizes and sorts, so A->B and B->A share one row.
    """
    # Degenerate pair: same normalized name.
    if pt.normalize_place_name(a) == pt.normalize_place_name(b):
        return _degenerate_leg(a, b)

    cache_key = pt.leg_cache_key(a, b)

    # Try cache.
    cached = None
    if use_cache:
        cached = _cache_read(cache_key)
        # Estimated rows are haversine fallbacks from a failed live fetch;
        # treat them as a miss so live directions are retried (self-healing).
        if cached and cached.get("estimated", False):
            cached = None

    walk_minutes = None
    transit_minutes = None
    drive_minutes = None
    distance_km = None
    estimated = False

    if cached:
        walk_minutes = cached.get("walk_minutes")
        transit_minutes = cached.get("transit_minutes")
        drive_minutes = cached.get("drive_minutes")
        distance_km = cached.get("distance_km")
        estimated = cached.get("estimated", False)

        # Partial miss: all three None or the row predates a mode we need.
        # If walk is present and <= 20, we can use the cache as-is (no refill).
        # Otherwise we need transit + drive; if either is NULL, refill.
        if walk_minutes is not None and walk_minutes <= pt.WALK_MAX_MINUTES:
            # Cache is complete enough — walk is the chosen mode.
            chosen_mode, chosen_minutes, est_flag = choose_mode(
                walk_minutes, transit_minutes, drive_minutes
            )
            return {
                "from_name": a,
                "to_name": b,
                "walk_minutes": walk_minutes,
                "transit_minutes": transit_minutes,
                "drive_minutes": drive_minutes,
                "distance_km": distance_km,
                "estimated": estimated or est_flag,
                "chosen_mode": chosen_mode,
                "chosen_minutes": chosen_minutes,
            }
        # Walk is None or > 20: we need transit + drive. If both present, use cache.
        if transit_minutes is not None and drive_minutes is not None:
            chosen_mode, chosen_minutes, est_flag = choose_mode(
                walk_minutes, transit_minutes, drive_minutes
            )
            return {
                "from_name": a,
                "to_name": b,
                "walk_minutes": walk_minutes,
                "transit_minutes": transit_minutes,
                "drive_minutes": drive_minutes,
                "distance_km": distance_km,
                "estimated": estimated,
                "chosen_mode": chosen_mode,
                "chosen_minutes": chosen_minutes,
            }
        # Partial miss — fall through to real fetch and refill.

    # ---- Fetch real durations from SerpApi ----

    # Walk
    walk_result = _serp_directions(a, b, "walking", city)
    if walk_result and walk_result.get("minutes") is not None:
        walk_minutes = walk_result["minutes"]
        if walk_result.get("distance_km") is not None:
            distance_km = walk_result["distance_km"]
    else:
        # Fallback to haversine if coords present.
        w_min, w_km, w_est = _haversine_leg("walk", lat_a, lng_a, lat_b, lng_b)
        if w_min is not None:
            walk_minutes = w_min
            if distance_km is None:
                distance_km = w_km
            estimated = True  # haversine fallback was used for walk

    # If walk is feasible (<= 20), skip transit/drive (saves quota).
    if walk_minutes is not None and walk_minutes <= pt.WALK_MAX_MINUTES:
        chosen_mode, chosen_minutes, est_flag = choose_mode(
            walk_minutes, None, None
        )
        # est_flag is False here because walk is real; but if walk was
        # haversine-estimated, estimated is already True.
        _cache_write(cache_key, a, b, walk_minutes, None, None,
                     distance_km, estimated)
        return {
            "from_name": a,
            "to_name": b,
            "walk_minutes": walk_minutes,
            "transit_minutes": None,
            "drive_minutes": None,
            "distance_km": distance_km,
            "estimated": estimated,
            "chosen_mode": "walk",
            "chosen_minutes": walk_minutes,
        }

    # Walk is None or > 20: fetch transit + driving.
    transit_result = _serp_directions(a, b, "transit", city)
    if transit_result and transit_result.get("minutes") is not None:
        transit_minutes = transit_result["minutes"]
        if transit_result.get("distance_km") is not None:
            distance_km = transit_result["distance_km"]
    else:
        t_min, t_km, t_est = _haversine_leg("transit", lat_a, lng_a, lat_b, lng_b)
        if t_min is not None:
            transit_minutes = t_min
            if distance_km is None:
                distance_km = t_km
            estimated = True

    drive_result = _serp_directions(a, b, "driving", city)
    if drive_result and drive_result.get("minutes") is not None:
        drive_minutes = drive_result["minutes"]
        if drive_result.get("distance_km") is not None:
            distance_km = drive_result["distance_km"]
    else:
        d_min, d_km, d_est = _haversine_leg("drive", lat_a, lng_a, lat_b, lng_b)
        if d_min is not None:
            drive_minutes = d_min
            if distance_km is None:
                distance_km = d_km
            estimated = True

    # If we still have no distance_km, try haversine as last resort.
    if distance_km is None and None not in (lat_a, lng_a, lat_b, lng_b):
        distance_km = pt.haversine_km(lat_a, lng_a, lat_b, lng_b)

    chosen_mode, chosen_minutes, est_flag = choose_mode(
        walk_minutes, transit_minutes, drive_minutes
    )
    # est_flag is True only for the all-None penalty case; for real values
    # it's False. But our `estimated` flag is True if ANY value was haversine.
    final_estimated = estimated or est_flag

    _cache_write(cache_key, a, b, walk_minutes, transit_minutes, drive_minutes,
                 distance_km, final_estimated)

    return {
        "from_name": a,
        "to_name": b,
        "walk_minutes": walk_minutes,
        "transit_minutes": transit_minutes,
        "drive_minutes": drive_minutes,
        "distance_km": distance_km,
        "estimated": final_estimated,
        "chosen_mode": chosen_mode,
        "chosen_minutes": chosen_minutes,
    }


# ---------------------------------------------------------------------------
# Node entry point: compute_legs
# ---------------------------------------------------------------------------
def compute_legs(ctx: dict) -> list[dict]:
    """Compute leg times for every unordered pair of resolved pins.

    ctx has: pins (list of pin dicts), destination (city name), session_id.
    Returns a list of Leg dicts. Never raises for per-leg issues.
    """
    pins = ctx.get("pins", [])
    city = ctx.get("destination", "")

    # Filter to resolved pins with usable names; dedupe by normalized name.
    seen = {}
    usable = []
    for pin in pins:
        if not pin.get("resolved"):
            continue
        name = pin.get("name", "").strip()
        if not name:
            continue
        norm = pt.normalize_place_name(name)
        if norm in seen:
            continue
        seen[norm] = True
        usable.append(pin)

    # Sort by seq for deterministic ordering.
    usable.sort(key=lambda p: p.get("seq", 0))

    legs = []
    n = len(usable)
    for i in range(n):
        for j in range(i + 1, n):
            pin_i = usable[i]
            pin_j = usable[j]
            a = pin_i["name"]
            b = pin_j["name"]
            try:
                leg = get_leg(
                    a, b, city,
                    lat_a=pin_i.get("lat"), lng_a=pin_i.get("lng"),
                    lat_b=pin_j.get("lat"), lng_b=pin_j.get("lng"),
                    use_cache=True,
                )
                legs.append(leg)
            except Exception as e:
                print(f"[planner_logistics] Per-leg error for ({a}, {b}): {type(e).__name__}: {e}")
                # Fallback: haversine estimate if coords exist, else 45-min penalty.
                lat_a, lng_a = pin_i.get("lat"), pin_i.get("lng")
                lat_b, lng_b = pin_j.get("lat"), pin_j.get("lng")
                if None not in (lat_a, lng_a, lat_b, lng_b):
                    km = pt.haversine_km(lat_a, lng_a, lat_b, lng_b)
                    est_min = pt.estimated_minutes("drive", km)
                    legs.append({
                        "from_name": a,
                        "to_name": b,
                        "walk_minutes": None,
                        "transit_minutes": None,
                        "drive_minutes": est_min,
                        "distance_km": km,
                        "estimated": True,
                        "chosen_mode": "drive",
                        "chosen_minutes": est_min,
                    })
                else:
                    legs.append({
                        "from_name": a,
                        "to_name": b,
                        "walk_minutes": None,
                        "transit_minutes": None,
                        "drive_minutes": 45.0,
                        "distance_km": None,
                        "estimated": True,
                        "chosen_mode": "drive",
                        "chosen_minutes": 45.0,
                    })

    return legs


# ---------------------------------------------------------------------------
# Node wrapper: run_logistics
# ---------------------------------------------------------------------------
def run_logistics(ctx: dict) -> dict:
    """Logistics node entry point for the graph runner.

    Computes all-pairs leg times, stores them in ctx["legs"], and returns
    a summary dict with counts. Never raises.
    """
    legs = compute_legs(ctx)
    ctx["legs"] = legs
    estimated_count = sum(1 for leg in legs if leg.get("estimated"))
    return {
        "legs_count": len(legs),
        "serp_calls": _serp_calls_made,
        "estimated_legs": estimated_count,
        "serp_errors": list(_serp_errors),
    }
