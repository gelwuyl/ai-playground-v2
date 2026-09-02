"""Mr. Bounce — thin SerpApi client for Google Maps geocoding, hours, and directions.

No DB imports, no LLM imports. Honors USE_FIXTURES=1 (repo convention).
Degrades gracefully: returns None on any failure (missing key, network error,
unexpected response shape). Never raises.
"""
import os
import httpx

from services.planner_types import normalize_place_name, leg_cache_key

USE_FIXTURES = os.environ.get("USE_FIXTURES", "0") == "1"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Reason for the most recent failure on this worker (reset at each call).
# Runtime logs are not exposed on the Hobby plan, so callers surface this
# through result payloads instead of relying on stdout.
LAST_ERROR: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _serp_key() -> str | None:
    """Read SERPAPI_KEY lazily from the environment at call time."""
    return os.environ.get("SERPAPI_KEY")


def _fixture_lookup(name: str) -> dict | None:
    """Find a fixture place by normalized name; return a copy or None."""
    from services import planner_fixtures
    norm = normalize_place_name(name)
    for key, entry in planner_fixtures.FIXTURE_PLACES.items():
        if normalize_place_name(key) == norm:
            return dict(entry)
    return None


# Maps-engine searches without a location bias (ll) can return empty
# local_results for far-away cities. Cache one GPS origin per city.
_CITY_LL: dict[str, str] = {}


def _city_ll(city: str) -> str | None:
    key = (city or "").strip().lower()
    if not key:
        return None
    if key in _CITY_LL:
        return _CITY_LL[key] or None
    k = _serp_key()
    if not k:
        return None
    try:
        resp = httpx.get(
            "https://serpapi.com/search.json",
            params={"engine": "google_maps", "type": "search", "q": city, "api_key": k, "hl": "en"},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            _CITY_LL[key] = ""
            return None
        results = data.get("local_results") or data.get("places") or []
        gps = (results[0].get("gps_coordinates") if results else None) or {}
        lat, lng = gps.get("latitude"), gps.get("longitude")
        ll = f"@{lat},{lng},13z" if lat is not None and lng is not None else ""
        _CITY_LL[key] = ll
        return ll or None
    except Exception:
        _CITY_LL[key] = ""
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def geocode_place(name: str, city: str) -> dict | None:
    """Geocode a place via SerpApi Google Maps engine. Returns None on any failure.

    Returns dict with keys: name, lat, lng, address, place_id, raw_hours.
    """
    global LAST_ERROR
    LAST_ERROR = None
    if USE_FIXTURES:
        entry = _fixture_lookup(name)
        if entry is None:
            return None
        return {
            "name": entry["name"],
            "lat": entry.get("lat"),
            "lng": entry.get("lng"),
            "address": entry.get("address"),
            "place_id": entry.get("place_id"),
            "raw_hours": entry.get("raw_hours"),
        }

    key = _serp_key()
    if not key:
        print("geocode_place: SERPAPI_KEY not set")
        LAST_ERROR = "SERPAPI_KEY not set"
        return None

    try:
        base_params = {
            "engine": "google_maps",
            "type": "search",
            "q": f"{name} {city}",
            "api_key": key,
            "hl": "en",
        }
        data = None
        results: list = []
        for ll in (None, _city_ll(city)):
            params = dict(base_params)
            if ll:
                params["ll"] = ll
            resp = httpx.get(
                "https://serpapi.com/search.json",
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("error"):
                LAST_ERROR = str(data["error"])
                return None
            results = data.get("local_results") or data.get("places") or []
            if results:
                break
        if not results:
            LAST_ERROR = f"no results for {name!r}; response keys: {sorted(data.keys())}"
            return None
        r = results[0]
        gps = r.get("gps_coordinates") or {}
        return {
            "name": r.get("title"),
            "lat": gps.get("latitude"),
            "lng": gps.get("longitude"),
            "address": r.get("address"),
            "place_id": r.get("place_id"),
            "raw_hours": r.get("operating_hours") or r.get("hours"),
        }
    except Exception as e:
        print(f"geocode_place failed: {type(e).__name__}: {e}")
        LAST_ERROR = f"geocode {type(e).__name__}: {e}"
        return None


def place_hours(name: str, city: str, place_id: str | None) -> dict | None:
    """Fetch opening hours for a place via SerpApi. Returns None on any failure.

    Tries the 'place' engine with place_id first; falls back to a 'search' query.
    Returns the raw hours passthrough (whatever shape SerpApi returns) or None.
    """
    global LAST_ERROR
    LAST_ERROR = None
    if USE_FIXTURES:
        entry = _fixture_lookup(name)
        if entry is None:
            return None
        return entry.get("raw_hours")

    key = _serp_key()
    if not key:
        print("place_hours: SERPAPI_KEY not set")
        LAST_ERROR = "SERPAPI_KEY not set"
        return None

    try:
        # Try the place engine first if we have a place_id
        if place_id:
            resp = httpx.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google_maps",
                    "type": "place",
                    "place_id": place_id,
                    "api_key": key,
                    "hl": "en",
                },
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("local_results", [])
            if results:
                r = results[0]
                hours = r.get("operating_hours") or r.get("hours")
                if hours is not None:
                    return hours

        # Fall back to the same search query as geocode_place
        resp = httpx.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_maps",
                "type": "search",
                "q": f"{name} {city}",
                "api_key": key,
                "hl": "en",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            LAST_ERROR = str(data["error"])
            return None
        results = data.get("local_results", [])
        if not results:
            LAST_ERROR = f"no local_results for {name!r}"
            return None
        r = results[0]
        return r.get("operating_hours") or r.get("hours")
    except Exception as e:
        print(f"place_hours failed: {type(e).__name__}: {e}")
        LAST_ERROR = f"hours {type(e).__name__}: {e}"
        return None


def directions(start: str, end: str, mode: str, city: str) -> dict | None:
    """Fetch directions between two places via SerpApi Google Directions engine.

    mode is one of "walking", "transit", "driving".
    Returns dict with "distance_km" (float|None) and "minutes" (float|None),
    or None on any failure.
    """
    global LAST_ERROR
    LAST_ERROR = None
    if USE_FIXTURES:
        from services import planner_fixtures
        ck = f"{leg_cache_key(start, end)}:{mode}"
        entry = planner_fixtures.FIXTURE_DIRECTIONS.get(ck)
        if entry is None:
            return None
        return {"distance_km": entry["distance_km"], "minutes": entry["minutes"]}

    key = _serp_key()
    if not key:
        print("directions: SERPAPI_KEY not set")
        LAST_ERROR = "SERPAPI_KEY not set"
        return None

    try:
        resp = httpx.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_directions",
                "start_addr": f"{start}, {city}",
                "end_addr": f"{end}, {city}",
                "mode": mode,
                "api_key": key,
                "hl": "en",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            LAST_ERROR = str(data["error"])
            return None
        directions_list = data.get("directions")
        if not directions_list or not isinstance(directions_list, list):
            LAST_ERROR = f"no directions for {mode} leg"
            return None
        el = directions_list[0]

        # distance_km: try fixed_distance_km, then distance_m/1000, then distance_km
        distance_km = el.get("fixed_distance_km")
        if distance_km is None:
            dm = el.get("distance_m")
            if dm:
                distance_km = dm / 1000.0
        if distance_km is None:
            distance_km = el.get("distance_km")

        # minutes: try fixed_duration_min, duration_min, duration_minutes, duration_s/60
        minutes = el.get("fixed_duration_min")
        if minutes is None:
            minutes = el.get("duration_min")
        if minutes is None:
            minutes = el.get("duration_minutes")
        if minutes is None:
            ds = el.get("duration_s")
            if ds:
                minutes = ds / 60.0

        return {
            "distance_km": float(distance_km) if distance_km is not None else None,
            "minutes": float(minutes) if minutes is not None else None,
        }
    except Exception as e:
        print(f"directions failed: {type(e).__name__}: {e}")
        LAST_ERROR = f"directions {type(e).__name__}: {e}"
        return None
