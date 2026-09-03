"""OSRM directions client — free, no API key.

Only used for driving and walking modes (OSRM has no transit). The public
demo server (router.project-osrm.org) has a fair-use policy; the leg cache
keeps our call count low (one request per unordered pin pair per mode).
Degrades gracefully: returns None on any failure, never raises. Error text
is kept free of secrets (there are none — no key).
"""
import httpx

OSRM_BASE_URL = "https://router.project-osrm.org/route/v1"
_TIMEOUT = 20

# profile per travel mode
_PROFILES = {"driving": "driving", "walking": "foot"}


def osrm_directions(lat_a, lng_a, lat_b, lng_b, mode: str) -> dict | None:
    """Fetch one route via OSRM.

    Returns {"minutes": float, "distance_km": float} or None on any failure.
    """
    profile = _PROFILES.get(mode)
    if profile is None:
        return None
    if None in (lat_a, lng_a, lat_b, lng_b):
        return None
    url = f"{OSRM_BASE_URL}/{profile}/{lng_a},{lat_a};{lng_b},{lat_b}"
    try:
        resp = httpx.get(
            url,
            params={"overview": "false", "alternatives": "false", "steps": "false"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        routes = data.get("routes") or []
        if not routes:
            return None
        route = routes[0]
        duration_s = route.get("duration")
        distance_m = route.get("distance")
        if duration_s is None or distance_m is None:
            return None
        return {
            "minutes": float(duration_s) / 60.0,
            "distance_km": float(distance_m) / 1000.0,
        }
    except Exception:
        return None
