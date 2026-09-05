"""Free geocoding + hours sources: Photon (primary), Nominatim, Overpass.

No API keys, no quota. All functions degrade gracefully (None on failure),
never raise, and never print secret values (there are none — keyless APIs).
Photon needs a User-Agent + Referer to avoid 403s from its nginx.
"""
import re

import httpx

_UA = {"User-Agent": "MrBounceTripPlanner/1.0 (personal project)",
       "Referer": "https://mrbounce.example"}
_TIMEOUT = 20

# ---------------------------------------------------------------------------
# Photon — POI-grade forward geocoding (free, keyless)
# ---------------------------------------------------------------------------
def photon_geocode(name: str, city: str) -> dict | None:
    """Geocode "name + city" via Photon (OSM POI index).

    Returns {"name", "lat", "lng", "address"} (address None) or None.
    Only accepts results that carry a real place name — coordinate dumps
    and street-only matches are rejected so callers can fall through.
    """
    q = f"{name} {city}".strip()
    if not q:
        return None
    try:
        import httpx
        resp = httpx.get("https://photon.komoot.io/api/",
                         params={"q": q, "limit": 1}, headers=_UA, timeout=_TIMEOUT)
        resp.raise_for_status()
        features = (resp.json() or {}).get("features") or []
        if not features:
            return None
        f = features[0]
        props = f.get("properties") or {}
        coords = (f.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            return None
        osm_name = props.get("name")
        if not osm_name:
            # Street/city-only match: not a POI hit — let the caller fall through.
            return None
        return {
            "name": osm_name,
            "lat": float(coords[1]),
            "lng": float(coords[0]),
            "address": props.get("street") and f"{props.get('street')}, {props.get('city') or ''}".strip(", ") or None,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Nominatim — forward (second fallback) + reverse (naming)
# ---------------------------------------------------------------------------
def nominatim_geocode(name: str, city: str) -> dict | None:
    """Forward geocode via Nominatim. Same return shape as photon_geocode."""
    q = f"{name} {city}".strip()
    if not q:
        return None
    try:
        import httpx
        resp = httpx.get("https://nominatim.openstreetmap.org/search",
                         params={"q": q, "format": "json", "limit": 1},
                         headers=_UA, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() or []
        if not data:
            return None
        first = data[0]
        return {
            "name": (first.get("display_name") or "").split(",")[0] or name,
            "lat": float(first["lat"]),
            "lng": float(first["lon"]),
            "address": first.get("display_name"),
        }
    except Exception:
        return None


def nominatim_reverse(lat: float, lng: float) -> dict | None:
    """Reverse geocode via Nominatim -> {"name", "address"} or None."""
    try:
        import httpx
        resp = httpx.get("https://nominatim.openstreetmap.org/reverse",
                         params={"lat": lat, "lon": lng, "format": "json", "zoom": 18},
                         headers=_UA, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        display = data.get("display_name")
        if not display:
            return None
        addr = data.get("address") or {}
        short = (addr.get("amenity") or addr.get("shop") or addr.get("road")
                 or addr.get("suburb") or addr.get("city") or "")
        return {
            "name": f"{short} ({lat:.4f}, {lng:.4f})" if short else f"Pin @ {lat:.4f}, {lng:.4f}",
            "address": display,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Overpass — best-effort opening_hours from OSM
# ---------------------------------------------------------------------------
def overpass_hours(lat: float, lng: float, name: str) -> dict | None:
    """Best-effort opening hours from OSM via Overpass.

    Searches nodes/ways within 100 m whose name fuzzy-matches *name*.
    Returns {"hours": str, "source": "osm"} or None. Never raises.
    """
    if None in (lat, lng) or not name:
        return None
    esc = re.sub(r"[^\w\s&\-']", "", name).strip()
    if not esc:
        return None
    q = (f'[out:json][timeout:10];'
         f'nwr(around:100,{lat},{lng})["name"~"{re.escape(esc)}",i];'
         f'out tags 3;')
    try:
        import httpx
        resp = httpx.post("https://overpass-api.de/api/interpreter",
                          data={"data": q}, headers=_UA, timeout=30)
        resp.raise_for_status()
        elements = (resp.json() or {}).get("elements") or []
        for el in elements:
            tags = el.get("tags") or {}
            hours = tags.get("opening_hours")
            if hours:
                return {"hours": hours, "source": "osm"}
        return None
    except Exception:
        return None
