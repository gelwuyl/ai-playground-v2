"""Mr. Bounce -- Ingest tool node.

First node of the 7-node crew. Takes the user's pin inputs (one Google Maps
short link per place OR pasted-text place names) and resolves them into Pin
dicts persisted to planner_pins.

Pure parsing is separated from network resolution and DB persistence so the
core logic stays deterministic and unit-testable without any external service.
"""
import os
import re
import sys
import uuid
import urllib.parse
from typing import Any

import httpx

# Lazy imports: database and planner_db are only needed inside save_pins,
# so the module stays importable (and unit-testable) without DATABASE_URL.
# from services.database import get_conn          # imported in save_pins
# from services.planner_db import _ensure_tables   # imported in save_pins

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# Google now serves an interstitial HTML page (no redirect) to desktop
# agents; a mobile UA still gets the classic 302 to the full maps URL.
SHORT_LINK_USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
)


def _use_fixtures() -> bool:
    """Check the USE_FIXTURES flag from planner_serp if importable, else env.

    Uses sys.modules.get so that monkeypatched stub modules are respected
    in tests (``from services import planner_serp`` would bypass the stub
    when the real module is already cached as a package attribute).
    """
    mod = sys.modules.get("services.planner_serp")
    if mod is not None:
        if getattr(mod, "USE_FIXTURES", False):
            return True
    else:
        try:
            from services import planner_serp
            if getattr(planner_serp, "USE_FIXTURES", False):
                return True
        except Exception:
            pass
    return os.environ.get("USE_FIXTURES") == "1"


# ---------------------------------------------------------------------------
# Pure parsing -- parse_pin_inputs
# ---------------------------------------------------------------------------
def parse_pin_inputs(payload: dict) -> list[dict]:
    """Parse a user payload dict into an ordered list of pin specs.

    Each spec is {"seq": int, "source": "short_link" | "text", "raw_input": str}.
    Pure and deterministic -- no network, no side effects.

    Accepted payload shapes:
      {"pins": [{"source": "short_link", "raw_input": "<url>"}, {"source": "text", "raw_input": "<name>"}, ...]}
      {"pins": ["https://maps.app.goo.gl/abc", "Gardens by the Bay", ...]}
      {"pins": "Gardens by the Bay\\nMarina Bay Sands\\n# this is a comment\\n"}
    """
    pins_raw = payload.get("pins", [])
    items: list[Any] = []

    # If pins is a single string, split on newlines (pasted text blob).
    if isinstance(pins_raw, str):
        for line in pins_raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    elif isinstance(pins_raw, list):
        for item in pins_raw:
            if isinstance(item, str):
                stripped = item.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                items.append(stripped)
            elif isinstance(item, dict):
                raw = (item.get("raw_input") or "").strip()
                if not raw:
                    continue
                source = item.get("source")
                if source not in ("short_link", "text"):
                    # Infer from the raw_input if source is missing/invalid.
                    source = _infer_source(raw)
                items.append({"source": source, "raw_input": raw})
    # else: no pins -> empty list

    # Normalise every item to a {"seq", "source", "raw_input"} dict.
    specs: list[dict] = []
    for i, item in enumerate(items):
        if isinstance(item, dict):
            source = item["source"]
            raw = item["raw_input"]
        else:
            source = _infer_source(item)
            raw = item
        specs.append({"seq": i, "source": source, "raw_input": raw})
    return specs


def _infer_source(text: str) -> str:
    """Return 'short_link' if the text looks like a URL, else 'text'."""
    return "short_link" if text.startswith(("http://", "https://")) else "text"


# ---------------------------------------------------------------------------
# Short-link resolution
# ---------------------------------------------------------------------------
def _parse_final_maps_url(url: str) -> dict | None:
    """Parse a final Google Maps URL for name and/or coordinates.

    Returns {"name": str | None, "lat": float | None, "lng": float | None,
             "address": str | None} or None if nothing useful can be extracted.

    When coords are found but no name, returns a dict with the special
    ``_coords_only`` flag set to True so the caller can route to the
    SerpApi geocode fallback to recover a name.
    """
    # Try to find coords anywhere in the URL: @lat,lng
    coord_match = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", url)
    lat = float(coord_match.group(1)) if coord_match else None
    lng = float(coord_match.group(2)) if coord_match else None

    # Try the /place/<Name>/ segment
    name: str | None = None
    place_match = re.search(r"/place/([^/]+)", url)
    if place_match:
        name = urllib.parse.unquote_plus(place_match.group(1))

    # Try maps.google.com/?q=<...>&ll=lat,lng or ?q=lat,lng
    if lat is None or lng is None:
        ll_match = re.search(r"[?&]ll=(-?\d+\.\d+),(-?\d+\.\d+)", url)
        if ll_match:
            lat = float(ll_match.group(1))
            lng = float(ll_match.group(2))

    # If there's a ?q= parameter but no /place/ name, try to extract a name from q.
    if name is None:
        q_match = re.search(r"[?&]q=([^&]+)", url)
        if q_match:
            q_val = urllib.parse.unquote_plus(q_match.group(1))
            # If q looks like "lat,lng" don't use it as a name -- but do parse coords.
            if re.fullmatch(r"(-?\d+\.\d+),(-?\d+\.\d+)", q_val):
                if lat is None:
                    lat = float(q_val.split(",")[0])
                if lng is None:
                    lng = float(q_val.split(",")[1])
            else:
                name = q_val

    if lat is None and lng is None and name is None:
        return None
    # If we have a name AND coords, return immediately.
    # If we have coords but no name, fall through to the SerpApi geocode
    # fallback (caller handles it) -- same as name-but-no-coords below.
    if name is not None and (lat is not None or lng is not None):
        return {"name": name, "lat": lat, "lng": lng, "address": None}
    # Signal that coords were found but a name is missing so the caller can
    # try the geocode fallback; if the fallback also fails, the caller can
    # return the coords with name=None so run_ingest persists them with a
    # resolve_error instead of silently treating the URL as a name.
    if lat is not None or lng is not None:
        return {"name": None, "lat": lat, "lng": lng, "address": None,
                "_coords_only": True}
    # name only, no coords -- SerpApi fallback runs with the q= name.
    return {"name": name, "lat": None, "lng": None, "address": None}


def _geocode_fallback(final_url: str, city: str,
                      coords: dict | None = None) -> dict | None:
    """Try to recover a place name via SerpApi geocode when the URL
    had no /place/ slug or the slug-only parse failed.

    If *coords* is provided (coords-only case), and the geocode fallback
    also fails or is absent, return the coords with name=None so the
    caller can persist them with a resolve_error rather than dropping
    the pin entirely.
    """
    # Try to derive a query from the /place/ slug of the final URL.
    query = None
    place_match = re.search(r"/place/([^/]+)", final_url)
    if place_match:
        query = urllib.parse.unquote_plus(place_match.group(1)).replace("+", " ")
    # A q= name parsed from the URL (no coords) is itself a geocodable query.
    if not query and coords and coords.get("name") and coords.get("lat") is None:
        query = coords["name"]
    # If no slug, try a "lat,lng" query from the coords we already parsed.
    if not query and coords and coords.get("lat") is not None:
        query = f"{coords['lat']},{coords['lng']}"

    if query:
        serp_mod = sys.modules.get("services.planner_serp")
        if serp_mod is None:
            try:
                from services import planner_serp
                serp_mod = planner_serp
            except Exception:
                serp_mod = None
        if serp_mod is not None:
            try:
                result = serp_mod.geocode_place(query, city)
                if result:
                    return {
                        "name": result.get("name") or query,
                        "lat": result.get("lat") or (coords.get("lat") if coords else None),
                        "lng": result.get("lng") or (coords.get("lng") if coords else None),
                        "address": result.get("address"),
                    }
            except Exception:
                pass

    # Geocode failed or unavailable. If we have coords, return them with
    # name=None so run_ingest persists the pin with a resolve_error.
    if coords and (coords.get("lat") is not None or coords.get("lng") is not None):
        return {
            "name": None,
            "lat": coords.get("lat"),
            "lng": coords.get("lng"),
            "address": None,
        }
    return None


def resolve_short_link(url: str, city: str) -> dict | None:
    """Resolve a Google Maps short link (or final URL) into place info.

    Returns {"name", "lat", "lng", "address"} or None.
    Never raises -- falls back to SerpApi geocode on failure.
    """
    # --- Fixtures path ---
    if _use_fixtures():
        try:
            from services import planner_fixtures
            fixture_map = getattr(planner_fixtures, "FIXTURE_SHORT_LINKS", {})
            if url in fixture_map:
                final_url = fixture_map[url]
                parsed = _parse_final_maps_url(final_url)
                if parsed and not parsed.get("_coords_only"):
                    # Full resolution (name + coords) from the fixture URL.
                    return parsed
                # Coords-only or unparseable: try geocode fallback with the
                # /place/ slug if present, else a "lat,lng" query.
                coords = parsed if parsed else None
                return _geocode_fallback(final_url, city, coords)
        except Exception:
            pass
        # If not in fixtures and we're in fixture mode, return None.
        return None

    # --- Live path ---
    final_url = url
    try:
        r = httpx.get(
            url,
            follow_redirects=True,
            timeout=20,
            headers={"User-Agent": SHORT_LINK_USER_AGENT},
        )
        # Prefer the final URL after redirects.
        candidate_url = str(r.url)
        # Also check the Location header in case redirects didn't update r.url.
        location = r.headers.get("location")
        if location and "@" in location and "@" not in candidate_url:
            candidate_url = location
        final_url = candidate_url
    except Exception:
        # If the HTTP fetch failed entirely, try to derive a query from the
        # original URL slug for a SerpApi fallback.
        pass

    parsed = _parse_final_maps_url(final_url)
    if parsed and not parsed.get("_coords_only") and parsed.get("lat") is not None:
        # Full resolution (name + coords) from the final URL.
        return parsed
    # Name-only, coords-only, or nothing usable: try the SerpApi geocode
    # fallback (it geocodes the q= name or the coords).
    # If the fallback also fails, return coords (name=None) so run_ingest
    # persists them with a resolve_error instead of silently using the URL.
    result = _geocode_fallback(final_url, city, parsed)
    if result:
        return result
    # Nothing worked — log WHY (interstitial vs network error vs unparseable
    # link shape); the function's contract is None on failure, never raise.
    print(
        f"resolve_short_link failed: url={url[:80]!r} "
        f"final_url={final_url[:120]!r} "
        f"http_fetch={'failed' if final_url == url else 'ok'} "
        f"parse={'none' if parsed is None else 'coords_only'}"
    )
    return None


# ---------------------------------------------------------------------------
# Text resolution
# ---------------------------------------------------------------------------
def resolve_text_pin(name: str, city: str) -> dict | None:
    """Resolve a plain-text place name into coordinates via SerpApi geocode.

    Returns {"name", "lat", "lng", "address"} or None.
    Never raises.
    """
    serp_mod = sys.modules.get("services.planner_serp")
    if serp_mod is None:
        try:
            from services import planner_serp
            serp_mod = planner_serp
        except Exception:
            serp_mod = None
    if serp_mod is not None:
        try:
            result = serp_mod.geocode_place(name, city)
            if result:
                return {
                    "name": result.get("name") or name,
                    "lat": result.get("lat"),
                    "lng": result.get("lng"),
                    "address": result.get("address"),
                }
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_pins(session_id: str, pins: list[dict]) -> None:
    """INSERT pin dicts into planner_pins in one transaction.

    Thin DB-write layer -- only exercised in the E2E, not unit-tested here.
    """
    from services.database import get_conn
    from services.planner_db import _ensure_tables

    _ensure_tables()
    with get_conn() as conn:
        with conn.cursor() as cur:
            for p in pins:
                cur.execute(
                    """
                    INSERT INTO planner_pins
                        (pin_id, session_id, seq, name, source, raw_input,
                         lat, lng, address, resolved, resolve_error)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        p["pin_id"],
                        session_id,
                        p["seq"],
                        p["name"],
                        p["source"],
                        p["raw_input"],
                        p.get("lat"),
                        p.get("lng"),
                        p.get("address"),
                        p.get("resolved", False),
                        p.get("resolve_error"),
                    ),
                )
        conn.commit()


# ---------------------------------------------------------------------------
# Orchestration (what the graph node calls)
# ---------------------------------------------------------------------------
def run_ingest(ctx: dict) -> dict:
    """Run the Ingest node: parse, resolve, persist, return summary.

    ctx keys: session_id, destination (city), payload (the original user
    payload dict with a "pins" key).
    Sets ctx["pins"] = list of Pin dicts. Returns a summary dict.
    """
    session_id = ctx["session_id"]
    city = ctx.get("destination") or ctx.get("city") or ""
    payload = ctx.get("payload", {})

    specs = parse_pin_inputs(payload)
    pins: list[dict] = []
    failed: list[str] = []
    failed_details: list[dict] = []

    for spec in specs:
        pin_id = str(uuid.uuid4())
        resolved_data: dict | None = None
        resolve_error: str | None = None
        try:
            if spec["source"] == "short_link":
                resolved_data = resolve_short_link(spec["raw_input"], city)
            else:
                resolved_data = resolve_text_pin(spec["raw_input"], city)
        except Exception as exc:
            resolve_error = f"{type(exc).__name__}: {exc}"

        if resolved_data:
            resolved_name = resolved_data.get("name")
            name = resolved_name or spec["raw_input"]
            lat = resolved_data.get("lat")
            lng = resolved_data.get("lng")
            address = resolved_data.get("address")
            # A pin is fully resolved only when BOTH coords and a real name
            # came back from the resolver. When coords were found but the
            # name is None (coords-only URL), keep the coords but mark the
            # pin as unresolved with a human-readable error.
            resolved = lat is not None and resolved_name is not None
            if not resolved:
                if lat is not None and resolved_name is None:
                    resolve_error = resolve_error or "coords found but name unresolved"
                else:
                    serp_mod = sys.modules.get("services.planner_serp")
                    serp_reason = getattr(serp_mod, "LAST_ERROR", None)
                    if serp_reason:
                        resolve_error = resolve_error or f"serpapi: {serp_reason}"
                    resolve_error = resolve_error or f"could not resolve {spec['raw_input']}"
        else:
            name = spec["raw_input"]
            lat = None
            lng = None
            address = None
            resolved = False
            serp_mod = sys.modules.get("services.planner_serp")
            serp_reason = getattr(serp_mod, "LAST_ERROR", None)
            if serp_reason:
                resolve_error = resolve_error or f"serpapi: {serp_reason}"
            resolve_error = resolve_error or f"could not resolve {spec['raw_input']}"

        pin = {
            "pin_id": pin_id,
            "session_id": session_id,
            "seq": spec["seq"],
            "name": name,
            "source": spec["source"],
            "raw_input": spec["raw_input"],
            "lat": lat,
            "lng": lng,
            "address": address,
            "resolved": resolved,
            "resolve_error": resolve_error,
        }
        pins.append(pin)
        if not resolved:
            failed.append(name)
            failed_details.append({"raw_input": spec["raw_input"], "source": spec["source"], "error": resolve_error})

    save_pins(session_id, pins)
    ctx["pins"] = pins

    n_resolved = sum(1 for p in pins if p["resolved"])
    return {
        "pins_resolved": n_resolved,
        "pins_total": len(pins),
        "failed": failed,
        "failed_details": failed_details,
    }
