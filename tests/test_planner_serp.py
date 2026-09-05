"""Unit tests for the SerpApi client's directions parsing (no network).

httpx.get is monkeypatched with fake responses; SERPAPI_KEY is fake.
Run: .venv/bin/python -m pytest tests/test_planner_serp.py -q
"""
import pytest

import services.planner_serp as planner_serp


class _FakeResp:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise planner_serp.httpx.HTTPStatusError(
                f"Client error '{self.status_code}'",
                request=None,
                response=None,
            )

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _no_fixtures(monkeypatch):
    monkeypatch.setattr(planner_serp, "USE_FIXTURES", False)


def test_directions_parses_meters_and_seconds(monkeypatch):
    """SerpApi google_maps_directions: distance in meters, duration in seconds."""
    payload = {
        "directions": [
            {
                "travel_mode": "Walking",
                "distance": 850,
                "duration": 660,
                "formatted_distance": "0.9 km",
                "formatted_duration": "11 min",
            }
        ]
    }
    monkeypatch.setattr(planner_serp.httpx, "get", lambda *a, **k: _FakeResp(payload))
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    result = planner_serp.directions("A", "B", "walking", "Singapore")
    assert result == {"distance_km": 0.85, "minutes": 11.0}


def test_directions_uses_correct_engine_and_travel_mode(monkeypatch):
    """Engine is google_maps_directions with numeric travel_mode (transit=3)."""
    captured = {}

    def fake_get(url, params=None, **k):
        captured["params"] = params
        return _FakeResp({"directions": [{"distance": 1000, "duration": 300}]})

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    planner_serp.directions("A", "B", "transit", "Singapore")
    assert captured["params"]["engine"] == "google_maps_directions"
    assert captured["params"]["travel_mode"] == "3"


def test_directions_walking_maps_to_travel_mode_2(monkeypatch):
    captured = {}

    def fake_get(url, params=None, **k):
        captured["params"] = params
        return _FakeResp({"directions": [{"distance": 500, "duration": 400}]})

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    planner_serp.directions("A", "B", "walking", "Singapore")
    assert captured["params"]["travel_mode"] == "2"


def test_directions_engine_error_surfaces_in_last_error(monkeypatch):
    """SerpApi returns HTTP 200 with an error field; it must surface."""
    payload = {"error": "Unsupported `wrong_engine` search engine."}

    def fake_get(*a, **k):
        return _FakeResp(payload, text=str(payload))

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    result = planner_serp.directions("A", "B", "walking", "Singapore")
    assert result is None
    assert "Unsupported" in planner_serp.LAST_ERROR


def test_directions_redacts_api_key_in_exception_text(monkeypatch):
    """HTTPStatusError messages embed the request URL incl. api_key — redacted."""
    req = planner_serp.httpx.Request(
        "GET", "https://serpapi.com/search.json?api_key=supersecret"
    )
    resp = planner_serp.httpx.Response(400, text='{"error": "bad"}', request=req)

    def fake_get(*a, **k):
        raise planner_serp.httpx.HTTPStatusError(
            "Client error '400' for url https://serpapi.com/search.json?api_key=supersecret",
            request=req,
            response=resp,
        )

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "supersecret")
    result = planner_serp.directions("A", "B", "walking", "Singapore")
    assert result is None
    assert "supersecret" not in planner_serp.LAST_ERROR
    assert "api_key=REDACTED" in planner_serp.LAST_ERROR


def test_directions_missing_fields_reports_item_keys(monkeypatch):
    """A directions item without distance/duration reports its keys, no crash."""
    payload = {"directions": [{"travel_mode": "Walking"}]}

    def fake_get(*a, **k):
        return _FakeResp(payload, text=str(payload))

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    result = planner_serp.directions("A", "B", "walking", "Singapore")
    assert result is None
    assert "item keys" in planner_serp.LAST_ERROR


# ---------------------------------------------------------------------------
# reverse_geocode
# ---------------------------------------------------------------------------
def test_reverse_geocode_parses_local_results(monkeypatch):
    """local_results[0] is parsed into the frozen contract shape."""
    payload = {
        "local_results": [
            {
                "title": "Gardens by the Bay",
                "address": "18 Marina Gardens Dr, Singapore 018953",
                "gps_coordinates": {"latitude": 1.2816, "longitude": 103.8636},
                "place_id": "0x31da175a4e0e1b3d:0x8c",
            }
        ]
    }

    def fake_get(*a, **k):
        return _FakeResp(payload)

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    result = planner_serp.reverse_geocode(1.2816, 103.8636)
    assert result["name"] == "Gardens by the Bay"
    assert result["address"] == "18 Marina Gardens Dr, Singapore 018953"
    assert result["lat"] == 1.2816
    assert result["lng"] == 103.8636
    assert result["place_id"] == "0x31da175a4e0e1b3d:0x8c"
    assert set(result) == {"name", "address", "lat", "lng", "place_id"}


def test_reverse_geocode_falls_back_to_place_results(monkeypatch):
    """A single place_results object is used when local_results is absent."""
    payload = {
        "place_results": {
            "title": "Merlion Park",
            "address": "1 Fullerton Rd, Singapore 049213",
            "gps_coordinates": {"latitude": 1.2868, "longitude": 103.8545},
            "place_id": "ChIJX5g0U5gZ2DSK",
        }
    }

    def fake_get(*a, **k):
        return _FakeResp(payload)

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    result = planner_serp.reverse_geocode(1.2868, 103.8545)
    assert result["name"] == "Merlion Park"
    assert result["address"] == "1 Fullerton Rd, Singapore 049213"
    assert result["lat"] == 1.2868
    assert result["lng"] == 103.8545
    assert result["place_id"] == "ChIJX5g0U5gZ2DSK"


def test_reverse_geocode_serp_error_surfaces_in_last_error(monkeypatch):
    """SerpApi returns HTTP 200 with an error field; it must surface."""
    payload = {"error": "Invalid `ll` parameter value."}

    def fake_get(*a, **k):
        return _FakeResp(payload, text=str(payload))

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    result = planner_serp.reverse_geocode(1.0, 103.0)
    assert result is None
    assert "Invalid" in planner_serp.LAST_ERROR


def test_reverse_geocode_redacts_api_key_in_exception_text(monkeypatch):
    """HTTPStatusError messages embed the request URL incl. api_key — redacted."""
    req = planner_serp.httpx.Request(
        "GET", "https://serpapi.com/search.json?api_key=supersecret"
    )
    resp = planner_serp.httpx.Response(400, text='{"error": "bad"}', request=req)

    def fake_get(*a, **k):
        raise planner_serp.httpx.HTTPStatusError(
            "Client error '400' for url https://serpapi.com/search.json?api_key=supersecret",
            request=req,
            response=resp,
        )

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "supersecret")
    result = planner_serp.reverse_geocode(1.0, 103.0)
    assert result is None
    assert "supersecret" not in planner_serp.LAST_ERROR
    assert "api_key=REDACTED" in planner_serp.LAST_ERROR


def test_reverse_geocode_no_results_reports_reason(monkeypatch):
    """No local_results / place_results yields a 'no results near' reason."""
    payload = {"search_metadata": {"status": "Success"}}

    def fake_get(*a, **k):
        return _FakeResp(payload, text=str(payload))

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    result = planner_serp.reverse_geocode(1.0, 103.0)
    assert result is None
    assert "no results near 1.0,103.0" in planner_serp.LAST_ERROR


# ---------------------------------------------------------------------------
# search_places
# ---------------------------------------------------------------------------
def _item(title, i):
    return {
        "title": title,
        "address": f"{i} Test Rd, Singapore 00000{i}",
        "gps_coordinates": {"latitude": 1.2 + i / 10.0, "longitude": 103.8 + i / 10.0},
        "place_id": f"place-{i}",
    }


def test_search_places_parses_local_results(monkeypatch):
    """local_results items are parsed into the frozen contract shape."""
    payload = {"local_results": [_item("Gardens by the Bay", 1), _item("Marina Bay Sands", 2), _item("Merlion Park", 3)]}

    def fake_get(*a, **k):
        return _FakeResp(payload)

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    result = planner_serp.search_places("Gardens by the Bay", "Singapore")
    assert len(result) == 3
    first = result[0]
    assert first["name"] == "Gardens by the Bay"
    assert first["address"] == "1 Test Rd, Singapore 000001"
    assert first["lat"] == pytest.approx(1.3)
    assert first["lng"] == pytest.approx(103.9)
    assert first["place_id"] == "place-1"
    assert set(first) == {"name", "address", "lat", "lng", "place_id"}


def test_search_places_caps_at_six(monkeypatch):
    """More than 6 local_results items yields exactly 6 results."""
    payload = {"local_results": [_item(f"Place {i}", i) for i in range(8)]}

    def fake_get(*a, **k):
        return _FakeResp(payload)

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    result = planner_serp.search_places("restaurants", "Singapore")
    assert len(result) == 6


def test_search_places_falls_back_to_place_results(monkeypatch):
    """A single place_results object yields a one-item list."""
    payload = {"place_results": _item("Merlion Park", 1)}

    def fake_get(*a, **k):
        return _FakeResp(payload)

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    result = planner_serp.search_places("Merlion Park", "Singapore")
    assert len(result) == 1
    assert result[0]["name"] == "Merlion Park"


def test_search_places_empty_local_results_is_success(monkeypatch):
    """Empty local_results returns [] (success), not None; LAST_ERROR stays None."""
    payload = {"local_results": []}

    def fake_get(*a, **k):
        return _FakeResp(payload, text=str(payload))

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    result = planner_serp.search_places("nowhere place", "Singapore")
    assert result == []
    assert planner_serp.LAST_ERROR is None


def test_search_places_serp_error_surfaces_in_last_error(monkeypatch):
    """SerpApi returns HTTP 200 with an error field; it must surface as None."""
    payload = {"error": "Invalid search query."}

    def fake_get(*a, **k):
        return _FakeResp(payload, text=str(payload))

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    result = planner_serp.search_places("restaurants", "Singapore")
    assert result is None
    assert "Invalid" in planner_serp.LAST_ERROR


def test_search_places_redacts_api_key_in_exception_text(monkeypatch):
    """HTTPStatusError messages embed api_key — it must be redacted."""
    req = planner_serp.httpx.Request(
        "GET", "https://serpapi.com/search.json?api_key=supersecret"
    )
    resp = planner_serp.httpx.Response(400, text='{"error": "bad"}', request=req)

    def fake_get(*a, **k):
        raise planner_serp.httpx.HTTPStatusError(
            "Client error '400' for url https://serpapi.com/search.json?api_key=supersecret",
            request=req,
            response=resp,
        )

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "supersecret")
    result = planner_serp.search_places("restaurants", "Singapore")
    assert result is None
    assert "supersecret" not in planner_serp.LAST_ERROR
    assert "api_key=REDACTED" in planner_serp.LAST_ERROR


def test_search_places_blank_query_does_not_crash(monkeypatch):
    """A blank query is handled sanely (no crash); backend still runs."""
    captured = {}

    def fake_get(url, params=None, **k):
        captured["params"] = params
        return _FakeResp({"local_results": []})

    monkeypatch.setattr(planner_serp.httpx, "get", fake_get)
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    result = planner_serp.search_places("", "")
    assert result == []
    assert captured["params"]["q"] == ""

