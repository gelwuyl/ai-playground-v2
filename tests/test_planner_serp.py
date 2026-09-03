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
