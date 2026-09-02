"""Tests for services.planner_ingest — deterministic, offline, no network.

All network paths are monkeypatched. save_pins is not tested here
(it requires a live Postgres connection).
"""
import sys
import types

import pytest

from services import planner_ingest
from services.planner_ingest import (
    parse_pin_inputs,
    resolve_short_link,
    resolve_text_pin,
    run_ingest,
    _parse_final_maps_url,
)


# ---------------------------------------------------------------------------
# parse_pin_inputs
# ---------------------------------------------------------------------------
class TestParsePinInputs:
    def test_mixed_dicts_and_bare_strings(self):
        payload = {
            "pins": [
                {"source": "short_link", "raw_input": "https://maps.app.goo.gl/abc"},
                "Gardens by the Bay",
                {"source": "text", "raw_input": "Marina Bay Sands"},
                "https://maps.app.goo.gl/xyz",
            ]
        }
        specs = parse_pin_inputs(payload)
        assert len(specs) == 4
        assert specs[0] == {"seq": 0, "source": "short_link", "raw_input": "https://maps.app.goo.gl/abc"}
        assert specs[1] == {"seq": 1, "source": "text", "raw_input": "Gardens by the Bay"}
        assert specs[2] == {"seq": 2, "source": "text", "raw_input": "Marina Bay Sands"}
        assert specs[3] == {"seq": 3, "source": "short_link", "raw_input": "https://maps.app.goo.gl/xyz"}

    def test_newline_blob_string(self):
        payload = {"pins": "Gardens by the Bay\nMarina Bay Sands\nArtScience Museum"}
        specs = parse_pin_inputs(payload)
        assert len(specs) == 3
        assert specs[0]["seq"] == 0
        assert specs[0]["source"] == "text"
        assert specs[0]["raw_input"] == "Gardens by the Bay"
        assert specs[1]["seq"] == 1
        assert specs[1]["raw_input"] == "Marina Bay Sands"
        assert specs[2]["seq"] == 2
        assert specs[2]["raw_input"] == "ArtScience Museum"

    def test_newline_blob_skips_empty_and_comments(self):
        payload = {"pins": "Gardens by the Bay\n\n# this is a comment\nMarina Bay Sands\n#another\n"}
        specs = parse_pin_inputs(payload)
        assert len(specs) == 2
        assert specs[0]["raw_input"] == "Gardens by the Bay"
        assert specs[1]["raw_input"] == "Marina Bay Sands"

    def test_list_skips_empty_and_comments(self):
        payload = {"pins": ["Gardens by the Bay", "", "  ", "# comment", "Marina Bay Sands"]}
        specs = parse_pin_inputs(payload)
        assert len(specs) == 2
        assert specs[0]["raw_input"] == "Gardens by the Bay"
        assert specs[1]["raw_input"] == "Marina Bay Sands"

    def test_http_detection(self):
        payload = {"pins": ["https://maps.app.goo.gl/abc", "http://short.link/xyz", "Just a name"]}
        specs = parse_pin_inputs(payload)
        assert specs[0]["source"] == "short_link"
        assert specs[1]["source"] == "short_link"
        assert specs[2]["source"] == "text"

    def test_seq_ordering(self):
        payload = {"pins": ["A", "B", "C", "D", "E"]}
        specs = parse_pin_inputs(payload)
        seqs = [s["seq"] for s in specs]
        assert seqs == [0, 1, 2, 3, 4]

    def test_empty_payload(self):
        assert parse_pin_inputs({}) == []
        assert parse_pin_inputs({"pins": []}) == []
        assert parse_pin_inputs({"pins": ""}) == []
        assert parse_pin_inputs({"pins": "\n\n  \n"}) == []

    def test_dict_missing_source_infers(self):
        payload = {"pins": [{"raw_input": "https://maps.app.goo.gl/abc"}, {"raw_input": "Some Place"}]}
        specs = parse_pin_inputs(payload)
        assert specs[0]["source"] == "short_link"
        assert specs[1]["source"] == "text"

    def test_dict_empty_raw_skipped(self):
        payload = {"pins": [{"raw_input": ""}, {"raw_input": "  "}, {"raw_input": "Real Place"}]}
        specs = parse_pin_inputs(payload)
        assert len(specs) == 1
        assert specs[0]["raw_input"] == "Real Place"


# ---------------------------------------------------------------------------
# _parse_final_maps_url (pure, no network)
# ---------------------------------------------------------------------------
class TestParseFinalMapsUrl:
    def test_place_name_with_coords(self):
        url = "https://www.google.com/maps/place/Gardens+by+the+Bay/@1.2816,103.8636,15z"
        result = _parse_final_maps_url(url)
        assert result is not None
        assert result["name"] == "Gardens by the Bay"
        assert result["lat"] == pytest.approx(1.2816)
        assert result["lng"] == pytest.approx(103.8636)

    def test_coords_only_no_name(self):
        url = "https://www.google.com/maps/@1.2816,103.8636,15z"
        result = _parse_final_maps_url(url)
        assert result is not None
        assert result["name"] is None
        assert result["lat"] == pytest.approx(1.2816)
        assert result["lng"] == pytest.approx(103.8636)

    def test_q_with_lat_lng(self):
        url = "https://maps.google.com/?q=1.2816,103.8636"
        result = _parse_final_maps_url(url)
        assert result is not None
        # q is lat,lng so name should not be set from q
        assert result["lat"] == pytest.approx(1.2816)
        assert result["lng"] == pytest.approx(103.8636)

    def test_q_with_name_and_ll(self):
        url = "https://maps.google.com/?q=Gardens+by+the+Bay&ll=1.2816,103.8636"
        result = _parse_final_maps_url(url)
        assert result is not None
        assert result["name"] == "Gardens by the Bay"
        assert result["lat"] == pytest.approx(1.2816)
        assert result["lng"] == pytest.approx(103.8636)

    def test_no_coords_no_name_returns_none(self):
        url = "https://www.google.com/maps"
        result = _parse_final_maps_url(url)
        assert result is None

    def test_negative_coords(self):
        url = "https://www.google.com/maps/place/Sydney+Opera+House/@-33.8568,151.2153,17z"
        result = _parse_final_maps_url(url)
        assert result is not None
        assert result["lat"] == pytest.approx(-33.8568)
        assert result["lng"] == pytest.approx(151.2153)
        assert result["name"] == "Sydney Opera House"


# ---------------------------------------------------------------------------
# resolve_short_link with monkeypatched httpx
# ---------------------------------------------------------------------------
class FakeResp:
    """Minimal fake httpx.Response for redirect/final URL tests."""

    def __init__(self, url, headers=None, status_code=200):
        self.url = url
        self.headers = headers or {}
        self.status_code = status_code


class TestResolveShortLink:
    def test_place_name_and_coords(self, monkeypatch):
        """httpx.get returns a final URL with /place/Name/@lat,lng."""
        final = "https://www.google.com/maps/place/Gardens+by+the+Bay/@1.2816,103.8636,15z"

        def fake_get(url, **kwargs):
            return FakeResp(url=final)

        monkeypatch.setattr(planner_ingest.httpx, "get", fake_get)
        monkeypatch.setattr(planner_ingest, "_use_fixtures", lambda: False)
        result = resolve_short_link("https://maps.app.goo.gl/abc", "Singapore")
        assert result is not None
        assert result["name"] == "Gardens by the Bay"
        assert result["lat"] == pytest.approx(1.2816)
        assert result["lng"] == pytest.approx(103.8636)

    def test_coords_only_no_name(self, monkeypatch):
        """Final URL has @lat,lng but no /place/ segment -> name is None."""
        final = "https://www.google.com/maps/@1.2816,103.8636,15z"

        def fake_get(url, **kwargs):
            return FakeResp(url=final)

        monkeypatch.setattr(planner_ingest.httpx, "get", fake_get)
        monkeypatch.setattr(planner_ingest, "_use_fixtures", lambda: False)
        result = resolve_short_link("https://maps.app.goo.gl/abc", "Singapore")
        assert result is not None
        assert result["name"] is None
        assert result["lat"] == pytest.approx(1.2816)
        assert result["lng"] == pytest.approx(103.8636)

    def test_q_lat_lng(self, monkeypatch):
        """maps.google.com/?q=lat,lng shape."""
        final = "https://maps.google.com/?q=1.2816,103.8636"

        def fake_get(url, **kwargs):
            return FakeResp(url=final)

        monkeypatch.setattr(planner_ingest.httpx, "get", fake_get)
        monkeypatch.setattr(planner_ingest, "_use_fixtures", lambda: False)
        result = resolve_short_link("https://maps.app.goo.gl/abc", "Singapore")
        assert result is not None
        assert result["lat"] == pytest.approx(1.2816)
        assert result["lng"] == pytest.approx(103.8636)

    def test_redirect_chain_location_header(self, monkeypatch):
        """URL has no @ but Location header does — use the header."""
        final_no_coords = "https://maps.app.goo.gl/redirect"
        location_header = "https://www.google.com/maps/place/Gardens+by+the+Bay/@1.2816,103.8636,15z"

        class FakeRespWithLocation:
            def __init__(self):
                self.url = final_no_coords
                self.headers = {"location": location_header}
                self.status_code = 302

        def fake_get(url, **kwargs):
            return FakeRespWithLocation()

        monkeypatch.setattr(planner_ingest.httpx, "get", fake_get)
        monkeypatch.setattr(planner_ingest, "_use_fixtures", lambda: False)
        result = resolve_short_link("https://maps.app.goo.gl/abc", "Singapore")
        assert result is not None
        assert result["name"] == "Gardens by the Bay"
        assert result["lat"] == pytest.approx(1.2816)

    def test_use_fixtures_path(self, monkeypatch):
        """USE_FIXTURES path via planner_serp + planner_fixtures stubs."""
        # Stub planner_serp module with USE_FIXTURES=True
        serp_stub = types.ModuleType("services.planner_serp")
        serp_stub.USE_FIXTURES = True
        monkeypatch.setitem(sys.modules, "services.planner_serp", serp_stub)

        # Stub planner_fixtures with FIXTURE_SHORT_LINKS
        fixtures_stub = types.ModuleType("services.planner_fixtures")
        fixtures_stub.FIXTURE_SHORT_LINKS = {
            "https://maps.app.goo.gl/X": (
                "https://www.google.com/maps/place/"
                "Gardens+by+the+Bay/@1.2816,103.8636,15z"
            ),
        }
        monkeypatch.setitem(sys.modules, "services.planner_fixtures", fixtures_stub)

        # _use_fixtures should pick up USE_FIXTURES from the stub
        assert planner_ingest._use_fixtures() is True

        result = resolve_short_link("https://maps.app.goo.gl/X", "Singapore")
        assert result is not None
        assert result["name"] == "Gardens by the Bay"
        assert result["lat"] == pytest.approx(1.2816)
        assert result["lng"] == pytest.approx(103.8636)

    def test_use_fixtures_not_in_dict_returns_none(self, monkeypatch):
        """When USE_FIXTURES but URL not in FIXTURE_SHORT_LINKS -> None."""
        serp_stub = types.ModuleType("services.planner_serp")
        serp_stub.USE_FIXTURES = True
        monkeypatch.setitem(sys.modules, "services.planner_serp", serp_stub)

        fixtures_stub = types.ModuleType("services.planner_fixtures")
        fixtures_stub.FIXTURE_SHORT_LINKS = {}
        monkeypatch.setitem(sys.modules, "services.planner_fixtures", fixtures_stub)

        # httpx.get should NOT be called in fixtures mode — but if it is,
        # it would fail. We prevent that by asserting it wasn't called.
        call_count = {"n": 0}

        def fake_get(url, **kwargs):
            call_count["n"] += 1
            return FakeResp(url="https://example.com")

        monkeypatch.setattr(planner_ingest.httpx, "get", fake_get)
        result = resolve_short_link("https://maps.app.goo.gl/unknown", "Singapore")
        assert result is None
        assert call_count["n"] == 0  # no network in fixtures mode

    def test_httpx_exception_returns_none(self, monkeypatch):
        """If httpx.get raises, and no SerpApi fallback, return None."""
        def fake_get(url, **kwargs):
            raise ConnectionError("network down")

        monkeypatch.setattr(planner_ingest.httpx, "get", fake_get)
        monkeypatch.setattr(planner_ingest, "_use_fixtures", lambda: False)
        # No planner_serp module available -> no fallback.
        monkeypatch.delitem(sys.modules, "services.planner_serp", raising=False)
        result = resolve_short_link("https://maps.app.goo.gl/broken", "Singapore")
        assert result is None

    def test_httpx_exception_falls_back_to_serp(self, monkeypatch):
        """If httpx.get fails but the URL has a /place/ slug, fall back to SerpApi."""
        def fake_get(url, **kwargs):
            raise ConnectionError("network down")

        monkeypatch.setattr(planner_ingest.httpx, "get", fake_get)
        monkeypatch.setattr(planner_ingest, "_use_fixtures", lambda: False)

        # Stub planner_serp with a geocode_place that returns a result.
        serp_stub = types.ModuleType("services.planner_serp")
        serp_stub.USE_FIXTURES = False
        serp_stub.geocode_place = lambda query, city: {
            "name": "Gardens by the Bay",
            "lat": 1.2816,
            "lng": 103.8636,
            "address": "18 Marina Gardens Dr, Singapore",
        }
        monkeypatch.setitem(sys.modules, "services.planner_serp", serp_stub)

        # Use a URL that has a /place/ segment so the fallback can derive a query.
        result = resolve_short_link(
            "https://www.google.com/maps/place/Gardens+by+the+Bay",
            "Singapore",
        )
        assert result is not None
        assert result["name"] == "Gardens by the Bay"
        assert result["lat"] == pytest.approx(1.2816)


# ---------------------------------------------------------------------------
# resolve_text_pin
# ---------------------------------------------------------------------------
class TestResolveTextPin:
    def test_geocode_returns_dict(self, monkeypatch):
        serp_stub = types.ModuleType("services.planner_serp")
        serp_stub.geocode_place = lambda name, city: {
            "name": "Gardens by the Bay",
            "lat": 1.2816,
            "lng": 103.8636,
            "address": "18 Marina Gardens Dr, Singapore",
        }
        monkeypatch.setitem(sys.modules, "services.planner_serp", serp_stub)

        result = resolve_text_pin("Gardens by the Bay", "Singapore")
        assert result is not None
        assert result["name"] == "Gardens by the Bay"
        assert result["lat"] == pytest.approx(1.2816)
        assert result["lng"] == pytest.approx(103.8636)

    def test_geocode_returns_none(self, monkeypatch):
        serp_stub = types.ModuleType("services.planner_serp")
        serp_stub.geocode_place = lambda name, city: None
        monkeypatch.setitem(sys.modules, "services.planner_serp", serp_stub)

        result = resolve_text_pin("Nonexistent Place", "Singapore")
        assert result is None

    def test_geocode_raises_returns_none(self, monkeypatch):
        serp_stub = types.ModuleType("services.planner_serp")

        def boom(name, city):
            raise RuntimeError("serp error")

        serp_stub.geocode_place = boom
        monkeypatch.setitem(sys.modules, "services.planner_serp", serp_stub)

        result = resolve_text_pin("Some Place", "Singapore")
        assert result is None

    def test_no_planner_serp_module(self, monkeypatch):
        """If planner_serp is not importable, return None gracefully."""
        monkeypatch.delitem(sys.modules, "services.planner_serp", raising=False)
        # Also make the import fail by injecting an import blocker.
        original_import = __import__

        def blocking_import(name, *args, **kwargs):
            if name == "services.planner_serp":
                raise ImportError("no such module")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", blocking_import)
        result = resolve_text_pin("Some Place", "Singapore")
        assert result is None


# ---------------------------------------------------------------------------
# run_ingest (orchestration, no DB — save_pins monkeypatched)
# ---------------------------------------------------------------------------
class TestRunIngest:
    def test_mixed_pins_resolved(self, monkeypatch):
        """End-to-end with monkeypatched resolution functions and save_pins."""
        monkeypatch.setattr(planner_ingest, "save_pins", lambda sid, pins: None)
        monkeypatch.setattr(planner_ingest, "_use_fixtures", lambda: False)

        # Stub resolve_short_link to return a resolved dict.
        monkeypatch.setattr(
            planner_ingest,
            "resolve_short_link",
            lambda url, city: {"name": "Gardens by the Bay", "lat": 1.28, "lng": 103.86, "address": None},
        )
        # Stub resolve_text_pin for the text pin.
        monkeypatch.setattr(
            planner_ingest,
            "resolve_text_pin",
            lambda name, city: {"name": "Marina Bay Sands", "lat": 1.28, "lng": 103.86, "address": "10 Bayfront Ave"},
        )

        ctx = {
            "session_id": "test-123",
            "destination": "Singapore",
            "payload": {
                "pins": [
                    {"source": "short_link", "raw_input": "https://maps.app.goo.gl/abc"},
                    {"source": "text", "raw_input": "Marina Bay Sands"},
                ],
            },
        }
        result = run_ingest(ctx)
        assert result["pins_total"] == 2
        assert result["pins_resolved"] == 2
        assert result["failed"] == []
        assert "pins" in ctx
        assert len(ctx["pins"]) == 2
        assert ctx["pins"][0]["seq"] == 0
        assert ctx["pins"][1]["seq"] == 1
        assert ctx["pins"][0]["resolved"] is True
        assert ctx["pins"][1]["resolved"] is True
        assert ctx["pins"][0]["pin_id"]  # UUID string present
        assert ctx["pins"][1]["address"] == "10 Bayfront Ave"

    def test_unresolved_pin_persists_with_error(self, monkeypatch):
        """A pin that can't be resolved still persists with resolved=False."""
        monkeypatch.setattr(planner_ingest, "save_pins", lambda sid, pins: None)
        monkeypatch.setattr(planner_ingest, "_use_fixtures", lambda: False)

        monkeypatch.setattr(
            planner_ingest,
            "resolve_short_link",
            lambda url, city: None,
        )
        monkeypatch.setattr(
            planner_ingest,
            "resolve_text_pin",
            lambda name, city: None,
        )

        ctx = {
            "session_id": "test-456",
            "destination": "Singapore",
            "payload": {"pins": ["Unknown Place"]},
        }
        result = run_ingest(ctx)
        assert result["pins_total"] == 1
        assert result["pins_resolved"] == 0
        assert "Unknown Place" in result["failed"]
        assert ctx["pins"][0]["resolved"] is False
        assert ctx["pins"][0]["resolve_error"] is not None
        assert ctx["pins"][0]["name"] == "Unknown Place"

    def test_one_bad_pin_does_not_kill_node(self, monkeypatch):
        """If resolve_short_link raises for one pin, others still resolve."""
        monkeypatch.setattr(planner_ingest, "save_pins", lambda sid, pins: None)
        monkeypatch.setattr(planner_ingest, "_use_fixtures", lambda: False)

        call_count = {"n": 0}

        def flaky_resolve(url, city):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("boom")
            return {"name": "Good Place", "lat": 1.0, "lng": 2.0, "address": None}

        monkeypatch.setattr(planner_ingest, "resolve_short_link", flaky_resolve)
        monkeypatch.setattr(
            planner_ingest,
            "resolve_text_pin",
            lambda name, city: None,
        )

        ctx = {
            "session_id": "test-789",
            "destination": "Singapore",
            "payload": {
                "pins": [
                    {"source": "short_link", "raw_input": "https://maps.app.goo.gl/bad"},
                    {"source": "short_link", "raw_input": "https://maps.app.goo.gl/good"},
                ],
            },
        }
        result = run_ingest(ctx)
        assert result["pins_total"] == 2
        # The first pin failed (exception caught), second succeeded.
        assert result["pins_resolved"] == 1
        assert len(result["failed"]) == 1
        assert ctx["pins"][0]["resolved"] is False
        assert ctx["pins"][1]["resolved"] is True

    def test_empty_payload(self, monkeypatch):
        monkeypatch.setattr(planner_ingest, "save_pins", lambda sid, pins: None)
        ctx = {"session_id": "test-empty", "destination": "Singapore", "payload": {}}
        result = run_ingest(ctx)
        assert result["pins_total"] == 0
        assert result["pins_resolved"] == 0
        assert result["failed"] == []
        assert ctx["pins"] == []
