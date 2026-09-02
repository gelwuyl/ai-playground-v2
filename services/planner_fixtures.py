"""Mr. Bounce — deterministic static fixtures for the Singapore 8-pin trip.

Pure data: zero imports, zero computation. Used by USE_FIXTURES=1 and unit tests
so the planner runs end-to-end without a SerpApi key or network access.

Canonical trip: start_date 2026-09-07 (a Monday), 2 days, Singapore.
National Gallery Singapore is CLOSED MONDAYS so closed-day repair is exercised.
Sultan Mosque has split-hours so the scheduler copes with multi-window days.
"""

FIXTURE_CITY = "Singapore"

FIXTURE_SHORT_LINKS = {
    "https://maps.app.goo.gl/AbCdEfGh12345678": "https://www.google.com/maps/place/Gardens+by+the+Bay/@1.2816,103.8636,15z",
    "https://maps.app.goo.gl/ZyXwVuTs87654321": "https://www.google.com/maps/place/Sultan+Mosque/@1.3020,103.8590,17z",
}

# ---------------------------------------------------------------------------
# 8 places — keyed by plain name, value is a dict with the SerpApi shape.
# raw_hours uses 7 lowercase day-of-week keys; values are strings as SerpApi
# prints them. The EN DASH (–) separates open/close times.
# ---------------------------------------------------------------------------
FIXTURE_PLACES = {
    "Gardens by the Bay": {
        "name": "Gardens by the Bay",
        "lat": 1.2816,
        "lng": 103.8636,
        "address": "18 Marina Gardens Drive, Marina Bay, Singapore 018953",
        "place_id": "0x31da1b9c5e000001:0xc51b123456789abc",
        "raw_hours": {
            "monday": "9–9 PM",
            "tuesday": "9–9 PM",
            "wednesday": "9–9 PM",
            "thursday": "9–9 PM",
            "friday": "9–9 PM",
            "saturday": "9–9 PM",
            "sunday": "9–9 PM",
        },
    },
    "ArtScience Museum": {
        "name": "ArtScience Museum",
        "lat": 1.2862,
        "lng": 103.8596,
        "address": "6 Bayfront Avenue, Marina Bay, Singapore 018974",
        "place_id": "0x31da1b9c5e000002:0xc51b23456789abcd",
        "raw_hours": {
            "monday": "10 AM–7 PM",
            "tuesday": "10 AM–7 PM",
            "wednesday": "10 AM–7 PM",
            "thursday": "10 AM–7 PM",
            "friday": "10 AM–7 PM",
            "saturday": "10 AM–7 PM",
            "sunday": "10 AM–7 PM",
        },
    },
    "National Gallery Singapore": {
        "name": "National Gallery Singapore",
        "lat": 1.2866,
        "lng": 103.8535,
        "address": "1 St Andrew's Road, City Hall, Singapore 178957",
        "place_id": "0x31da1b9c5e000003:0xc51b3456789abcde",
        "raw_hours": {
            "monday": "Closed",
            "tuesday": "10 AM–7 PM",
            "wednesday": "10 AM–7 PM",
            "thursday": "10 AM–7 PM",
            "friday": "10 AM–7 PM",
            "saturday": "10 AM–7 PM",
            "sunday": "10 AM–7 PM",
        },
    },
    "St Andrew's Cathedral": {
        "name": "St Andrew's Cathedral",
        "lat": 1.2876,
        "lng": 103.8520,
        "address": "61 St Andrew's Road, City Hall, Singapore 178958",
        "place_id": "0x31da1b9c5e000004:0xc51b456789abcdef",
        "raw_hours": {
            "monday": "10 AM–4:30 PM",
            "tuesday": "10 AM–4:30 PM",
            "wednesday": "10 AM–4:30 PM",
            "thursday": "10 AM–4:30 PM",
            "friday": "10 AM–4:30 PM",
            "saturday": "10 AM–4:30 PM",
            "sunday": "10 AM–4:30 PM",
        },
    },
    "Buddha Tooth Relic Temple": {
        "name": "Buddha Tooth Relic Temple",
        "lat": 1.2814,
        "lng": 103.8447,
        "address": "288 South Bridge Road, Chinatown, Singapore 058840",
        "place_id": "0x31da1b9c5e000005:0xc51b56789abcdef0",
        "raw_hours": {
            "monday": "7 AM–5 PM",
            "tuesday": "7 AM–5 PM",
            "wednesday": "7 AM–5 PM",
            "thursday": "7 AM–5 PM",
            "friday": "7 AM–5 PM",
            "saturday": "7 AM–5 PM",
            "sunday": "7 AM–5 PM",
        },
    },
    "Maxwell Food Centre": {
        "name": "Maxwell Food Centre",
        "lat": 1.2803,
        "lng": 103.8444,
        "address": "11 Maxwell Road, Chinatown, Singapore 069192",
        "place_id": "0x31da1b9c5e000006:0xc51b6789abcdef01",
        "raw_hours": {
            "monday": "8 AM–10 PM",
            "tuesday": "8 AM–10 PM",
            "wednesday": "8 AM–10 PM",
            "thursday": "8 AM–10 PM",
            "friday": "8 AM–10 PM",
            "saturday": "8 AM–10 PM",
            "sunday": "8 AM–10 PM",
        },
    },
    "Singapore Botanic Gardens": {
        "name": "Singapore Botanic Gardens",
        "lat": 1.3138,
        "lng": 103.8159,
        "address": "1 Cluny Road, Botanic Gardens, Singapore 259569",
        "place_id": "0x31da1b9c5e000007:0xc51b789abcdef012",
        "raw_hours": {
            "monday": "5 AM–12 AM",
            "tuesday": "5 AM–12 AM",
            "wednesday": "5 AM–12 AM",
            "thursday": "5 AM–12 AM",
            "friday": "5 AM–12 AM",
            "saturday": "5 AM–12 AM",
            "sunday": "5 AM–12 AM",
        },
    },
    "Sultan Mosque": {
        "name": "Sultan Mosque",
        "lat": 1.3020,
        "lng": 103.8590,
        "address": "3 Muscat Street, Kampong Glam, Singapore 198833",
        "place_id": "0x31da1b9c5e000008:0xc51b89abcdef0123",
        "raw_hours": {
            "monday": "10 AM–12 PM, 2–4:30 PM",
            "tuesday": "10 AM–12 PM, 2–4:30 PM",
            "wednesday": "10 AM–12 PM, 2–4:30 PM",
            "thursday": "10 AM–12 PM, 2–4:30 PM",
            "friday": "10 AM–12 PM, 2–4:30 PM",
            "saturday": "10 AM–12 PM, 2–4:30 PM",
            "sunday": "10 AM–12 PM, 2–4:30 PM",
        },
    },
}

# ---------------------------------------------------------------------------
# 84 direction entries — 28 unordered place pairs x 3 modes (walking/driving/transit).
# Keyed by f"{leg_cache_key(name_a, name_b)}:{mode}" where leg_cache_key sorts
# the normalized names alphabetically and joins with "|".
# distance_km = straight-line * 1.4 (road correction); minutes per spec formula.
# ---------------------------------------------------------------------------
FIXTURE_DIRECTIONS = {
    "artscience museum|gardens by the bay:walking": {"distance_km": 0.9, "minutes": 12.0},
    "artscience museum|gardens by the bay:driving": {"distance_km": 0.9, "minutes": 2.5},
    "artscience museum|gardens by the bay:transit": {"distance_km": 0.9, "minutes": 10.6},
    "gardens by the bay|national gallery singapore:walking": {"distance_km": 1.8, "minutes": 24.0},
    "gardens by the bay|national gallery singapore:driving": {"distance_km": 1.8, "minutes": 4.9},
    "gardens by the bay|national gallery singapore:transit": {"distance_km": 1.8, "minutes": 14.2},
    "gardens by the bay|st andrew's cathedral:walking": {"distance_km": 2.0, "minutes": 26.7},
    "gardens by the bay|st andrew's cathedral:driving": {"distance_km": 2.0, "minutes": 5.5},
    "gardens by the bay|st andrew's cathedral:transit": {"distance_km": 2.0, "minutes": 15.0},
    "buddha tooth relic temple|gardens by the bay:walking": {"distance_km": 2.9, "minutes": 38.7},
    "buddha tooth relic temple|gardens by the bay:driving": {"distance_km": 2.9, "minutes": 7.9},
    "buddha tooth relic temple|gardens by the bay:transit": {"distance_km": 2.9, "minutes": 18.6},
    "gardens by the bay|maxwell food centre:walking": {"distance_km": 3.0, "minutes": 40.0},
    "gardens by the bay|maxwell food centre:driving": {"distance_km": 3.0, "minutes": 8.2},
    "gardens by the bay|maxwell food centre:transit": {"distance_km": 3.0, "minutes": 19.0},
    "gardens by the bay|singapore botanic gardens:walking": {"distance_km": 9.0, "minutes": 120.0},
    "gardens by the bay|singapore botanic gardens:driving": {"distance_km": 9.0, "minutes": 24.5},
    "gardens by the bay|singapore botanic gardens:transit": {"distance_km": 9.0, "minutes": 43.0},
    "gardens by the bay|sultan mosque:walking": {"distance_km": 3.3, "minutes": 44.0},
    "gardens by the bay|sultan mosque:driving": {"distance_km": 3.3, "minutes": 9.0},
    "gardens by the bay|sultan mosque:transit": {"distance_km": 3.3, "minutes": 20.2},
    "artscience museum|national gallery singapore:walking": {"distance_km": 1.0, "minutes": 13.3},
    "artscience museum|national gallery singapore:driving": {"distance_km": 1.0, "minutes": 2.7},
    "artscience museum|national gallery singapore:transit": {"distance_km": 1.0, "minutes": 11.0},
    "artscience museum|st andrew's cathedral:walking": {"distance_km": 1.2, "minutes": 16.0},
    "artscience museum|st andrew's cathedral:driving": {"distance_km": 1.2, "minutes": 3.3},
    "artscience museum|st andrew's cathedral:transit": {"distance_km": 1.2, "minutes": 11.8},
    "artscience museum|buddha tooth relic temple:walking": {"distance_km": 2.4, "minutes": 32.0},
    "artscience museum|buddha tooth relic temple:driving": {"distance_km": 2.4, "minutes": 6.5},
    "artscience museum|buddha tooth relic temple:transit": {"distance_km": 2.4, "minutes": 16.6},
    "artscience museum|maxwell food centre:walking": {"distance_km": 2.5, "minutes": 33.3},
    "artscience museum|maxwell food centre:driving": {"distance_km": 2.5, "minutes": 6.8},
    "artscience museum|maxwell food centre:transit": {"distance_km": 2.5, "minutes": 17.0},
    "artscience museum|singapore botanic gardens:walking": {"distance_km": 8.0, "minutes": 106.7},
    "artscience museum|singapore botanic gardens:driving": {"distance_km": 8.0, "minutes": 21.8},
    "artscience museum|singapore botanic gardens:transit": {"distance_km": 8.0, "minutes": 39.0},
    "artscience museum|sultan mosque:walking": {"distance_km": 2.5, "minutes": 33.3},
    "artscience museum|sultan mosque:driving": {"distance_km": 2.5, "minutes": 6.8},
    "artscience museum|sultan mosque:transit": {"distance_km": 2.5, "minutes": 17.0},
    "national gallery singapore|st andrew's cathedral:walking": {"distance_km": 0.3, "minutes": 4.0},
    "national gallery singapore|st andrew's cathedral:driving": {"distance_km": 0.3, "minutes": 0.8},
    "national gallery singapore|st andrew's cathedral:transit": {"distance_km": 0.3, "minutes": 8.2},
    "buddha tooth relic temple|national gallery singapore:walking": {"distance_km": 1.6, "minutes": 21.3},
    "buddha tooth relic temple|national gallery singapore:driving": {"distance_km": 1.6, "minutes": 4.4},
    "buddha tooth relic temple|national gallery singapore:transit": {"distance_km": 1.6, "minutes": 13.4},
    "maxwell food centre|national gallery singapore:walking": {"distance_km": 1.7, "minutes": 22.7},
    "maxwell food centre|national gallery singapore:driving": {"distance_km": 1.7, "minutes": 4.6},
    "maxwell food centre|national gallery singapore:transit": {"distance_km": 1.7, "minutes": 13.8},
    "national gallery singapore|singapore botanic gardens:walking": {"distance_km": 7.2, "minutes": 96.0},
    "national gallery singapore|singapore botanic gardens:driving": {"distance_km": 7.2, "minutes": 19.6},
    "national gallery singapore|singapore botanic gardens:transit": {"distance_km": 7.2, "minutes": 35.8},
    "national gallery singapore|sultan mosque:walking": {"distance_km": 2.5, "minutes": 33.3},
    "national gallery singapore|sultan mosque:driving": {"distance_km": 2.5, "minutes": 6.8},
    "national gallery singapore|sultan mosque:transit": {"distance_km": 2.5, "minutes": 17.0},
    "buddha tooth relic temple|st andrew's cathedral:walking": {"distance_km": 1.5, "minutes": 20.0},
    "buddha tooth relic temple|st andrew's cathedral:driving": {"distance_km": 1.5, "minutes": 4.1},
    "buddha tooth relic temple|st andrew's cathedral:transit": {"distance_km": 1.5, "minutes": 13.0},
    "maxwell food centre|st andrew's cathedral:walking": {"distance_km": 1.6, "minutes": 21.3},
    "maxwell food centre|st andrew's cathedral:driving": {"distance_km": 1.6, "minutes": 4.4},
    "maxwell food centre|st andrew's cathedral:transit": {"distance_km": 1.6, "minutes": 13.4},
    "singapore botanic gardens|st andrew's cathedral:walking": {"distance_km": 6.9, "minutes": 92.0},
    "singapore botanic gardens|st andrew's cathedral:driving": {"distance_km": 6.9, "minutes": 18.8},
    "singapore botanic gardens|st andrew's cathedral:transit": {"distance_km": 6.9, "minutes": 34.6},
    "st andrew's cathedral|sultan mosque:walking": {"distance_km": 2.5, "minutes": 33.3},
    "st andrew's cathedral|sultan mosque:driving": {"distance_km": 2.5, "minutes": 6.8},
    "st andrew's cathedral|sultan mosque:transit": {"distance_km": 2.5, "minutes": 17.0},
    "buddha tooth relic temple|maxwell food centre:walking": {"distance_km": 0.2, "minutes": 2.7},
    "buddha tooth relic temple|maxwell food centre:driving": {"distance_km": 0.2, "minutes": 0.5},
    "buddha tooth relic temple|maxwell food centre:transit": {"distance_km": 0.2, "minutes": 7.8},
    "buddha tooth relic temple|singapore botanic gardens:walking": {"distance_km": 6.7, "minutes": 89.3},
    "buddha tooth relic temple|singapore botanic gardens:driving": {"distance_km": 6.7, "minutes": 18.3},
    "buddha tooth relic temple|singapore botanic gardens:transit": {"distance_km": 6.7, "minutes": 33.8},
    "buddha tooth relic temple|sultan mosque:walking": {"distance_km": 3.9, "minutes": 52.0},
    "buddha tooth relic temple|sultan mosque:driving": {"distance_km": 3.9, "minutes": 10.6},
    "buddha tooth relic temple|sultan mosque:transit": {"distance_km": 3.9, "minutes": 22.6},
    "maxwell food centre|singapore botanic gardens:walking": {"distance_km": 6.8, "minutes": 90.7},
    "maxwell food centre|singapore botanic gardens:driving": {"distance_km": 6.8, "minutes": 18.5},
    "maxwell food centre|singapore botanic gardens:transit": {"distance_km": 6.8, "minutes": 34.2},
    "maxwell food centre|sultan mosque:walking": {"distance_km": 4.1, "minutes": 54.7},
    "maxwell food centre|sultan mosque:driving": {"distance_km": 4.1, "minutes": 11.2},
    "maxwell food centre|sultan mosque:transit": {"distance_km": 4.1, "minutes": 23.4},
    "singapore botanic gardens|sultan mosque:walking": {"distance_km": 7.0, "minutes": 93.3},
    "singapore botanic gardens|sultan mosque:driving": {"distance_km": 7.0, "minutes": 19.1},
    "singapore botanic gardens|sultan mosque:transit": {"distance_km": 7.0, "minutes": 35.0},
}

# ---------------------------------------------------------------------------
# 8 pin inputs — 6 text pins + 2 short-link pins (Gardens by the Bay, Sultan Mosque)
# ---------------------------------------------------------------------------
FIXTURE_PIN_INPUTS = [
    {"source": "short_link", "raw_input": "https://maps.app.goo.gl/AbCdEfGh12345678"},
    {"source": "text", "raw_input": "ArtScience Museum"},
    {"source": "text", "raw_input": "National Gallery Singapore"},
    {"source": "text", "raw_input": "St Andrew's Cathedral"},
    {"source": "text", "raw_input": "Buddha Tooth Relic Temple"},
    {"source": "text", "raw_input": "Maxwell Food Centre"},
    {"source": "text", "raw_input": "Singapore Botanic Gardens"},
    {"source": "short_link", "raw_input": "https://maps.app.goo.gl/ZyXwVuTs87654321"},
]
