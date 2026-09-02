-- Mr. Bounce — Trip Orchestrator tables (planner crew)
-- Canonical schema; services/planner_db.py mirrors these statements verbatim in
-- _ensure_tables() so a missing migration self-heals on serverless cold starts.

-- One row per trip-planning run. Stage-per-request: current_node names the next
-- node the graph runner should execute; critic_round bounds the Critic loop.
CREATE TABLE IF NOT EXISTS planner_sessions (
    session_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    destination  TEXT NOT NULL,                -- single city (v1 scope)
    start_date   DATE NOT NULL,
    num_days     INT NOT NULL DEFAULT 2,
    raw_input    JSONB NOT NULL DEFAULT '{}'::jsonb,  -- original user payload
    status       TEXT NOT NULL DEFAULT 'pending',     -- pending|running|completed|failed
    current_node TEXT,                              -- next graph node to run
    critic_round INT NOT NULL DEFAULT 0,
    error        TEXT
);

-- Pins resolved by the Ingest tool (one Google Maps short link or pasted name per place)
CREATE TABLE IF NOT EXISTS planner_pins (
    pin_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID NOT NULL REFERENCES planner_sessions(session_id) ON DELETE CASCADE,
    seq           INT NOT NULL,                   -- user-given order (tie-breaker)
    name          TEXT NOT NULL,
    source        TEXT NOT NULL,                  -- short_link | text
    raw_input     TEXT NOT NULL,
    lat           DOUBLE PRECISION,
    lng           DOUBLE PRECISION,
    address       TEXT,
    resolved      BOOLEAN NOT NULL DEFAULT FALSE, -- got usable name+coords
    resolve_error TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_planner_pins_session ON planner_pins(session_id);

-- Leg-time cache keyed by an unordered place pair (sorted normalized names),
-- so A->B and B->A share one row. Durations per mode; the Logistics tool
-- decides walk vs transit vs drive per leg when reading/writing.
CREATE TABLE IF NOT EXISTS planner_leg_cache (
    cache_key       TEXT PRIMARY KEY,             -- "name-a|name-b" (sorted, normalized)
    from_name       TEXT NOT NULL,
    to_name         TEXT NOT NULL,
    walk_minutes    DOUBLE PRECISION,
    transit_minutes DOUBLE PRECISION,
    drive_minutes   DOUBLE PRECISION,
    distance_km     DOUBLE PRECISION,
    estimated       BOOLEAN NOT NULL DEFAULT FALSE, -- TRUE = haversine fallback, not real directions
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Full node trace: every graph node run (input, output, status, timing)
CREATE TABLE IF NOT EXISTS planner_trace (
    trace_id    BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES planner_sessions(session_id) ON DELETE CASCADE,
    seq         INT NOT NULL,                    -- global run order within the session
    node_name   TEXT NOT NULL,
    node_type   TEXT NOT NULL,                   -- agent | tool
    round       INT NOT NULL DEFAULT 0,          -- Critic loop round (0 = first pass)
    status      TEXT NOT NULL,                   -- ok | failed
    input_json  JSONB,
    output_json JSONB,
    error       TEXT,
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    duration_ms INT
);

CREATE INDEX IF NOT EXISTS idx_planner_trace_session ON planner_trace(session_id);

-- Final Compiler output: structured JSON + human-readable markdown
CREATE TABLE IF NOT EXISTS planner_itinerary (
    session_id     UUID PRIMARY KEY REFERENCES planner_sessions(session_id) ON DELETE CASCADE,
    itinerary_json JSONB NOT NULL,
    markdown       TEXT NOT NULL,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);
