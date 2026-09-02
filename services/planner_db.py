"""Mr. Bounce — planner schema self-healing (mirrors sql/005_trip_planner.sql).

Serverless deploys have no migration hook (the 004 incident), so the same
CREATE TABLE IF NOT EXISTS statements are applied once per cold start.
"""
from services.database import get_conn

_DDL = """
CREATE TABLE IF NOT EXISTS planner_sessions (
    session_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    destination  TEXT NOT NULL,
    start_date   DATE NOT NULL,
    num_days     INT NOT NULL DEFAULT 2,
    raw_input    JSONB NOT NULL DEFAULT '{}'::jsonb,
    status       TEXT NOT NULL DEFAULT 'pending',
    current_node TEXT,
    critic_round INT NOT NULL DEFAULT 0,
    error        TEXT
);
CREATE TABLE IF NOT EXISTS planner_pins (
    pin_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID NOT NULL REFERENCES planner_sessions(session_id) ON DELETE CASCADE,
    seq           INT NOT NULL,
    name          TEXT NOT NULL,
    source        TEXT NOT NULL,
    raw_input     TEXT NOT NULL,
    lat           DOUBLE PRECISION,
    lng           DOUBLE PRECISION,
    address       TEXT,
    resolved      BOOLEAN NOT NULL DEFAULT FALSE,
    resolve_error TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_planner_pins_session ON planner_pins(session_id);
CREATE TABLE IF NOT EXISTS planner_leg_cache (
    cache_key       TEXT PRIMARY KEY,
    from_name       TEXT NOT NULL,
    to_name         TEXT NOT NULL,
    walk_minutes    DOUBLE PRECISION,
    transit_minutes DOUBLE PRECISION,
    drive_minutes   DOUBLE PRECISION,
    distance_km     DOUBLE PRECISION,
    estimated       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS planner_trace (
    trace_id    BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES planner_sessions(session_id) ON DELETE CASCADE,
    seq         INT NOT NULL,
    node_name   TEXT NOT NULL,
    node_type   TEXT NOT NULL,
    round       INT NOT NULL DEFAULT 0,
    status      TEXT NOT NULL,
    input_json  JSONB,
    output_json JSONB,
    error       TEXT,
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    duration_ms INT
);
CREATE INDEX IF NOT EXISTS idx_planner_trace_session ON planner_trace(session_id);
CREATE TABLE IF NOT EXISTS planner_itinerary (
    session_id     UUID PRIMARY KEY REFERENCES planner_sessions(session_id) ON DELETE CASCADE,
    itinerary_json JSONB NOT NULL,
    markdown       TEXT NOT NULL,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);
"""

_tables_ready = False


def _ensure_tables():
    """Apply the 005 schema once per serverless instance (no-op afterwards)."""
    global _tables_ready
    if _tables_ready:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
    _tables_ready = True
