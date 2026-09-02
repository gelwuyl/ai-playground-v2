"""Mr. Bounce API — consolidated trip-planner endpoints.

Vercel's Hobby plan allows at most 12 Serverless Functions per deployment, so
the four planner endpoints share one function (the 11th overall). vercel.json
rewrites map the original paths here with an ``action`` query parameter:

    POST /api/planner_start   -> action=start
    POST /api/planner_step    -> action=step
    GET  /api/planner_status   -> action=status
    POST /api/planner_delete  -> action=delete
"""
from __future__ import annotations

import datetime
import json
from urllib.parse import parse_qs, urlparse

from services.database import get_conn
from services.planner_db import _ensure_tables
from services.vercel_handler import VercelHandler


# ---------------------------------------------------------------------------
# ctx snapshot keys — the subset of ctx persisted inside each trace row's
# output_json["_ctx"] so the next serverless invocation can rehydrate full
# state without a schema change.
# ---------------------------------------------------------------------------
_CTX_KEYS = (
    "pins", "research", "legs", "schedule",
    "critic", "alternatives", "itinerary", "errors",
)


class handler(VercelHandler):
    def do_POST(self):
        action = self._action()
        if action == "start":
            return self._start()
        if action == "step":
            return self._step()
        if action == "delete":
            return self._delete()
        return self.json_response(
            {"detail": "Unknown POST action. Use action=start|step|delete."}, 400
        )

    def do_GET(self):
        if self._action() == "status":
            return self._status()
        return self.json_response(
            {"detail": "Unknown GET action. Use action=status."}, 400
        )

    # ------------------------------------------------------------------ helpers

    def _query_param(self, name: str) -> str:
        query = parse_qs(urlparse(self.path).query)
        return (query.get(name) or [""])[0].strip()

    def _action(self) -> str:
        return self._query_param("action")

    def _body(self):
        """Parse the JSON body, responding 400 and returning None on failure."""
        try:
            return self.read_json()
        except Exception:
            self.json_response({"detail": "Invalid JSON body."}, 400)
            return None

    # ------------------------------------------------------------------ actions

    def _start(self):
        body = self._body()
        if body is None:
            return

        destination = (body.get("destination") or "").strip()
        start_date_raw = (body.get("start_date") or "").strip()
        pins_raw = body.get("pins")
        num_days = body.get("num_days", 2)

        # --- required fields ---
        if not destination or not start_date_raw or not pins_raw:
            return self.json_response(
                {"detail": "Missing required fields: destination, start_date, or pins."},
                400,
            )

        # --- start_date parse ---
        try:
            datetime.date.fromisoformat(start_date_raw)
        except (ValueError, TypeError):
            return self.json_response(
                {"detail": "start_date must be YYYY-MM-DD."}, 400
            )

        # --- num_days ---
        try:
            num_days = int(num_days)
        except (ValueError, TypeError):
            return self.json_response(
                {"detail": "num_days must be an integer between 1 and 7."}, 400
            )
        if not (1 <= num_days <= 7):
            return self.json_response(
                {"detail": "num_days must be between 1 and 7."}, 400
            )

        # --- pins: list or newline-blob string ---
        if isinstance(pins_raw, str):
            pin_items = [
                line.strip() for line in pins_raw.splitlines() if line.strip()
            ]
            pins_list = pin_items
        elif isinstance(pins_raw, list):
            pins_list = pins_raw
        else:
            return self.json_response(
                {"detail": "pins must be a list or a newline-separated string."}, 400
            )

        if not pins_list:
            return self.json_response(
                {"detail": "At least one pin is required."}, 400
            )

        # --- create session ---
        _ensure_tables()
        payload = {
            "destination": destination,
            "start_date": start_date_raw,
            "num_days": num_days,
            "pins": pins_raw,
        }
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO planner_sessions
                        (destination, start_date, num_days, raw_input,
                         status, current_node, critic_round)
                    VALUES (%s, %s, %s, %s, 'pending', 'ingest', 0)
                    RETURNING session_id
                    """,
                    (
                        destination,
                        start_date_raw,
                        num_days,
                        json.dumps(payload),
                    ),
                )
                row = cur.fetchone()
            conn.commit()

        session_id = str(row[0])
        return self.json_response({
            "session_id": session_id,
            "status": "started",
            "next_step": "ingest",
        })

    def _step(self):
        body = self._body()
        if body is None:
            return

        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return self.json_response({"detail": "Missing session_id."}, 400)

        try:
            _ensure_tables()

            # --- load session row ---
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT destination, start_date, num_days, raw_input,
                               status, current_node, critic_round
                        FROM planner_sessions WHERE session_id = %s
                        """,
                        (session_id,),
                    )
                    res = cur.fetchone()
                conn.commit()

            if not res:
                return self.json_response(
                    {"detail": "Session not found."}, 404
                )

            destination, start_date, num_days, raw_input, status, current_node, critic_round = res

            # --- completed check ---
            if status == "completed" or current_node is None:
                return self.json_response(
                    {"detail": "Session already completed."}, 400
                )

            # --- rehydrate ctx from prior trace rows ---
            ctx = self._build_ctx(
                session_id, destination, start_date, num_days,
                raw_input, current_node, critic_round,
            )

            # --- build registry ---
            registry = build_registry()

            # --- run one node ---
            from services.planner_graph import advance, PostgresSink
            sink = PostgresSink(session_id)
            row = advance(ctx, registry, sink)

            # --- update session ---
            new_node = ctx.get("current_node")
            new_status = "completed" if new_node is None else "running"
            new_critic_round = ctx.get("critic_round", critic_round)
            last_error = ctx["errors"][-1] if ctx.get("errors") else None

            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE planner_sessions
                        SET current_node = %s, status = %s,
                            critic_round = %s, error = %s
                        WHERE session_id = %s
                        """,
                        (new_node, new_status, new_critic_round, last_error, session_id),
                    )
                conn.commit()

            # --- persist itinerary if compiler just ran ---
            if row["node_name"] == "compiler" and ctx.get("itinerary") is not None:
                self._persist_itinerary(session_id, row, ctx)

            # --- respond ---
            # Strip _ctx from the node's output for the response payload.
            node_output = dict(row.get("output", {}))
            node_output.pop("_ctx", None)

            return self.json_response({
                "status": new_status,
                "node": row["node_name"],
                "next_node": new_node,
                "result": node_output,
                "critic_round": ctx.get("critic_round", 0),
            })

        except Exception as e:
            # Best-effort: mark the session as failed.
            try:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE planner_sessions SET status = 'failed', error = %s WHERE session_id = %s",
                            (str(e), session_id),
                        )
                    conn.commit()
            except Exception:
                pass
            return self.json_response({"detail": str(e)}, 500)

    def _status(self):
        session_id = self._query_param("session_id")
        if not session_id:
            return self.json_response({"detail": "Missing session_id."}, 400)

        _ensure_tables()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT destination, start_date, num_days, status,
                           current_node, critic_round
                    FROM planner_sessions WHERE session_id = %s
                    """,
                    (session_id,),
                )
                res = cur.fetchone()
                if not res:
                    return self.json_response(
                        {"detail": "Session not found."}, 404
                    )
                destination, start_date, num_days, status, current_node, critic_round = res

                # --- trace rows ---
                cur.execute(
                    """
                    SELECT seq, node_name, node_type, round, status,
                           input_json, output_json, error,
                           started_at, finished_at, duration_ms
                    FROM planner_trace WHERE session_id = %s
                    ORDER BY trace_id
                    """,
                    (session_id,),
                )
                trace_rows = cur.fetchall()

                # --- itinerary ---
                cur.execute(
                    """
                    SELECT itinerary_json, markdown
                    FROM planner_itinerary WHERE session_id = %s
                    """,
                    (session_id,),
                )
                itin_row = cur.fetchone()
            conn.commit()

        trace = []
        for tr in trace_rows:
            (seq, node_name, node_type, round_num, tr_status,
             input_json, output_json, error,
             started_at, finished_at, duration_ms) = tr

            out_parsed = _as_json(output_json)
            if isinstance(out_parsed, dict):
                out_parsed = dict(out_parsed)
                out_parsed.pop("_ctx", None)

            trace.append({
                "seq": seq,
                "node_name": node_name,
                "node_type": node_type,
                "round": round_num,
                "status": tr_status,
                "input_json": _as_json(input_json),
                "output_json": out_parsed,
                "error": error,
                "started_at": _iso(started_at),
                "finished_at": _iso(finished_at),
                "duration_ms": duration_ms,
            })

        itin_json = None
        itin_md = None
        if itin_row:
            itin_json = _as_json(itin_row[0])
            itin_md = itin_row[1]

        return self.json_response({
            "status": status,
            "destination": destination,
            "start_date": _iso_date(start_date),
            "num_days": num_days,
            "current_node": current_node,
            "critic_round": critic_round,
            "trace": trace,
            "itinerary_json": itin_json,
            "markdown": itin_md,
        })

    def _delete(self):
        body = self._body()
        if body is None:
            return

        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return self.json_response({"detail": "Missing session_id."}, 400)

        _ensure_tables()
        with get_conn() as conn:
            with conn.cursor() as cur:
                # CASCADE wipes pins, trace, itinerary. planner_leg_cache is
                # shared and intentionally NOT deleted.
                cur.execute(
                    "DELETE FROM planner_sessions WHERE session_id = %s",
                    (session_id,),
                )
            conn.commit()

        return self.json_response({"status": "deleted"})

    # ------------------------------------------------------------------ private

    def _build_ctx(self, session_id, destination, start_date, num_days,
                    raw_input, current_node, critic_round) -> dict:
        """Build the ctx dict, rehydrating prior node outputs from the trace."""
        # Parse raw_input (JSONB comes back as dict from psycopg).
        payload = raw_input if isinstance(raw_input, dict) else {}
        if not payload and isinstance(raw_input, str):
            try:
                payload = json.loads(raw_input)
            except (ValueError, TypeError):
                payload = {}

        ctx = {
            "session_id": session_id,
            "destination": destination,
            "start_date": start_date if isinstance(start_date, str) else (
                start_date.isoformat() if start_date else ""
            ),
            "num_days": num_days,
            "payload": payload,
            "pins": [],
            "research": [],
            "legs": [],
            "schedule": None,
            "critic": None,
            "alternatives": None,
            "itinerary": None,
            "critic_round": critic_round or 0,
            "current_node": current_node,
            "errors": [],
        }

        # Rehydrate from the last ok trace row's _ctx snapshot.
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT output_json FROM planner_trace
                    WHERE session_id = %s AND status = 'ok'
                    ORDER BY trace_id DESC
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
            conn.commit()

        for r in rows:
            out = r[0]
            if out is None:
                continue
            # output_json may be dict (psycopg jsonb) or str.
            if isinstance(out, str):
                try:
                    out = json.loads(out)
                except (ValueError, TypeError):
                    continue
            if not isinstance(out, dict):
                continue
            saved_ctx = out.get("_ctx")
            if isinstance(saved_ctx, dict):
                for key in _CTX_KEYS:
                    if key in saved_ctx and saved_ctx[key] is not None:
                        # Don't overwrite pins from an empty _ctx (pins come
                        # from the ingest node and live in ctx, not in the
                        # node's return value).
                        if key == "pins" and not saved_ctx[key]:
                            continue
                        ctx[key] = saved_ctx[key]
                # Take the first (most recent) ok row's _ctx — but we want
                # the LATEST state. Since we ordered DESC, the first row
                # with a _ctx is the latest. Break after merging.
                break

        return ctx

    def _persist_itinerary(self, session_id: str, trace_row: dict, ctx: dict):
        """Upsert the compiled itinerary into planner_itinerary."""
        itinerary_json = ctx.get("itinerary")
        # The compiler wrapper's return value carries markdown.
        node_output = trace_row.get("output", {})
        markdown = ""
        if isinstance(node_output, dict):
            md = node_output.get("markdown")
            if isinstance(md, str) and md:
                markdown = md
            else:
                print(
                    "planner: compiler node ran but markdown was empty; "
                    "persisting with empty markdown"
                )

        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO planner_itinerary
                            (session_id, itinerary_json, markdown)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET
                            itinerary_json = EXCLUDED.itinerary_json,
                            markdown = EXCLUDED.markdown
                        """,
                        (
                            session_id,
                            json.dumps(itinerary_json) if not isinstance(itinerary_json, str) else itinerary_json,
                            markdown,
                        ),
                    )
                conn.commit()
        except Exception as exc:
            print(f"planner: itinerary persist failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Registry builder — lazy imports so the module loads even if some
# downstream modules (planner_agents) are momentarily absent.
# ---------------------------------------------------------------------------
def build_registry() -> dict:
    """Return {node_name: callable(ctx) -> dict} with _ctx-injecting wrappers."""
    import services.planner_ingest as _ingest
    import services.planner_logistics as _logistics
    import services.planner_scheduler as _scheduler

    # --- planner_agents (guarded) ---
    _scout = _critic = _alternatives = _compiler = None
    try:
        from services import planner_agents
        _scout = planner_agents.run_scout
        _critic = planner_agents.run_critic
        _alternatives = planner_agents.run_alternatives
        _compiler = planner_agents.run_compiler
    except ImportError:
        pass

    def _wrap(fn):
        """Wrap a node function so its return dict gets a _ctx snapshot.

        The snapshot is taken *before* the result is mutated so that nodes
        which assign their return value into ctx (e.g. run_critic sets
        ctx["critic"] = output) don't create a circular reference.
        """
        def _wrapped(ctx: dict) -> dict:
            result = fn(ctx)
            if not isinstance(result, dict):
                result = {"result": result}
            # Build a JSON-safe snapshot — deepcopy via JSON round-trip
            # breaks any shared references between result and ctx values.
            raw_snapshot = {k: ctx.get(k) for k in _CTX_KEYS}
            try:
                snapshot = json.loads(json.dumps(raw_snapshot, default=str))
            except (TypeError, ValueError):
                # Fall back: shallow copies without _ctx contamination.
                snapshot = {
                    k: _shallow_copy_safe(ctx.get(k)) for k in _CTX_KEYS
                }
            result["_ctx"] = snapshot
            return result
        return _wrapped

    def _run_scheduler_adapter(ctx: dict) -> dict:
        """Adapt ctx-based calling to planner_scheduler.run_schedule signature.

        run_schedule takes (pins, legs, start_date, num_days) — explicit args,
        not ctx. We also merge research fields into the pin dicts so the
        scheduler has opening_hours, dwell_minutes, category, etc.
        """
        research = ctx.get("research") or []
        pins = ctx.get("pins") or []

        # If we have research, merge research fields into pins by pin_id.
        if research:
            research_map = {}
            for r in research:
                pid = r.get("pin_id")
                if pid:
                    research_map[pid] = r
            merged_pins = []
            for pin in pins:
                pid = pin.get("pin_id")
                if pid and pid in research_map:
                    r = research_map[pid]
                    merged_pin = dict(pin)
                    for field in (
                        "opening_hours", "dwell_minutes", "category",
                        "neighborhood", "booking_required", "tip",
                        "hours_verified",
                    ):
                        if field in r:
                            merged_pin[field] = r[field]
                    merged_pins.append(merged_pin)
                else:
                    merged_pins.append(pin)
            sched_pins = merged_pins
        else:
            sched_pins = pins

        # Unresolved pins have no coordinates - they cannot be routed or
        # slotted sanely. Skip them with an explicit repair note instead of
        # letting None coordinates poison clustering and ordering.
        resolved_sched = []
        skipped_unresolved = []
        for p in sched_pins:
            if p.get("lat") is not None and p.get("lng") is not None:
                resolved_sched.append(p)
            else:
                skipped_unresolved.append(p.get("name") or p.get("raw_input") or "?")
        sched_pins = resolved_sched

        legs = ctx.get("legs") or []
        start_date = ctx.get("start_date", "")
        num_days = ctx.get("num_days", 2)

        schedule = _scheduler.run_schedule(sched_pins, legs, start_date, num_days)
        ctx["schedule"] = schedule
        for name in skipped_unresolved:
            schedule.setdefault("repairs", []).append(
                f"skipped unresolved pin {name!r} (could not geocode) - not scheduled"
            )
        return {
            "days": len(schedule.get("days", [])),
            "unplaced": len(schedule.get("unplaced", [])),
            "repairs": schedule.get("repairs", []),
            "stats": schedule.get("stats", {}),
            "skipped_unresolved": skipped_unresolved,
        }

    registry = {
        "ingest": _wrap(_ingest.run_ingest),
        "logistics": _wrap(_logistics.run_logistics),
        "scheduler": _wrap(_run_scheduler_adapter),
    }

    # Agent nodes — guard against missing module.
    if _scout is not None:
        registry["scout"] = _wrap(_scout)
    else:
        def _no_scout(ctx: dict) -> dict:
            raise RuntimeError("scout stage unavailable (planner_agents not found)")
        registry["scout"] = _wrap(_no_scout)

    if _critic is not None:
        registry["critic"] = _wrap(_critic)
    else:
        def _no_critic(ctx: dict) -> dict:
            raise RuntimeError("critic stage unavailable (planner_agents not found)")
        registry["critic"] = _wrap(_no_critic)

    if _alternatives is not None:
        registry["alternatives"] = _wrap(_alternatives)
    else:
        def _no_alternatives(ctx: dict) -> dict:
            raise RuntimeError("alternatives stage unavailable (planner_agents not found)")
        registry["alternatives"] = _wrap(_no_alternatives)

    if _compiler is not None:
        registry["compiler"] = _wrap(_compiler)
    else:
        def _no_compiler(ctx: dict) -> dict:
            raise RuntimeError("compiler stage unavailable (planner_agents not found)")
        registry["compiler"] = _wrap(_no_compiler)

    return registry


# ---------------------------------------------------------------------------
# Small JSON / datetime helpers
# ---------------------------------------------------------------------------
def _as_json(val):
    """Parse a value that might be a JSON string or already parsed.

    psycopg returns JSONB columns as native Python objects, but defensive
    handling covers the str case (e.g. if the column was text).
    """
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (ValueError, TypeError):
            return val
    return val


def _shallow_copy_safe(val):
    """Shallow copy a dict/list, stripping any '_ctx' key to break cycles."""
    if isinstance(val, dict):
        return {k: v for k, v in val.items() if k != "_ctx"}
    if isinstance(val, list):
        return list(val)
    return val


def _iso(dt) -> str | None:
    """Safely convert a datetime to ISO string."""
    if dt is None:
        return None
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


def _iso_date(d) -> str | None:
    """Safely convert a date/date to ISO string."""
    if d is None:
        return None
    if hasattr(d, "isoformat"):
        return d.isoformat()
    return str(d)
