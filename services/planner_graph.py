"""Mr. Bounce — Trip Orchestrator graph runner.

Loads the declarative graph from ``planner_graph.yaml`` (repo root), validates
it, and advances one node per call (stage-per-request pattern).  The runner
never imports the real agent/tool implementations — ``api/planner.py`` wires a
registry of ``dict[str, Callable[[dict], dict]]`` so the runner stays testable
with trivial fakes.

Routing rules (centralised here):
  * terminal node  -> done (current_node = None).
  * conditional node (critic): inspect output[conditional.on];
      == pass_value -> pass_next;
      else          -> fail_next and critic_round += 1.
      Bound: scheduler runs are allowed at critic_round 0, 1, 2.  When the
      critic fails at critic_round 2 (the 3rd critic run) the graph
      force-proceeds to pass_next instead of looping again.
  * normal edge     -> the ``to`` node.
  * A failed node routes the same way as a successful one, EXCEPT a failed
    critic routes forward to pass_next (so the pipeline can still finish).
  * A failed terminal -> done.
"""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from typing import Callable

import yaml

# ---------------------------------------------------------------------------
# Graph path — resolved relative to the repo root (one level up from this file).
# ---------------------------------------------------------------------------
GRAPH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "planner_graph.yaml",
)

# ---------------------------------------------------------------------------
# Load + validate
# ---------------------------------------------------------------------------

_VALID_TYPES = {"agent", "tool"}


def load_graph(path: str | None = None) -> dict:
    """Load and validate the YAML graph definition.

    Returns the parsed graph dict.  Raises ``ValueError`` with a clear message
    on any structural problem.
    """
    resolved = path or GRAPH_PATH
    with open(resolved, "r", encoding="utf-8") as fh:
        graph = yaml.safe_load(fh)

    # --- top-level keys ---------------------------------------------------
    name = graph.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("graph: missing or non-string 'name'")

    entry = graph.get("entry")
    if not entry or not isinstance(entry, str):
        raise ValueError("graph: missing or non-string 'entry'")

    max_rounds = graph.get("max_critic_rounds")
    if not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("graph: max_critic_rounds must be an int >= 1")

    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("graph: 'nodes' must be a non-empty list")

    # --- node table (pass 1: collect names + validate types) ------------
    node_names: set[str] = set()
    terminal_count = 0
    conditional_nodes: dict[str, dict] = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("graph: each node must be a mapping")
        nname = node.get("name")
        if not nname or not isinstance(nname, str):
            raise ValueError("graph: node missing 'name'")
        if nname in node_names:
            raise ValueError(f"graph: duplicate node name {nname!r}")
        node_names.add(nname)

        ntype = node.get("type")
        if ntype not in _VALID_TYPES:
            raise ValueError(
                f"graph: node {nname!r} has unknown type {ntype!r}; "
                f"must be one of {sorted(_VALID_TYPES)}"
            )

        if node.get("terminal"):
            terminal_count += 1

        cond = node.get("conditional")
        if cond is not None:
            if not isinstance(cond, dict):
                raise ValueError(f"graph: node {nname!r} conditional must be a mapping")
            conditional_nodes[nname] = cond

    # --- pass 2: validate conditional refs (now all names are known) -----
    for cname, cond in conditional_nodes.items():
        for key in ("on", "pass_value", "pass_next", "fail_next"):
            if key not in cond:
                raise ValueError(
                    f"graph: node {cname!r} conditional missing {key!r}"
                )
        for ref in ("pass_next", "fail_next"):
            if cond[ref] not in node_names:
                raise ValueError(
                    f"graph: node {cname!r} conditional {ref!r} "
                    f"references unknown node {cond[ref]!r}"
                )

    if entry not in node_names:
        raise ValueError(f"graph: entry {entry!r} is not a known node")

    if terminal_count != 1:
        raise ValueError(
            f"graph: expected exactly one terminal node, found {terminal_count}"
        )

    # --- edges ------------------------------------------------------------
    edges = graph.get("edges")
    if not isinstance(edges, list):
        raise ValueError("graph: 'edges' must be a list")

    # Build outgoing edge index: node_name -> list of edge dicts
    outgoing: dict[str, list[dict]] = {n: [] for n in node_names}
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("graph: each edge must be a mapping")
        frm = edge.get("from")
        to = edge.get("to")
        if frm not in node_names:
            raise ValueError(f"graph: edge from unknown node {frm!r}")
        if to not in node_names:
            raise ValueError(f"graph: edge to unknown node {to!r}")
        outgoing[frm].append(edge)

    # Every non-terminal node must have at least one outgoing edge.
    for node in nodes:
        nname = node["name"]
        if not node.get("terminal") and not outgoing[nname]:
            raise ValueError(f"graph: node {nname!r} has no outgoing edges")

    # Stash computed fields for the runner.
    graph["_node_map"] = {n["name"]: n for n in nodes}
    graph["_outgoing"] = outgoing
    graph["_conditional"] = conditional_nodes
    return graph


# ---------------------------------------------------------------------------
# Trace sinks
# ---------------------------------------------------------------------------

class InMemorySink(list):
    """Trivial list subclass that just appends trace rows."""

    def append(self, row: dict) -> None:  # noqa: D401
        list.append(self, row)


class PostgresSink:
    """Writes trace rows into the ``planner_trace`` table.

    DB imports are lazy so the module stays importable without a database.
    A trace failure must never kill the run — ``append`` catches all
    exceptions and prints a warning.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._ensured = False

    def append(self, row: dict) -> None:
        try:
            from services.database import get_conn  # lazy
            from services import planner_db           # lazy

            if not self._ensured:
                planner_db._ensure_tables()
                self._ensured = True

            # Row is mutated, so copy — the caller may reuse the dict.
            row = dict(row)
            with get_conn() as conn:
                with conn.cursor() as cur:
                    # A fresh sink is created per serverless invocation, so the
                    # in-memory seq from advance() restarts at 0 every request.
                    # Derive a true global order from the table instead.
                    cur.execute(
                        "SELECT COALESCE(MAX(seq), -1) + 1 FROM planner_trace WHERE session_id = %s",
                        (self.session_id,),
                    )
                    row["seq"] = cur.fetchone()[0]
                    cur.execute(
                        """
                        INSERT INTO planner_trace
                            (session_id, seq, node_name, node_type, round,
                             status, input_json, output_json, error,
                             started_at, finished_at, duration_ms)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            self.session_id,
                            row.get("seq"),
                            row.get("node_name"),
                            row.get("node_type"),
                            row.get("round", 0),
                            row.get("status"),
                            json.dumps(row.get("input")) if row.get("input") is not None else None,
                            json.dumps(row.get("output")) if row.get("output") is not None else None,
                            row.get("error"),
                            row.get("started_at"),
                            row.get("finished_at"),
                            row.get("duration_ms"),
                        ),
                    )
                conn.commit()
        except Exception as exc:  # never raise out
            print(f"PostgresSink: trace write failed: {exc}")


# ---------------------------------------------------------------------------
# Node input projection (small, JSON-serializable snapshot of ctx)
# ---------------------------------------------------------------------------

def _node_input_snapshot(ctx: dict) -> dict:
    """Return a small projection of the session context for the trace row.

    Keeps the trace lean — counts and flags only, no giant blobs.
    """
    return {
        "pins": len(ctx.get("pins") or []),
        "research": len(ctx.get("research") or []),
        "legs": len(ctx.get("legs") or []),
        "has_schedule": bool(ctx.get("schedule")),
        "has_critic": bool(ctx.get("critic")),
        "has_alternatives": bool(ctx.get("alternatives")),
        "has_itinerary": bool(ctx.get("itinerary")),
        "critic_round": ctx.get("critic_round", 0),
        "current_node": ctx.get("current_node"),
    }


# ---------------------------------------------------------------------------
# advance — run exactly one node (stage-per-request)
# ---------------------------------------------------------------------------

def advance(
    ctx: dict,
    registry: dict[str, Callable[[dict], dict]],
    sink: "InMemorySink | PostgresSink",
    graph: dict | None = None,
) -> dict:
    """Run exactly one graph node and return its trace row.

    The node named by ``ctx["current_node"]`` is executed (initialised to the
    graph entry if absent).  After the run the routing logic updates
    ``ctx["current_node"]`` to the next node (or ``None`` when done) and
    embeds ``next_node`` in the trace row's ``output`` dict.
    """
    if graph is None:
        graph = load_graph()

    node_map: dict[str, dict] = graph["_node_map"]
    conditional: dict[str, dict] = graph["_conditional"]
    outgoing: dict[str, list[dict]] = graph["_outgoing"]
    max_rounds: int = graph["max_critic_rounds"]

    # Initialise current_node if missing.
    if ctx.get("current_node") is None:
        ctx["current_node"] = graph["entry"]

    node_name = ctx["current_node"]
    node_def = node_map[node_name]
    is_conditional = node_name in conditional
    is_terminal = bool(node_def.get("terminal"))

    # --- trace row scaffold -----------------------------------------------
    started_at = datetime.now(timezone.utc).isoformat()
    input_snap = _node_input_snapshot(ctx)

    # The trace row is built incrementally and appended at the end.
    round_num = ctx.get("critic_round", 0)

    # --- run the node ------------------------------------------------------
    output: dict
    status: str
    error: str | None = None
    fn = registry.get(node_name)
    started_dt = datetime.now(timezone.utc)
    if fn is None:
        # No implementation wired — treat as a soft failure.
        output = {"error": f"no implementation for node {node_name!r}"}
        status = "failed"
        error = output["error"]
        ctx.setdefault("errors", []).append(error)
    else:
        try:
            output = fn(ctx)
            status = "ok"
        except Exception as exc:
            status = "failed"
            error = str(exc)
            traceback.print_exc()
            output = {"error": error}
            ctx.setdefault("errors", []).append(error)

    finished_dt = datetime.now(timezone.utc)
    duration_ms = int((finished_dt - started_dt).total_seconds() * 1000)

    # --- routing -----------------------------------------------------------
    next_node: str | None

    if is_terminal:
        next_node = None
    elif is_conditional:
        cond = conditional[node_name]
        # A failed conditional node routes forward to pass_next so the
        # pipeline can still finish.
        if status == "failed":
            next_node = cond["pass_next"]
        else:
            verdict = output.get(cond["on"])
            if verdict == cond["pass_value"]:
                next_node = cond["pass_next"]
            else:
                # Critic FAIL — would route to fail_next, but check the bound.
                current_round = ctx.get("critic_round", 0)
                if current_round >= max_rounds - 1:
                    # 3rd critic failure (round 2) — force-proceed.
                    next_node = cond["pass_next"]
                    ctx.setdefault("errors", []).append(
                        "critic: max rounds reached; proceeding with issues"
                    )
                else:
                    next_node = cond["fail_next"]
                    ctx["critic_round"] = current_round + 1
    else:
        # Normal edge: take the first (and usually only) outgoing edge.
        edges = outgoing[node_name]
        next_node = edges[0]["to"]

    ctx["current_node"] = next_node

    # --- build + emit trace row -------------------------------------------
    trace_output = dict(output)
    trace_output["next_node"] = next_node

    row = {
        "seq": len(sink) if hasattr(sink, "__len__") else 0,
        "node_name": node_name,
        "node_type": node_def["type"],
        "round": round_num,
        "status": status,
        "input": input_snap,
        "output": trace_output,
        "error": error,
        "started_at": started_at,
        "finished_at": finished_dt.isoformat(),
        "duration_ms": duration_ms,
    }
    sink.append(row)
    return row


# ---------------------------------------------------------------------------
# Helper — session status snapshot
# ---------------------------------------------------------------------------

def run_status(ctx: dict) -> dict:
    """Return a small status dict for the status endpoint."""
    current = ctx.get("current_node")
    return {
        "current_node": current,
        "critic_round": ctx.get("critic_round", 0),
        "is_completed": current is None,
        "errors": list(ctx.get("errors", [])),
    }
