"""Mr. Bounce — Trip Orchestrator graph runner (Reasoner Crew v2).

Loads the declarative graph from ``planner_graph.yaml`` (repo root), validates
it, and advances one node per call (stage-per-request pattern).  The runner
never imports the real agent/tool implementations — ``api/planner.py`` wires a
registry of ``dict[str, Callable[[dict], dict]]`` (agent nodes) and a tools
dict (tool functions owned by agents) so the runner stays testable with
trivial fakes.

Routing rules (centralised here):
  * terminal node                -> done (current_node = None).
  * back marker (``_back_to``)   -> if the previous node routed over an edge
      with ``back: <node>``, the consulted node returns to that node and the
      marker is cleared.  This is how the Reasoner's consult / re-research
      bounces come back for one more draft.
  * conditional edge (``when``)  -> the node's output dict is inspected; the
      first edge whose ``when`` key is truthy AND whose round budget
      (``counter`` < ``max_rounds``) is not exhausted wins.  Taking the edge
      increments the counter and, if the edge carries ``back``, sets the back
      marker.  When every conditional edge is exhausted/skipped, the first
      edge WITHOUT a ``when`` is the fallback (proceed forward).
  * node-level ``conditional`` block (legacy) -> ``output[on] == pass_value``
      routes to pass_next else fail_next, bounded by max_reasoner_rounds on
      the critic_round counter (kept for backward compatibility with synthetic
      graphs; the production graph is edge-driven).
  * normal edge                  -> the ``to`` node.
  * A failed node routes forward (fallback edge / back marker), so the
    pipeline can still finish and the failure is surfaced in the trace/errors.
  * A failed terminal            -> done.

Tool rows: when an agent node declares ``tools``, the runner injects
``ctx["_call_tool"](name, **kwargs)``.  Every invocation emits its OWN trace
row with node_type "tool", node_name "{agent}.{tool}", a "parent" field naming
the owning agent, and the agent's current round.  Tool failures emit a failed
tool row and then propagate so the owning agent node fails cleanly.
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

# Round counters the runner may bump during routing.  These are merged back
# into the node output's ``_ctx`` snapshot so they survive the stage-per-request
# model without a DB schema change.
_ROUND_COUNTER_KEYS = (
    "reasoner_round",
    "critic_round",
    "re_research_round",
    "consult_round",
    "logistics_round",
)


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

    # Reasoner review-round bound (legacy key max_critic_rounds accepted).
    max_rounds = graph.get("max_reasoner_rounds", graph.get("max_critic_rounds"))
    if not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError(
            "graph: max_reasoner_rounds (or max_critic_rounds) must be an int >= 1"
        )
    graph["max_reasoner_rounds"] = max_rounds

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

        # Tools owned by an agent node (optional).
        tools = node.get("tools")
        if tools is not None:
            if not isinstance(tools, list) or not tools:
                raise ValueError(
                    f"graph: node {nname!r} 'tools' must be a non-empty list"
                )
            if not all(isinstance(t, str) and t for t in tools):
                raise ValueError(
                    f"graph: node {nname!r} 'tools' must be a list of non-empty strings"
                )

        # Round counter key for the node's trace rows (optional).
        round_key = node.get("round_key")
        if round_key is not None and (not isinstance(round_key, str) or not round_key):
            raise ValueError(
                f"graph: node {nname!r} 'round_key' must be a non-empty string"
            )

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
        for key in ("when", "counter", "back"):
            if key in edge and (not isinstance(edge[key], str) or not edge[key]):
                raise ValueError(
                    f"graph: edge {frm}->{to} {key!r} must be a non-empty string"
                )
        if "max_rounds" in edge and (
            not isinstance(edge["max_rounds"], int) or edge["max_rounds"] < 0
        ):
            raise ValueError(
                f"graph: edge {frm}->{to} 'max_rounds' must be an int >= 0"
            )
        if edge.get("back") is not None and edge["back"] not in node_names:
            raise ValueError(
                f"graph: edge {frm}->{to} 'back' references unknown node "
                f"{edge['back']!r}"
            )
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
        "has_reasoner": bool(ctx.get("reasoner")),
        "has_alternatives": bool(ctx.get("alternatives")),
        "has_itinerary": bool(ctx.get("itinerary")),
        "reasoner_round": ctx.get("reasoner_round", 0),
        "re_research_round": ctx.get("re_research_round", 0),
        "consult_round": ctx.get("consult_round", 0),
        "current_node": ctx.get("current_node"),
    }


# ---------------------------------------------------------------------------
# Tool dispatch — agents own tools; each invocation emits its own trace row
# ---------------------------------------------------------------------------

def _make_tool_caller(
    ctx: dict,
    agent_node: str,
    tool_names: list[str],
    tools: dict[str, Callable],
    sink: "InMemorySink | PostgresSink",
    round_key: str | None,
) -> Callable:
    """Return ``_call(name, **kwargs)`` bound to the given agent's tools.

    Each invocation runs the tool function against the same ctx, emits a
    tool trace row (node_type "tool", node_name "{agent}.{tool}", "parent"
    = agent), and returns the tool's output dict.  Tool failures emit a
    failed row and then propagate so the owning agent node fails cleanly.
    """
    def _call(name: str, **kwargs) -> dict:
        if name not in tool_names:
            raise KeyError(f"{agent_node}: unknown tool {name!r}")
        fn = tools.get(name)
        started_at = datetime.now(timezone.utc).isoformat()
        started_dt = datetime.now(timezone.utc)
        input_snap = _node_input_snapshot(ctx)
        if kwargs:
            input_snap["tool_args"] = {
                k: (v if isinstance(v, (int, float, str, bool, type(None)))
                    else f"<{type(v).__name__} n={len(v) if hasattr(v, '__len__') else '?'}>")
                for k, v in kwargs.items()
            }
        round_num = ctx.get(round_key, 0) if round_key else 0

        output: dict
        status: str
        error: str | None = None
        if fn is None:
            output = {"error": f"no implementation for tool {name!r}"}
            status = "failed"
            error = output["error"]
            ctx.setdefault("errors", []).append(error)
        else:
            try:
                output = fn(ctx, **kwargs)
                status = "ok"
            except Exception as exc:
                status = "failed"
                error = str(exc)
                traceback.print_exc()
                output = {"error": error}
                ctx.setdefault("errors", []).append(error)
                finished_dt = datetime.now(timezone.utc)
                row = {
                    "seq": len(sink) if hasattr(sink, "__len__") else 0,
                    "node_name": f"{agent_node}.{name}",
                    "node_type": "tool",
                    "round": round_num,
                    "status": status,
                    "input": input_snap,
                    "output": dict(output),
                    "error": error,
                    "parent": agent_node,
                    "started_at": started_at,
                    "finished_at": finished_dt.isoformat(),
                    "duration_ms": int((finished_dt - started_dt).total_seconds() * 1000),
                }
                sink.append(row)
                raise

        finished_dt = datetime.now(timezone.utc)
        # The row's output carries "parent" so the owning agent is visible
        # through the status endpoint (output_json is persisted); the agent
        # itself receives the clean output dict.
        trace_output = dict(output)
        trace_output["parent"] = agent_node
        row = {
            "seq": len(sink) if hasattr(sink, "__len__") else 0,
            "node_name": f"{agent_node}.{name}",
            "node_type": "tool",
            "round": round_num,
            "status": status,
            "input": input_snap,
            "output": trace_output,
            "error": error,
            "parent": agent_node,
            "started_at": started_at,
            "finished_at": finished_dt.isoformat(),
            "duration_ms": int((finished_dt - started_dt).total_seconds() * 1000),
        }
        sink.append(row)
        return output

    return _call


# ---------------------------------------------------------------------------
# advance — run exactly one node (stage-per-request)
# ---------------------------------------------------------------------------

def _fallback_edge(edges: list[dict]) -> dict:
    """The first edge without a ``when`` condition (else the first edge)."""
    for edge in edges:
        if not edge.get("when"):
            return edge
    return edges[0]


def _route(
    node_name: str,
    output: dict,
    status: str,
    ctx: dict,
    graph: dict,
) -> str | None:
    """Compute the next node after ``node_name`` ran, mutating counters/back.

    Order of precedence:
      1. terminal node -> None
      2. back marker (``_back_to``) -> return to that node, clear marker
      3. legacy node-level ``conditional`` block (pass_value/fail_next)
      4. conditional edges (``when`` + bounded ``counter``), in YAML order
      5. fallback edge (first without ``when``)
    """
    node_map: dict[str, dict] = graph["_node_map"]
    conditional: dict[str, dict] = graph["_conditional"]
    outgoing: dict[str, list[dict]] = graph["_outgoing"]
    max_rounds: int = graph["max_reasoner_rounds"]

    node_def = node_map[node_name]
    if node_def.get("terminal"):
        return None

    # 2. Back marker — a consulted/throw-back node returns to its owner.
    back_target = ctx.get("_back_to")
    if back_target is not None:
        ctx.pop("_back_to", None)
        return back_target

    # 3. Legacy node-level conditional block (kept for synthetic graphs).
    if node_name in conditional:
        cond = conditional[node_name]
        if status == "failed":
            return cond["pass_next"]
        verdict = output.get(cond["on"])
        if verdict == cond["pass_value"]:
            return cond["pass_next"]
        current_round = ctx.get("critic_round", 0)
        if current_round >= max_rounds - 1:
            ctx.setdefault("errors", []).append(
                f"{node_name}: max rounds reached; proceeding with issues"
            )
            return cond["pass_next"]
        ctx["critic_round"] = current_round + 1
        return cond["fail_next"]

    # 4. Conditional edges — first match with budget wins.
    if status == "ok":
        for edge in outgoing[node_name]:
            when_key = edge.get("when")
            if not when_key:
                continue
            if not bool(output.get(when_key)):
                continue
            counter = edge.get("counter") or f"{when_key}_round"
            cap = edge.get("max_rounds", 1)
            used = ctx.get(counter, 0)
            if used >= cap:
                continue  # budget exhausted -> try the next edge
            ctx[counter] = used + 1
            if edge.get("back"):
                ctx["_back_to"] = edge["back"]
            return edge["to"]

    # 5. Fallback edge (forward progress).
    return _fallback_edge(outgoing[node_name])["to"]


def _persist_round_counters(output: dict, ctx: dict) -> dict:
    """Refresh the node output's ``_ctx`` snapshot with routing counters.

    The api layer snapshots ``_CTX_KEYS`` at the end of the node function,
    BEFORE the runner's routing mutates counters, so the runner merges the
    final counter values back into ``_ctx`` to survive the next request.
    """
    if isinstance(output, dict):
        snap = output.get("_ctx")
        if isinstance(snap, dict):
            for key in _ROUND_COUNTER_KEYS:
                if key in ctx:
                    snap[key] = ctx.get(key)
    return output


def advance(
    ctx: dict,
    registry: dict[str, Callable[[dict], dict]],
    sink: "InMemorySink | PostgresSink",
    graph: dict | None = None,
    tools: dict[str, Callable] | None = None,
) -> dict:
    """Run exactly one graph node and return its trace row.

    The node named by ``ctx["current_node"]`` is executed (initialised to the
    graph entry if absent).  After the run the routing logic updates
    ``ctx["current_node"]`` to the next node (or ``None`` when done) and
    embeds ``next_node`` in the trace row's ``output`` dict.

    ``tools`` maps tool names to ``callable(ctx, **kwargs)`` functions that the
    agent nodes invoke through the injected ``ctx["_call_tool"]`` dispatcher.
    """
    if graph is None:
        graph = load_graph()

    node_map: dict[str, dict] = graph["_node_map"]
    outgoing: dict[str, list[dict]] = graph["_outgoing"]
    max_rounds: int = graph["max_reasoner_rounds"]

    # Initialise current_node if missing.
    if ctx.get("current_node") is None:
        ctx["current_node"] = graph["entry"]

    node_name = ctx["current_node"]
    node_def = node_map[node_name]
    is_terminal = bool(node_def.get("terminal"))

    # --- inject tool dispatcher for agent nodes that own tools -------------
    tool_names = node_def.get("tools") or []
    if tool_names:
        round_key = node_def.get("round_key")
        ctx["_call_tool"] = _make_tool_caller(
            ctx, node_name, tool_names, tools or {}, sink, round_key
        )
        ctx["_max_reasoner_rounds"] = max_rounds
    else:
        ctx.pop("_call_tool", None)

    # --- trace row scaffold -----------------------------------------------
    started_at = datetime.now(timezone.utc).isoformat()
    input_snap = _node_input_snapshot(ctx)
    rkey = node_def.get("round_key")
    if rkey:
        round_num = ctx.get(rkey, 0)
    elif node_def.get("conditional"):
        round_num = ctx.get("critic_round", 0)  # legacy conditional nodes
    else:
        round_num = 0

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
    next_node = _route(node_name, output, status, ctx, graph)
    ctx["current_node"] = next_node

    # Refresh _ctx counters (see _persist_round_counters) before persisting.
    output = _persist_round_counters(output, ctx)

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
    reasoner_round = ctx.get("reasoner_round", 0)
    return {
        "current_node": current,
        "reasoner_round": reasoner_round,
        "critic_round": reasoner_round,  # back-compat alias
        "re_research_round": ctx.get("re_research_round", 0),
        "consult_round": ctx.get("consult_round", 0),
        "is_completed": current is None,
        "errors": list(ctx.get("errors", [])),
    }
