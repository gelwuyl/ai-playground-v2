# -*- coding: utf-8 -*-
"""Deterministic unit tests for InterviewService (fake DB + fake LLM)."""
from __future__ import annotations

import re
from typing import Any, Dict

import pytest

# The service lives one level up (package import).
from services import interview_service as svc
from services.interview_service import InterviewService

# ----------------------------------------------------------------------
# In-memory fake DB state
# ----------------------------------------------------------------------
_STATE: Dict[str, Dict] = {
    "sessions": {},
    "roles": {},
    "questions": {},
}


def _uid() -> str:
    return "fake-id"


class _FakeCursor:
    """Mimics a psycopg Cursor well enough for the service's SQL.

    Recognises the SQL statements the service issues and returns canned
    results.  Unknown SELECTs return empty; INSERT/UPDATE are recorded in
    _STATE.  The handlers below are aligned to the exact SQL the service
    emits (column order, literal values, param positions).
    """

    def __init__(self, conn):
        self._conn = conn
        self.last_sql = None
        self.last_params = None
        self._results = None

    def execute(self, sql, params=None):
        self.last_sql = sql
        self.last_params = tuple(params) if params else ()
        up = (sql or "").upper()
        p = list(self.last_params)

        # ---- INSERT INTO INTERVIEW_PREP_SESSIONS ----
        if "INSERT INTO INTERVIEW_PREP_SESSIONS" in up:
            _sid = p[0]  # service-generated session_id passed in
            _STATE["sessions"][_sid] = {
                "session_id": _sid,
                "created_at": "2026-09-01T00:00:00+00:00",
                "target_role": p[1],
                "target_companies": p[2],
                "user_resume": p[3],
                "current_stage": p[4] if len(p) > 4 else "prospecting",
                "final_guide": p[5] if len(p) > 5 else None,
                "researching_round": int(p[6]) if len(p) > 6 and p[6] is not None else 0,
            }
            self._results = [(_sid,)]

        # ---- INSERT INTO INTERVIEW_PREP_ROLES ----
        elif "INSERT INTO INTERVIEW_PREP_ROLES" in up:
            _rid = _uid()
            _STATE["roles"][_rid] = {
                "role_id": _rid,
                "session_id": p[0],
                "company_name": p[1],
                "job_title": p[2],
                "responsibilities": p[3],
                "required_skills": p[4],
            }
            self._results = [(_rid,)]

        # ---- INSERT INTO INTERVIEW_PREP_QUESTIONS ----
        # Two patterns:
        #  A) Service path:  VALUES (%s, %s, %s)              params=(role_id, question, type)
        #  B) Test helper:   VALUES ((SELECT role_id FROM ... WHERE session_id=%s), %s, %s)
        #     params=(session_id, question, type) — p[0] is session_id, NOT role_id
        elif "INSERT INTO INTERVIEW_PREP_QUESTIONS" in up:
            _qid = _uid()
            if "SELECT ROLE_ID FROM" in up:
                # Test-helper subquery pattern — look up the real role_id.
                _sid = p[0]
                _role_id = None
                if sql:
                    _m = re.search(
                        r"WHERE\s+session_id\s*=\s*%s"
                        r"(?:\s+AND\s+company_name\s*=\s*'([^']+?)')?",
                        sql,
                        re.IGNORECASE,
                    )
                    if _m:
                        _company = _m.group(1)
                        for r in _STATE["roles"].values():
                            if r["session_id"] == _sid and (
                                _company is None or r["company_name"] == _company
                            ):
                                _role_id = r["role_id"]
                                break
                _role_id = _role_id or _uid()
            else:
                _role_id = p[0]
            _STATE["questions"][_qid] = {
                "question_id": _qid,
                "role_id": _role_id,
                "question_text": p[1],
                "question_type": p[2],
            }
            self._results = [(_qid,)]

        # ---- SELECT CURRENT_STAGE FROM ----
        elif up.startswith("SELECT CURRENT_STAGE FROM"):
            _sid = p[0] if p else None
            row = _STATE["sessions"].get(_sid)
            self._results = [(row["current_stage"],)] if row else []

        # ---- SELECT CURRENT_STAGE, FINAL_GUIDE, RESEARCHING_ROUND FROM ----
        elif "SELECT CURRENT_STAGE, FINAL_GUIDE, RESEARCHING_ROUND FROM" in up:
            _sid = p[0] if p else None
            row = _STATE["sessions"].get(_sid)
            if row:
                self._results = [(row["current_stage"], row["final_guide"], row["researching_round"])]
            else:
                self._results = []

        # ---- SELECT TARGET_ROLE, TARGET_COMPANIES FROM ----
        elif up.startswith("SELECT TARGET_ROLE, TARGET_COMPANIES"):
            _sid = p[0] if p else None
            row = _STATE["sessions"].get(_sid)
            self._results = [(row["target_role"], row["target_companies"])] if row else []

        # ---- SELECT role_id, company_name, job_title, responsibilities, required_skills ----
        elif "SELECT ROLE_ID, COMPANY_NAME, JOB_TITLE, RESPONSIBILITIES, REQUIRED_SKILLS" in up:
            _sid = p[0] if p else None
            self._results = [
                (r["role_id"], r["company_name"], r["job_title"], r["responsibilities"], r["required_skills"])
                for r in _STATE["roles"].values()
                if r["session_id"] == _sid
            ]

        # ---- SELECT role_id, company_name, job_title (3-col, for _persist_questions lookup) ----
        elif "SELECT ROLE_ID, COMPANY_NAME, JOB_TITLE FROM" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            self._results = [
                (r["role_id"], r["company_name"], r["job_title"])
                for r in _STATE["roles"].values()
                if r["session_id"] == _sid
            ]

        # ---- SELECT COUNT(*) FROM INTERVIEW_PREP_ROLES WHERE SESSION_ID ----
        elif "SELECT COUNT(*) FROM INTERVIEW_PREP_ROLES" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            self._results = [(sum(1 for r in _STATE["roles"].values() if r["session_id"] == _sid),)]

        # ---- SELECT COUNT(*) FROM INTERVIEW_PREP_QUESTIONS Q JOIN ----
        elif "SELECT COUNT(*) FROM INTERVIEW_PREP_QUESTIONS Q JOIN" in up:
            self._results = [(len(_STATE["questions"]),)]

        # ---- SELECT company_name, job_title (2-col) ----
        elif "SELECT COMPANY_NAME, JOB_TITLE FROM" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            self._results = [
                (r["company_name"], r["job_title"])
                for r in _STATE["roles"].values()
                if r["session_id"] == _sid
            ]

        # ---- SELECT responsibilities, required_skills (2-col) ----
        elif "SELECT RESPONSIBILITIES, REQUIRED_SKILLS FROM" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            self._results = [
                (r["responsibilities"], r["required_skills"])
                for r in _STATE["roles"].values()
                if r["session_id"] == _sid
            ]

        # ---- UPDATE INTERVIEW_PREP_SESSIONS SET CURRENT_STAGE ----
        # Two patterns:
        #  A) Standalone:  UPDATE ... SET current_stage = 'X' WHERE session_id = %s
        #     params=(session_id,)  — stage is a SQL literal
        #  B) Combined:    UPDATE ... SET final_guide = %s, current_stage = 'completed' WHERE session_id = %s
        #     params=(guide, session_id)  — stage is a SQL literal, session_id is p[1]
        elif "UPDATE INTERVIEW_PREP_SESSIONS SET CURRENT_STAGE" in up:
            if "FINAL_GUIDE" in up:
                # Combined update: params = (guide, session_id)
                _sid = p[1] if len(p) > 1 else None
                if _sid in _STATE["sessions"]:
                    _STATE["sessions"][_sid]["current_stage"] = "completed"
            else:
                # Standalone update: params = (session_id,), stage literal in SQL
                _sid = p[0] if p else None
                _stage = None
                if sql:
                    _m = re.search(r"CURRENT_STAGE\s*=\s*'([^']+)'", sql, re.IGNORECASE)
                    if _m:
                        _stage = _m.group(1)
                if _sid in _STATE["sessions"] and _stage:
                    _STATE["sessions"][_sid]["current_stage"] = _stage
            self._results = [()]

        # ---- UPDATE INTERVIEW_PREP_SESSIONS SET FINAL_GUIDE ----
        # (Standalone FINAL_GUIDE updates are not emitted by the service;
        #  the combined form is handled by the CURRENT_STAGE handler above.)
        elif "UPDATE INTERVIEW_PREP_SESSIONS SET FINAL_GUIDE" in up:
            _sid = p[0] if p else None
            guide = p[1] if len(p) > 1 else None
            if _sid in _STATE["sessions"]:
                _STATE["sessions"][_sid]["final_guide"] = guide
            self._results = [()]

        # ---- UPDATE INTERVIEW_PREP_SESSIONS SET RESEARCHING_ROUND ----
        # Two patterns:
        #  A) Backfill:  UPDATE ... SET researching_round = 0 WHERE session_id = %s
        #     params=(session_id,)  — p[0] is session_id (UUID string)
        #  B) Increment: UPDATE ... SET researching_round = %s WHERE session_id = %s
        #     params=(round_value, session_id)  — p[0] = int, p[1] = session_id
        elif "UPDATE INTERVIEW_PREP_SESSIONS SET RESEARCHING_ROUND" in up:
            if len(p) >= 2 and p[1] is not None:
                # Pattern B: two params
                _sid = p[1]
                val = int(p[0]) if p[0] is not None else 0
            else:
                # Pattern A: one param (session_id)
                _sid = p[0] if p else None
                val = 0
            if _sid in _STATE["sessions"]:
                _STATE["sessions"][_sid]["researching_round"] = val
            self._results = [()]

        # ---- SELECT RESEARCHING_ROUND FROM ----
        elif "SELECT RESEARCHING_ROUND FROM" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            row = _STATE["sessions"].get(_sid)
            self._results = [(row["researching_round"],)] if row else []

        # ---- SELECT USER_RESUME FROM ----
        elif "SELECT USER_RESUME FROM" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            row = _STATE["sessions"].get(_sid)
            self._results = [(row["user_resume"],)] if row else []

        # ---- SELECT r.company_name, r.job_title, q.question_text, q.question_type ----
        # Writing-stage JOIN (4 columns incl. question_type).
        elif "SELECT R.COMPANY_NAME, R.JOB_TITLE, Q.QUESTION_TEXT, Q.QUESTION_TYPE" in up:
            _sid = p[0] if p else None
            rows = []
            for q in _STATE["questions"].values():
                r = _STATE["roles"].get(q["role_id"])
                if r and r["session_id"] == _sid:
                    rows.append((r["company_name"], r["job_title"], q["question_text"], q["question_type"]))
            rows.sort(key=lambda x: (x[0], x[1], x[2]))
            self._results = rows

        # ---- SELECT r.company_name, r.job_title, q.question_text ----
        # Researching-stage "avoid duplicates" JOIN (3 columns, NO question_type).
        elif "SELECT R.COMPANY_NAME, R.JOB_TITLE, Q.QUESTION_TEXT FROM" in up and "JOIN" in up and "WHERE R.SESSION_ID" in up:
            _sid = p[0] if p else None
            rows = []
            for q in _STATE["questions"].values():
                r = _STATE["roles"].get(q["role_id"])
                if r and r["session_id"] == _sid:
                    rows.append((r["company_name"], r["job_title"], q["question_text"]))
            rows.sort(key=lambda x: (x[0], x[1], x[2]))
            self._results = rows

        # ---- SELECT role_id FROM INTERVIEW_PREP_ROLES WHERE SESSION_ID ----
        elif "SELECT ROLE_ID FROM INTERVIEW_PREP_ROLES" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            self._results = [(r["role_id"],) for r in _STATE["roles"].values() if r["session_id"] == _sid]

        # ---- SELECT COUNT(*) FROM INTERVIEW_PREP_SESSIONS WHERE SESSION_ID ----
        elif "SELECT COUNT(*) FROM INTERVIEW_PREP_SESSIONS" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            self._results = [(1 if _sid in _STATE["sessions"] else 0,)]

        # ---- Unknown SELECT → empty ----
        elif up.strip().startswith("SELECT"):
            self._results = []

        # ---- Everything else (non-SELECT) → empty row ----
        else:
            self._results = [()]

    def fetchone(self):
        return self._results[0] if self._results else None

    def fetchall(self):
        return list(self._results) if self._results else []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConnection:
    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_get_conn():
    return _FakeConnection()


def _reset_state():
    _STATE["roles"].clear()
    _STATE["questions"].clear()
    _STATE["sessions"].clear()
