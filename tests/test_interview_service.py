"""Deterministic tests for the Mr. Brave interview pipeline (v2).

All tests use small hand-built fixtures and injectable fakes — no network,
no real DB (an in-memory fake Connection/Cursor replaces InterviewService.get_conn),
no real LLM (a fake function replaces InterviewService.llm).

The service's LLM is InterviewService.llm (a class attribute, default: the real
OpenRouter call). Tests override it directly with a fake callable — no patching.
The service's DB is InterviewService.get_conn (a class attribute, default: the
real Postgres pool). Tests override it with a fake that mimics psycopg's
Connection.cursor() context-manager protocol.

Run: .venv/bin/python -m pytest tests/test_interview_service.py -q
"""

import os
import re
import json
import uuid

# Set env BEFORE importing interview_service (database.py reads DATABASE_URL at
# import time and will raise KeyError otherwise). Test env is harmless here.
os.environ["DATABASE_URL"] = "postgres://dummy:***@localhost/dummy"
os.environ["OPENROUTER_API_KEY"] = "test"

import pytest

from services.interview_service import InterviewService, PROSPECTOR_SYSTEM, RESEARCHER_SYSTEM, WRITER_SYSTEM


# ==============================================================================
# In-memory fake DB (replaces InterviewService.get_conn for test assertions)
# ==============================================================================

_STATE = {"roles": {}, "questions": {}, "sessions": {}}
_UID_COUNTER = 0


def _uid() -> str:
    global _UID_COUNTER
    _UID_COUNTER += 1
    return str(uuid.uuid4())


class _FakeCursor:
    """Mimics psycopg's Cursor context-manager protocol.

    Recognises the SQL statements the service issues and returns canned results.
    Unknown SELECTs return empty; INSERT/UPDATE are recorded in _STATE.
    """

    def __init__(self, conn):
        self._conn = conn
        self.last_sql = None
        self.last_params = None
        self._results = None

    def execute(self, sql, params=None):
        self.last_sql = sql
        self.last_params = tuple(params) if params else ()
        up = re.sub(r'\s+', ' ', (sql or "").upper())
        p = list(self.last_params)

        if "INSERT INTO INTERVIEW_PREP_SESSIONS" in up:
            # Service start_session only inserts 4 cols:
            #   (session_id, target_role, target_companies, user_resume)
            #   params = (session_id, target_role, target_companies, user_resume)
            _sid = p[0] if p else None
            _STATE["sessions"][_sid] = {
                "session_id": _sid,
                "created_at": "2026-09-01T00:00:00+00:00",
                "target_role": p[1] if len(p) > 1 else None,
                "target_companies": p[2] if len(p) > 2 else None,
                "user_resume": p[3] if len(p) > 3 else None,
                "current_stage": "prospecting",
                "final_guide": None,
                "researching_round": 0,
            }
            self._results = [(_sid,)]

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

        elif "INSERT INTO INTERVIEW_PREP_QUESTIONS" in up:
            _qid = _uid()
            # Two INSERT shapes:
            #  A) Service path:  VALUES (%s, %s, %s)
            #     params=(role_id, question_text, question_type)  -> p[0]=role_id
            #  B) Test helper:   VALUES ((SELECT role_id FROM INTERVIEW_PREP_ROLES
            #     WHERE session_id = %s [AND company_name = 'X']), %s, %s)
            #     params=(session_id, question_text, question_type)  -> p[0]=session_id
            if "SELECT ROLE_ID FROM" in up:
                _sid = p[0]
                _company = None
                if sql:
                    _m = re.search(r"AND\s+company_name\s*=\s*'([^']+)'", sql, re.IGNORECASE)
                    if _m:
                        _company = _m.group(1)
                _rid = None
                for r in _STATE["roles"].values():
                    if r["session_id"] == _sid and (_company is None or r["company_name"] == _company):
                        _rid = r["role_id"]
                        break
                _role_id = _rid if _rid is not None else _uid()
            else:
                _role_id = p[0]
            _STATE["questions"][_qid] = {
                "question_id": _qid,
                "role_id": _role_id,
                "question_text": p[1],
                "question_type": p[2],
            }
            self._results = [(_qid,)]

        elif up.startswith("SELECT CURRENT_STAGE FROM"):
            _sid = p[0] if p else None
            row = _STATE["sessions"].get(_sid)
            self._results = [(row["current_stage"],)] if row else []

        elif "SELECT CURRENT_STAGE, FINAL_GUIDE, RESEARCHING_ROUND FROM" in up:
            _sid = p[0] if p else None
            row = _STATE["sessions"].get(_sid)
            if row:
                self._results = [(row["current_stage"], row["final_guide"], row["researching_round"])]
            else:
                self._results = []

        elif up.startswith("SELECT TARGET_ROLE, TARGET_COMPANIES"):
            _sid = p[0] if p else None
            row = _STATE["sessions"].get(_sid)
            self._results = [(row["target_role"], row["target_companies"])] if row else []

        elif "SELECT ROLE_ID, COMPANY_NAME, JOB_TITLE, RESPONSIBILITIES, REQUIRED_SKILLS" in up:
            _sid = p[0] if p else None
            self._results = [
                (r["role_id"], r["company_name"], r["job_title"], r["responsibilities"], r["required_skills"])
                for r in _STATE["roles"].values()
                if r["session_id"] == _sid
            ]

        elif "SELECT ROLE_ID, COMPANY_NAME, JOB_TITLE FROM" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            self._results = [
                (r["role_id"], r["company_name"], r["job_title"])
                for r in _STATE["roles"].values()
                if r["session_id"] == _sid
            ]

        elif "SELECT COUNT(*) FROM INTERVIEW_PREP_ROLES" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            self._results = [(sum(1 for r in _STATE["roles"].values() if r["session_id"] == _sid),)]

        elif "SELECT COUNT(*) FROM INTERVIEW_PREP_QUESTIONS" in up and "JOIN" in up and "WHERE" in up:
            self._results = [(len(_STATE["questions"]),)]

        elif "SELECT COMPANY_NAME, JOB_TITLE FROM" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            self._results = [
                (r["company_name"], r["job_title"])
                for r in _STATE["roles"].values()
                if r["session_id"] == _sid
            ]

        elif "SELECT RESPONSIBILITIES, REQUIRED_SKILLS FROM" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            self._results = [
                (r["responsibilities"], r["required_skills"])
                for r in _STATE["roles"].values()
                if r["session_id"] == _sid
            ]

        elif "UPDATE INTERVIEW_PREP_SESSIONS SET CURRENT_STAGE" in up:
            # Two patterns:
            #  A) Standalone:  UPDATE ... SET current_stage = 'X' WHERE session_id = %s
            #     params=(session_id,)  — p[0] is session_id, stage is a SQL literal
            #  B) Combined:    UPDATE ... SET final_guide = %s, current_stage = 'completed' WHERE session_id = %s
            #     params=(guide, session_id)  — p[1] is session_id, stage is a SQL literal
            if "FINAL_GUIDE" in up:
                _sid = p[1] if len(p) > 1 else None
                if _sid in _STATE["sessions"]:
                    _STATE["sessions"][_sid]["current_stage"] = "completed"
            else:
                _sid = p[0] if p else None
                _stage = None
                if sql:
                    _m = re.search(r"CURRENT_STAGE\s*=\s*'([^']+)'", sql, re.IGNORECASE)
                    if _m:
                        _stage = _m.group(1)
                if _sid in _STATE["sessions"] and _stage:
                    _STATE["sessions"][_sid]["current_stage"] = _stage
            self._results = [()]

        elif "UPDATE INTERVIEW_PREP_SESSIONS SET FINAL_GUIDE" in up:
            # Combined UPDATE: SET final_guide = %s, current_stage = 'completed'
            #   params = (guide, session_id)  — p[0]=guide, p[1]=session_id
            _sid = p[1] if len(p) > 1 else None
            guide = p[0] if p else None
            if _sid in _STATE["sessions"]:
                _STATE["sessions"][_sid]["final_guide"] = guide
                _STATE["sessions"][_sid]["current_stage"] = "completed"
            self._results = [()]

        elif "UPDATE INTERVIEW_PREP_SESSIONS SET RESEARCHING_ROUND" in up:
            # Two patterns:
            #  A) Increment: UPDATE ... SET researching_round = %s WHERE session_id = %s
            #     params=(round_value, session_id)  — p[0]=int, p[1]=session_id
            #  B) Backfill:  UPDATE ... SET researching_round = 0 WHERE session_id = %s
            #     params=(session_id,)  — p[0]=session_id (UUID string), round is SQL literal
            if len(p) >= 2 and p[1] is not None:
                # Pattern A: two params — p[1] is the session_id
                _sid = p[1]
                val = int(p[0]) if p[0] is not None else 0
            else:
                # Pattern B: one param — p[0] is the session_id
                _sid = p[0] if p else None
                val = 0
                if sql:
                    _m = re.search(r"RESEARCHING_ROUND\s*=\s*(\d+)", sql, re.IGNORECASE)
                    if _m:
                        val = int(_m.group(1))
            if _sid in _STATE["sessions"]:
                _STATE["sessions"][_sid]["researching_round"] = val
            self._results = [()]

        # ---- SELECT r.company_name, r.job_title, q.question_text ----
        # Researching-stage "avoid duplicates" JOIN (3 columns, no question_type).
        elif (
            "SELECT" in up
            and "COMPANY_NAME" in up
            and "JOB_TITLE" in up
            and "QUESTION_TEXT" in up
            and "QUESTION_TYPE" not in up
            and "JOIN" in up
            and "WHERE" in up
        ):
            _sid = p[0] if p else None
            rows = []
            for q in _STATE["questions"].values():
                r = _STATE["roles"].get(q["role_id"])
                if r and r["session_id"] == _sid:
                    rows.append((r["company_name"], r["job_title"], q["question_text"]))
            self._results = rows

        elif "SELECT RESEARCHING_ROUND FROM" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            row = _STATE["sessions"].get(_sid)
            self._results = [(row["researching_round"],)] if row else []

        elif "SELECT USER_RESUME FROM" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            row = _STATE["sessions"].get(_sid)
            self._results = [(row["user_resume"],)] if row else []

        elif "SELECT R.COMPANY_NAME, R.JOB_TITLE, Q.QUESTION_TEXT FROM" in up and "JOIN" in up and "WHERE R.SESSION_ID" in up:
            _sid = p[0] if p else None
            rows = []
            for q in _STATE["questions"].values():
                r = _STATE["roles"].get(q["role_id"])
                if r and r["session_id"] == _sid:
                    rows.append((r["company_name"], r["job_title"], q["question_text"]))
            rows.sort(key=lambda x: (x[0], x[1], x[2]))
            self._results = rows

        elif "SELECT R.COMPANY_NAME, R.JOB_TITLE, Q.QUESTION_TEXT, Q.QUESTION_TYPE" in up:
            _sid = p[0] if p else None
            rows = []
            for q in _STATE["questions"].values():
                r = _STATE["roles"].get(q["role_id"])
                if r and r["session_id"] == _sid:
                    rows.append((r["company_name"], r["job_title"], q["question_text"], q["question_type"]))
            rows.sort(key=lambda x: (x[0], x[1], x[2]))
            self._results = rows

        elif "SELECT ROLE_ID FROM INTERVIEW_PREP_ROLES" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            self._results = [(r["role_id"],) for r in _STATE["roles"].values() if r["session_id"] == _sid]

        elif "SELECT COUNT(*) FROM INTERVIEW_PREP_SESSIONS" in up and "WHERE SESSION_ID" in up:
            _sid = p[0] if p else None
            self._results = [(1 if _sid in _STATE["sessions"] else 0,)]

        elif up.strip().startswith("SELECT"):
            self._results = []
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


# ==============================================================================
# Pytest fixtures — inject fakes via the class attributes the service actually uses
# ==============================================================================

@pytest.fixture(autouse=True)
def _fake_db():
    """Every test runs with InterviewService.get_conn pointing at the in-memory fake."""
    _reset_state()
    InterviewService.get_conn = _fake_get_conn
    yield
    _reset_state()


def _fake_llm(prompt, system_prompt, schema, model):
    """Default fake LLM — returns a structure matching whatever schema is requested.

    For prospect/research/writer shapes, return a valid-but-boring payload. For
    unknown schemas, return {} (the service's except clause then uses its fallback).
    """
    props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
    if "roles" in props:
        return {
            "roles": [
                {"company": "Acme", "title": "Data Scientist", "responsibilities": "ML", "skills": ["python"]},
                {"company": "Globex", "title": "ML Engineer", "responsibilities": "Deploy", "skills": ["python"]},
            ]
        }
    if "role_questions" in props or "quality_score" in props:
        return {
            "roleCompany": "Acme",
            "roleTitle": "Data Scientist",
            "questions": [
                {"question": "Q1", "type": "technical", "rationale": "core"},
                {"question": "Q2", "type": "behavioral", "rationale": "fit"},
            ],
            "quality_score": 5,
        }
    if "answers" in props:
        return {
            "role": "Acme — Data Scientist",
            "answers": [
                {"question": "Tell me about yourself", "type": "behavioral", "answer": "STAR: led a team, reduced churn 15%.", "critique": "Good numbers; consider more action ownership."},
                {"question": "How do you handle outliers?", "type": "technical", "answer": "I cap at 3 SD and log-transform.", "critique": ""},
            ],
        }
    return {}


# ==============================================================================
# Contract tests (pure Pydantic — no DB, no LLM)
# ==============================================================================

class TestProspectOutputContract:
    def test_model_json_schema_is_dict(self):
        schema = InterviewService.ProspectOutput.model_json_schema()
        assert isinstance(schema, dict)
        assert "roles" in schema["properties"]

    def test_validate_valid_data(self):
        out = InterviewService.ProspectOutput.model_validate(
            {"roles": [{"company": "Acme", "title": "Data Scientist", "responsibilities": "Build models", "skills": ["python"]}]}
        )
        assert len(out.roles) == 1
        assert out.roles[0].company == "Acme"
        assert out.roles[0].title == "Data Scientist"
        assert out.roles[0].skills == ["python"]

    def test_skills_defaults_to_empty_list(self):
        out = InterviewService.ProspectOutput.model_validate(
            {"roles": [{"company": "Acme", "title": "Data Scientist", "responsibilities": "Build models"}]}
        )
        assert out.roles[0].skills == []

    def test_min_length_enforced(self):
        with pytest.raises(Exception):
            InterviewService.ProspectOutput.model_validate({"roles": []})


class TestRoleQuestionsContract:
    def test_model_json_schema_is_dict(self):
        schema = InterviewService.RoleQuestions.model_json_schema()
        assert isinstance(schema, dict)
        assert "quality_score" in schema["properties"]

    def test_quality_score_clamped_by_pydantic(self):
        with pytest.raises(Exception):
            InterviewService.RoleQuestions.model_validate(
                {
                    "roleCompany": "Acme",
                    "roleTitle": "Data Scientist",
                    "questions": [{"question": "Q", "type": "technical"}],
                    "quality_score": 6,
                }
            )

    def test_valid_low_score_accepted(self):
        rq = InterviewService.RoleQuestions(
            roleCompany="Acme",
            roleTitle="Data Scientist",
            questions=[{"question": "Q", "type": "technical", "rationale": "R"}],
            quality_score=1,
        )
        assert rq.quality_score == 1


class TestResearchOutputContract:
    def test_overall_quality_score_required(self):
        schema = InterviewService.ResearchOutput.model_json_schema()
        assert "overall_quality_score" in schema["properties"]
        assert schema["required"] == ["role_questions", "overall_quality_score"]


class TestWriterOutputContract:
    def test_critique_field_optional(self):
        wo = InterviewService.WriterOutput(
            role="Acme — Data Scientist",
            answers=[
                InterviewService.WriterAnswer(question="Tell me about yourself", type="behavioral", answer="STAR answer here", critique="Missing numbers"),
            ],
        )
        assert wo.answers[0].critique == "Missing numbers"

    def test_critique_defaults_empty(self):
        wo = InterviewService.WriterOutput(
            role="Acme — Data Scientist",
            answers=[InterviewService.WriterAnswer(question="Q", type="technical", answer="A")],
        )
        assert wo.answers[0].critique == ""


# ==============================================================================
# run_prospecting
# ==============================================================================

class TestRunProspecting:
    def test_start_session_creates_uuid(self):
        sid = InterviewService.start_session("Data Scientist", "Acme, Globex", "My resume")
        assert isinstance(sid, str)
        assert len(sid) == 36

    def test_run_prospecting_calls_llm_with_schema(self, monkeypatch):
        captured = {}

        def mock_json(prompt, system_prompt, schema, model):
            captured["prompt"] = prompt
            captured["system"] = system_prompt
            captured["schema"] = schema
            captured["model"] = model
            return {
                "roles": [
                    {"company": "Acme", "title": "Data Scientist", "responsibilities": "Build ML", "skills": ["python", "sql"]},
                    {"company": "Globex", "title": "ML Engineer", "responsibilities": "Deploy models", "skills": ["python"]},
                ]
            }

        # Inject the fake LLM directly via the class attribute.
        original = InterviewService.llm
        InterviewService.llm = mock_json
        try:
            session_id = InterviewService.start_session("Data Scientist", "Acme, Globex", "My resume")
            result = InterviewService.run_prospecting(session_id)

            assert result["status"] == "completed"
            assert result["roles_count"] == 2
            assert "Search for open roles at Acme, Globex that match the profile of a Data Scientist" in captured["prompt"]
            assert isinstance(captured["schema"], dict)
            assert "roles" in captured["schema"]["properties"]

            # Persisted: two roles in the fake DB.
            with InterviewService.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM interview_prep_roles WHERE session_id = %s", (session_id,))
                    count = cur.fetchone()[0]
            assert count == 2

            # Stage advanced to researching.
            status = InterviewService.get_status(session_id)
            assert status["stage"] == "researching"
        finally:
            InterviewService.llm = original

    def test_run_prospecting_fallback_on_bad_json(self, monkeypatch):
        def raise_fn(prompt, system_prompt, schema, model):
            raise ValueError("bad json")

        original = InterviewService.llm
        InterviewService.llm = raise_fn
        try:
            session_id = InterviewService.start_session("Data Scientist", "Acme", "Resume")
            result = InterviewService.run_prospecting(session_id)

            assert result["status"] == "completed"
            assert result["roles_count"] == 1

            with InterviewService.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT company_name, job_title FROM interview_prep_roles WHERE session_id = %s", (session_id,))
                    row = cur.fetchone()
            assert row[0] == "Unknown"
            assert row[1] == "Unknown"
        finally:
            InterviewService.llm = original


# ==============================================================================
# run_researching — fan-out + bounded retry
# ==============================================================================

class TestRunResearchingFanout:
    def _seed_roles(self, session_id, role_specs):
        with InterviewService.get_conn() as conn:
            with conn.cursor() as cur:
                for spec in role_specs:
                    cur.execute(
                        """INSERT INTO interview_prep_roles
                           (session_id, company_name, job_title, responsibilities, required_skills)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (
                            session_id,
                            spec["company"],
                            spec["title"],
                            spec.get("responsibilities", ""),
                            ", ".join(spec.get("skills", [])),
                        ),
                    )
            conn.commit()

    def test_fan_out_one_call_per_role(self, monkeypatch):
        call_count = {"n": 0}
        role_order = []

        def mock_per_role(prompt, system_prompt, schema, model):
            call_count["n"] += 1
            lines = prompt.split("\n")
            company = next((l.replace("Company: ", "") for l in lines if l.startswith("Company: ")), "??")
            title = next((l.replace("Role: ", "") for l in lines if l.startswith("Role: ")), "??")
            role_order.append((company, title))
            return {
                "roleCompany": company,
                "roleTitle": title,
                "questions": [
                    {"question": f"{company} Q1", "type": "technical", "rationale": "core"},
                    {"question": f"{company} Q2", "type": "behavioral", "rationale": "fit"},
                ],
                "quality_score": 5,
            }

        original = InterviewService.llm
        InterviewService.llm = mock_per_role
        try:
            session_id = InterviewService.start_session("Data Scientist", "Acme", "Resume")
            self._seed_roles(session_id, [
                {"company": "Acme", "title": "Data Scientist", "responsibilities": "ML", "skills": ["python"]},
                {"company": "Globex", "title": "ML Engineer", "responsibilities": "Deploy", "skills": ["python"]},
            ])

            result = InterviewService.run_researching(session_id)

            assert call_count["n"] == 2
            assert len(role_order) == 2
            assert role_order[0] == ("Acme", "Data Scientist")
            assert role_order[1] == ("Globex", "ML Engineer")
            assert result["status"] == "completed"
            assert result["stage"] == "researching"
            assert result["overall_quality_score"] == 5
            assert result["questions_count"] == 4

            with InterviewService.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT COUNT(*) FROM interview_prep_questions q
                           JOIN interview_prep_roles r ON q.role_id = r.role_id
                           WHERE r.session_id = %s""",
                        (session_id,),
                    )
                    count = cur.fetchone()[0]
            assert count == 4
        finally:
            InterviewService.llm = original

    def test_researching_retry_when_quality_low(self, monkeypatch):
        call_count = {"n": 0}

        def flaky_llm(prompt, system_prompt, schema, model):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"roleCompany": "Acme", "roleTitle": "Data Scientist", "questions": [{"question": "Q1", "type": "technical"}], "quality_score": 2}
            return {
                "roleCompany": "Acme",
                "roleTitle": "Data Scientist",
                "questions": [{"question": "Q1", "type": "technical"}, {"question": "Q2", "type": "behavioral"}, {"question": "Q3", "type": "technical"}],
                "quality_score": 5,
            }

        original = InterviewService.llm
        InterviewService.llm = flaky_llm
        try:
            session_id = InterviewService.start_session("Data Scientist", "Acme", "Resume")
            self._seed_roles(session_id, [
                {"company": "Acme", "title": "Data Scientist", "responsibilities": "ML", "skills": ["python"]},
            ])

            result1 = InterviewService.run_researching(session_id)
            assert result1["status"] == "retry"
            assert result1["stage"] == "researching"
            assert result1["current_round"] == 1
            assert result1["overall_quality_score"] == 2
            assert call_count["n"] == 1

            result2 = InterviewService.run_researching(session_id)
            assert result2["status"] == "completed"
            assert result2["stage"] == "researching"
            assert result2["overall_quality_score"] == 5
            assert call_count["n"] == 2

            status = InterviewService.get_status(session_id)
            assert status["researching_round"] == 0
        finally:
            InterviewService.llm = original

    def test_researching_exhausts_max_rounds(self, monkeypatch):
        call_count = {"n": 0}

        def always_low(prompt, system_prompt, schema, model):
            call_count["n"] += 1
            return {"roleCompany": "Acme", "roleTitle": "Data Scientist", "questions": [{"question": "Q1", "type": "technical"}], "quality_score": 1}

        original_llm = InterviewService.llm
        original_max = InterviewService.max_rounds
        original_qual = InterviewService.quality_threshold
        InterviewService.llm = always_low
        InterviewService.max_rounds = 1
        InterviewService.quality_threshold = 4
        try:
            session_id = InterviewService.start_session("Data Scientist", "Acme", "Resume")
            self._seed_roles(session_id, [
                {"company": "Acme", "title": "Data Scientist", "responsibilities": "ML", "skills": ["python"]},
            ])

            r1 = InterviewService.run_researching(session_id)
            assert r1["status"] == "retry"
            assert r1["current_round"] == 1
            assert r1["max_rounds"] == 2

            r2 = InterviewService.run_researching(session_id)
            assert r2["status"] == "retry"
            assert r2["current_round"] == 2
            assert r2["max_rounds"] == 2

            r3 = InterviewService.run_researching(session_id)
            assert r3["status"] == "completed"
            assert r3["stage"] == "researching"
            assert call_count["n"] == 3
        finally:
            InterviewService.llm = original_llm
            InterviewService.max_rounds = original_max
            InterviewService.quality_threshold = original_qual

    def test_retry_context_appended_on_second_call(self, monkeypatch):
        prompts_seen = []

        def tracking_llm(prompt, system_prompt, schema, model):
            prompts_seen.append(prompt)
            return {"roleCompany": "Acme", "roleTitle": "Data Scientist", "questions": [{"question": "Q1", "type": "technical"}], "quality_score": 2}

        original = InterviewService.llm
        InterviewService.llm = tracking_llm
        try:
            session_id = InterviewService.start_session("Data Scientist", "Acme", "Resume")
            self._seed_roles(session_id, [
                {"company": "Acme", "title": "Data Scientist", "responsibilities": "ML", "skills": ["python"]},
            ])

            InterviewService.run_researching(session_id)
            assert len(prompts_seen) == 1
            assert "Previously generated questions" not in prompts_seen[0]

            InterviewService.run_researching(session_id)
            assert len(prompts_seen) == 2
            assert "Previously generated questions" in prompts_seen[1]
            assert "Acme / Data Scientist" in prompts_seen[1]
        finally:
            InterviewService.llm = original


# ==============================================================================
# run_writing
# ==============================================================================

class TestRunWriting:
    def _seed_session(self, session_id, company, title, resume, qa_pairs):
        with InterviewService.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO interview_prep_roles
                       (session_id, company_name, job_title, responsibilities, required_skills)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (session_id, company, title, "ML", "python"),
                )
                for question, qtype in qa_pairs:
                    cur.execute(
                        """INSERT INTO interview_prep_questions
                           (role_id, question_text, question_type)
                           VALUES ((SELECT role_id FROM interview_prep_roles WHERE session_id = %s), %s, %s)""",
                        (session_id, question, qtype),
                    )
            conn.commit()

    def test_writing_produces_structured_guide_with_critiques(self, monkeypatch):
        def mock_writer(prompt, system_prompt, schema, model):
            return {
                "role": "Acme — Data Scientist",
                "answers": [
                    {"question": "Tell me about yourself", "type": "behavioral", "answer": "STAR: led a team, reduced churn 15%.", "critique": "Good numbers; consider more action ownership."},
                    {"question": "How do you handle outliers?", "type": "technical", "answer": "I cap at 3 SD and log-transform.", "critique": ""},
                ],
            }

        original = InterviewService.llm
        InterviewService.llm = mock_writer
        try:
            session_id = InterviewService.start_session("Data Scientist", "Acme", "My resume")
            self._seed_session(session_id, "Acme", "Data Scientist", "My resume", [
                ("Tell me about yourself", "behavioral"),
                ("How do you handle outliers?", "technical"),
            ])

            result = InterviewService.run_writing(session_id)

            assert result["status"] == "completed"
            assert result["stage"] == "writing"
            assert result["role_count"] == 1
            assert result["answers_count"] == 2
            assert "Critique:" in result["guide"]
            assert "reduced churn 15%" in result["guide"]
            assert "led a team" in result["guide"]

            status = InterviewService.get_status(session_id)
            assert status["is_completed"] is True
            assert "STAR" in status["final_guide"]
        finally:
            InterviewService.llm = original

    def test_writing_fallback_when_llm_fails(self, monkeypatch):
        def raise_fn(prompt, system_prompt, schema, model):
            raise ValueError("LLM unavailable")

        original = InterviewService.llm
        InterviewService.llm = raise_fn
        try:
            session_id = InterviewService.start_session("Data Scientist", "Acme", "Resume")
            self._seed_session(session_id, "Acme", "Data Scientist", "Resume", [("Tell me about yourself", "behavioral")])

            result = InterviewService.run_writing(session_id)

            assert result["status"] == "completed"
            assert "did not return a structured response" in result["guide"]
            assert "Fallback" in result["guide"]
        finally:
            InterviewService.llm = original

    def test_writing_groups_by_role(self, monkeypatch):
        prompts_seen = []

        def mock_writer(prompt, system_prompt, schema, model):
            roles_mentioned = sorted(set(
                line.replace("## ", "").strip()
                for line in prompt.split("\n")
                if line.startswith("Role: ")
            ))
            prompts_seen.append(roles_mentioned)
            company_title = roles_mentioned[0] if roles_mentioned else "Acme — Data Scientist"
            return {"role": company_title, "answers": [{"question": "Q1", "type": "technical", "answer": f"STAR for {company_title}", "critique": ""}]}

        original = InterviewService.llm
        InterviewService.llm = mock_writer
        try:
            session_id = InterviewService.start_session("Data Scientist", "Acme, Globex", "Resume")
            self._seed_session(session_id, "Acme", "Data Scientist", "Resume", [
                ("Tell me about yourself", "behavioral"),
                ("How do you handle outliers?", "technical"),
            ])
            # Second role in a different company.
            with InterviewService.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO interview_prep_roles
                           (session_id, company_name, job_title, responsibilities, required_skills)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (session_id, "Globex", "ML Engineer", "Deploy", "python"),
                    )
                    cur.execute(
                        """INSERT INTO interview_prep_questions
                           (role_id, question_text, question_type)
                           VALUES ((SELECT role_id FROM interview_prep_roles WHERE session_id = %s AND company_name = 'Globex'), %s, %s)""",
                        (session_id, "Design a pipeline", "technical"),
                    )
                conn.commit()

            result = InterviewService.run_writing(session_id)

            assert len(prompts_seen) == 2
            assert any("Acme — Data Scientist" in r for r in prompts_seen)
            assert any("Globex — ML Engineer" in r for r in prompts_seen)
            assert result["role_count"] == 2
        finally:
            InterviewService.llm = original


# ==============================================================================
# Prompts carry the structural guarantees we want
# ==============================================================================

class TestPromptContent:
    def test_researching_prompt_mentions_avoid_duplicates_when_round_gt_0(self, monkeypatch):
        captured = {}

        def mock(prompt, system_prompt, schema, model):
            captured["prompt"] = prompt
            return {"roleCompany": "Acme", "roleTitle": "Data Scientist", "questions": [{"question": "Q", "type": "technical"}], "quality_score": 5}

        original = InterviewService.llm
        InterviewService.llm = mock
        try:
            session_id = InterviewService.start_session("Data Scientist", "Acme", "Resume")
            with InterviewService.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO interview_prep_roles
                           (session_id, company_name, job_title, responsibilities, required_skills)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (session_id, "Acme", "Data Scientist", "ML", "python"),
                    )
                conn.commit()
            with InterviewService.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE interview_prep_sessions SET researching_round = 1 WHERE session_id = %s", (session_id,))
                conn.commit()

            InterviewService.run_researching(session_id)

            assert "Previously generated questions" in captured["prompt"]
            assert "avoid duplicating" in captured["prompt"]
        finally:
            InterviewService.llm = original

    def test_writing_prompt_includes_critique_instruction(self, monkeypatch):
        captured = {}

        def mock(prompt, system_prompt, schema, model):
            captured["prompt"] = prompt
            return {"role": "Acme — Data Scientist", "answers": [{"question": "Tell me about yourself", "type": "behavioral", "answer": "STAR", "critique": ""}]}

        original = InterviewService.llm
        InterviewService.llm = mock
        try:
            session_id = InterviewService.start_session("Data Scientist", "Acme", "Resume")
            with InterviewService.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO interview_prep_roles
                           (session_id, company_name, job_title, responsibilities, required_skills)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (session_id, "Acme", "Data Scientist", "ML", "python"),
                    )
                    cur.execute(
                        """INSERT INTO interview_prep_questions
                           (role_id, question_text, question_type)
                           VALUES ((SELECT role_id FROM interview_prep_roles WHERE session_id = %s), %s, %s)""",
                        (session_id, "Tell me about yourself", "behavioral"),
                    )
                conn.commit()

            InterviewService.run_writing(session_id)

            assert "critique" in captured["prompt"].lower()
            assert "STAR" in captured["prompt"]
        finally:
            InterviewService.llm = original


# ==============================================================================
# Schema seam — call (the injected LLM) receives a dict schema
# ==============================================================================

class TestSchemaSeam:
    def test_pydantic_schema_is_plain_dict(self):
        schema = InterviewService.ProspectOutput.model_json_schema()
        assert isinstance(schema, dict)
        payload = json.dumps(schema)
        loaded = json.loads(payload)
        assert isinstance(loaded, dict)
