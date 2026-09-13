"""Mr. Brave — three-stage interview preparation pipeline (v2).

Each stage is one agent persona (system prompt) plus one or more LLM calls,
persisted to Postgres. Structured output contracts are Pydantic models whose
json_schema() drives the prompt's explicit JSON instruction and is also passed
through the LLM call's schema param (ready for any future json_schema
response_format upgrade).

Pipeline:
  prospecting  ->  researching (fan-out per role, bounded retry on quality_score)
                 ->  writing (structured STAR answers + inline per-question critique)

DI note: InterviewService.llm and InterviewService.get_conn are class attributes
with defaults wired to the real OpenRouter / Postgres implementations. Tests swap
them for fakes directly (no monkeypatching). Config values (models, thresholds,
max_rounds) are likewise class attributes so tests can set them inline.

Schema note: sql/004_interview_prep.sql is the canonical schema, but migrations
are applied manually (see README) and serverless deploys have no migration hook —
_ensure_tables() re-applies the same CREATE TABLE IF NOT EXISTS statements once
per cold start so a missing migration self-heals. v2 adds researching_round to
the sessions table (self-healed the same way).
"""

import os
import uuid
from typing import Any, Callable, Dict, Type

from pydantic import BaseModel, Field

from services.database import get_conn as _real_get_conn
from services.openrouter_service import OPENROUTER_MODEL, call_openrouter_json as _real_llm

# ==============================================================================
# SCHEMA (mirrors sql/004_interview_prep.sql; v2 adds researching_round)
# ==============================================================================

_DDL = """
CREATE TABLE IF NOT EXISTS interview_prep_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    target_role TEXT NOT NULL,
    target_companies TEXT NOT NULL,
    user_resume TEXT NOT NULL,
    current_stage TEXT DEFAULT 'prospecting',
    final_guide TEXT,
    researching_round INT DEFAULT 0
);
CREATE TABLE IF NOT EXISTS interview_prep_roles (
    role_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES interview_prep_sessions(session_id) ON DELETE CASCADE,
    company_name TEXT NOT NULL,
    job_title TEXT NOT NULL,
    responsibilities TEXT,
    required_skills TEXT
);
CREATE TABLE IF NOT EXISTS interview_prep_questions (
    question_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_id UUID REFERENCES interview_prep_roles(role_id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    question_type TEXT
);
"""

_tables_ready = False


def _ensure_tables(get_conn: Callable) -> None:
    """Apply the 004 schema once per serverless instance (no-op afterwards)."""
    global _tables_ready
    if _tables_ready:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
    _tables_ready = True


# ==============================================================================
# CONFIGURATION (class attributes — injectable by tests)
# ==============================================================================

class InterviewService:
    """Three-stage interview preparation pipeline.

    DI via class attributes (defaults wired to the real implementations):
        llm            — callable(prompt, system_prompt, schema, model) -> dict
        get_conn       — callable() -> context-manager connection
        prospector_model
        researcher_model
        writer_model
        max_rounds     — max researching retry rounds
        quality_threshold — researching quality_score below which we retry
    """

    llm = _real_llm
    get_conn = _real_get_conn
    prospector_model = os.environ.get("INTERVIEW_PROSPECTOR_MODEL", OPENROUTER_MODEL)
    researcher_model = os.environ.get("INTERVIEW_RESEARCHER_MODEL", OPENROUTER_MODEL)
    writer_model = os.environ.get("INTERVIEW_WRITER_MODEL", OPENROUTER_MODEL)
    max_rounds = int(os.environ.get("INTERVIEW_RESEARCHING_MAX_ROUNDS", "2"))
    quality_threshold = int(os.environ.get("INTERVIEW_QUALITY_THRESHOLD", "4"))

    # ==========================================================================
    # OUTPUT CONTRACTS (Pydantic — drives prompt schema + LLM schema)
    # ==========================================================================

    class ProspectRole(BaseModel):
        company: str = Field(..., description="Company name")
        title: str = Field(..., description="Job title")
        responsibilities: str = Field(..., description="Key responsibilities")
        skills: list[str] = Field(default_factory=list, description="Required skills")

    class ProspectOutput(BaseModel):
        roles: list["InterviewService.ProspectRole"] = Field(
            ..., min_length=1, description="3 to 6 distinct roles"
        )

    class RoleQuestions(BaseModel):
        roleCompany: str = Field(..., description="Company this question set targets")
        roleTitle: str = Field(..., description="Job title this question set targets")
        questions: list[dict] = Field(
            ...,
            description="Each item: {question, type, rationale}",
        )
        quality_score: int = Field(
            ...,
            ge=1,
            le=5,
            description="1-5 confidence that these questions will actually appear",
        )

    class ResearchOutput(BaseModel):
        role_questions: list["InterviewService.RoleQuestions"] = Field(
            ..., description="One entry per role"
        )
        overall_quality_score: int = Field(
            ..., ge=1, le=5, description="Worst (most conservative) role score"
        )
        retry_context: str = Field(
            default="",
            description="Prior questions to avoid, used when retrying",
        )

    class WriterAnswer(BaseModel):
        question: str = Field(..., description="The interview question")
        type: str = Field(..., description="technical or behavioral")
        answer: str = Field(..., description="STAR response")
        critique: str = Field(
            default="",
            description="One-line critique: vague claims, missing numbers, action ownership, competency match",
        )

    class WriterOutput(BaseModel):
        role: str = Field(..., description="Company + title this answer set targets")
        answers: list["InterviewService.WriterAnswer"] = Field(..., min_length=1)

    # ==========================================================================
    # HELPERS
    # ==========================================================================

    @staticmethod
    def _prompt_schema_str(model: Type[BaseModel]) -> str:
        """Return a compact human-readable JSON-schema-ish string for the prompt."""
        return str(model.model_json_schema())

    # ==========================================================================
    # prospecting
    # ==========================================================================

    @classmethod
    def start_session(cls, target_role: str, target_companies: str, user_resume: str) -> str:
        """Creates a new interview preparation session."""
        _ensure_tables(cls.get_conn)
        session_id = str(uuid.uuid4())
        with cls.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO interview_prep_sessions
                       (session_id, target_role, target_companies, user_resume)
                       VALUES (%s, %s, %s, %s)""",
                    (session_id, target_role, target_companies, user_resume),
                )
            conn.commit()
        return session_id

    @classmethod
    def run_prospecting(cls, session_id: str) -> Dict[str, Any]:
        """Stage 1: Prospect for roles (structured output contract)."""
        _ensure_tables(cls.get_conn)
        with cls.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT target_role, target_companies FROM interview_prep_sessions WHERE session_id = %s",
                    (session_id,),
                )
                session = cur.fetchone()
                if not session:
                    raise ValueError("Session not found")
                target_role, target_companies = session

        schema_str = cls._prompt_schema_str(cls.ProspectOutput)
        prompt = (
            f"Search for open roles at {target_companies} that match the profile of a {target_role}. "
            "Extract job title, responsibilities, and skills for each. "
            "Return 3 to 6 distinct roles.\n\n"
            f"Return JSON matching this exact structure (no extra keys):\n{schema_str}\n"
        )
        try:
            data = cls.llm(
                prompt,
                PROSPECTOR_SYSTEM,
                schema=cls.ProspectOutput.model_json_schema(),
                model=cls.prospector_model,
            )
            output = cls.ProspectOutput.model_validate(data)
            roles_data = output.roles
        except Exception:
            roles_data = [
                cls.ProspectRole(
                    company="Unknown",
                    title="Unknown",
                    responsibilities="The model did not return structured role data. Retry this stage.",
                    skills=["Unknown"],
                )
            ]

        roles_data = [r for r in roles_data if isinstance(r, dict) or isinstance(r, cls.ProspectRole)]
        serialised = [
            {
                "company": r.company if isinstance(r, cls.ProspectRole) else r.get("company"),
                "title": r.title if isinstance(r, cls.ProspectRole) else r.get("title"),
                "responsibilities": r.responsibilities
                if isinstance(r, cls.ProspectRole)
                else r.get("responsibilities"),
                "skills": r.skills if isinstance(r, cls.ProspectRole) else r.get("skills", []),
            }
            for r in roles_data
        ]

        with cls.get_conn() as conn:
            with conn.cursor() as cur:
                for role in serialised:
                    cur.execute(
                        """INSERT INTO interview_prep_roles
                           (session_id, company_name, job_title, responsibilities, required_skills)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (
                            session_id,
                            role["company"],
                            role["title"],
                            role["responsibilities"],
                            ", ".join(role["skills"]) if isinstance(role["skills"], list) else role["skills"],
                        ),
                    )
                cur.execute(
                    "UPDATE interview_prep_sessions SET current_stage = 'researching' WHERE session_id = %s",
                    (session_id,),
                )
            conn.commit()

        return {"status": "completed", "roles_count": len(serialised)}

    # ==========================================================================
    # researching
    # ==========================================================================

    @classmethod
    def _fetch_roles(cls, session_id: str):
        """Return list of (role_id, company_name, job_title, responsibilities, required_skills)."""
        _ensure_tables(cls.get_conn)
        with cls.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT role_id, company_name, job_title, responsibilities, required_skills
                       FROM interview_prep_roles WHERE session_id = %s""",
                    (session_id,),
                )
                roles = cur.fetchall()
                if not roles:
                    raise ValueError("No roles found for this session")
                return roles

    @classmethod
    def _persist_questions(cls, session_id: str, role_questions: list["InterviewService.RoleQuestions"]) -> list[dict]:
        """Insert questions per role (by signature) and return counts per role."""
        _ensure_tables(cls.get_conn)
        with cls.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role_id, company_name, job_title FROM interview_prep_roles WHERE session_id = %s",
                    (session_id,),
                )
                rows = cur.fetchall()

            role_index_by_signature = {(r[1], r[2]): r[0] for r in rows}
            all_role_ids = [r[0] for r in rows]
            counts: dict[str, int] = {}
            for rq in role_questions:
                sig = (rq.roleCompany, rq.roleTitle)
                role_id = role_index_by_signature.get(sig)
                if role_id is None:
                    role_id = all_role_ids[len(counts) % len(all_role_ids)] if all_role_ids else None
                if role_id is None:
                    continue
                qlist = rq.questions if isinstance(rq.questions, list) else []
                for q in qlist:
                    if not isinstance(q, dict) or not q.get("question"):
                        continue
                    cur.execute(
                        """INSERT INTO interview_prep_questions
                           (role_id, question_text, question_type)
                           VALUES (%s, %s, %s)""",
                        (role_id, q.get("question"), q.get("type", "general")),
                    )
                counts[sig[0]] = counts.get(sig[0], 0) + len(qlist)

            cur.execute(
                "UPDATE interview_prep_sessions SET current_stage = 'writing' WHERE session_id = %s",
                (session_id,),
            )
            conn.commit()
        return [{"company": k, "questions": v} for k, v in counts.items()]

    @classmethod
    def run_researching(cls, session_id: str) -> Dict[str, Any]:
        """Stage 2: Predict questions, fan-out per role, bounded retry on quality_score."""
        _ensure_tables(cls.get_conn)

        with cls.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT researching_round FROM interview_prep_sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
                current_round = int(row[0]) if row and row[0] is not None else 0

        roles = cls._fetch_roles(session_id)

        avoid_context = ""
        if current_round > 0:
            with cls.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT r.company_name, r.job_title, q.question_text
                           FROM interview_prep_questions q
                           JOIN interview_prep_roles r ON q.role_id = r.role_id
                           WHERE r.session_id = %s""",
                        (session_id,),
                    )
                    prior = cur.fetchall()
            if prior:
                avoid_context = (
                    "\n\nPreviously generated questions (avoid duplicating these):\n"
                    + "\n".join(f"- [{p[0]} / {p[1]}] {p[2]}" for p in prior)
                )

        schema_str = cls._prompt_schema_str(cls.RoleQuestions)

        def _call_per_role(role: tuple) -> "InterviewService.RoleQuestions":
            """One Researcher call per role."""
            _role_id, company, title, responsibilities, skills = role
            context = (
                f"Company: {company}\n"
                f"Role: {title}\n"
                f"Responsibilities: {responsibilities or ''}\n"
                f"Required skills: {skills or ''}\n"
            )
            prompt = (
                f"Based on this role:\n{context}\n"
                "Generate 3 to 4 of the most likely interview questions for this specific role. "
                "Include a mix of technical and behavioral questions.\n"
                + avoid_context
                + "\n\n"
                f"Return JSON matching this exact structure (no extra keys):\n{schema_str}\n"
            )
            try:
                data = cls.llm(
                    prompt,
                    RESEARCHER_SYSTEM,
                    schema=cls.RoleQuestions.model_json_schema(),
                    model=cls.researcher_model,
                )
                return cls.RoleQuestions.model_validate(data)
            except Exception:
                return cls.RoleQuestions(
                    roleCompany=company,
                    roleTitle=title,
                    questions=[
                        {
                            "question": "The model did not return structured questions for this role. Retry this stage.",
                            "type": "general",
                            "rationale": "Fallback",
                        }
                    ],
                    quality_score=1,
                )

        role_questions: list["InterviewService.RoleQuestions"] = []
        for role in roles:
            try:
                rq = _call_per_role(role)
                role_questions.append(rq)
            except Exception as e:
                _role_id, company, title, *_ = role
                role_questions.append(
                    cls.RoleQuestions(
                        roleCompany=company,
                        roleTitle=title,
                        questions=[
                            {
                                "question": f"Research stage failed for this role: {e}",
                                "type": "general",
                                "rationale": "Error",
                            }
                        ],
                        quality_score=1,
                    )
                )

        scores = [rq.quality_score for rq in role_questions if isinstance(rq.quality_score, int)]
        overall = min(scores) if scores else 1

        cls._persist_questions(session_id, role_questions)

        retry_context = ""
        next_round = current_round + 1
        if overall < cls.quality_threshold and next_round <= cls.max_rounds + 1:
            retry_items = []
            for rq in role_questions:
                for q in (rq.questions if isinstance(rq.questions, list) else []):
                    if isinstance(q, dict) and q.get("question"):
                        retry_items.append(f"- [{rq.roleCompany} / {rq.roleTitle}] {q['question']}")
            retry_context = (
                "\n\nPreviously generated questions (avoid duplicating these):\n"
                + "\n".join(retry_items)
                if retry_items
                else ""
            )
            with cls.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE interview_prep_sessions SET researching_round = %s WHERE session_id = %s",
                        (next_round, session_id),
                    )
                conn.commit()
            return {
                "status": "retry",
                "stage": "researching",
                "overall_quality_score": overall,
                "current_round": next_round,
                "max_rounds": cls.max_rounds + 1,
                "message": (
                    f"Quality score {overall} below threshold {cls.quality_threshold}. "
                    f"Re-running researching (round {next_round} of {cls.max_rounds + 1})."
                ),
                "role_questions": [
                    {"company": rq.roleCompany, "title": rq.roleTitle, "quality_score": rq.quality_score}
                    for rq in role_questions
                ],
            }

        with cls.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE interview_prep_sessions SET researching_round = 0 WHERE session_id = %s",
                    (session_id,),
                )
            conn.commit()

        return {
            "status": "completed",
            "stage": "researching",
            "overall_quality_score": overall,
            "questions_count": sum(
                len(rq.questions) for rq in role_questions if isinstance(rq.questions, list)
            ),
            "role_questions": [
                {"company": rq.roleCompany, "title": rq.roleTitle, "quality_score": rq.quality_score}
                for rq in role_questions
            ],
        }

    # ==========================================================================
    # writing
    # ==========================================================================

    @classmethod
    def run_writing(cls, session_id: str) -> Dict[str, Any]:
        """Stage 3: Draft STAR responses + inline per-question critique (structured)."""
        _ensure_tables(cls.get_conn)
        with cls.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_resume FROM interview_prep_sessions WHERE session_id = %s",
                    (session_id,),
                )
                resume = cur.fetchone()[0]

                cur.execute(
                    """SELECT r.company_name, r.job_title, q.question_text, q.question_type
                       FROM interview_prep_questions q
                       JOIN interview_prep_roles r ON q.role_id = r.role_id
                       WHERE r.session_id = %s
                       ORDER BY r.company_name, r.job_title, q.question_id""",
                    (session_id,),
                )
                qa_pairs = cur.fetchall()

        if not qa_pairs:
            raise ValueError("No questions found for this session")

        from collections import OrderedDict

        grouped: OrderedDict[tuple, list[tuple]] = OrderedDict()
        for company, title, question, qtype in qa_pairs:
            grouped.setdefault((company, title), []).append((question, qtype))

        schema_str = cls._prompt_schema_str(cls.WriterOutput)
        writer_outputs: list["InterviewService.WriterOutput"] = []

        for (company, title), questions in grouped.items():
            context = "\n".join(
                f"[{company} / {title}] {q} (type: {t})" for q, t in questions
            )
            prompt = (
                f"User Resume:\n{resume}\n\n"
                f"Role: {company} — {title}\n\n"
                f"Questions:\n{context}\n\n"
                "Draft high-impact STAR responses for each question. "
                "For each, also write a one-line critique noting vague claims without numbers, "
                "missing action ownership ('we' vs 'I'), results lacking measurable outcomes, "
                "or answers that don't match the question's stated competency.\n\n"
                f"Return JSON matching this exact structure (no extra keys):\n{schema_str}\n"
            )
            try:
                data = cls.llm(
                    prompt,
                    WRITER_SYSTEM,
                    schema=cls.WriterOutput.model_json_schema(),
                    model=cls.writer_model,
                )
                wo = cls.WriterOutput.model_validate(data)
                wo.role = f"{company} — {title}"
                writer_outputs.append(wo)
            except Exception:
                answers = [
                    cls.WriterAnswer(
                        question=q,
                        type=t or "general",
                        answer="The model did not return a structured response for this question. Retry this stage.",
                        critique="Fallback — no structured STAR answer generated.",
                    )
                    for q, t in questions
                ]
                writer_outputs.append(cls.WriterOutput(role=f"{company} — {title}", answers=answers))

        guide_lines: list[str] = []
        guide_lines.append("# Interview Preparation Guide\n")
        guide_lines.append(f"## Resume\n{resume}\n")
        for wo in writer_outputs:
            guide_lines.append(f"## {wo.role}\n")
            for a in wo.answers:
                guide_lines.append(f"### {a.question}  *({a.type})*\n")
                guide_lines.append(f"{a.answer}\n")
                if a.critique:
                    guide_lines.append(f"**Critique:** {a.critique}\n")
                guide_lines.append("")

        guide = "\n".join(guide_lines)

        with cls.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE interview_prep_sessions
                       SET final_guide = %s, current_stage = 'completed'
                       WHERE session_id = %s""",
                    (guide, session_id),
                )
            conn.commit()

        return {
            "status": "completed",
            "stage": "writing",
            "guide": guide,
            "role_count": len(writer_outputs),
            "answers_count": sum(len(wo.answers) for wo in writer_outputs),
        }

    # ==========================================================================
    # status / delete
    # ==========================================================================

    @classmethod
    def get_status(cls, session_id: str) -> Dict[str, Any]:
        """Return current stage + final guide (used by GET /api/interview?action=status)."""
        _ensure_tables(cls.get_conn)
        with cls.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT current_stage, final_guide, researching_round FROM interview_prep_sessions WHERE session_id = %s",
                    (session_id,),
                )
                res = cur.fetchone()
                if not res:
                    raise ValueError("Session not found")
                stage, guide, researching_round = res
        return {
            "stage": stage,
            "final_guide": guide,
            "researching_round": int(researching_round or 0),
            "is_completed": stage == "completed",
        }

    @classmethod
    def delete_session(cls, session_id: str):
        """Delete a session and all its related data."""
        _ensure_tables(cls.get_conn)
        with cls.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM interview_prep_sessions WHERE session_id = %s", (session_id,)
                )
            conn.commit()


# ==============================================================================
# AGENT PERSONAS (module-level system prompts — referenced by the class above)
# ==============================================================================

PROSPECTOR_SYSTEM = """You are the Company Directory Prospector — an expert scout. You excel at
navigating corporate career portals and LinkedIn to identify open roles that match a specific
professional profile. Extract specific job roles and requirements from the target company
directories and career pages the user gives you."""  # noqa: E501

RESEARCHER_SYSTEM = """You are the Interview Strategist — a seasoned recruiter and industry analyst.
You can look at a job description and immediately identify the "gotcha" questions and the core
competencies. Analyze job roles to predict the most likely technical and behavioral interview
questions. For each question, also return a quality_score (1-5) rating your confidence that these
questions will actually appear in an interview for this role, and a short rationale."""  # noqa: E501

WRITER_SYSTEM = """You are the Professional Communications Expert — a communications coach for
C-suite executives. You know how to frame professional experience using the STAR method
(Situation, Task, Action, Result). Draft high-impact interview responses based on the user's
resume and critique them."""  # noqa: E501
