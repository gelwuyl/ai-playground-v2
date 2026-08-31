"""Mr. Brave — three-stage interview preparation pipeline.

Each stage is one agent persona (system prompt) plus one OpenRouter call,
persisted to Postgres. Originally written against the crewAI framework, but
crewAI's dependency stack (chromadb, onnxruntime, litellm, kubernetes client)
pushes the Vercel serverless bundle past its 500 MB limit, so the crew is
orchestrated directly here: Prospector -> Researcher -> Writer, one stage per
/api/interview?action=step call.

Schema note: sql/004_interview_prep.sql is the canonical schema, but migrations
are applied manually (see README) and serverless deploys have no migration
hook — 004 initially never reached the production DB and every interview
endpoint 500'd. _ensure_tables() re-applies the same CREATE TABLE IF NOT
EXISTS statements once per cold start so a missing migration self-heals.
"""
import os
import uuid
from typing import Any, Dict

from services.database import get_conn
from services.openrouter_service import call_openrouter, call_openrouter_json

# ==============================================================================
# SCHEMA (mirrors sql/004_interview_prep.sql)
# ==============================================================================

_DDL = """
CREATE TABLE IF NOT EXISTS interview_prep_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    target_role TEXT NOT NULL,
    target_companies TEXT NOT NULL,
    user_resume TEXT NOT NULL,
    current_stage TEXT DEFAULT 'prospecting',
    final_guide TEXT
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


def _ensure_tables():
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
# CONFIGURATION
# ==============================================================================

# OpenRouter model ids per stage. (The crewAI version used LiteLLM-style
# "gemini-1.5-*" names, which are not valid OpenRouter ids.) Overridable via
# environment variables in Vercel/local .env.
PROSPECTOR_MODEL = os.environ.get("INTERVIEW_PROSPECTOR_MODEL", "google/gemini-2.5-flash")
RESEARCHER_MODEL = os.environ.get("INTERVIEW_RESEARCHER_MODEL", "google/gemini-2.5-pro")
WRITER_MODEL = os.environ.get("INTERVIEW_WRITER_MODEL", "google/gemini-2.5-pro")

# ==============================================================================
# AGENT PERSONAS (former crewAI role/goal/backstory, kept as system prompts)
# ==============================================================================

PROSPECTOR_SYSTEM = """You are the Company Directory Prospector — an expert scout. You excel at
navigating corporate career portals and LinkedIn to identify open roles that match a specific
professional profile. Extract specific job roles and requirements from the target company
directories and career pages the user gives you."""

RESEARCHER_SYSTEM = """You are the Interview Strategist — a seasoned recruiter and industry analyst.
You can look at a job description and immediately identify the "gotcha" questions and the core
competencies. Analyze job roles to predict the most likely technical and behavioral interview
questions."""

WRITER_SYSTEM = """You are the Professional Communications Expert — a communications coach for
C-suite executives. You know how to frame professional experience using the STAR method
(Situation, Task, Action, Result). Draft high-impact interview responses based on the user's
resume and critique them."""

# ==============================================================================
# SERVICE LOGIC
# ==============================================================================

class InterviewService:
    @staticmethod
    def start_session(target_role: str, target_companies: str, user_resume: str) -> str:
        """Creates a new interview preparation session."""
        _ensure_tables()
        session_id = str(uuid.uuid4())
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO interview_prep_sessions (session_id, target_role, target_companies, user_resume) VALUES (%s, %s, %s, %s)",
                    (session_id, target_role, target_companies, user_resume)
                )
            conn.commit()
        return session_id

    @staticmethod
    def run_prospecting(session_id: str) -> Dict[str, Any]:
        """Stage 1: Prospect for roles."""
        _ensure_tables()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT target_role, target_companies FROM interview_prep_sessions WHERE session_id = %s", (session_id,))
                session = cur.fetchone()
                if not session: raise ValueError("Session not found")
                target_role, target_companies = session

        prompt = (
            f"Search for open roles at {target_companies} that match the profile of a {target_role}. "
            "Extract job title, responsibilities, and skills for each. "
            "Return 3 to 6 distinct roles.\n\n"
            'Return JSON: {"roles": [{"company": "...", "title": "...", "responsibilities": "...", "skills": "..."}]}'
        )
        try:
            data = call_openrouter_json(prompt, PROSPECTOR_SYSTEM, schema={}, model=PROSPECTOR_MODEL)
            if isinstance(data, list):
                roles_data = data
            elif isinstance(data, dict):
                roles_data = data.get("roles") or []
            else:
                roles_data = []
            roles_data = [r for r in roles_data if isinstance(r, dict)]
        except ValueError:
            # Fallback if the model doesn't return parseable JSON
            roles_data = [{"company": "Unknown", "title": "Unknown",
                           "responsibilities": "The model did not return structured role data. Retry this stage.",
                           "skills": "Unknown"}]

        with get_conn() as conn:
            with conn.cursor() as cur:
                for role in roles_data:
                    cur.execute(
                        "INSERT INTO interview_prep_roles (session_id, company_name, job_title, responsibilities, required_skills) VALUES (%s, %s, %s, %s, %s)",
                        (session_id, role.get('company'), role.get('title'), role.get('responsibilities'), role.get('skills'))
                    )
                cur.execute("UPDATE interview_prep_sessions SET current_stage = 'researching' WHERE session_id = %s", (session_id,))
            conn.commit()

        return {"status": "completed", "roles_count": len(roles_data)}

    @staticmethod
    def run_researching(session_id: str) -> Dict[str, Any]:
        """Stage 2: Predict questions."""
        _ensure_tables()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT company_name, job_title, responsibilities, required_skills FROM interview_prep_roles WHERE session_id = %s", (session_id,))
                roles = cur.fetchall()
                if not roles: raise ValueError("No roles found for this session")

        # Prepare context for the researcher
        roles_context = "\n".join([f"Company: {r[0]}, Role: {r[1]}, Req: {r[2]}, Skills: {r[3]}" for r in roles])

        prompt = (
            f"Based on these roles:\n{roles_context}\n"
            "Generate 8 to 12 of the most likely interview questions across these roles. "
            "Include a mix of technical and behavioral questions.\n\n"
            'Return JSON: {"questions": [{"question": "...", "type": "technical|behavioral"}]}'
        )
        try:
            data = call_openrouter_json(prompt, RESEARCHER_SYSTEM, schema={}, model=RESEARCHER_MODEL)
            if isinstance(data, list):
                questions_data = data
            elif isinstance(data, dict):
                questions_data = data.get("questions") or []
            else:
                questions_data = []
            questions_data = [q for q in questions_data if isinstance(q, dict) and q.get("question")]
        except ValueError:
            questions_data = [{"question": "Model did not return structured questions. Retry this stage.", "type": "general"}]

        with get_conn() as conn:
            with conn.cursor() as cur:
                # Simple mapping: distribute questions across the session's roles
                # round-robin. In production, we'd make the agent return the role_id.
                cur.execute("SELECT role_id FROM interview_prep_roles WHERE session_id = %s", (session_id,))
                all_role_ids = [r[0] for r in cur.fetchall()]

                for i, q in enumerate(questions_data):
                    role_id = all_role_ids[i % len(all_role_ids)]
                    cur.execute(
                        "INSERT INTO interview_prep_questions (role_id, question_text, question_type) VALUES (%s, %s, %s)",
                        (role_id, q.get('question'), q.get('type'))
                    )
                cur.execute("UPDATE interview_prep_sessions SET current_stage = 'writing' WHERE session_id = %s", (session_id,))
            conn.commit()

        return {"status": "completed", "questions_count": len(questions_data)}

    @staticmethod
    def run_writing(session_id: str) -> Dict[str, Any]:
        """Stage 3: Draft responses."""
        _ensure_tables()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT user_resume FROM interview_prep_sessions WHERE session_id = %s", (session_id,))
                resume = cur.fetchone()[0]

                cur.execute("""
                    SELECT r.company_name, r.job_title, q.question_text
                    FROM interview_prep_questions q
                    JOIN interview_prep_roles r ON q.role_id = r.role_id
                    WHERE r.session_id = %s
                """, (session_id,))
                qa_pairs = cur.fetchall()

        context = "\n".join([f"Company: {p[0]}, Role: {p[1]}, Question: {p[2]}" for p in qa_pairs])

        prompt = (
            f"User Resume:\n{resume}\n\nQuestions:\n{context}\n\n"
            "Draft high-impact STAR responses for each. Add a 'Critic's Note' for each. "
            "Return the guide in Markdown format."
        )
        guide = call_openrouter(prompt, WRITER_SYSTEM)

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE interview_prep_sessions SET final_guide = %s, current_stage = 'completed' WHERE session_id = %s",
                            (guide, session_id))
            conn.commit()

        return {"status": "completed", "guide": guide}

    @staticmethod
    def delete_session(session_id: str):
        """Delete a session and all its related data."""
        _ensure_tables()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM interview_prep_sessions WHERE session_id = %s", (session_id,))
            conn.commit()
