import os
import json
import uuid
from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew, Process, LLM
from services.database import get_conn

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# We use LiteLLM naming convention for CrewAI LLMs.
# These environment variables should be set in Vercel/local .env
# OPENROUTER_API_KEY or GOOGLE_API_KEY
PROSPECTOR_MODEL = os.environ.get("INTERVIEW_PROSPECTOR_MODEL", "gemini-1.5-flash")
RESEARCHER_MODEL = os.environ.get("INTERVIEW_RESEARCHER_MODEL", "gemini-1.5-pro")
WRITER_MODEL = os.environ.get("INTERVIEW_WRITER_MODEL", "gemini-1.5-pro")

# Initialize LLMs
prospector_llm = LLM(model=PROSPECTOR_MODEL)
researcher_llm = LLM(model=RESEARCHER_MODEL)
writer_llm = LLM(model=WRITER_MODEL)

# ==============================================================================
# AGENT DEFINITIONS
# ==============================================================================

def create_prospector():
    return Agent(
        role='Company Directory Prospector',
        goal='Extract specific job roles and requirements from target company directories and career pages.',
        backstory="""You are an expert scout. You excel at navigating corporate career portals
        and LinkedIn to identify open roles that match a specific professional profile.""",
        llm=prospector_llm,
        verbose=True,
        allow_delegation=False
    )

def create_researcher():
    return Agent(
        role='Interview Strategist',
        goal='Analyze job roles to predict the most likely technical and behavioral interview questions.',
        backstory="""You are a seasoned recruiter and industry analyst. You can look at a job
        description and immediately identify the "gotcha" questions and the core competencies.""",
        llm=researcher_llm,
        verbose=True,
        allow_delegation=False
    )

def create_writer():
    return Agent(
        role='Professional Communications Expert',
        goal='Draft high-impact interview responses based on the user\'s resume and critique them.',
        backstory="""You are a communications coach for C-suite executives. You know how to
        frame professional experience using the STAR method (Situation, Task, Action, Result).""",
        llm=writer_llm,
        verbose=True,
        allow_delegation=False
    )

# ==============================================================================
# SERVICE LOGIC
# ==============================================================================

class InterviewService:
    @staticmethod
    def start_session(target_role: str, target_companies: str, user_resume: str) -> str:
        """Creates a new interview preparation session."""
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
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT target_role, target_companies FROM interview_prep_sessions WHERE session_id = %s", (session_id,))
                session = cur.fetchone()
                if not session: raise ValueError("Session not found")
                target_role, target_companies = session

        agent = create_prospector()
        task = Task(
            description=f"Search for open roles at {target_companies} that match the profile of a {target_role}. Extract job title, responsibilities, and skills.",
            expected_output="A structured list of roles in JSON format: [{'company': '...', 'title': '...', 'responsibilities': '...', 'skills': '...'}]",
            agent=agent
        )

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()

        # Parse result and save to DB
        try:
            roles_data = json.loads(result.raw)
        except:
            # Fallback if the LLM doesn't return perfect JSON
            roles_data = [{"company": "Unknown", "title": "Unknown", "responsibilities": result.raw, "skills": "Unknown"}]

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
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT company_name, job_title, responsibilities, required_skills FROM interview_prep_roles WHERE session_id = %s", (session_id,))
                roles = cur.fetchall()
                if not roles: raise ValueError("No roles found for this session")

        # Prepare context for the researcher
        roles_context = "\n".join([f"Company: {r[0]}, Role: {r[1]}, Req: {r[2]}, Skills: {r[3]}" for r in roles])

        agent = create_researcher()
        task = Task(
            description=f"Based on these roles:\n{roles_context}\nGenerate a list of the most likely interview questions. Include technical and behavioral questions.",
            expected_output="A structured list of questions in JSON format: [{'role_id': '...', 'question': '...', 'type': '...'}]",
            agent=agent
        )

        # Note: In a real scenario, we'd need the role_id for the mapping.
        # For simplicity, we'll map them back based on company/title.
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()

        try:
            questions_data = json.loads(result.raw)
        except:
            questions_data = [{"question": result.raw, "type": "general"}]

        with get_conn() as conn:
            with conn.cursor() as cur:
                # Simple mapping: find the first role that matches the question context
                # This is a simplified version; in production, we'd make the agent return the role_id.
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

        agent = create_writer()
        task = Task(
            description=f"User Resume: {resume}\n\nQuestions:\n{context}\n\nDraft high-impact STAR responses for each. Add a 'Critic's Note' for each.",
            expected_output="A professional interview guide in Markdown format.",
            agent=agent
        )

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE interview_prep_sessions SET final_guide = %s, current_stage = 'completed' WHERE session_id = %s",
                            (result.raw, session_id))
            conn.commit()

        return {"status": "completed", "guide": result.raw}

    @staticmethod
    def delete_session(session_id: str):
        """Delete a session and all its related data."""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM interview_prep_sessions WHERE session_id = %s", (session_id,))
            conn.commit()
