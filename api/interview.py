"""Mr. Brave API — consolidated interview endpoints.

Vercel's Hobby plan allows at most 12 Serverless Functions per deployment, so
the four interview endpoints share one function. vercel.json rewrites map the
original paths here with an `action` query parameter:

    POST /api/interview_start   -> action=start
    POST /api/interview_step    -> action=step
    GET  /api/interview_status  -> action=status
    POST /api/interview_delete  -> action=delete
"""
from urllib.parse import parse_qs, urlparse

from services.database import get_conn
from services.interview_service import InterviewService, _ensure_tables
from services.vercel_handler import VercelHandler


class handler(VercelHandler):
    def do_POST(self):
        action = self._action()
        if action == "start":
            return self._start()
        if action == "step":
            return self._step()
        if action == "delete":
            return self._delete()
        return self.json_response({"detail": "Unknown POST action. Use action=start|step|delete."}, 400)

    def do_GET(self):
        if self._action() == "status":
            return self._status()
        return self.json_response({"detail": "Unknown GET action. Use action=status."}, 400)

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

        target_role = (body.get("target_role") or "").strip()
        target_companies = (body.get("target_companies") or "").strip()
        user_resume = (body.get("user_resume") or "").strip()

        if not all([target_role, target_companies, user_resume]):
            return self.json_response({"detail": "Missing required fields: target_role, target_companies, or user_resume."}, 400)

        session_id = InterviewService.start_session(target_role, target_companies, user_resume)

        return self.json_response({"session_id": session_id, "status": "started", "next_step": "prospecting"})

    def _step(self):
        body = self._body()
        if body is None:
            return

        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return self.json_response({"detail": "Missing session_id."}, 400)

        try:
            # Let the service determine the next step based on current_stage in DB
            # This is essentially the 'tick' for the agent crew
            _ensure_tables()
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT current_stage FROM interview_prep_sessions WHERE session_id = %s", (session_id,))
                    res = cur.fetchone()
                    if not res:
                        return self.json_response({"detail": "Session not found."}, 404)
                    stage = res[0]

            if stage == 'prospecting':
                result = InterviewService.run_prospecting(session_id)
                next_stage = 'researching'
            elif stage == 'researching':
                result = InterviewService.run_researching(session_id)
                next_stage = 'writing'
            elif stage == 'writing':
                result = InterviewService.run_writing(session_id)
                next_stage = 'completed'
            else:
                return self.json_response({"detail": "Session already completed."}, 400)

            return self.json_response({
                "status": "completed",
                "stage": stage,
                "next_stage": next_stage,
                "result": result
            })

        except Exception as e:
            return self.json_response({"detail": str(e)}, 500)

    def _status(self):
        session_id = self._query_param("session_id")
        if not session_id:
            return self.json_response({"detail": "Missing session_id."}, 400)

        _ensure_tables()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_stage, final_guide FROM interview_prep_sessions WHERE session_id = %s", (session_id,))
                res = cur.fetchone()
                if not res:
                    return self.json_response({"detail": "Session not found."}, 404)

                stage, guide = res

        return self.json_response({
            "stage": stage,
            "final_guide": guide,
            "is_completed": stage == 'completed'
        })

    def _delete(self):
        body = self._body()
        if body is None:
            return

        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return self.json_response({"detail": "Missing session_id."}, 400)

        InterviewService.delete_session(session_id)
        return self.json_response({"status": "deleted"})
