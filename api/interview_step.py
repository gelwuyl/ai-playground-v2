"""POST /api/interview_step — execute the next phase of the crew."""
from services.interview_service import InterviewService
from services.vercel_handler import VercelHandler

class handler(VercelHandler):
    def do_POST(self):
        try:
            body = self.read_json()
        except Exception:
            return self.json_response({"detail": "Invalid JSON body."}, 400)

        session_id = (body.get("session_id") or "").strip()
        if not session_id:
            return self.json_response({"detail": "Missing session_id."}, 400)

        try:
            # We'll let the service determine the next step based on current_stage in DB
            # This is essentially the a 'tick' for the agent crew
            from services.database import get_conn
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT current_stage FROM interview_prep_sessions WHERE session_id = %s", (session_id,))
                    res = cur.fetchone()
                    if not res: return self.json_response({"detail": "Session not found."}, 404)
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
