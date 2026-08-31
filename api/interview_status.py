"""GET /api/interview_status — check progress or fetch the final guide."""
from services.database import get_conn
from services.vercel_handler import VercelHandler

class handler(VercelHandler):
    def do_GET(self):
        session_id = self.request.args.get("session_id")
        if not session_id:
            return self.json_response({"detail": "Missing session_id."}, 400)

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_stage, final_guide FROM interview_prep_sessions WHERE session_id = %s", (session_id,))
                res = cur.fetchone()
                if not res: return self.json_response({"detail": "Session not found."}, 404)

                stage, guide = res

        return self.json_response({
            "stage": stage,
            "final_guide": guide,
            "is_completed": stage == 'completed'
        })
