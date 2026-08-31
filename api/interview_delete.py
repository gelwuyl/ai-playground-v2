"""POST /api/interview_delete — erase a session from the database."""
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

        InterviewService.delete_session(session_id)
        return self.json_response({"status": "deleted"})
