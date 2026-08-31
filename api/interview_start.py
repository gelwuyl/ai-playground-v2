"""POST /api/interview_start — initialize a preparation session."""
from services.interview_service import InterviewService
from services.vercel_handler import VercelHandler

class handler(VercelHandler):
    def do_POST(self):
        try:
            body = self.read_json()
        except Exception:
            return self.json_response({"detail": "Invalid JSON body."}, 400)

        target_role = (body.get("target_role") or "").strip()
        target_companies = (body.get("target_companies") or "").strip()
        user_resume = (body.get("user_resume") or "").strip()

        if not all([target_role, target_companies, user_resume]):
            return self.json_response({"detail": "Missing required fields: target_role, target_companies, or user_resume."}, 400)

        session_id = InterviewService.start_session(target_role, target_companies, user_resume)

        return self.json_response({"session_id": session_id, "status": "started", "next_step": "prospecting"})
