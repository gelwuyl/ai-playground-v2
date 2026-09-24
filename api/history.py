"""GET /api/history — return recent question-log interactions.
DELETE /api/history?id=... — delete an interaction from history.
"""
import urllib.parse

from services.interaction_service import delete_interaction, fetch_recent_history
from services.vercel_handler import VercelHandler


class handler(VercelHandler):
    def do_GET(self):
        return self.json_response(fetch_recent_history())

    def do_DELETE(self):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        id_str = (query.get("id") or [""])[0].strip()

        if not id_str:
            return self.json_response({"detail": "Missing id parameter."}, 400)

        try:
            interaction_id = int(id_str)
        except ValueError:
            return self.json_response({"detail": "Invalid id parameter."}, 400)

        if delete_interaction(interaction_id):
            return self.json_response({"status": "deleted", "id": interaction_id})
        return self.json_response({"detail": "Record not found."}, 404)
