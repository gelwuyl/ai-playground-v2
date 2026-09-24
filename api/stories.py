"""GET /api/stories — return recent bedtime stories.
DELETE /api/stories?id=... — delete a story from history.
"""
import urllib.parse

from services.story_service import delete_story, fetch_recent_stories
from services.vercel_handler import VercelHandler


class handler(VercelHandler):
    def do_GET(self):
        return self.json_response(fetch_recent_stories())

    def do_DELETE(self):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        id_str = (query.get("id") or [""])[0].strip()

        if not id_str:
            return self.json_response({"detail": "Missing id parameter."}, 400)

        try:
            story_id = int(id_str)
        except ValueError:
            return self.json_response({"detail": "Invalid id parameter."}, 400)

        if delete_story(story_id):
            return self.json_response({"status": "deleted", "id": story_id})
        return self.json_response({"detail": "Record not found."}, 404)
