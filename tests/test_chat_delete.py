"""Unit tests for chat deletion (services and HTTP handlers).

Covers:
- delete_interaction in services/interaction_service.py
- delete_story in services/story_service.py
- do_DELETE in api/history.py
- do_DELETE in api/stories.py
"""
import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATABASE_URL", "postgres://dummy:***@localhost/dummy")
os.environ.setdefault("OPENROUTER_MODEL", "openrouter/free")

import api.history as history_api
import api.stories as stories_api
import services.interaction_service as interaction_svc
import services.story_service as story_svc


class TestInteractionServiceDelete(unittest.TestCase):
    @patch("services.interaction_service.get_conn")
    def test_delete_interaction_success(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 1
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        result = interaction_svc.delete_interaction(42)

        self.assertTrue(result)
        mock_cur.execute.assert_called_once_with(
            "DELETE FROM interactions WHERE id = %s", (42,)
        )
        mock_conn.commit.assert_called_once()

    @patch("services.interaction_service.get_conn")
    def test_delete_interaction_not_found(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 0
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        result = interaction_svc.delete_interaction(999)

        self.assertFalse(result)
        mock_cur.execute.assert_called_once_with(
            "DELETE FROM interactions WHERE id = %s", (999,)
        )
        mock_conn.commit.assert_called_once()


class TestStoryServiceDelete(unittest.TestCase):
    @patch("services.story_service.get_conn")
    def test_delete_story_success(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 1
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        result = story_svc.delete_story(7)

        self.assertTrue(result)
        mock_cur.execute.assert_called_once_with(
            "DELETE FROM stories WHERE id = %s", (7,)
        )
        mock_conn.commit.assert_called_once()

    @patch("services.story_service.get_conn")
    def test_delete_story_not_found(self, mock_get_conn):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.rowcount = 0
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        result = story_svc.delete_story(999)

        self.assertFalse(result)
        mock_cur.execute.assert_called_once_with(
            "DELETE FROM stories WHERE id = %s", (999,)
        )
        mock_conn.commit.assert_called_once()


class DummyRequestHandler:
    """Helper to exercise BaseHTTPRequestHandler subclasses without a real socket."""

    def __init__(self, handler_cls, path):
        self.handler_cls = handler_cls
        self.path = path
        self.sent_status = None
        self.sent_headers = {}
        self.written_bytes = io.BytesIO()

    def invoke_delete(self):
        instance = self.handler_cls.__new__(self.handler_cls)
        instance.path = self.path
        instance.headers = {}
        instance.rfile = io.BytesIO(b"")
        instance.wfile = self.written_bytes

        def send_response(status):
            self.sent_status = status

        def send_header(k, v):
            self.sent_headers[k] = v

        def end_headers():
            pass

        instance.send_response = send_response
        instance.send_header = send_header
        instance.end_headers = end_headers

        instance.do_DELETE()

        self.written_bytes.seek(0)
        body = self.written_bytes.read().decode("utf-8")
        parsed_body = json.loads(body) if body else None
        return self.sent_status, parsed_body


class TestHistoryApiDelete(unittest.TestCase):
    @patch("api.history.delete_interaction")
    def test_do_DELETE_success(self, mock_delete):
        mock_delete.return_value = True
        dummy = DummyRequestHandler(history_api.handler, "/api/history?id=42")
        status, body = dummy.invoke_delete()

        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "deleted", "id": 42})
        mock_delete.assert_called_once_with(42)

    @patch("api.history.delete_interaction")
    def test_do_DELETE_not_found(self, mock_delete):
        mock_delete.return_value = False
        dummy = DummyRequestHandler(history_api.handler, "/api/history?id=999")
        status, body = dummy.invoke_delete()

        self.assertEqual(status, 404)
        self.assertEqual(body, {"detail": "Record not found."})

    def test_do_DELETE_missing_id(self):
        dummy = DummyRequestHandler(history_api.handler, "/api/history")
        status, body = dummy.invoke_delete()

        self.assertEqual(status, 400)
        self.assertIn("Missing id parameter", body["detail"])

    def test_do_DELETE_invalid_id(self):
        dummy = DummyRequestHandler(history_api.handler, "/api/history?id=not_an_int")
        status, body = dummy.invoke_delete()

        self.assertEqual(status, 400)
        self.assertIn("Invalid id parameter", body["detail"])


class TestStoriesApiDelete(unittest.TestCase):
    @patch("api.stories.delete_story")
    def test_do_DELETE_success(self, mock_delete):
        mock_delete.return_value = True
        dummy = DummyRequestHandler(stories_api.handler, "/api/stories?id=15")
        status, body = dummy.invoke_delete()

        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "deleted", "id": 15})
        mock_delete.assert_called_once_with(15)

    @patch("api.stories.delete_story")
    def test_do_DELETE_not_found(self, mock_delete):
        mock_delete.return_value = False
        dummy = DummyRequestHandler(stories_api.handler, "/api/stories?id=999")
        status, body = dummy.invoke_delete()

        self.assertEqual(status, 404)
        self.assertEqual(body, {"detail": "Record not found."})

    def test_do_DELETE_missing_id(self):
        dummy = DummyRequestHandler(stories_api.handler, "/api/stories")
        status, body = dummy.invoke_delete()

        self.assertEqual(status, 400)
        self.assertIn("Missing id parameter", body["detail"])

    def test_do_DELETE_invalid_id(self):
        dummy = DummyRequestHandler(stories_api.handler, "/api/stories?id=abc")
        status, body = dummy.invoke_delete()

        self.assertEqual(status, 400)
        self.assertIn("Invalid id parameter", body["detail"])


if __name__ == "__main__":
    unittest.main()
