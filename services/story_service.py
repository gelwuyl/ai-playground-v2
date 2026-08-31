"""Database operations for the Bedtime Story Generator app."""
import os

import psycopg
from psycopg.rows import dict_row

from services.database import get_conn

MODEL_NAME = os.environ.get("OPENROUTER_MODEL", "openrouter/free")


def save_story(prompt: str, story: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO stories (prompt, story, model_name) "
                "VALUES (%s, %s, %s)",
                (prompt, story, MODEL_NAME),
            )
        conn.commit()


def fetch_recent_stories(limit: int = 10) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, prompt, story, model_name, "
                "       to_char(created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at "
                "FROM stories ORDER BY id DESC LIMIT %s",
                (limit,),
            )
            return cur.fetchall()