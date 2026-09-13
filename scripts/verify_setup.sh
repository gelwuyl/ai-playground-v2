#!/usr/bin/env bash
# verify_setup.sh — Module 0 acceptance gate (macOS / Linux).
# Exits 0 only when every dependency is reachable and the venv is active.
# Run from the repo root, after activating the project venv.

set -u

ok()   { echo "✓ $1"; }
fail() { echo "✗ $1"; exit 1; }

# 1. Virtual environment active.
[ -n "${VIRTUAL_ENV:-}" ] \
    || fail "Virtual environment not active. Run: source venv/bin/activate"
ok "Virtual environment active ($VIRTUAL_ENV)"

# 2. Python 3.11+.
py_version=$(python --version 2>&1 | awk '{print $2}')
case "$py_version" in
    3.11.*|3.12.*|3.13.*) ok "Python $py_version" ;;
    *) fail "Python 3.11+ required, found $py_version" ;;
esac

# 3. requirements.txt present at repo root.
[ -f requirements.txt ] \
    || fail "requirements.txt not found in current directory. Run from the repo root."
ok "requirements.txt present"

# 4. Postgres reachable.
pg_isready -h localhost -p 5432 >/dev/null 2>&1 \
    || fail "Postgres not reachable at localhost:5432. Start it: brew services start postgresql@17 (macOS) | sudo systemctl start postgresql (Linux)"
ok "Postgres reachable at localhost:5432"

# 5. Database llm_question_log reachable as the app will connect.
# Use the same connection string the app uses (TCP + postgres user + password)
# so a green check means the app can connect, not just that psql can.
APP_DSN="postgresql://postgres:postgres@localhost:5432/llm_question_log"
psql "$APP_DSN" -tAc "SELECT 1" >/dev/null 2>&1 \
    || fail "Cannot connect to llm_question_log as user 'postgres'. Either the database is missing (createdb llm_question_log) or the postgres role is missing (psql -d postgres -c \"CREATE USER postgres WITH PASSWORD 'postgres' SUPERUSER;\")"
ok "Database llm_question_log reachable as user 'postgres'"

# 6. Table interactions exists.
table_exists=$(psql "$APP_DSN" -tAc \
    "SELECT 1 FROM information_schema.tables WHERE table_name='interactions'")
[ "$table_exists" = "1" ] \
    || fail "Table 'interactions' not found. Apply schema: psql \"$APP_DSN\" -f sql/001_create_tables.sql"
ok "Table interactions exists"

# 7. Ollama reachable.
curl -sf http://localhost:11434/api/tags >/dev/null \
    || fail "Ollama not reachable at localhost:11434. Start it: ollama serve"
ok "Ollama reachable at localhost:11434"

# 8. Model gemma4:12b-mlx available.
curl -s http://localhost:11434/api/tags | grep -q '"name":"gemma4:12b-mlx' \
    || fail "Model 'gemma4:12b-mlx' not pulled. Pull it: ollama pull gemma4:12b-mlx"
ok "Model gemma4:12b-mlx available"

echo
echo "All checks passed. You're ready for Module 1."
