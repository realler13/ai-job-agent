#!/bin/bash
# Start self-hosted LangFuse. Secrets live in secrets.compose next to this
# script (gitignored). Create it before the first start: one NAME=value per
# line for every variable marked CHANGEME in docker-compose.yml (SALT,
# ENCRYPTION_KEY, NEXTAUTH_SECRET, the database, ClickHouse, MinIO and Redis
# passwords). After creating a project in the LangFuse UI, put its
# LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST into
# langfuse-keys.conf (also gitignored) or into the main env file.
# Usage: ./up.sh            start detached
#        ./up.sh down       stop
#        ./up.sh logs -f    follow logs
cd "$(dirname "$0")" || exit 1
if [ ! -f secrets.compose ]; then
    echo "secrets.compose not found: create it first (see the header of this script)." >&2
    exit 1
fi
if [ $# -eq 0 ]; then
    exec docker compose --env-file secrets.compose up -d
fi
exec docker compose --env-file secrets.compose "$@"
