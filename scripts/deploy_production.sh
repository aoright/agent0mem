#!/usr/bin/env bash
set -euo pipefail

SSH_KEY="/Users/liuyukai/CREATE/PandaAI/nunu/admin_key"
REMOTE_HOST="root@47.97.127.223"

echo "[1/4] Executing mandatory pre-deploy SQLite hot snapshot..."
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_HOST}" '
  mkdir -p /var/lib/agent0mem/backups
  TS=$(date +%Y%m%d_%H%M%S)
  SNAPSHOT_DB="/var/lib/agent0mem/backups/hot_snapshot_${TS}.sqlite3"
  sqlite3 /var/lib/agent0mem/data/memories.sqlite3 "
    PRAGMA wal_checkpoint(PASSIVE);
    ATTACH DATABASE \"${SNAPSHOT_DB}\" AS snap;
    CREATE TABLE snap.schema_info AS SELECT sql FROM sqlite_master WHERE type=\"table\";
    CREATE TABLE snap.memories_recent AS SELECT * FROM memories WHERE rowid > (SELECT max(rowid) - 2000 FROM memories);
    DETACH DATABASE snap;
  "
  ls -lh "${SNAPSHOT_DB}"
'

echo "[2/4] Syncing application code to /opt/agent0mem/app/ with strict state protection filters..."
rsync -avz \
  --filter=":- /Users/liuyukai/CREATE/agent0mem/.gitignore" \
  --filter="protect *.db" \
  --filter="protect *.sqlite*" \
  --filter="protect /data/**" \
  --filter="protect /backups/**" \
  --exclude="*.db*" \
  --exclude="*.sqlite*" \
  --exclude="data/" \
  --exclude="backups/" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  -e "ssh -o StrictHostKeyChecking=no -i ${SSH_KEY}" \
  /Users/liuyukai/CREATE/agent0mem/app/ \
  "${REMOTE_HOST}:/opt/agent0mem/app/"

echo "[3/4] Restarting agent0mem service on remote server..."
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_HOST}" \
  'systemctl restart agent0mem && sleep 1 && systemctl is-active agent0mem'

echo "[4/4] Verifying remote service health..."
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_HOST}" \
  'curl -s http://127.0.0.1:8288/health'

echo ""
echo "Production deployment and verification completed successfully."
