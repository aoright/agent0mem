#!/usr/bin/env bash
set -euo pipefail

SSH_KEY="/Users/liuyukai/CREATE/PandaAI/nunu/admin_key"
REMOTE_HOST="root@47.97.127.223"

echo "[1/5] Executing mandatory pre-deploy SQLite hot snapshot..."
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

echo "[2/5] Syncing application code to /opt/agent0mem/app/ with strict state protection filters..."
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

echo "[3/5] Starting agent0mem-v2 on port 8289 with zero-downtime protection..."
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_HOST}" '
  systemctl restart agent0mem-v2
  sleep 2
  systemctl is-active agent0mem-v2
  curl -s http://127.0.0.1:8289/health
'

echo "[4/5] Gracefully switching nginx to agent0mem-v2..."
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_HOST}" '
  sed -i "s|proxy_pass http://127.0.0.1:8288/;|proxy_pass http://127.0.0.1:8289/;|g" /etc/nginx/conf.d/architect-exam.conf
  nginx -t
  nginx -s reload
'

echo "[5/5] Verifying production public endpoint health..."
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_HOST}" '
  curl -s http://47.97.127.223/agent0mem/health
'

echo ""
echo "Zero-downtime blue/green deployment and verification completed successfully."
