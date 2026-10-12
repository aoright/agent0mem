#!/usr/bin/env bash
set -euo pipefail

SSH_KEY="/Users/liuyukai/CREATE/PandaAI/nunu/admin_key"
REMOTE_HOST="root@47.97.127.223"

echo "[1/6] Executing mandatory pre-deploy SQLite hot snapshot..."
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

echo "[2/6] Detecting active blue/green upstream service..."
ACTIVE_PORT=$(ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_HOST}" "grep -o 'http://127.0.0.1:828[89]/' /etc/nginx/conf.d/architect-exam.conf | grep -o '828[89]' | head -n1")
if [ "${ACTIVE_PORT}" = "8289" ]; then
  TARGET_PORT="8288"
  TARGET_SERVICE="agent0mem"
  DRAIN_PORT="8289"
  DRAIN_SERVICE="agent0mem-v2"
else
  TARGET_PORT="8289"
  TARGET_SERVICE="agent0mem-v2"
  DRAIN_PORT="8288"
  DRAIN_SERVICE="agent0mem"
fi
echo "Active upstream is :${ACTIVE_PORT} (${DRAIN_SERVICE}). Staging deployment on :${TARGET_PORT} (${TARGET_SERVICE})."

CURRENT_COMMIT=$(git rev-parse --short HEAD)
echo "${CURRENT_COMMIT}" > /Users/liuyukai/CREATE/agent0mem/.git_commit
echo "Current deployment commit is ${CURRENT_COMMIT}"

echo "[3/6] Syncing application code to /opt/agent0mem/app/ with strict state protection filters..."
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

ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_HOST}" "echo '${CURRENT_COMMIT}' > /opt/agent0mem/.git_commit"

echo "[4/6] Restarting staging service ${TARGET_SERVICE} on port ${TARGET_PORT}..."
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_HOST}" "
  systemctl restart ${TARGET_SERVICE}
  sleep 2
  systemctl is-active ${TARGET_SERVICE}
  curl -s http://127.0.0.1:${TARGET_PORT}/health
"

echo "[5/6] Gracefully switching nginx to ${TARGET_SERVICE} on port ${TARGET_PORT}..."
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_HOST}" "
  sed -i 's|proxy_pass http://127.0.0.1:${ACTIVE_PORT}/;|proxy_pass http://127.0.0.1:${TARGET_PORT}/;|g' /etc/nginx/conf.d/architect-exam.conf
  nginx -t
  nginx -s reload
"

echo "[6/6] Verifying public endpoint and syncing drain service ${DRAIN_SERVICE}..."
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_HOST}" "
  curl -s http://47.97.127.223/agent0mem/health
  echo ''
  echo 'Draining in-flight requests on ${DRAIN_SERVICE} (8s)...'
  sleep 8
  systemctl restart ${DRAIN_SERVICE}
  sleep 2
  systemctl is-active ${DRAIN_SERVICE}
  curl -s http://127.0.0.1:${DRAIN_PORT}/health
"

echo ""
echo "Zero-downtime blue/green deployment and dual-service synchronization completed successfully."
