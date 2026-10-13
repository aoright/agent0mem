#!/usr/bin/env bash
set -euo pipefail

SSH_KEY="/Users/liuyukai/CREATE/PandaAI/nunu/admin_key"
REMOTE_HOST="root@47.97.127.223"

echo "=================================================="
echo "Agent0Mem Production SQLite Full Backup Pipeline"
echo "Target: ${REMOTE_HOST}"
echo "=================================================="

ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_HOST}" '
  set -euo pipefail

  DATA_DIR="/var/lib/agent0mem/data"
  DB_PATH="${DATA_DIR}/memories.sqlite3"
  BACKUP_DIR="/var/lib/agent0mem/backups"
  mkdir -p "${BACKUP_DIR}"

  TS=$(date +%Y%m%d_%H%M%S)
  RAW_BACKUP="${BACKUP_DIR}/memories_full_${TS}.sqlite3"
  ZST_BACKUP="${RAW_BACKUP}.zst"

  echo "[1/6] Checking free disk space..."
  FREE_KB=$(df "${DATA_DIR}" | awk "NR==2 {print \$4}")
  FREE_GB=$((FREE_KB / 1024 / 1024))
  echo "Available disk space: ${FREE_GB} GB"
  if [ "${FREE_GB}" -lt 70 ]; then
    echo "ERROR: Insufficient disk space (${FREE_GB} GB < 70 GB) for safe full backup. Aborting."
    exit 1
  fi

  echo "[2/6] Flushing SQLite WAL checkpoint (PASSIVE)..."
  sqlite3 "${DB_PATH}" "PRAGMA wal_checkpoint(PASSIVE);"

  echo "[3/6] Running online SQLite .backup (non-blocking)..."
  START_TIME=$(date +%s)
  sqlite3 "${DB_PATH}" ".backup ${RAW_BACKUP}"
  BACKUP_DUR=$(( $(date +%s) - START_TIME ))
  echo "Online backup finished in ${BACKUP_DUR}s. Size:"
  ls -lh "${RAW_BACKUP}"

  echo "[4/6] Verifying SQLite database integrity with PRAGMA quick_check..."
  CHECK_RESULT=$(sqlite3 "${RAW_BACKUP}" "PRAGMA quick_check;")
  if [ "${CHECK_RESULT}" != "ok" ]; then
    echo "FATAL: Backup integrity check failed: ${CHECK_RESULT}"
    rm -f "${RAW_BACKUP}"
    exit 1
  fi
  echo "Integrity check PASSED: ok"

  echo "[5/6] Compressing with zstd (2 threads, removing raw file on success)..."
  Z_START=$(date +%s)
  zstd -T2 --rm "${RAW_BACKUP}" -o "${ZST_BACKUP}"
  Z_DUR=$(( $(date +%s) - Z_START ))
  echo "Compression finished in ${Z_DUR}s. Compressed size:"
  ls -lh "${ZST_BACKUP}"

  echo "[6/6] Verifying zstd archive integrity..."
  zstd -t "${ZST_BACKUP}"
  echo "Archive integrity verified successfully."

  echo "Rotating older backups (keeping latest 2)..."
  ls -t ${BACKUP_DIR}/memories_full_*.sqlite3.zst | tail -n +3 | xargs -r rm -f || true
  echo "Current backups in ${BACKUP_DIR}:"
  ls -lh ${BACKUP_DIR}/memories_full_*.sqlite3.zst
'

echo "=================================================="
echo "Full SQLite backup pipeline completed successfully."
echo "=================================================="
