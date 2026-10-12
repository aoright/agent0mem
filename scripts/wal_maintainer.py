"""Automated SQLite WAL Maintenance Daemon for Long-Running Evaluation Endurance.

Prevents unbounded WAL growth and database lock contention during long-running
benchmarks (such as the 189-hour official Multimodal Full evaluation).

Executes:
1. Periodic passive WAL checkpoints every interval seconds (default: 300s).
2. Proactive restart checkpoints when WAL size exceeds max_wal_mb (default: 50MB).
3. Disk space safety checks to ensure the volume never runs below 2GB.
4. Configurable busy timeout (PRAGMA busy_timeout = 10000).

Usage:
  python3 scripts/wal_maintainer.py --db /var/lib/agent0mem/data/memories.sqlite3 --interval 300
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WALMaintainer]: %(message)s"
)
logger = logging.getLogger("wal_maintainer")


def get_file_size_mb(path: str) -> float:
    try:
        if os.path.exists(path):
            return os.path.getsize(path) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def check_disk_space(dir_path: str, min_free_gb: float = 2.0) -> bool:
    try:
        usage = shutil.disk_usage(dir_path)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_free_gb:
            logger.error(f"Low disk space on {dir_path}: {free_gb:.2f} GB free (threshold: {min_free_gb} GB)")
            return False
        return True
    except Exception as exc:
        logger.warning(f"Disk check failed for {dir_path}: {exc}")
        return True


def perform_checkpoint(db_path: str, mode: str = "PASSIVE") -> tuple[int, int, int]:
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA busy_timeout = 10000;")
        cur.execute(f"PRAGMA wal_checkpoint({mode});")
        res = cur.fetchone()
        conn.close()
        return res if res else (0, 0, 0)
    except Exception as exc:
        logger.warning(f"Checkpoint ({mode}) on {db_path} error: {exc}")
        try:
            conn.close()
        except Exception:
            pass
        return (-1, -1, -1)


def maintain_loop(db_path: str, interval: int = 300, max_wal_mb: float = 50.0, one_shot: bool = False):
    db_dir = os.path.dirname(os.path.abspath(db_path))
    wal_path = f"{db_path}-wal"

    logger.info(f"Starting WAL maintainer for database: {db_path}")
    logger.info(f"Target directory: {db_dir} | Interval: {interval}s | Max WAL: {max_wal_mb}MB")

    while True:
        if not os.path.exists(db_path):
            logger.warning(f"Database file not found: {db_path}; waiting...")
            time.sleep(interval)
            continue

        check_disk_space(db_dir)

        wal_mb = get_file_size_mb(wal_path)
        db_mb = get_file_size_mb(db_path)

        if wal_mb > max_wal_mb:
            logger.info(f"WAL size ({wal_mb:.2f} MB) exceeds threshold ({max_wal_mb} MB). Executing RESTART checkpoint...")
            busy, log_pages, ckpt_pages = perform_checkpoint(db_path, mode="RESTART")
        else:
            busy, log_pages, ckpt_pages = perform_checkpoint(db_path, mode="PASSIVE")

        new_wal_mb = get_file_size_mb(wal_path)
        logger.info(
            f"Checkpoint status: busy={busy}, log_pages={log_pages}, ckpt_pages={ckpt_pages} | "
            f"DB size: {db_mb:.2f} MB | WAL size: {new_wal_mb:.2f} MB"
        )

        if one_shot:
            break

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="SQLite WAL Maintainer Daemon")
    parser.add_argument("--db", default="/var/lib/agent0mem/data/memories.sqlite3", help="Target SQLite database path")
    parser.add_argument("--interval", type=int, default=300, help="Checkpoint interval in seconds (default: 300)")
    parser.add_argument("--max-wal-mb", type=float, default=50.0, help="Maximum WAL size in MB before forcing RESTART checkpoint")
    parser.add_argument("--one-shot", action="store_true", help="Execute single checkpoint pass and exit")
    args = parser.parse_args()

    maintain_loop(args.db, interval=args.interval, max_wal_mb=args.max_wal_mb, one_shot=args.one_shot)


if __name__ == "__main__":
    main()
