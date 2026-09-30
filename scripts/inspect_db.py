import sqlite3
import json

conn = sqlite3.connect("/var/lib/agent0mem/data/memories.sqlite3")
cursor = conn.cursor()

cursor.execute("""
    SELECT user_id, count(*), max(created_at)
    FROM memories
    GROUP BY user_id
    ORDER BY max(created_at) DESC
    LIMIT 6
""")
print("=== Latest Users in DB ===")
users = []
for row in cursor.fetchall():
    print(row)
    users.append(row[0])

if users:
    target_user = users[0]
    print(f"\n=== Memories for latest user: {target_user} ===")
    cursor.execute("""
        SELECT id, text, metadata, created_at
        FROM memories
        WHERE user_id = ?
        LIMIT 5
    """, (target_user,))
    for row in cursor.fetchall():
        print(f"ID: {row[0]}")
        print(f"Text: {row[1][:160]}")
        print(f"Meta: {row[2]}")
        print("-" * 40)
