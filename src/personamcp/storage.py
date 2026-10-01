from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from personamcp.config import Config
from personamcp.models import Conversation, stable_id

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE conversations (
 id TEXT PRIMARY KEY, platform TEXT NOT NULL, external_id TEXT NOT NULL,
 title TEXT NOT NULL, created_at TEXT, context TEXT NOT NULL DEFAULT '', metadata TEXT NOT NULL,
 UNIQUE(platform, external_id)
);
CREATE TABLE participants (
 conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
 name TEXT NOT NULL, name_key TEXT NOT NULL, PRIMARY KEY(conversation_id, name_key)
);
CREATE TABLE messages (
 id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
 sender TEXT NOT NULL, sender_key TEXT NOT NULL, sender_id TEXT, text TEXT NOT NULL,
 timestamp TEXT NOT NULL, reply_to TEXT, is_user INTEGER NOT NULL, kind TEXT NOT NULL,
 platform TEXT NOT NULL, metadata TEXT NOT NULL, sequence INTEGER NOT NULL
);
CREATE INDEX messages_conversation_time ON messages(conversation_id,timestamp,sequence);
CREATE INDEX messages_owner ON messages(is_user,kind);
CREATE TABLE interactions (
 id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
 incoming TEXT NOT NULL, user_reply TEXT NOT NULL, person TEXT, person_key TEXT,
 platform TEXT NOT NULL, timestamp TEXT NOT NULL, context TEXT NOT NULL,
 message_ids TEXT NOT NULL
);
CREATE INDEX interactions_person ON interactions(person_key,platform);
CREATE VIRTUAL TABLE interactions_fts USING fts5(
 incoming,content='interactions',content_rowid='rowid');
CREATE TRIGGER interactions_ai AFTER INSERT ON interactions BEGIN
 INSERT INTO interactions_fts(rowid,incoming) VALUES(new.rowid,new.incoming);
END;
CREATE TRIGGER interactions_ad AFTER DELETE ON interactions BEGIN
 INSERT INTO interactions_fts(interactions_fts,rowid,incoming)
 VALUES('delete',old.rowid,old.incoming);
END;
CREATE TABLE profiles (scope TEXT PRIMARY KEY, data TEXT NOT NULL, generated_at TEXT NOT NULL);
CREATE TABLE imports (
 hash TEXT PRIMARY KEY, platform TEXT NOT NULL, imported_at TEXT NOT NULL,
 message_count INTEGER NOT NULL
);
CREATE TABLE embeddings (
 interaction_id TEXT NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
 provider TEXT NOT NULL, dimensions INTEGER NOT NULL, vector BLOB NOT NULL,
 PRIMARY KEY(interaction_id,provider)
);
CREATE VIRTUAL TABLE messages_fts USING fts5(text,content='messages',content_rowid='rowid');
CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
 INSERT INTO messages_fts(rowid,text) VALUES(new.rowid,new.text);
END;
CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
 INSERT INTO messages_fts(messages_fts,rowid,text) VALUES('delete',old.rowid,old.text);
END;
CREATE TRIGGER messages_au AFTER UPDATE OF text ON messages BEGIN
 INSERT INTO messages_fts(messages_fts,rowid,text) VALUES('delete',old.rowid,old.text);
 INSERT INTO messages_fts(rowid,text) VALUES(new.rowid,new.text);
END;
"""


class Store:
    def __init__(self, home: Path):
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = home / "persona.sqlite3"
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA secure_delete=ON")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            self.db.close()
            raise ValueError("Database is newer than this version of PersonaMCP")
        if version == 0:
            self.db.executescript("BEGIN;" + SCHEMA + "PRAGMA user_version=1;COMMIT;")
        if os.name != "nt":
            self.path.chmod(0o600)

    def close(self) -> None:
        self.db.close()

    def rows(self, sql: str, args: Iterable[Any] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute(sql, tuple(args))]

    def stats(self) -> dict[str, Any]:
        counts = {}
        for table in ("messages", "conversations", "interactions", "embeddings"):
            counts[table] = self.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        counts["user_messages"] = self.db.execute(
            "SELECT count(*) FROM messages WHERE is_user=1 AND kind='text' AND text!=''"
        ).fetchone()[0]
        counts["people"] = self.db.execute(
            "SELECT count(DISTINCT sender_key) FROM messages WHERE is_user=0"
        ).fetchone()[0]
        counts["platforms"] = dict(
            self.db.execute("SELECT platform,count(*) FROM messages GROUP BY platform")
        )
        counts["style_profile"] = bool(self.db.execute("SELECT 1 FROM profiles LIMIT 1").fetchone())
        return counts

    def invalidate(self) -> None:
        self.db.execute("DELETE FROM profiles")
        # Also remove generated text artifacts; no stale profile after identity or data changes.
        for name in ("persona.md", "persona-profile.json"):
            (self.path.parent / name).unlink(missing_ok=True)

    def import_conversations(
        self, conversations: list[Conversation], digest: str, platform: str, config: Config
    ) -> dict[str, int]:
        if self.db.execute("SELECT 1 FROM imports WHERE hash=?", (digest,)).fetchone():
            return {"added": 0, "duplicate_files": 1}
        before = self.stats()["messages"]
        touched: set[str] = set()
        with self.db:
            for conv in conversations:
                touched.add(conv.id)
                self.db.execute(
                    "INSERT OR IGNORE INTO conversations VALUES(?,?,?,?,?,?,?)",
                    (
                        conv.id,
                        conv.platform,
                        conv.external_id,
                        conv.title,
                        min((m.timestamp for m in conv.messages), default=None),
                        conv.metadata.get("context", ""),
                        json.dumps(conv.metadata),
                    ),
                )
                for person in set(conv.participants) | {m.sender for m in conv.messages}:
                    self.db.execute(
                        "INSERT OR IGNORE INTO participants VALUES(?,?,?)",
                        (conv.id, person, person.strip().casefold()),
                    )
                occurrences: Counter[str] = Counter()
                for sequence, msg in enumerate(conv.messages):
                    key = stable_id(conv.id, msg.sender, msg.timestamp, msg.text)
                    occurrences[key] += 1
                    message_id = (
                        stable_id(conv.id, msg.external_id)
                        if msg.external_id
                        else stable_id(key, occurrences[key])
                    )
                    is_user = msg.sender.strip().casefold() in config.identities(conv.platform)
                    if msg.sender_id:
                        is_user = is_user or msg.sender_id.casefold() in config.identities(
                            conv.platform
                        )
                    self.db.execute(
                        "INSERT OR IGNORE INTO messages VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            message_id,
                            conv.id,
                            msg.sender,
                            msg.sender.strip().casefold(),
                            msg.sender_id,
                            msg.text,
                            msg.timestamp,
                            msg.reply_to,
                            int(is_user),
                            msg.metadata.get("kind", "text"),
                            conv.platform,
                            json.dumps(msg.metadata),
                            int(msg.metadata.get("export_sequence", sequence)),
                        ),
                    )
            for conv_id in touched:
                self.rebuild_interactions(conv_id, config)
            added = self.stats()["messages"] - before
            self.db.execute(
                "INSERT INTO imports VALUES(?,?,?,?)",
                (digest, platform, datetime.now().astimezone().isoformat(), added),
            )
            if added:
                self.invalidate()
        return {"added": added, "duplicate_files": 0}

    def rebuild_interactions(self, conversation_id: str, config: Config) -> None:
        conv = self.rows("SELECT * FROM conversations WHERE id=?", (conversation_id,))[0]
        others = [
            p
            for p in self.rows(
                "SELECT * FROM participants WHERE conversation_id=?", (conversation_id,)
            )
            if p["name_key"] not in config.identities(conv["platform"])
        ]
        # One-to-one only; group history is not safe evidence about an individual recipient.
        person = others[0]["name"] if len(others) == 1 else None
        person_key = person.strip().casefold() if person else None
        self.db.execute("DELETE FROM interactions WHERE conversation_id=?", (conversation_id,))
        messages = self.rows(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY timestamp,sequence,id",
            (conversation_id,),
        )
        incoming: list[dict[str, Any]] = []
        replies: list[dict[str, Any]] = []

        def flush() -> None:
            if incoming and replies:
                ids = [m["id"] for m in replies]
                self.db.execute(
                    "INSERT INTO interactions VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        stable_id("interaction", *ids),
                        conversation_id,
                        "\n".join(m["text"] for m in incoming[-3:])[-1200:],
                        "\n".join(m["text"] for m in replies)[:2000],
                        person,
                        person_key,
                        conv["platform"],
                        replies[0]["timestamp"],
                        conv["context"],
                        json.dumps(ids),
                    ),
                )

        previous: dict[str, Any] | None = None
        for msg in messages:
            gap = (
                (
                    datetime.fromisoformat(msg["timestamp"])
                    - datetime.fromisoformat(previous["timestamp"])
                ).total_seconds()
                if previous
                else 0
            )
            if gap > 7200 or msg["kind"] != "text" or not msg["text"].strip():
                flush()
                incoming, replies = [], []
            if msg["kind"] != "text" or not msg["text"].strip():
                previous = msg
                continue
            if msg["is_user"]:
                if replies and (gap > 300 or len(replies) >= 8):
                    flush()
                    incoming, replies = [], []
                replies.append(msg)
            else:
                if replies:
                    flush()
                    incoming, replies = [], []
                incoming.append(msg)
                incoming = incoming[-3:]
            previous = msg
        flush()

    def reidentify(self, config: Config) -> None:
        with self.db:
            for row in self.rows("SELECT id,sender_key,sender_id,platform FROM messages"):
                identities = config.identities(row["platform"])
                own = row["sender_key"] in identities or (
                    row["sender_id"] and row["sender_id"].casefold() in identities
                )
                self.db.execute("UPDATE messages SET is_user=? WHERE id=?", (bool(own), row["id"]))
            for conv in self.rows("SELECT id FROM conversations"):
                self.rebuild_interactions(conv["id"], config)
            self.invalidate()

    def delete_conversations(self, ids: list[str]) -> int:
        if not ids:
            return 0
        with self.db:
            for conv_id in ids:
                self.db.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
            self.db.execute("DELETE FROM imports")  # Explicit re-import after deletion is possible.
            self.invalidate()
            self.db.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
            self.db.execute("INSERT INTO interactions_fts(interactions_fts) VALUES('rebuild')")
        self.db.execute("VACUUM")
        return len(ids)
