"""Adapters parse data, never execute export scripts or follow links."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from zipfile import ZipFile

from personamcp.importers.html import instagram_html, snapchat_html
from personamcp.models import Conversation, Message, stable_id, timestamp

MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_IMPORT_BYTES = 512 * 1024 * 1024


class Importer(Protocol):
    def parse(self, text: str, source: str) -> list[Conversation]: ...


def read_sources(path: Path, platform: str) -> Iterator[tuple[str, str]]:
    if path.is_symlink():
        raise ValueError("Symlink import roots are not supported")
    root = path.resolve(strict=True)
    if not root.is_file() and not root.is_dir():
        raise ValueError("Import path must be a regular file or directory")
    total = 0
    files = sorted(root.rglob("*")) if root.is_dir() else [root]
    for file in files:
        if file.is_symlink() or not file.is_file():
            continue
        if root.is_dir() and not file.resolve().is_relative_to(root):
            raise ValueError("Import path escapes the selected directory")
        suffix = file.suffix.lower()
        if suffix == ".zip" and platform == "snapchat":
            with ZipFile(file) as archive:
                json_history = any(
                    "chat_history" in i.filename and i.filename.endswith(".json")
                    for i in archive.infolist()
                )
                for info in sorted(archive.infolist(), key=lambda i: i.filename):
                    name = PurePosixPath(info.filename)
                    if (
                        name.is_absolute()
                        or ".." in name.parts
                        or "\\" in info.filename
                        or re.match(r"^[a-zA-Z]:", info.filename)
                    ):
                        raise ValueError("Unsafe archive member path")
                    if not (
                        "chat_history" in info.filename.lower()
                        and name.suffix.lower() in {".html", ".json"}
                    ):
                        continue
                    if json_history and name.suffix.lower() == ".html":
                        continue
                    total += info.file_size
                    if info.file_size > MAX_FILE_BYTES or total > MAX_IMPORT_BYTES:
                        raise ValueError("Chat data exceeds safe import size limits")
                    yield info.filename, archive.read(info).decode("utf-8-sig")
            continue
        allowed = {
            "instagram": {".json", ".html"},
            "snapchat": {".json", ".html"},
            "whatsapp": {".txt"},
            "json": {".json", ".jsonl"},
        }[platform]
        if suffix not in allowed:
            continue
        if (
            root.is_dir()
            and platform == "instagram"
            and not re.fullmatch(r"message_\d+\.(json|html)", file.name)
        ):
            continue
        if root.is_dir() and platform == "snapchat" and "chat_history" not in str(file):
            continue
        size = file.stat().st_size
        total += size
        if size > MAX_FILE_BYTES or total > MAX_IMPORT_BYTES:
            raise ValueError("Chat data exceeds safe import size limits")
        yield (
            file.relative_to(root).as_posix() if root.is_dir() else file.name,
            file.read_text(encoding="utf-8-sig"),
        )


def meta_text(text: str) -> str:
    # Meta JSON can contain UTF-8 bytes decoded as Latin-1. Leave genuine Unicode alone.
    if any(char in text for char in ("Ã", "Â", "ð")):
        try:
            return text.encode("latin1").decode("utf-8")
        except (UnicodeError, ValueError):
            pass
    return text


def json_data(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON at line {exc.lineno}, column {exc.colno}") from None
    except RecursionError:
        raise ValueError("JSON nesting exceeds safe parser depth") from None


class InstagramImporter:
    def parse(self, text: str, source: str) -> list[Conversation]:
        if source.endswith(".html"):
            return instagram_html(text, source)
        data = json_data(text)
        if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
            raise ValueError("Expected an Instagram message_N.json conversation")
        people = data.get("participants", [])
        if not isinstance(people, list) or not all(
            isinstance(p, dict) and isinstance(p.get("name"), str) for p in people
        ):
            raise ValueError("Instagram participants must contain named objects")
        participants = [meta_text(p["name"]) for p in people]
        messages: list[Message] = []
        page_match = re.search(r"message_(\d+)\.json$", source)
        page = int(page_match.group(1)) if page_match else 1
        for row in data["messages"]:
            if not isinstance(row, dict) or not isinstance(row.get("sender_name"), str):
                raise ValueError("Instagram message has no sender_name")
            if not isinstance(row.get("content", ""), str):
                raise ValueError("Instagram message content must be a string")
            body = meta_text(str(row.get("content", "")))
            # Media/system records remain for ordering, but never enter style statistics.
            kind = "text" if body and row.get("type", "Generic") == "Generic" else "nontext"
            messages.append(
                Message(
                    meta_text(row["sender_name"]),
                    body,
                    timestamp(row["timestamp_ms"]),
                    external_id=str(row["id"]) if "id" in row else None,
                    metadata={
                        "kind": kind,
                        "export_sequence": -(page * 1_000_000_000 + len(messages)),
                    },
                )
            )
        external = str(data.get("thread_path") or str(PurePosixPath(source).parent))
        return [
            Conversation(
                "instagram",
                external,
                meta_text(data.get("title", external)),
                participants,
                messages,
            )
        ]


class GenericImporter:
    def parse(self, text: str, source: str) -> list[Conversation]:
        data = (
            [json_data(line) for line in text.splitlines() if line.strip()]
            if source.endswith(".jsonl")
            else json_data(text)
        )
        if isinstance(data, dict):
            if "conversations" in data:
                conversations = data["conversations"]
            elif "messages" in data:
                conversations = [data]
            else:
                raise ValueError("JSON requires messages or conversations")
        elif isinstance(data, list):
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in data:
                if not isinstance(row, dict):
                    raise ValueError("Each JSONL/JSON message must be an object")
                grouped.setdefault(str(row.get("conversation_id", "default")), []).append(row)
            conversations = [{"id": key, "messages": rows} for key, rows in grouped.items()]
        else:
            raise ValueError("Expected a JSON object or array")
        result = []
        if not isinstance(conversations, list):
            raise ValueError("conversations must be an array")
        for conv in conversations:
            if not isinstance(conv, dict) or not isinstance(conv.get("messages"), list):
                raise ValueError("Conversation messages must be an array")
            rows = conv["messages"]
            messages = []
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("sender"), str):
                    raise ValueError("Message requires a string sender")
                if not isinstance(row.get("text"), str) or "timestamp" not in row:
                    raise ValueError("Message requires text and timestamp")
                messages.append(
                    Message(
                        row["sender"],
                        row["text"],
                        timestamp(row["timestamp"]),
                        sender_id=str(row["sender_id"]) if row.get("sender_id") else None,
                        external_id=str(row["id"]) if row.get("id") else None,
                        reply_to=str(row["reply_to"]) if row.get("reply_to") else None,
                        metadata={"kind": "text"},
                    )
                )
            people = conv.get("participants") or sorted({m.sender for m in messages})
            if not isinstance(people, list) or not all(isinstance(p, str) for p in people):
                raise ValueError("participants must be an array of names")
            result.append(
                Conversation(
                    str(conv.get("platform", "json")),
                    str(conv.get("id", "default")),
                    str(conv.get("title", "Imported conversation")),
                    people,
                    messages,
                    {"context": str(conv.get("context", ""))},
                )
            )
        return result


class SnapchatImporter:
    def parse(self, text: str, source: str) -> list[Conversation]:
        if source.endswith(".html"):
            return snapchat_html(text, source)
        data = json_data(text)
        if not isinstance(data, dict):
            raise ValueError("Expected Snapchat chat-history JSON object")
        grouped: dict[str, list[Message]] = {}
        sections = [data] if any(isinstance(v, list) for v in data.values()) else data.values()
        for threads in sections:
            if not isinstance(threads, dict):
                continue
            for person, rows in threads.items():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        raise ValueError("Snapchat chat record must be an object")
                    sender = row.get("From") or row.get("Sender")
                    date = row.get("Created") or row.get("Date")
                    if not isinstance(sender, str) or not sender.strip() or not date:
                        raise ValueError("Unsupported Snapchat JSON record; missing From/Created")
                    if not isinstance(row.get("Content", ""), str):
                        raise ValueError("Snapchat Content must be a string")
                    body = str(row.get("Content", ""))
                    kind = str(row.get("Media Type", "TEXT")).upper()
                    grouped.setdefault(str(person), []).append(
                        Message(
                            str(sender),
                            body if kind == "TEXT" else "",
                            timestamp(date),
                            metadata={"kind": "text" if kind == "TEXT" else "nontext"},
                        )
                    )
        if not grouped:
            raise ValueError("Unsupported Snapchat JSON layout")
        return [
            Conversation(
                "snapchat",
                person,
                person,
                sorted({person, *(m.sender for m in messages)}),
                messages,
            )
            for person, messages in grouped.items()
        ]


class WhatsAppImporter:
    pattern = re.compile(
        r"^\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s+"
        r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*(?:-\s*)?(.*)$"
    )

    def __init__(self, day_first: bool = True, conversation_id: str | None = None):
        self.day_first = day_first
        self.conversation_id = conversation_id

    def parse(self, text: str, source: str) -> list[Conversation]:
        messages: list[Message] = []
        current: Message | None = None
        for line in text.splitlines():
            line = line.lstrip("\u200e\u200f").replace("\u202f", " ")
            match = self.pattern.match(line)
            if match:
                current = None
                date, time, content = match.groups()
                if ": " not in content:
                    continue  # System notices are not authored messages.
                sender, body = content.split(": ", 1)
                current = Message(
                    sender,
                    body,
                    timestamp(f"{date} {time}", day_first=self.day_first),
                    metadata={"kind": "text"},
                )
                if body in {"<Media omitted>", "image omitted", "video omitted"}:
                    current.text = ""
                    current.metadata["kind"] = "nontext"
                messages.append(current)
            elif current is not None:
                current.text += "\n" + line
        if not messages:
            raise ValueError("No WhatsApp messages found; supported dates use slash separators")
        external = self.conversation_id or stable_id("whatsapp-file", PurePosixPath(source).name)
        return [
            Conversation(
                "whatsapp",
                external,
                Path(source).stem,
                sorted({m.sender for m in messages}),
                messages,
            )
        ]


def importer_for(
    platform: str, *, day_first: bool = True, conversation_id: str | None = None
) -> Importer:
    adapters: dict[str, Importer] = {
        "instagram": InstagramImporter(),
        "snapchat": SnapchatImporter(),
        "whatsapp": WhatsAppImporter(day_first, conversation_id),
        "json": GenericImporter(),
    }
    if platform not in adapters:
        raise ValueError("Supported imports: instagram, snapchat, whatsapp, json")
    return adapters[platform]
