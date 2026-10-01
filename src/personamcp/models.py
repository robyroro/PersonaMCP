from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def stable_id(*parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def timestamp(value: object, *, day_first: bool = True) -> str:
    if isinstance(value, (int, float)):
        seconds = value / 1000 if abs(value) > 10**11 else value
        dt = datetime.fromtimestamp(seconds, UTC)
    else:
        text = (
            str(value)
            .strip()
            .replace("\u202f", " ")
            .replace("\u00a0", " ")
            .replace(" UTC", "+00:00")
            .replace("Z", "+00:00")
        )
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            formats = ["%b %d, %Y %I:%M %p", "%B %d, %Y %I:%M %p", "%Y-%m-%d %H:%M:%S"]
            formats += ["%b %d, %Y, %I:%M %p", "%b %d, %Y, %I:%M:%S %p"]
            date = "%d/%m/%Y" if day_first else "%m/%d/%Y"
            formats += [
                f"{date} %H:%M",
                f"{date} %H:%M:%S",
                f"{date} %I:%M %p",
                f"{date} %I:%M:%S %p",
            ]
            short = date.replace("%Y", "%y")
            formats += [
                f"{short} %H:%M",
                f"{short} %H:%M:%S",
                f"{short} %I:%M %p",
                f"{short} %I:%M:%S %p",
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(
                    "Unsupported timestamp format; choose the correct export/date format"
                )
    if dt.tzinfo is None:
        # Offsetless exports retain wall-clock ordering, using a documented UTC convention.
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


@dataclass
class Message:
    sender: str
    text: str
    timestamp: str
    sender_id: str | None = None
    external_id: str | None = None
    reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    platform: str
    external_id: str
    title: str
    participants: list[str]
    messages: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return stable_id(self.platform, self.external_id)
