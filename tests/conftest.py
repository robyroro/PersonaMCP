from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from personamcp.config import Config
from personamcp.models import Conversation, Message
from personamcp.storage import Store


@pytest.fixture
def config() -> Config:
    return Config(name="Owner", aliases=["my-id"], min_profile_messages=3)


def conversation(
    person: str = "Alex",
    *,
    platform: str = "json",
    count: int = 30,
    context: str = "casual",
    group: bool = False,
) -> Conversation:
    messages = []
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(count):
        when = start + timedelta(days=i)
        messages.extend(
            [
                Message(person, "mai vii azi la cafea?", when.isoformat()),
                Message(
                    "Owner",
                    f"da gen vin acu =)) {person.casefold()}",
                    (when + timedelta(seconds=10)).isoformat(),
                ),
                Message("Owner", "poate mai tarziu", (when + timedelta(seconds=20)).isoformat()),
            ]
        )
    return Conversation(
        platform,
        f"chat-{person}",
        person,
        ["Owner", person, *(["Blair"] if group else [])],
        messages,
        {"context": context},
    )


@pytest.fixture
def store(tmp_path: Path, config: Config) -> Any:
    config.save(tmp_path)
    db = Store(tmp_path)
    db.import_conversations(
        [conversation(), conversation("Blair", context="business")], "initial", "json", config
    )
    yield db
    db.close()


class TestEmbedding:
    """Deterministic vector fixture to exercise indexing/filtering; not a semantic model."""

    identity = "test-vector-v1"

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float("cafea" in t), float("pizza" in t), 0.1] for t in texts]
