from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from platformdirs import user_data_path


def data_dir() -> Path:
    return Path(os.environ.get("PERSONAMCP_HOME", user_data_path("personamcp", appauthor=False)))


@dataclass
class Config:
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    platform_aliases: dict[str, list[str]] = field(default_factory=dict)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_revision: str | None = None
    min_profile_messages: int = 20

    def identities(self, platform: str) -> set[str]:
        values = self.platform_aliases.get(platform, [self.name, *self.aliases])
        return {v.strip().casefold() for v in values if v.strip()}

    @classmethod
    def load(cls, home: Path) -> Config:
        path = home / "config.json"
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, home: Path) -> None:
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = home / "config.json"
        temporary = home / "config.json.tmp"
        temporary.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
        if os.name != "nt":
            path.chmod(0o600)
