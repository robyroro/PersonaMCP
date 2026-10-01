from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from personamcp.config import Config


class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class LocalEmbeddingProvider:
    """Offline multilingual MiniLM. Model preparation is a separate, explicit action."""

    def __init__(self, home: Path, config: Config):
        self.home = home
        self.config = config
        self._model: Any = None
        self._lock = threading.RLock()

    @property
    def identity(self) -> str:
        return f"sentence-transformers:{self.config.embedding_model}@{self.config.model_revision}"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            return self._encode(texts)

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.config.model_revision or not (self.home / "model").is_dir():
            raise ValueError("Local model unavailable; run persona model prepare explicitly")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ValueError(
                "Install the semantic extra: pip install 'personamcp[semantic]'"
            ) from None
        if self._model is None:
            self._model = SentenceTransformer(
                str(self.home / "model"),
                local_files_only=True,
                trust_remote_code=False,
                device="cpu",
            )
        return self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=32
        ).tolist()


def prepare_model(home: Path, config: Config) -> str:
    """Downloads model files only, never receives conversation text."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        raise ValueError("Install the semantic extra before preparing a model") from None
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    revision = HfApi().model_info(config.embedding_model).sha
    if not revision:
        raise ValueError("Could not resolve an immutable model revision")
    snapshot_download(
        config.embedding_model,
        revision=revision,
        local_dir=home / "model",
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "1_Pooling/*"],
    )
    config.model_revision = revision
    config.save(home)
    return revision
