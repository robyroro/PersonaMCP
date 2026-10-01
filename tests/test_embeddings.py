from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from personamcp.config import Config
from personamcp.embeddings import LocalEmbeddingProvider, prepare_model


def test_unprepared_model_fails_without_network(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prepare"):
        LocalEmbeddingProvider(tmp_path, Config()).encode(["private text"])


def test_local_model_loads_only_offline_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "model").mkdir()
    loads: list[dict[str, Any]] = []

    class FakeModel:
        def __init__(self, path: str, **kwargs: Any):
            assert Path(path) == tmp_path / "model"
            loads.append(kwargs)

        def encode(self, texts: list[str], **kwargs: Any) -> Any:
            assert kwargs["normalize_embeddings"]
            return SimpleNamespace(tolist=lambda: [[1.0, 0.0] for _ in texts])

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeModel)
    )
    provider = LocalEmbeddingProvider(tmp_path, Config(model_revision="pinned"))
    assert provider.encode(["sensitive text"]) == [[1.0, 0.0]]
    provider.encode(["other text"])
    assert loads == [{"local_files_only": True, "trust_remote_code": False, "device": "cpu"}]


def test_prepare_downloads_only_pinned_model_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class FakeApi:
        def model_info(self, name: str) -> Any:
            return SimpleNamespace(sha="immutable-revision")

    def download(name: str, **kwargs: Any) -> None:
        calls.append((name, kwargs))

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi, snapshot_download=download)
    )
    config = Config(name="Sensitive Owner")
    assert prepare_model(tmp_path, config) == "immutable-revision"
    assert calls[0][1]["revision"] == "immutable-revision"
    assert "Sensitive Owner" not in str(calls)
    assert Config.load(tmp_path).model_revision == "immutable-revision"
