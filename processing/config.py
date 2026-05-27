from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _find_project_config() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parents[1] / "config.yaml"


def _load_config_data() -> dict[str, Any]:
    config_path = _find_project_config()
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {config_path}")

    return data


def _require_config_value(data: dict[str, Any], config_key: str) -> Any:
    if config_key not in data:
        raise KeyError(f"Missing required config key: {config_key}")
    value = data[config_key]
    if value is None:
        raise ValueError(f"Config key cannot be null: {config_key}")
    return value


@dataclass(frozen=True)
class Settings:
    collection_name: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    embed_model: str = ""
    chunk_max_tokens: int = 0
    chunk_overlap_tokens: int = 0
    embedding_dim: int = 0
    database_url: str = ""

    @classmethod
    def from_yaml(cls) -> "Settings":
        data = _load_config_data()
        return cls(
            collection_name=str(_require_config_value(data, "collection_name")),
            qdrant_url=str(_require_config_value(data, "qdrant_url")),
            qdrant_api_key=str(_require_config_value(data, "qdrant_api_key")),
            embed_model=str(_require_config_value(data, "embed_model")),
            chunk_max_tokens=int(_require_config_value(data, "chunk_max_tokens")),
            chunk_overlap_tokens=int(_require_config_value(data, "chunk_overlap_tokens")),
            embedding_dim=int(_require_config_value(data, "embedding_dim")),
            database_url=str(_require_config_value(data, "database_url")),
        )


settings = Settings.from_yaml()