from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _find_project_config() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parents[2] / "config.yaml"


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


def _require_config_list(data: dict[str, Any], config_key: str) -> list[str]:
    value = _require_config_value(data, config_key)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    raise TypeError(f"Config key must be a list or comma-separated string: {config_key}")


@dataclass(frozen=True)
class Settings:
    collection_name: str
    groq_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    frontend_origins: list[str] = field(default_factory=list)
    llm_model: str = ""
    embed_model: str = ""
    chunk_max_tokens: int = 0
    chunk_overlap_tokens: int = 0
    embedding_dim: int = 0
    default_top_k: int = 0
    minimum_relevance_score: float = 0.0

    @classmethod
    def from_yaml(cls) -> "Settings":
        data = _load_config_data()
        return cls(
            collection_name=str(_require_config_value(data, "collection_name")),
            groq_api_key=str(_require_config_value(data, "groq_api_key")),
            qdrant_url=str(_require_config_value(data, "qdrant_url")),
            qdrant_api_key=str(_require_config_value(data, "qdrant_api_key")),
            frontend_origins=_require_config_list(data, "frontend_origins"),
            llm_model=str(_require_config_value(data, "llm_model")),
            embed_model=str(_require_config_value(data, "embed_model")),
            chunk_max_tokens=int(_require_config_value(data, "chunk_max_tokens")),
            chunk_overlap_tokens=int(_require_config_value(data, "chunk_overlap_tokens")),
            embedding_dim=int(_require_config_value(data, "embedding_dim")),
            default_top_k=int(_require_config_value(data, "default_top_k")),
            minimum_relevance_score=float(_require_config_value(data, "minimum_relevance_score")),
        )


settings = Settings.from_yaml()
