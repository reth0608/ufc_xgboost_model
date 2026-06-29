from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    fighter_a_id: str | None = None
    fighter_b_id: str | None = None
    fighter_a_ref_no: int | None = None
    fighter_b_ref_no: int | None = None


class ShapFeature(BaseModel):
    feature: str
    impact: float


class PredictResponse(BaseModel):
    fighter_a_win_prob: float
    fighter_b_win_prob: float
    top_shap_features: list[ShapFeature]
    confidence: Literal["low", "medium", "high"]


class FighterSearchResult(BaseModel):
    fighter_id: str
    name: str
    ref_no: int | None = None


class FighterStatsResponse(BaseModel):
    model_config = {"extra": "allow"}

    fighter_id: str
    stats: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
