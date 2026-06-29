from __future__ import annotations

import json
import pickle
import sqlite3
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import FighterSearchResult, FighterStatsResponse, HealthResponse, PredictRequest, PredictResponse, ShapFeature


ARTIFACT_DIR = Path("models/artifacts")
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "ufc_raw.db"


class AppState:
    model: Any | None = None
    fill_values: pd.Series | None = None
    feature_cols: list[str] = []
    career_stats: pd.DataFrame = pd.DataFrame()
    fighters_df: pd.DataFrame = pd.DataFrame()
    fighter_refs: pd.DataFrame = pd.DataFrame()
    elo_table: dict[str, float] = {}
    composite_norm_params: dict[str, float] = {}
    shap_explainer: Any | None = None


state = AppState()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_fighters() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["fighter_id", "name", "height_cm", "reach_cm", "dob"])
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT fighter_id, name, height_cm, reach_cm, dob FROM fighters", conn)


def load_fighter_refs(fighters_df: pd.DataFrame) -> pd.DataFrame:
    refs_path = DATA_DIR / "fighter_refs.csv"
    if refs_path.exists():
        refs = pd.read_csv(refs_path)
        if "fighter_id" not in refs.columns and not fighters_df.empty:
            refs = refs.merge(fighters_df[["fighter_id", "name"]], on="name", how="left")
        return refs
    if fighters_df.empty:
        return pd.DataFrame(columns=["ref_no", "fighter_id", "name"])
    refs = fighters_df[["fighter_id", "name"]].drop_duplicates("fighter_id").sort_values(["name", "fighter_id"]).reset_index(drop=True)
    refs.insert(0, "ref_no", range(1, len(refs) + 1))
    return refs


def build_elo_table(features: pd.DataFrame) -> dict[str, float]:
    if features.empty:
        return {}
    table: dict[str, float] = {}
    ordered = features.sort_values("date")
    for _, row in ordered.iterrows():
        table[str(row["fighter_a_id"])] = float(row.get("fighter_a_elo", 1500.0))
        table[str(row["fighter_b_id"])] = float(row.get("fighter_b_elo", 1500.0))
    return table


def load_state() -> None:
    required = [ARTIFACT_DIR / "xgb_model.pkl", ARTIFACT_DIR / "fill_values.pkl", ARTIFACT_DIR / "feature_cols.json"]
    if all(path.exists() for path in required):
        state.model = load_pickle(required[0])
        state.fill_values = load_pickle(required[1])
        state.feature_cols = json.loads(required[2].read_text(encoding="utf-8"))
        try:
            import shap

            state.shap_explainer = shap.TreeExplainer(state.model)
        except ImportError:
            state.shap_explainer = None

    career_path = DATA_DIR / "fighter_career_stats.csv"
    state.career_stats = pd.read_csv(career_path, parse_dates=["date"]) if career_path.exists() else pd.DataFrame()
    features_path = DATA_DIR / "fights_features.csv"
    features = pd.read_csv(features_path, parse_dates=["date"]) if features_path.exists() else pd.DataFrame()
    state.elo_table = build_elo_table(features)
    state.fighters_df = load_fighters()
    if not state.fighters_df.empty:
        state.fighters_df["dob"] = pd.to_datetime(state.fighters_df["dob"], errors="coerce")
    state.fighter_refs = load_fighter_refs(state.fighters_df)
    norm_path = DATA_DIR / "composite_norm_params.json"
    state.composite_norm_params = json.loads(norm_path.read_text(encoding="utf-8")) if norm_path.exists() else {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_state()
    yield


app = FastAPI(title="UFC Fight Predictor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_model() -> None:
    if state.model is None or state.fill_values is None or not state.feature_cols:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded. Run python -m models.train first.")


def resolve_fighter_id(identifier: str | int | None, *, field_name: str = "fighter") -> str:
    if identifier is None or str(identifier).strip() == "":
        raise HTTPException(status_code=422, detail=f"Missing {field_name}.")
    value = str(identifier).strip()
    if not state.fighter_refs.empty and value.isdigit():
        rows = state.fighter_refs[state.fighter_refs["ref_no"].astype(str) == value]
        if not rows.empty and pd.notna(rows.iloc[0].get("fighter_id")):
            return str(rows.iloc[0]["fighter_id"])
    if not state.fighters_df.empty:
        rows = state.fighters_df[state.fighters_df["fighter_id"].astype(str) == value]
        if not rows.empty:
            return value
    raise HTTPException(status_code=422, detail=f"Fighter not found: {identifier}")


def fighter_latest_stats(fighter_id: str) -> pd.Series:
    if state.career_stats.empty:
        raise HTTPException(status_code=422, detail=f"Fighter not found: {fighter_id}")
    rows = state.career_stats[state.career_stats["fighter_id"].astype(str) == str(fighter_id)].sort_values("date")
    if rows.empty:
        raise HTTPException(status_code=422, detail=f"Fighter not found: {fighter_id}")
    return rows.iloc[-1]


def fighter_bio(fighter_id: str) -> pd.Series | None:
    if state.fighters_df.empty:
        return None
    rows = state.fighters_df[state.fighters_df["fighter_id"].astype(str) == str(fighter_id)]
    return rows.iloc[0] if not rows.empty else None


def fighter_ref_row(fighter_id: str) -> pd.Series | None:
    if state.fighter_refs.empty:
        return None
    rows = state.fighter_refs[state.fighter_refs["fighter_id"].astype(str) == str(fighter_id)]
    return rows.iloc[0] if not rows.empty else None


def age_years(dob: Any) -> float:
    if dob is None or pd.isna(dob):
        return np.nan
    today = pd.Timestamp(date.today())
    return float((today - pd.Timestamp(dob)).days / 365.25)


def add_prefixed_stats(features: dict[str, float], row: pd.Series, prefix: str) -> None:
    for key, value in row.items():
        if key in {"fight_id", "event_id", "date", "fighter_id", "opponent_id", "pre_fight_elo"}:
            continue
        if pd.api.types.is_number(value):
            features[f"{prefix}_{key}"] = float(value)


def norm_value(key: str) -> float:
    value = state.composite_norm_params.get(key, 0.0)
    if value is None or pd.isna(value) or float(value) <= 0:
        return 1.0
    return float(value)


def component(features: dict[str, float], key: str) -> float:
    value = features.get(key, np.nan)
    return 0.0 if pd.isna(value) else float(value)


def stat_component(stats: dict[str, Any], key: str) -> float:
    value = stats.get(key, np.nan)
    return 0.0 if pd.isna(value) else float(value)


def add_composite_features(features: dict[str, float]) -> None:
    max_sig_str_landed = norm_value("max_sig_str_landed")
    max_sig_str_absorbed = norm_value("max_sig_str_absorbed")
    max_td_landed = norm_value("max_td_landed")
    max_sub_att = norm_value("max_sub_att")
    for side in ["a", "b"]:
        features[f"{side}_striking_score"] = (
            0.35 * component(features, f"{side}_avg_sig_str_pct_5f")
            + 0.25 * component(features, f"{side}_avg_sig_str_def_5f")
            + 0.25 * (component(features, f"{side}_avg_sig_str_landed_5f") / max_sig_str_landed)
            + 0.15 * (1.0 / (1.0 + component(features, f"{side}_avg_sig_str_absorbed_5f") / max_sig_str_absorbed))
        )
        features[f"{side}_grappling_score"] = (
            0.40 * component(features, f"{side}_avg_td_pct_5f")
            + 0.30 * component(features, f"{side}_avg_td_def_5f")
            + 0.20 * (component(features, f"{side}_avg_td_landed_5f") / max_td_landed)
            + 0.10 * (component(features, f"{side}_avg_sub_att_5f") / max_sub_att)
        )
    features["diff_striking_score"] = features["a_striking_score"] - features["b_striking_score"]
    features["diff_grappling_score"] = features["a_grappling_score"] - features["b_grappling_score"]


def add_single_fighter_scores(stats: dict[str, Any]) -> None:
    max_sig_str_landed = norm_value("max_sig_str_landed")
    max_sig_str_absorbed = norm_value("max_sig_str_absorbed")
    max_td_landed = norm_value("max_td_landed")
    max_sub_att = norm_value("max_sub_att")
    stats["striking_score"] = (
        0.35 * stat_component(stats, "avg_sig_str_pct_5f")
        + 0.25 * stat_component(stats, "avg_sig_str_def_5f")
        + 0.25 * (stat_component(stats, "avg_sig_str_landed_5f") / max_sig_str_landed)
        + 0.15 * (1.0 / (1.0 + stat_component(stats, "avg_sig_str_absorbed_5f") / max_sig_str_absorbed))
    )
    stats["grappling_score"] = (
        0.40 * stat_component(stats, "avg_td_pct_5f")
        + 0.30 * stat_component(stats, "avg_td_def_5f")
        + 0.20 * (stat_component(stats, "avg_td_landed_5f") / max_td_landed)
        + 0.10 * (stat_component(stats, "avg_sub_att_5f") / max_sub_att)
    )


def assemble_features(fighter_a_id: str, fighter_b_id: str) -> pd.DataFrame:
    a_stats = fighter_latest_stats(fighter_a_id)
    b_stats = fighter_latest_stats(fighter_b_id)
    features: dict[str, float] = {}
    add_prefixed_stats(features, a_stats, "a")
    add_prefixed_stats(features, b_stats, "b")

    for key in list(features):
        if key.startswith("a_avg_"):
            base = key.removeprefix("a_")
            b_key = f"b_{base}"
            if b_key in features:
                features[f"diff_{base}"] = features[key] - features[b_key]

    features["fighter_a_elo"] = state.elo_table.get(str(fighter_a_id), float(a_stats.get("pre_fight_elo", 1500.0)))
    features["fighter_b_elo"] = state.elo_table.get(str(fighter_b_id), float(b_stats.get("pre_fight_elo", 1500.0)))
    features["elo_diff"] = features["fighter_a_elo"] - features["fighter_b_elo"]
    features["diff_win_streak"] = features.get("a_win_streak", np.nan) - features.get("b_win_streak", np.nan)
    for key in ["finish_rate", "ko_rate", "sub_rate", "ko_loss_rate", "sub_loss_rate"]:
        features[f"diff_{key}"] = features.get(f"a_{key}", np.nan) - features.get(f"b_{key}", np.nan)

    a_bio = fighter_bio(fighter_a_id)
    b_bio = fighter_bio(fighter_b_id)
    features["reach_diff"] = (float(a_bio["reach_cm"]) if a_bio is not None and pd.notna(a_bio["reach_cm"]) else np.nan) - (
        float(b_bio["reach_cm"]) if b_bio is not None and pd.notna(b_bio["reach_cm"]) else np.nan
    )
    features["height_diff"] = (float(a_bio["height_cm"]) if a_bio is not None and pd.notna(a_bio["height_cm"]) else np.nan) - (
        float(b_bio["height_cm"]) if b_bio is not None and pd.notna(b_bio["height_cm"]) else np.nan
    )
    a_age = age_years(a_bio["dob"]) if a_bio is not None else np.nan
    b_age = age_years(b_bio["dob"]) if b_bio is not None else np.nan
    features["a_age"] = a_age
    features["b_age"] = b_age
    features["age_diff"] = a_age - b_age
    add_composite_features(features)

    return pd.DataFrame([{col: features.get(col, np.nan) for col in state.feature_cols}]).fillna(state.fill_values)


def shap_top_features(X: pd.DataFrame) -> list[ShapFeature]:
    if state.shap_explainer is None:
        return []
    shap_values = state.shap_explainer.shap_values(X)
    values = shap_values[1][0] if isinstance(shap_values, list) and len(shap_values) > 1 else np.asarray(shap_values)[0]
    order = np.argsort(np.abs(values))[::-1][:5]
    return [ShapFeature(feature=state.feature_cols[i], impact=round(float(values[i]), 6)) for i in order]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=state.model is not None)


@app.get("/fighters", response_model=list[FighterSearchResult])
def search_fighters(q: str = Query(default="", min_length=0)) -> list[FighterSearchResult]:
    if state.fighters_df.empty:
        return []
    query = q.strip().lower()
    df = state.fighters_df
    if query:
        df = df[df["name"].str.lower().str.contains(query, na=False)]
    if not state.fighter_refs.empty:
        df = df.merge(state.fighter_refs[["fighter_id", "ref_no"]], on="fighter_id", how="left")
    else:
        df = df.assign(ref_no=np.nan)
    return [
        FighterSearchResult(fighter_id=str(row.fighter_id), name=str(row.name), ref_no=None if pd.isna(row.ref_no) else int(row.ref_no))
        for row in df.sort_values("name").head(20).itertuples()
    ]


@app.get("/fighters/ref/{ref_no}", response_model=FighterSearchResult)
def get_fighter_by_ref(ref_no: int) -> FighterSearchResult:
    fighter_id = resolve_fighter_id(ref_no, field_name="ref_no")
    ref = fighter_ref_row(fighter_id)
    bio = fighter_bio(fighter_id)
    name = str(ref["name"]) if ref is not None and pd.notna(ref.get("name")) else str(bio["name"]) if bio is not None else fighter_id
    return FighterSearchResult(fighter_id=fighter_id, name=name, ref_no=ref_no)


@app.get("/fighter/{fighter_id}/stats", response_model=FighterStatsResponse)
def get_fighter_stats(fighter_id: str) -> FighterStatsResponse:
    fighter_id = resolve_fighter_id(fighter_id, field_name="fighter")
    try:
        row = fighter_latest_stats(fighter_id)
    except HTTPException as exc:
        raise HTTPException(status_code=404, detail="Fighter not found") from exc
    stats = {key: (None if pd.isna(value) else value) for key, value in row.to_dict().items()}
    bio = fighter_bio(fighter_id)
    if bio is not None:
        for key in ["name", "height_cm", "reach_cm", "dob"]:
            value = bio.get(key)
            stats[key] = None if pd.isna(value) else str(value) if key == "dob" else value
    ref = fighter_ref_row(fighter_id)
    if ref is not None and pd.notna(ref.get("ref_no")):
        stats["ref_no"] = int(ref["ref_no"])
    add_single_fighter_scores(stats)
    return FighterStatsResponse(fighter_id=fighter_id, stats=stats)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    require_model()
    fighter_a_id = resolve_fighter_id(request.fighter_a_id or request.fighter_a_ref_no, field_name="fighter_a")
    fighter_b_id = resolve_fighter_id(request.fighter_b_id or request.fighter_b_ref_no, field_name="fighter_b")
    if fighter_a_id == fighter_b_id:
        raise HTTPException(status_code=422, detail="Choose two different fighters.")
    X = assemble_features(fighter_a_id, fighter_b_id)
    prob_a = float(state.model.predict_proba(X)[0, 1])
    distance = abs(prob_a - 0.5)
    confidence = "high" if distance > 0.2 else "medium" if distance > 0.1 else "low"
    return PredictResponse(
        fighter_a_win_prob=round(prob_a, 4),
        fighter_b_win_prob=round(1.0 - prob_a, 4),
        top_shap_features=shap_top_features(X),
        confidence=confidence,
    )
