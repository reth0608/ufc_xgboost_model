from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DB_PATH = Path("data/ufc_raw.db")
DATA_DIR = Path("data")
RANDOM_SEED = 42
ROLLING_STATS = [
    "sig_str_pct",
    "td_pct",
    "kd",
    "ctrl_sec",
    "sub_att",
    "sig_str_landed",
    "td_landed",
    "sig_str_absorbed",
    "sig_str_def",
    "td_def",
]
WEIGHT_CLASS_KG = {
    "strawweight": 52,
    "flyweight": 57,
    "bantamweight": 61,
    "featherweight": 66,
    "lightweight": 70,
    "welterweight": 77,
    "middleweight": 84,
    "light heavyweight": 93,
    "heavyweight": 120,
}
DROPPED_FEATURES = {
    "b_win_streak",
    "a_avg_kd_5f",
    "a_avg_td_landed_3f",
    "b_avg_sub_att_5f",
    "b_avg_kd_5f",
    "b_avg_td_landed_3f",
    "diff_avg_td_landed_3f",
    "b_sub_rate",
    "a_finish_rate",
    "b_ko_rate",
    "a_sub_loss_rate",
    "a_avg_sub_att_5f",
    "diff_avg_kd_3f",
    "diff_avg_sub_att_5f",
    "b_sub_loss_rate",
    "a_avg_kd_3f",
    "b_finish_rate",
    "diff_avg_sub_att_3f",
    "b_avg_kd_3f",
    "a_avg_sub_att_3f",
    "b_avg_sub_att_3f",
    "is_title_fight",
}


def build_fighter_refs(fighters: pd.DataFrame) -> pd.DataFrame:
    refs = fighters[["fighter_id", "name"]].drop_duplicates("fighter_id").copy()
    refs = refs.sort_values(["name", "fighter_id"], na_position="last").reset_index(drop=True)
    refs.insert(0, "ref_no", np.arange(1, len(refs) + 1))
    return refs[["ref_no", "fighter_id", "name"]]


@dataclass
class FighterHistory:
    fights: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    streak: int = 0
    ko_wins: int = 0
    sub_wins: int = 0
    ko_losses: int = 0
    sub_losses: int = 0
    values: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))


def parse_landed_attempted(value: Any) -> tuple[int, int]:
    if value is None or pd.isna(value) or str(value).strip() == "---":
        return 0, 0
    match = re.search(r"(\d+)\s+of\s+(\d+)", str(value))
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def parse_pct(value: Any) -> float | None:
    if value is None or pd.isna(value) or str(value).strip() == "---":
        return None
    match = re.search(r"(\d+(?:\.\d+)?)%", str(value))
    return float(match.group(1)) / 100.0 if match else None


def parse_time_to_seconds(value: Any) -> int:
    if value is None or pd.isna(value) or str(value).strip() == "---":
        return 0
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        return 0
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return 0


def load_tables(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        fights = pd.read_sql_query("SELECT * FROM fights", conn)
        fighters = pd.read_sql_query("SELECT * FROM fighters", conn)
        events = pd.read_sql_query("SELECT * FROM events", conn)
        rounds = pd.read_sql_query("SELECT * FROM rounds", conn)
    return fights, fighters, events, rounds


def parse_raw_fight_columns(fights: pd.DataFrame) -> pd.DataFrame:
    df = fights.copy()
    for side in ["a", "b"]:
        for base in ["sig_str", "total_str", "td", "head", "body", "leg", "distance", "clinch", "ground"]:
            landed_attempted = df[f"{side}_{base}"].apply(parse_landed_attempted)
            df[f"{side}_{base}_landed"] = landed_attempted.apply(lambda x: x[0])
            df[f"{side}_{base}_att"] = landed_attempted.apply(lambda x: x[1])
        df[f"{side}_ctrl_sec"] = df[f"{side}_ctrl"].apply(parse_time_to_seconds)
        df[f"{side}_sig_str_pct"] = df[f"{side}_sig_str_pct"].apply(parse_pct)
        df[f"{side}_td_pct"] = df[f"{side}_td_pct"].apply(parse_pct)
    return df


def normalize_event_dates(fights: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    event_dates = events[["event_id", "date"]].rename(columns={"date": "event_date"})
    df = fights.merge(event_dates, on="event_id", how="left")
    df["date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.drop(columns=["event_date"])
    return df.sort_values(["date", "event_id", "fight_id"], na_position="last").reset_index(drop=True)


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent_rating - rating) / 400.0))


def is_ko(method: Any) -> bool:
    if method is None or pd.isna(method):
        return False
    text = str(method).lower()
    return "ko" in text or "tko" in text


def is_sub(method: Any) -> bool:
    if method is None or pd.isna(method):
        return False
    return "sub" in str(method).lower()


def rolling_snapshot(history: FighterHistory) -> dict[str, float | int | None]:
    row: dict[str, float | int | None] = {
        "fights_in_ufc": history.fights,
        "win_rate": history.wins / history.fights if history.fights else np.nan,
        "win_streak": history.streak,
        "finish_rate": (history.ko_wins + history.sub_wins) / history.wins if history.wins else 0.0,
        "ko_rate": history.ko_wins / history.fights if history.fights else 0.0,
        "sub_rate": history.sub_wins / history.fights if history.fights else 0.0,
        "ko_loss_rate": history.ko_losses / history.losses if history.losses else 0.0,
        "sub_loss_rate": history.sub_losses / history.losses if history.losses else 0.0,
    }
    for stat in ROLLING_STATS:
        values = list(history.values[stat])
        for window in [5, 3]:
            prior_values = values[-window:]
            row[f"avg_{stat}_{window}f"] = float(np.mean(prior_values)) if prior_values else np.nan
    return row


def update_history(history: FighterHistory, stats: dict[str, float], won: bool | None, method: str | None = None) -> None:
    history.fights += 1
    if won is True:
        history.wins += 1
        if is_ko(method):
            history.ko_wins += 1
        elif is_sub(method):
            history.sub_wins += 1
        history.streak = history.streak + 1 if history.streak > 0 else 1
    elif won is False:
        history.losses += 1
        if is_ko(method):
            history.ko_losses += 1
        elif is_sub(method):
            history.sub_losses += 1
        history.streak = history.streak - 1 if history.streak < 0 else -1
    else:
        history.draws += 1
    for stat in ROLLING_STATS:
        value = stats.get(stat)
        if value is not None and not pd.isna(value):
            history.values[stat].append(float(value))


def safe_defense_rate(landed: Any, attempted: Any) -> float | None:
    if attempted is None or pd.isna(attempted) or float(attempted) <= 0:
        return None
    if landed is None or pd.isna(landed):
        return None
    return 1.0 - (float(landed) / float(attempted))


def side_stats(row: pd.Series, side: str) -> dict[str, float]:
    opponent = "b" if side == "a" else "a"
    return {
        "sig_str_pct": row[f"{side}_sig_str_pct"],
        "td_pct": row[f"{side}_td_pct"],
        "kd": row[f"{side}_kd"],
        "ctrl_sec": row[f"{side}_ctrl_sec"],
        "sub_att": row[f"{side}_sub_att"],
        "sig_str_landed": row[f"{side}_sig_str_landed"],
        "td_landed": row[f"{side}_td_landed"],
        "sig_str_absorbed": row[f"{opponent}_sig_str_landed"],
        "sig_str_def": safe_defense_rate(row[f"{opponent}_sig_str_landed"], row[f"{opponent}_sig_str_att"]),
        "td_def": safe_defense_rate(row[f"{opponent}_td_landed"], row[f"{opponent}_td_att"]),
    }


def build_elo_rankings(ratings: dict[str, float], histories: dict[str, FighterHistory], career_rows: list[dict[str, Any]]) -> pd.DataFrame:
    last_fight_dates = (
        pd.DataFrame(career_rows)
        .sort_values("date")
        .groupby("fighter_id", as_index=True)["date"]
        .last()
        .to_dict()
        if career_rows
        else {}
    )
    rows: list[dict[str, Any]] = []
    for fighter_id, rating in ratings.items():
        history = histories[fighter_id]
        if history.fights == 0:
            continue
        rows.append(
            {
                "fighter_id": fighter_id,
                "elo": round(float(rating), 2),
                "fights": history.fights,
                "wins": history.wins,
                "losses": history.losses,
                "draws": history.draws,
                "win_rate": round(history.wins / history.fights, 4),
                "last_fight_date": last_fight_dates.get(fighter_id),
            }
        )
    rankings = pd.DataFrame(rows).sort_values(["elo", "fights"], ascending=[False, False]).reset_index(drop=True)
    rankings.insert(0, "rank", np.arange(1, len(rankings) + 1))
    return rankings


def add_elo_and_rolling(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings: defaultdict[str, float] = defaultdict(lambda: 1500.0)
    histories: defaultdict[str, FighterHistory] = defaultdict(FighterHistory)
    feature_rows: list[dict[str, Any]] = []
    career_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        a_id = row["fighter_a_id"]
        b_id = row["fighter_b_id"]
        a_elo = ratings[a_id]
        b_elo = ratings[b_id]
        a_prior = rolling_snapshot(histories[a_id])
        b_prior = rolling_snapshot(histories[b_id])

        feature_row = row.to_dict()
        feature_row["fighter_a_elo"] = a_elo
        feature_row["fighter_b_elo"] = b_elo
        feature_row["elo_diff"] = a_elo - b_elo
        for key, value in a_prior.items():
            feature_row[f"a_{key}"] = value
        for key, value in b_prior.items():
            feature_row[f"b_{key}"] = value
        feature_rows.append(feature_row)

        for fighter_id, opponent_id, prior, elo in [(a_id, b_id, a_prior, a_elo), (b_id, a_id, b_prior, b_elo)]:
            career_rows.append(
                {
                    "fight_id": row["fight_id"],
                    "event_id": row["event_id"],
                    "date": row["date"],
                    "fighter_id": fighter_id,
                    "opponent_id": opponent_id,
                    "pre_fight_elo": elo,
                    **prior,
                }
            )

        winner = row["winner_id"]
        a_score = 1.0 if winner == a_id else 0.0 if winner == b_id else 0.5
        b_score = 1.0 - a_score
        a_expected = expected_score(a_elo, b_elo)
        b_expected = expected_score(b_elo, a_elo)
        ratings[a_id] = a_elo + 32 * (a_score - a_expected)
        ratings[b_id] = b_elo + 32 * (b_score - b_expected)
        update_history(histories[a_id], side_stats(row, "a"), True if winner == a_id else False if winner == b_id else None, row.get("method"))
        update_history(histories[b_id], side_stats(row, "b"), True if winner == b_id else False if winner == a_id else None, row.get("method"))

    elo_rankings = build_elo_rankings(ratings, histories, career_rows)
    return pd.DataFrame(feature_rows), pd.DataFrame(career_rows), elo_rankings


def add_physical_features(df: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    bio = fighters[["fighter_id", "height_cm", "reach_cm", "dob"]].copy()
    bio["dob"] = pd.to_datetime(bio["dob"], errors="coerce")
    out = df.merge(bio.add_prefix("a_"), left_on="fighter_a_id", right_on="a_fighter_id", how="left")
    out = out.merge(bio.add_prefix("b_"), left_on="fighter_b_id", right_on="b_fighter_id", how="left")
    out["a_age"] = (out["date"] - out["a_dob"]).dt.days / 365.25
    out["b_age"] = (out["date"] - out["b_dob"]).dt.days / 365.25
    out["reach_diff"] = out["a_reach_cm"] - out["b_reach_cm"]
    out["height_diff"] = out["a_height_cm"] - out["b_height_cm"]
    out["age_diff"] = out["a_age"] - out["b_age"]
    return out.drop(columns=["a_fighter_id", "b_fighter_id", "a_dob", "b_dob"], errors="ignore")


def add_differentials(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        if not col.startswith("a_avg_"):
            continue
        base = col.removeprefix("a_")
        b_col = f"b_{base}"
        if b_col in out.columns:
            out[f"diff_{base}"] = out[col] - out[b_col]
    out["diff_win_streak"] = out["a_win_streak"] - out["b_win_streak"]
    return out


def randomize_fighter_sides(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(RANDOM_SEED)
    swap_mask = rng.random(len(out)) < 0.5
    pairs = [
        ("fighter_a_id", "fighter_b_id"),
        ("fighter_a_elo", "fighter_b_elo"),
    ]
    prefixes = [
        "avg_",
        "fights_in_ufc",
        "win_rate",
        "win_streak",
        "finish_rate",
        "ko_rate",
        "sub_rate",
        "ko_loss_rate",
        "sub_loss_rate",
        "age",
        "height_cm",
        "reach_cm",
    ]
    for col in out.columns:
        if col.startswith("a_"):
            other = "b_" + col[2:]
            if other in out.columns and any(col[2:].startswith(prefix) for prefix in prefixes):
                pairs.append((col, other))
    seen: set[tuple[str, str]] = set()
    for left, right in pairs:
        if (left, right) in seen or left not in out.columns or right not in out.columns:
            continue
        seen.add((left, right))
        temp = out.loc[swap_mask, left].copy()
        out.loc[swap_mask, left] = out.loc[swap_mask, right]
        out.loc[swap_mask, right] = temp

    out["elo_diff"] = out["fighter_a_elo"] - out["fighter_b_elo"]
    out["reach_diff"] = out["a_reach_cm"] - out["b_reach_cm"]
    out["height_diff"] = out["a_height_cm"] - out["b_height_cm"]
    out["age_diff"] = out["a_age"] - out["b_age"]
    out = add_differentials(out.drop(columns=[c for c in out.columns if c.startswith("diff_avg_")], errors="ignore"))
    out["diff_win_streak"] = out["a_win_streak"] - out["b_win_streak"]
    out["diff_finish_rate"] = out["a_finish_rate"] - out["b_finish_rate"]
    out["diff_ko_rate"] = out["a_ko_rate"] - out["b_ko_rate"]
    out["diff_sub_rate"] = out["a_sub_rate"] - out["b_sub_rate"]
    out["diff_ko_loss_rate"] = out["a_ko_loss_rate"] - out["b_ko_loss_rate"]
    out["diff_sub_loss_rate"] = out["a_sub_loss_rate"] - out["b_sub_loss_rate"]
    out["target"] = np.where(
        out["winner_id"] == out["fighter_a_id"],
        1.0,
        np.where(out["winner_id"] == out["fighter_b_id"], 0.0, np.nan),
    )
    return out


def norm_value(norm_params: dict[str, float], key: str) -> float:
    value = norm_params.get(key, 0.0)
    if value is None or pd.isna(value) or float(value) <= 0:
        return 1.0
    return float(value)


def filled_component(series: pd.Series) -> pd.Series:
    return series.fillna(0).astype(float)


def compute_composite_scores(df: pd.DataFrame, norm_params: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    max_sig_str_landed = norm_value(norm_params, "max_sig_str_landed")
    max_sig_str_absorbed = norm_value(norm_params, "max_sig_str_absorbed")
    max_td_landed = norm_value(norm_params, "max_td_landed")
    max_sub_att = norm_value(norm_params, "max_sub_att")

    for side in ["a", "b"]:
        sig_str_pct = filled_component(out[f"{side}_avg_sig_str_pct_5f"])
        sig_str_def = filled_component(out[f"{side}_avg_sig_str_def_5f"])
        sig_str_landed = filled_component(out[f"{side}_avg_sig_str_landed_5f"])
        sig_str_absorbed = filled_component(out[f"{side}_avg_sig_str_absorbed_5f"])
        td_pct = filled_component(out[f"{side}_avg_td_pct_5f"])
        td_def = filled_component(out[f"{side}_avg_td_def_5f"])
        td_landed = filled_component(out[f"{side}_avg_td_landed_5f"])
        sub_att = filled_component(out[f"{side}_avg_sub_att_5f"])

        out[f"{side}_striking_score"] = (
            0.35 * sig_str_pct
            + 0.25 * sig_str_def
            + 0.25 * (sig_str_landed / max_sig_str_landed)
            + 0.15 * (1.0 / (1.0 + sig_str_absorbed / max_sig_str_absorbed))
        )
        out[f"{side}_grappling_score"] = (
            0.40 * td_pct
            + 0.30 * td_def
            + 0.20 * (td_landed / max_td_landed)
            + 0.10 * (sub_att / max_sub_att)
        )

    out["diff_striking_score"] = out["a_striking_score"] - out["b_striking_score"]
    out["diff_grappling_score"] = out["a_grappling_score"] - out["b_grappling_score"]
    return out


def encode_weight_class(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).lower().replace("women's", "").replace("ufc", "").replace("bout", "").replace("title", "")
    text = re.sub(r"\s+", " ", text).strip()
    for label, kg in WEIGHT_CLASS_KG.items():
        if label in text:
            return float(kg)
    return np.nan


def build_rounds_agg(rounds: pd.DataFrame) -> pd.DataFrame:
    if rounds.empty:
        return pd.DataFrame(columns=["fight_id"])
    numeric_cols = [col for col in rounds.columns if col not in {"id", "fight_id", "fighter_id"}]
    agg = rounds.groupby("fight_id")[numeric_cols].agg(["mean", "sum", "max"])
    agg.columns = ["_".join(col).strip() for col in agg.columns.to_flat_index()]
    return agg.reset_index()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in DROPPED_FEATURES
        and c.startswith(
            (
                "diff_",
                "elo_",
                "a_avg",
                "b_avg",
                "reach_",
                "height_",
                "age_",
                "is_title",
                "weight_class_kg",
                "a_fights_in_ufc",
                "b_fights_in_ufc",
                "a_win_rate",
                "b_win_rate",
                "a_win_streak",
                "b_win_streak",
                "a_finish_rate",
                "b_finish_rate",
                "a_ko_rate",
                "b_ko_rate",
                "a_sub_rate",
                "b_sub_rate",
                "a_ko_loss_rate",
                "b_ko_loss_rate",
                "a_sub_loss_rate",
                "b_sub_loss_rate",
                "fighter_a_elo",
                "fighter_b_elo",
                "a_striking",
                "b_striking",
                "diff_striking",
                "a_grappling",
                "b_grappling",
                "diff_grappling",
            )
        )
    ]


def training_norm_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns or df["date"].dropna().empty:
        return df
    latest_year = df["date"].dt.year.max()
    train = df[df["date"].dt.year < latest_year]
    return train if not train.empty else df


def rolling_max(df: pd.DataFrame, stat: str) -> float:
    cols = [col for col in [f"a_avg_{stat}_5f", f"b_avg_{stat}_5f"] if col in df.columns]
    if not cols:
        return 0.0
    return float(df[cols].fillna(0).max().max())


def write_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        fallback = path.with_name(f"{path.stem}.new{path.suffix}")
        df.to_csv(fallback, index=False)
        print(f"Could not overwrite locked file {path}; wrote {fallback} instead.")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    fights, fighters, events, rounds = load_tables(DB_PATH)
    fighter_refs = build_fighter_refs(fighters)
    fights = parse_raw_fight_columns(fights)
    fights = normalize_event_dates(fights, events)
    features, career_stats, elo_rankings = add_elo_and_rolling(fights)
    fighter_names = fighters[["fighter_id", "name"]].drop_duplicates("fighter_id")
    elo_rankings = elo_rankings.merge(fighter_names, on="fighter_id", how="left")
    elo_rankings = elo_rankings.merge(fighter_refs[["fighter_id", "ref_no"]], on="fighter_id", how="left")
    ordered_cols = ["rank", "ref_no", "fighter_id", "name", "elo", "fights", "wins", "losses", "draws", "win_rate", "last_fight_date"]
    elo_rankings = elo_rankings[[col for col in ordered_cols if col in elo_rankings.columns]]
    career_stats = career_stats.merge(fighter_refs[["fighter_id", "ref_no", "name"]], on="fighter_id", how="left")
    features = add_physical_features(features, fighters)
    features = randomize_fighter_sides(features)
    norm_source = training_norm_frame(features)
    norm_params = {
        "max_sig_str_landed": rolling_max(norm_source, "sig_str_landed"),
        "max_sig_str_absorbed": rolling_max(norm_source, "sig_str_absorbed"),
        "max_td_landed": rolling_max(norm_source, "td_landed"),
        "max_sub_att": rolling_max(norm_source, "sub_att"),
    }
    features = compute_composite_scores(features, norm_params)
    features["weight_class_kg"] = features["weight_class"].apply(encode_weight_class)
    rounds_agg = build_rounds_agg(rounds)

    write_csv(features, DATA_DIR / "fights_features.csv")
    write_csv(career_stats, DATA_DIR / "fighter_career_stats.csv")
    write_csv(rounds_agg, DATA_DIR / "rounds_agg.csv")
    write_csv(fighter_refs[["ref_no", "name"]], DATA_DIR / "fighter_refs.csv")
    write_csv(elo_rankings.head(100), DATA_DIR / "top_100_elo.csv")
    with (DATA_DIR / "composite_norm_params.json").open("w", encoding="utf-8") as fh:
        json.dump(norm_params, fh, indent=2)

    target_rows = features["target"].notna()
    cols = feature_columns(features)
    missing_ratio = float(features[cols].isna().mean().mean()) if cols else 0.0
    print("Feature engineering complete")
    print(f"Total fights: {len(features)}")
    print(f"Fights with target: {int(target_rows.sum())}")
    print(f"Target mean: {features.loc[target_rows, 'target'].mean():.3f}")
    print(f"Total features: {len(cols)}")
    print(f"Missing ratio: {missing_ratio:.3f}")
    print(f"Date range: {features['date'].min()} -> {features['date'].max()}")
    print("New feature groups added: sig_str_absorbed, sig_str_def, td_def, striking_score, grappling_score")
    print("\nTop 10 ELO fighters")
    if elo_rankings.empty:
        print("No ELO rankings available.")
    else:
        print(
            elo_rankings.head(10).to_string(
                index=False,
                columns=["rank", "name", "elo", "fights", "wins", "losses", "draws"],
                formatters={"elo": lambda value: f"{value:.2f}"},
            )
        )


if __name__ == "__main__":
    main()
