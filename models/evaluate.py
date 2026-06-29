from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


DATA_PATH = Path("data/fights_features.csv")
ARTIFACT_DIR = Path("models/artifacts")
DEFAULT_OUTPUT_DIR = ARTIFACT_DIR / "evaluation"
MONEYLINE_CANDIDATES = {
    "a": ["fighter_a_odds", "a_odds", "a_moneyline", "fighter_a_moneyline", "moneyline_a"],
    "b": ["fighter_b_odds", "b_odds", "b_moneyline", "fighter_b_moneyline", "moneyline_b"],
}


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}. Run python -m models.train first.")
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_artifacts() -> tuple[Any, pd.Series, list[str]]:
    model = load_pickle(ARTIFACT_DIR / "xgb_model.pkl")
    fill_values = load_pickle(ARTIFACT_DIR / "fill_values.pkl")
    feature_path = ARTIFACT_DIR / "feature_cols.json"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing artifact: {feature_path}. Run python -m models.train first.")
    feature_cols = json.loads(feature_path.read_text(encoding="utf-8"))
    return model, fill_values, feature_cols


def load_dataset(feature_cols: list[str]) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run python -m scrapers.feature_engineering first.")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["target"]).sort_values("date").reset_index(drop=True)
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing trained feature columns: {missing_cols[:10]}")
    df["target"] = df["target"].astype(int)
    return df


def choose_eval_rows(df: pd.DataFrame, year: int | None, last_fraction: float) -> pd.DataFrame:
    if year is not None:
        rows = df[df["date"].dt.year == year].copy()
        if rows.empty:
            raise ValueError(f"No fights found for evaluation year {year}.")
        return rows
    start = max(0, int(len(df) * (1.0 - last_fraction)))
    rows = df.iloc[start:].copy()
    if rows.empty:
        raise ValueError("No rows selected for evaluation.")
    return rows


def classification_metrics(y_true: pd.Series, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "n_fights": float(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if y_true.nunique() == 2 else np.nan,
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, prob)),
        "mean_prob": float(np.mean(prob)),
        "target_mean": float(np.mean(y_true)),
    }


def calibration_table(y_true: pd.Series, prob: np.ndarray, bins: int) -> pd.DataFrame:
    frame = pd.DataFrame({"target": y_true.to_numpy(), "prob": prob})
    frame["bin"] = pd.cut(frame["prob"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    table = (
        frame.groupby("bin", observed=False)
        .agg(n=("target", "size"), avg_pred_prob=("prob", "mean"), actual_win_rate=("target", "mean"))
        .reset_index()
    )
    table["calibration_error"] = table["avg_pred_prob"] - table["actual_win_rate"]
    table["bin"] = table["bin"].astype(str)
    return table


def find_moneyline_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lower_map = {col.lower(): col for col in df.columns}
    a_col = next((lower_map[name] for name in MONEYLINE_CANDIDATES["a"] if name in lower_map), None)
    b_col = next((lower_map[name] for name in MONEYLINE_CANDIDATES["b"] if name in lower_map), None)
    return a_col, b_col


def american_odds_profit(odds: float, stake: float) -> float:
    if odds > 0:
        return stake * odds / 100.0
    return stake * 100.0 / abs(odds)


def implied_probability(odds: float) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def betting_sim(df: pd.DataFrame, prob: np.ndarray, edge: float, stake: float) -> pd.DataFrame:
    a_col, b_col = find_moneyline_columns(df)
    if a_col is None or b_col is None:
        return pd.DataFrame(
            columns=[
                "fight_id",
                "bet_side",
                "model_prob",
                "implied_prob",
                "edge",
                "odds",
                "stake",
                "profit",
                "bankroll",
            ]
        )

    rows: list[dict[str, Any]] = []
    bankroll = 0.0
    eval_df = df.reset_index(drop=True)
    for idx, row in eval_df.iterrows():
        prob_a = float(prob[idx])
        candidates = [
            ("A", prob_a, row[a_col], int(row["target"]) == 1),
            ("B", 1.0 - prob_a, row[b_col], int(row["target"]) == 0),
        ]
        best: tuple[str, float, float, bool, float] | None = None
        for side, model_prob, odds, won in candidates:
            if pd.isna(odds):
                continue
            odds = float(odds)
            side_edge = model_prob - implied_probability(odds)
            if side_edge >= edge and (best is None or side_edge > best[4]):
                best = (side, model_prob, odds, won, side_edge)
        if best is None:
            continue
        side, model_prob, odds, won, side_edge = best
        profit = american_odds_profit(odds, stake) if won else -stake
        bankroll += profit
        rows.append(
            {
                "fight_id": row.get("fight_id"),
                "bet_side": side,
                "model_prob": model_prob,
                "implied_prob": implied_probability(odds),
                "edge": side_edge,
                "odds": odds,
                "stake": stake,
                "profit": profit,
                "bankroll": bankroll,
            }
        )
    return pd.DataFrame(rows)


def shap_summary(model: Any, X: pd.DataFrame, output_dir: Path, sample_size: int) -> pd.DataFrame:
    try:
        import shap
    except ImportError:
        print("SHAP skipped: package is not installed.")
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    sample = X.head(sample_size)
    if sample.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    values = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
    summary = (
        pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "shap_summary.csv", index=False)
    return summary


def print_metrics(metrics: dict[str, float]) -> None:
    print("\nEvaluation metrics")
    for key, value in metrics.items():
        if key == "n_fights":
            print(f"{key:16s}: {int(value)}")
        else:
            print(f"{key:16s}: {value:.4f}")


def print_betting_summary(bets: pd.DataFrame, stake: float) -> None:
    print("\nBetting simulation")
    if bets.empty:
        print("No bets placed. Add American moneyline columns to fights_features.csv to enable this.")
        return
    total_staked = float(len(bets) * stake)
    profit = float(bets["profit"].sum())
    roi = profit / total_staked if total_staked else 0.0
    print(f"Bets placed     : {len(bets)}")
    print(f"Total staked    : {total_staked:.2f}")
    print(f"Profit          : {profit:.2f}")
    print(f"ROI             : {roi:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained UFC fight predictor.")
    parser.add_argument("--year", type=int, default=None, help="Evaluate a specific calendar year.")
    parser.add_argument("--last-fraction", type=float, default=0.2, help="Chronological tail fraction if --year is omitted.")
    parser.add_argument("--bins", type=int, default=10, help="Number of calibration bins.")
    parser.add_argument("--edge", type=float, default=0.03, help="Minimum model edge over implied odds for betting sim.")
    parser.add_argument("--stake", type=float, default=100.0, help="Flat stake per simulated bet.")
    parser.add_argument("--shap-sample", type=int, default=500, help="Rows to use for SHAP summary.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for evaluation CSV outputs.")
    args = parser.parse_args()

    if not 0 < args.last_fraction <= 1:
        raise ValueError("--last-fraction must be in the range (0, 1].")

    model, fill_values, feature_cols = load_artifacts()
    df = load_dataset(feature_cols)
    eval_df = choose_eval_rows(df, args.year, args.last_fraction)
    X = eval_df[feature_cols].fillna(fill_values)
    y = eval_df["target"]
    prob = model.predict_proba(X)[:, 1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = eval_df[["fight_id", "date", "fighter_a_id", "fighter_b_id", "winner_id", "target"]].copy()
    predictions["fighter_a_win_prob"] = prob
    predictions["prediction"] = (prob >= 0.5).astype(int)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)

    metrics = classification_metrics(y, prob)
    pd.DataFrame([metrics]).to_csv(args.output_dir / "metrics.csv", index=False)
    print_metrics(metrics)

    calibration = calibration_table(y, prob, args.bins)
    calibration.to_csv(args.output_dir / "calibration.csv", index=False)
    print("\nCalibration")
    print(calibration.to_string(index=False, formatters={
        "avg_pred_prob": lambda v: "" if pd.isna(v) else f"{v:.3f}",
        "actual_win_rate": lambda v: "" if pd.isna(v) else f"{v:.3f}",
        "calibration_error": lambda v: "" if pd.isna(v) else f"{v:.3f}",
    }))

    bets = betting_sim(eval_df, prob, args.edge, args.stake)
    bets.to_csv(args.output_dir / "betting_sim.csv", index=False)
    print_betting_summary(bets, args.stake)

    shap = shap_summary(model, X, args.output_dir, args.shap_sample)
    if not shap.empty:
        print(f"\nAll SHAP features ({len(shap)})")
        print(shap.to_string(index=False, formatters={"mean_abs_shap": lambda v: f"{v:.6f}"}))

    print(f"\nSaved evaluation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
