from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier


DATA_PATH = Path("data/fights_features.csv")
ARTIFACT_DIR = Path("models/artifacts")


def walk_forward_years(df: pd.DataFrame) -> list[int]:
    years = sorted(int(year) for year in df["date"].dt.year.dropna().unique())
    return [year for year in years if not df[df["date"].dt.year < year].empty]


def select_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.startswith(
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
            )
        )
    ]


def model_params(**overrides) -> dict:
    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(overrides)
    return params


def safe_metrics(y_true: pd.Series, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    metrics = {"accuracy": accuracy_score(y_true, pred)}
    metrics["roc_auc"] = roc_auc_score(y_true, prob) if y_true.nunique() == 2 else np.nan
    metrics["log_loss"] = log_loss(y_true, prob, labels=[0, 1])
    return metrics


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run python -m scrapers.feature_engineering first.")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["target"]).sort_values("date").reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    feature_cols = select_feature_cols(df)
    if not feature_cols:
        raise ValueError("No training feature columns found.")
    return df, feature_cols


def run_walk_forward(df: pd.DataFrame, feature_cols: list[str], params: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    rows: list[dict] = []
    last_test: pd.DataFrame | None = None
    for year in walk_forward_years(df):
        train = df[df["date"].dt.year < year]
        test = df[df["date"].dt.year == year]
        if train.empty or test.empty:
            rows.append({"year": year, "n_fights": len(test), "correct": np.nan, "accuracy": np.nan, "roc_auc": np.nan, "log_loss": np.nan})
            continue
        fill_values = train[feature_cols].median(numeric_only=True)
        X_train = train[feature_cols].fillna(fill_values)
        X_test = test[feature_cols].fillna(fill_values)
        y_train = train["target"]
        y_test = test["target"]
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = safe_metrics(y_test, prob)
        rows.append({"year": year, "n_fights": len(test), "correct": int((pred == y_test.to_numpy()).sum()), **metrics})
        last_test = test.assign(_prob=prob)
    return pd.DataFrame(rows), last_test


def tune_params(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required for --tune. Install optuna or run without --tune.") from exc

    recent_year = int(df["date"].dt.year.max())
    train = df[df["date"].dt.year < recent_year]
    valid = df[df["date"].dt.year == recent_year]
    if train.empty or valid.empty:
        raise ValueError("Not enough chronological data for Optuna tuning.")
    fill_values = train[feature_cols].median(numeric_only=True)
    X_train = train[feature_cols].fillna(fill_values)
    X_valid = valid[feature_cols].fillna(fill_values)
    y_train = train["target"]
    y_valid = valid["target"]

    def objective(trial) -> float:
        params = model_params(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        )
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_valid)[:, 1]
        return log_loss(y_valid, prob, labels=[0, 1])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    print(f"Best Optuna log_loss: {study.best_value:.4f}")
    return model_params(**study.best_params)


def print_walk_forward(results: pd.DataFrame) -> None:
    print("\nWalk-forward validation")
    display_cols = [col for col in ["year", "n_fights", "accuracy", "roc_auc", "log_loss"] if col in results.columns]
    print(results[display_cols].to_string(index=False, formatters={
        "accuracy": lambda v: "" if pd.isna(v) else f"{v:.3f}",
        "roc_auc": lambda v: "" if pd.isna(v) else f"{v:.3f}",
        "log_loss": lambda v: "" if pd.isna(v) else f"{v:.3f}",
    }))
    valid = results.dropna(subset=["correct", "n_fights"]).sort_values("year").tail(5)
    total_fights = int(valid["n_fights"].sum())
    total_correct = int(valid["correct"].sum())
    cumulative_accuracy = total_correct / total_fights if total_fights else np.nan
    year_start = int(valid["year"].min()) if not valid.empty else None
    year_end = int(valid["year"].max()) if not valid.empty else None
    if pd.isna(cumulative_accuracy):
        print("\nLast-5-years cumulative walk-forward accuracy: unavailable")
    else:
        print(
            f"\nLast-5-years cumulative walk-forward accuracy ({year_start}-{year_end}): "
            f"{cumulative_accuracy:.3f} ({total_correct}/{total_fights})"
        )


def print_shap_summary(model: XGBClassifier, sample: pd.DataFrame, feature_cols: list[str]) -> None:
    if sample is None or sample.empty:
        print("\nSHAP skipped: no held-out test rows available.")
        return
    try:
        import shap
    except ImportError:
        print("\nSHAP skipped: package is not installed.")
        return
    sample_X = sample[feature_cols].head(500)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_X)
    values = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
    mean_abs = np.abs(values).mean(axis=0)
    top = pd.Series(mean_abs, index=feature_cols).sort_values(ascending=False).head(15)
    print("\nTop 15 SHAP features")
    for feature, value in top.items():
        print(f"{feature:35s} {value:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UFC fight outcome model.")
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning before final training.")
    args = parser.parse_args()

    df, feature_cols = load_dataset()
    params = tune_params(df, feature_cols) if args.tune else model_params()
    walk_forward, last_test = run_walk_forward(df, feature_cols, params)
    print_walk_forward(walk_forward)

    fill_values = df[feature_cols].median(numeric_only=True)
    X = df[feature_cols].fillna(fill_values)
    y = df["target"]
    final_model = XGBClassifier(**params)
    final_model.fit(X, y)

    shap_sample = X.loc[last_test.index].head(500) if last_test is not None else pd.DataFrame(columns=feature_cols)
    print_shap_summary(final_model, shap_sample, feature_cols)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT_DIR / "xgb_model.pkl").open("wb") as fh:
        pickle.dump(final_model, fh)
    with (ARTIFACT_DIR / "fill_values.pkl").open("wb") as fh:
        pickle.dump(fill_values, fh)
    with (ARTIFACT_DIR / "feature_cols.json").open("w", encoding="utf-8") as fh:
        json.dump(feature_cols, fh, indent=2)
    walk_forward.to_csv(ARTIFACT_DIR / "walk_forward.csv", index=False)
    print(f"\nSaved artifacts to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
