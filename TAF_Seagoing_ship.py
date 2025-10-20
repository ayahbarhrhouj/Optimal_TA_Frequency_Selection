from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from typing import Dict, Tuple, List

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import shap
import xgboost as xgb


# ---------------------------- Config ----------------------------

CSV_PATH = "SeagoingShip-Sensor-data.csv"

FREQUENCIES = [10, 20, 30, 40, 50, 60]      # minutes
DENDRO_THRESHOLD = 0.2                      # distance threshold => corr >= 0.8
ALPHA = 0.8                                  # coherence weight
BETA = 0.2                                   # sensitivity weight
EPSILON = 1e-3                               # perturbation amplitude for sensitivity
RANDOM_STATE = 42

XGB_PARAMS = dict(
    random_state=RANDOM_STATE,
    n_jobs=1,            
    tree_method="hist"  
)

TARGET_COL = "3_ME_tot_FL"
DATETIME_COL = "0_datetime"


# ------------------------- Utilities ----------------------------

def _base_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATETIME_COL] = pd.to_datetime(out[DATETIME_COL], format="%m/%d/%Y %H:%M")
    out["13_ballast"] = out["13_ballast"].map({"laden": 1, "ballast": 0})
    out["27_Prop_slip"] = out["27_Prop_slip"].map({"a": 0, "b": 1})
    out = out.drop(columns=["7_Fuel_Index", "2_Fuel_eff"])
    return out


def make_aggregations(df_raw: pd.DataFrame, freqs: List[int]) -> Dict[int, pd.DataFrame]:
    df = _base_preprocess(df_raw)
    datasets = {}
    for f in freqs:
        agg = (
            df.set_index(DATETIME_COL)
              .groupby(pd.Grouper(freq=f"{f}Min"))
              .mean(numeric_only=True)
              .dropna()
        )
        datasets[f] = agg
    return datasets


def build_groups_df_for_similarity(df_raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = _base_preprocess(df_raw)
    # Exclude the target explicitly when calculating similarity between features
    if TARGET_COL in df.columns:
        df_features = df.drop(columns=[TARGET_COL])
    else:
        df_features = df.copy()

    corr = df_features.corr(numeric_only=True)
    distance = 1 - corr
    condensed = squareform(distance.values, checks=False)
    Z = linkage(condensed, method="single")
    clusters = fcluster(Z, t=threshold, criterion="distance")

    groups_df = pd.DataFrame({"Variable": corr.columns, "Groupe": clusters})
    return groups_df


def compute_coherence(shap_df: pd.DataFrame) -> float:
    if shap_df.empty or "Groupe" not in shap_df.columns:
        return float("nan")
    grp = shap_df.groupby("Groupe")["SHAP_value"].std()
    if grp.empty:
        return float("nan")
    return float(grp.mean())


def minmax_norm(d: Dict[int, float]) -> Dict[int, float]:
    vals = np.array(list(d.values()), dtype=float)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if np.any(np.isnan(vals)):
        vals = np.nan_to_num(vals, nan=vmax)
        vmin, vmax = float(vals.min()), float(vals.max())
    if np.isclose(vmax, vmin):
        return {k: 0.0 for k in d}
    return {k: (float(d[k]) - vmin) / (vmax - vmin) for k in d}


# ---------------------- SHAP & Sensitivity ----------------------

def shap_compute(
    data_agg: pd.DataFrame,
    groups_df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], shap.explainers._tree.Tree, np.ndarray, pd.DataFrame]:
    if TARGET_COL not in data_agg.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in aggregated data.")
    X = data_agg.drop(columns=[TARGET_COL])
    y = data_agg[TARGET_COL]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )

    # pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", xgb.XGBRegressor(**XGB_PARAMS)),
    ])

    pipe.fit(X_train, y_train)

    scaler: StandardScaler = pipe.named_steps["scaler"]
    model: xgb.XGBRegressor = pipe.named_steps["model"]

    X_test_scaled = scaler.transform(X_test)
    feature_names = list(X.columns)

    # Explainer
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_scaled)

    # SHAP mean |value| per feature
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    shap_df = pd.DataFrame({"Variable": feature_names, "SHAP_value": mean_abs})
    shap_df = shap_df.merge(groups_df, on="Variable", how="left")

    # keep only groups size >= 2
    group_sizes = groups_df["Groupe"].value_counts()
    valid_groups = group_sizes[group_sizes >= 2].index
    shap_df = shap_df[shap_df["Groupe"].isin(valid_groups)]

    return X_test, X_test_scaled, feature_names, explainer, shap_values, shap_df


def perturb_data_and_get_delta(X_scaled: np.ndarray, epsilon: float = EPSILON, seed: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perturb = rng.normal(0.0, epsilon, X_scaled.shape)
    Xp = X_scaled + perturb
    delta_per_sample = np.linalg.norm(perturb, axis=1)  # (n_samples,)
    # avoid delta=0
    delta_per_sample = np.where(delta_per_sample == 0, 1e-12, delta_per_sample)
    return Xp, delta_per_sample


def lipschitz_sensitivity_median(
    explainer: shap.explainers._tree.Tree,
    X_scaled: np.ndarray,
    epsilon: float = EPSILON,
    seed: int = RANDOM_STATE
) -> float:
    Xp, delta = perturb_data_and_get_delta(X_scaled, epsilon=epsilon, seed=seed)
    shap_ref = explainer.shap_values(X_scaled)
    shap_pert = explainer.shap_values(Xp)
    # L2 norm per sample in the SHAP features space
    diff = shap_ref - shap_pert
    per_sample = np.linalg.norm(diff, axis=1) / delta
    return float(np.median(per_sample))


# ----------------------------- Main -----------------------------

def main():
    # 1) Load raw data
    raw = pd.read_csv(CSV_PATH)

    # 2) Build aggregated datasets at different frequencies
    datasets = make_aggregations(raw, FREQUENCIES)
    for f, df in datasets.items():
        print(f"F={f} min -> {df.shape[0]} rows after aggregation")

    # 3) Compute similarity groups (once and for all)
    groups_df = build_groups_df_for_similarity(raw, threshold=DENDRO_THRESHOLD)
    useful_groups = (groups_df["Groupe"].value_counts() >= 2).sum()
    print(f"\n Variable groups (size >= 2): {useful_groups}")

    # 4) Frequency loop: SHAP, coherence, sensitivity
    coherences: Dict[int, float] = {}
    lipschitz: Dict[int, float] = {}

    for f, agg in datasets.items():
        try:
            X_test, X_test_scaled, feat_names, explainer, shap_vals, shap_df = shap_compute(
                agg, groups_df, random_state=RANDOM_STATE
            )
            coh = compute_coherence(shap_df)
            sens = lipschitz_sensitivity_median(explainer, X_test_scaled, epsilon=EPSILON, seed=RANDOM_STATE)

            coherences[f] = coh
            lipschitz[f] = sens

        except Exception as e:
            # If the dataset is too small at a particular frequency, log it and continue
            print(f"[WARN] F={f} : {e}")
            coherences[f] = np.nan
            lipschitz[f] = np.nan

    # 5) Standardization and cost
    norm_coh = minmax_norm(coherences)
    norm_lip = minmax_norm(lipschitz)

    costs = {f: ALPHA * norm_coh[f] + BETA * norm_lip[f] for f in FREQUENCIES}
    best_frequency = min(costs, key=costs.get)

    print("\n Coherences:", {k: float(v) for k, v in coherences.items()})
    print("Sensitivity:", {k: float(v) for k, v in lipschitz.items()})
    print("\n Coherences (normalized):", {k: float(v) for k, v in norm_coh.items()})
    print("Sensitivity (normalized):", {k: float(v) for k, v in norm_lip.items()})
    print("\n Reliability Cost:", {k: float(v) for k, v in costs.items()})
    print(f"\n>>> Optimal frequency: {best_frequency} minutes, cost={costs[best_frequency]:.6f}")



if __name__ == "__main__":
    main()

