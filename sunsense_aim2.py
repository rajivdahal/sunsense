"""
SunSense - Aim 2: next five-minute UVI category prediction
Owner: Sivani Ammisetti

Pipeline (follows the Aim 2 protocol):
  1. Load + audit data (sort, dedupe, invalid / negative / nonfinite flags)
  2. Aggregate to 5-min intervals (input here is already 5-min)
  3. Time-aligned labels: features use data at or before t, target = category of
     the mean UVI in (t, t+5 min]
  4. History features: intensity, short history, trend, variability, duration,
     dose (trapezoid, UVI/80 * dt J/m^2), time of day
  5. Chronological split (whole days when >= 3 days exist; otherwise contiguous
     time blocks inside one day, flagged as exploratory)
  6. Models: persistence, intensity-only, multinomial logistic regression,
     Random Forest, XGBoost (small, prespecified grids tuned on validation macro-F1)
  7. Ablation ladder A-E, metrics, confusion matrices, figures
  8. Minimal leakage / correctness tests

IMPORTANT: the supplied dataset is SYNTHETIC (pipeline testing only). Results
are not sensor measurements, calibration, or real-world accuracy.

Usage:
  python sunsense_aim2.py --data sunsense_synthetic_uv_dataset.csv --out results
"""

import argparse
import json
import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

warnings.filterwarnings("ignore")
SEED = 42
STEP_MIN = 5

# --------------------------------------------------------------------------
# Category definition (WHO UVI bands; labels describe intensity, not health risk)
# --------------------------------------------------------------------------
CATS = ["Low", "Moderate", "High", "Very high", "Extreme"]
EDGES = [0, 3, 6, 8, 11, np.inf]


def uvi_to_cat(u):
    """Map mean UVI to category index 0..4. Negative / NaN -> -1 (invalid)."""
    u = np.asarray(u, dtype=float)
    out = np.full(u.shape, -1, dtype=int)
    ok = np.isfinite(u) & (u >= 0)
    out[ok] = np.digitize(u[ok], EDGES[1:-1])
    return out


# --------------------------------------------------------------------------
# 1. Load and audit
# --------------------------------------------------------------------------
COLMAP = {  # dataset column -> pipeline name
    "uv_index_proxy_synthetic": "uvi",
    "uva_raw_synthetic": "raw_uva",
    "uvb_raw_synthetic": "raw_uvb",
    "environment_condition": "condition",
    "sensor_orientation": "orientation",
}


def load_and_audit(path):
    df = pd.read_csv(path).rename(columns=COLMAP)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    audit = {"rows_raw": len(df)}

    df = df.sort_values("timestamp")
    n0 = len(df)
    df = df.drop_duplicates(subset="timestamp", keep="first")  # deterministic
    audit["duplicates_removed"] = n0 - len(df)

    df["invalid_flag"] = ~np.isfinite(df["uvi"]) | (df["uvi"] < 0)
    audit["invalid_or_negative"] = int(df["invalid_flag"].sum())
    df.loc[df["invalid_flag"], "uvi"] = np.nan  # flagged, not relabeled

    if "day_id" not in df:
        df["day_id"] = df["timestamp"].dt.date.astype(str)
    if "session_id" not in df:
        df["session_id"] = df["day_id"]

    # gap check: expected 5-min spacing within a session
    gaps = df.groupby("session_id")["timestamp"].diff().dt.total_seconds() / 60
    audit["max_gap_min"] = float(np.nanmax(gaps)) if gaps.notna().any() else 0.0
    audit["unexpected_gaps"] = int((gaps > STEP_MIN).sum())
    audit["days"] = int(df["day_id"].nunique())
    audit["synthetic"] = bool(df.get("synthetic_data", pd.Series([False])).all())
    return df.reset_index(drop=True), audit


# --------------------------------------------------------------------------
# 2-4. Features + time-aligned target
# --------------------------------------------------------------------------
def trailing_slope(s, k):
    """Linear slope (UVI per minute) over the last k intervals, using past only."""
    x = np.arange(k) * STEP_MIN

    def f(w):
        if np.isnan(w).any():
            return np.nan
        return np.polyfit(x, w, 1)[0]

    return s.rolling(k, min_periods=k).apply(f, raw=True)


def build_features(df):
    out = []
    for sid, g in df.groupby("session_id", sort=False):
        g = g.sort_values("timestamp").copy()
        u = g["uvi"]
        f = pd.DataFrame(index=g.index)
        f["timestamp"] = g["timestamp"]
        f["day_id"] = g["day_id"]
        f["session_id"] = sid
        f["condition"] = g["condition"]  # diagnostics only, NOT a model input

        # Current intensity (interval ending at t)
        f["uvi_now"] = u
        # Short history: trailing 10/15/30 min means, 15-min min/max
        for k, m in [(2, 10), (3, 15), (6, 30)]:
            f[f"mean_{m}m"] = u.rolling(k, min_periods=k).mean()
        f["min_15m"] = u.rolling(3, min_periods=3).min()
        f["max_15m"] = u.rolling(3, min_periods=3).max()
        # Trend
        f["delta_5m"] = u.diff(1)
        f["slope_15m"] = trailing_slope(u, 3)
        # Variability
        f["std_15m"] = u.rolling(3, min_periods=3).std()
        f["range_15m"] = f["max_15m"] - f["min_15m"]
        f["absdiff_15m"] = u.diff().abs().rolling(3, min_periods=3).mean()
        # Duration: consecutive intervals with UVI >= 3, fraction >= 3 in last 30 min
        ge3 = (u >= 3).astype(int)
        run = ge3.groupby((ge3 == 0).cumsum()).cumsum()
        f["minutes_ge3"] = run * STEP_MIN
        f["frac_ge3_30m"] = ge3.rolling(6, min_periods=6).mean()
        # Dose (trapezoid, J/m^2 at sensor location): ((UVI_i + UVI_i+1)/80) * dt_s
        dt_s = g["timestamp"].diff().dt.total_seconds()
        inc = ((u + u.shift(1)) / 80.0) * dt_s
        inc[dt_s > STEP_MIN * 60] = np.nan  # never integrate across a gap
        f["dose_15m"] = inc.rolling(3, min_periods=3).sum()
        f["dose_30m"] = inc.rolling(6, min_periods=6).sum()
        f["dose_session"] = inc.fillna(0).cumsum()  # separate ablation only
        # Time of day (cyclic)
        h = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60
        f["hour_sin"] = np.sin(2 * np.pi * h / 24)
        f["hour_cos"] = np.cos(2 * np.pi * h / 24)

        # Target: category of the NEXT interval (t, t+5]; must be exactly 5 min later
        nxt_t = g["timestamp"].shift(-1)
        f["target_time"] = nxt_t
        f["target_uvi"] = u.shift(-1)
        ok = (nxt_t - g["timestamp"]).dt.total_seconds() == STEP_MIN * 60
        f.loc[~ok, "target_uvi"] = np.nan
        f["y"] = uvi_to_cat(f["target_uvi"])
        f["current_cat"] = uvi_to_cat(u)
        out.append(f)
    return pd.concat(out).reset_index(drop=True)


FEATURE_SETS = {  # ablation ladder (F = + quality flags, not available here)
    "B_intensity": ["uvi_now", "mean_10m"],
    "C_+history": ["uvi_now", "mean_10m", "mean_15m", "mean_30m", "min_15m",
                   "max_15m", "delta_5m", "slope_15m"],
    "D_+variability_duration": ["std_15m", "range_15m", "absdiff_15m",
                                "minutes_ge3", "frac_ge3_30m"],
    "E_+dose_time": ["dose_15m", "dose_30m", "hour_sin", "hour_cos"],
}
FEATURE_SETS["C_+history"] = FEATURE_SETS["C_+history"]
FEATURE_SETS["D_+variability_duration"] = (FEATURE_SETS["C_+history"]
                                           + FEATURE_SETS["D_+variability_duration"])
FEATURE_SETS["E_+dose_time"] = (FEATURE_SETS["D_+variability_duration"]
                                + FEATURE_SETS["E_+dose_time"])
FULL = FEATURE_SETS["E_+dose_time"]


# --------------------------------------------------------------------------
# 5. Chronological split
# --------------------------------------------------------------------------
def chronological_split(feat, frac=(0.6, 0.2, 0.2)):
    days = sorted(feat["day_id"].unique())
    if len(days) >= 3:
        n = len(days)
        n_tr = max(1, round(frac[0] * n))
        n_va = max(1, round(frac[1] * n))
        part = {d: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                for i, d in enumerate(days)}
        feat["split"] = feat["day_id"].map(part)
        mode = "whole days"
    else:
        # Single day: contiguous time blocks (exploratory only)
        order = feat["timestamp"].rank(method="first").astype(int) - 1
        n = len(feat)
        cut1, cut2 = int(frac[0] * n), int((frac[0] + frac[1]) * n)
        feat["split"] = np.where(order < cut1, "train",
                                 np.where(order < cut2, "val", "test"))
        mode = "contiguous time blocks within one day (exploratory)"
    # Drop rows whose history window reaches back into an earlier partition
    # or whose target crosses into the next one (no window crosses a boundary).
    feat = feat.sort_values("timestamp").reset_index(drop=True)
    boundary = feat["split"] != feat["split"].shift(1)
    blk = boundary.cumsum()
    pos_in_block = feat.groupby(blk).cumcount()
    last_in_block = feat["split"] != feat["split"].shift(-1)
    feat["eligible"] = (pos_in_block >= 6) & ~last_in_block  # 30-min history
    return feat, mode


# --------------------------------------------------------------------------
# 6. Models
# --------------------------------------------------------------------------
def metrics(y, p, labels):
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, p, labels=labels, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": round(accuracy_score(y, p), 3),
        "balanced_accuracy": round(balanced_accuracy_score(y, p), 3) if len(set(y)) > 1 else None,
        "macro_f1": round(f1_score(y, p, labels=labels, average="macro", zero_division=0), 3),
        # supplementary: macro-F1 over only the classes present in this partition
        "macro_f1_present": round(f1_score(y, p, labels=sorted(set(y)), average="macro",
                                           zero_division=0), 3),
        "per_class": {CATS[l]: {"precision": round(pr[i], 3), "recall": round(rc[i], 3),
                                "f1": round(f1[i], 3), "support": int(sup[i])}
                      for i, l in enumerate(labels)},
        "multi_category_errors": int((np.abs(np.asarray(y) - np.asarray(p)) > 1).sum()),
    }


def fit_predict(name, params, Xtr, ytr, Xev):
    classes = np.unique(ytr)
    if len(classes) == 1:  # degenerate: only one class seen in training
        return np.full(len(Xev), classes[0])
    if name == "logreg":
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=SEED))
    elif name == "rf":
        m = RandomForestClassifier(class_weight="balanced", random_state=SEED, **params)
    elif name == "xgb":
        # XGBoost needs labels 0..k-1: remap
        enc = {c: i for i, c in enumerate(classes)}
        m = XGBClassifier(random_state=SEED, verbosity=0, **params)  # objective inferred
        m.fit(Xtr, np.vectorize(enc.get)(ytr))
        return classes[m.predict(Xev)]
    m.fit(Xtr, ytr)
    return m.predict(Xev)


GRIDS = {
    "logreg": [{}],
    "rf": [dict(n_estimators=n, max_depth=d, min_samples_leaf=l)
           for n, d, l in product([100, 300], [3, 6, None], [2, 5, 10])],
    "xgb": [dict(max_depth=d, learning_rate=lr, n_estimators=n, subsample=s,
                 colsample_bytree=c)
            for d, lr, n, s, c in product([2, 3], [0.03, 0.1], [50, 150],
                                          [0.8, 1.0], [0.8, 1.0])],
}


def tune(name, cols, tr, va, labels):
    """Pick hyperparameters on validation macro-F1 (training data only for fitting)."""
    best, best_f1 = GRIDS[name][0], -1
    for p in GRIDS[name]:
        pred = fit_predict(name, p, tr[cols].values, tr["y"].values, va[cols].values)
        f1 = f1_score(va["y"], pred, labels=labels, average="macro", zero_division=0)
        if f1 > best_f1:
            best, best_f1 = p, f1
    return best, round(best_f1, 3)


# --------------------------------------------------------------------------
# 8. Minimal validation checks
# --------------------------------------------------------------------------
def run_tests(feat):
    res = {}
    res["target_after_cutoff"] = bool(
        (feat["target_time"].dropna() > feat.loc[feat["target_time"].notna(), "timestamp"]).all())

    # Step 2 -> 5: past features Low, future label Moderate
    t = pd.date_range("2026-01-01 10:00", periods=8, freq="5min")
    step = pd.DataFrame({"timestamp": t, "uvi": [2] * 7 + [5], "day_id": "x",
                         "session_id": "x", "condition": "test"})
    sf = build_features(step)
    r = sf.iloc[6]
    res["step_future_moderate_past_low"] = bool(
        r["y"] == 1 and r["current_cat"] == 0 and r["mean_30m"] < 3)

    # Gap: no dose integrated across a gap
    t2 = list(pd.date_range("2026-01-01 10:00", periods=4, freq="5min")) + \
         [pd.Timestamp("2026-01-01 10:30")]
    gap = pd.DataFrame({"timestamp": t2, "uvi": [4.0] * 5, "day_id": "x",
                        "session_id": "x", "condition": "test"})
    gf = build_features(gap)
    res["no_dose_across_gap"] = bool(np.isnan(gf["dose_15m"].iloc[-1]))
    res["no_target_across_gap"] = bool(np.isnan(gf["target_uvi"].iloc[3]))

    # Replay determinism
    res["replay_identical"] = bool(build_features(step).equals(sf))
    return res


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    df, audit = load_and_audit(data)
    feat = build_features(df)
    feat, split_mode = chronological_split(feat)
    tests = run_tests(feat)

    rows = feat[feat["eligible"] & (feat["y"] >= 0)].dropna(subset=FULL)
    tr, va, te = (rows[rows.split == s] for s in ("train", "val", "test"))
    labels = sorted(rows["y"].unique())  # categories actually observed

    support = {s: {CATS[c]: int((d["y"] == c).sum()) for c in range(5)}
               for s, d in [("train", tr), ("val", va), ("test", te)]}
    absent = [CATS[c] for c in range(5) if c not in labels]

    results, val_scores, chosen = {}, {}, {}
    # A: persistence (no fitting)
    results["A_persistence"] = metrics(te["y"], te["current_cat"], labels)
    val_scores["A_persistence"] = metrics(va["y"], va["current_cat"], labels)["macro_f1"]

    # Ablation ladder with Random Forest
    for fs, cols in FEATURE_SETS.items():
        p, vf = tune("rf", cols, tr, va, labels)
        pred = fit_predict("rf", p, tr[cols].values, tr["y"].values, te[cols].values)
        key = f"{fs}_RF"
        results[key], val_scores[key], chosen[key] = metrics(te["y"], pred, labels), vf, p

    # Full-feature model comparison
    for name in ["logreg"] + (["xgb"] if HAVE_XGB else []):
        p, vf = tune(name, FULL, tr, va, labels)
        pred = fit_predict(name, p, tr[FULL].values, tr["y"].values, te[FULL].values)
        key = f"E_full_{name}"
        results[key], val_scores[key], chosen[key] = metrics(te["y"], pred, labels), vf, p

    # Summary table
    summary = pd.DataFrame([
        {"model": k, "val_macro_f1": val_scores[k], "test_accuracy": v["accuracy"],
         "test_macro_f1": v["macro_f1"], "test_macro_f1_present": v["macro_f1_present"],
         "delta_macro_f1_vs_persistence": round(v["macro_f1"]
                                                - results["A_persistence"]["macro_f1"], 3)}
        for k, v in results.items()])
    summary.to_csv(os.path.join(outdir, "results_summary.csv"), index=False)

    # Confusion matrix of full RF on test
    best_rf = chosen["E_+dose_time_RF"]
    pred_rf = fit_predict("rf", best_rf, tr[FULL].values, tr["y"].values, te[FULL].values)
    cm = confusion_matrix(te["y"], pred_rf, labels=labels)
    feat.to_csv(os.path.join(outdir, "modeling_table.csv"), index=False)

    # Figure 1: trace + dose + split shading
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(df["timestamp"], df["uvi"], "o-", ms=3, color="#d97706", label="UVI proxy (5-min)")
    for e, c in zip(EDGES[1:-1], ["#999"] * 4):
        ax.axhline(e, ls=":", lw=0.8, color=c)
    colors = {"train": "#e5f4e4", "val": "#f9eeda", "test": "#e7f1fb"}
    for s, c in colors.items():
        d = feat[feat.split == s]["timestamp"]
        ax.axvspan(d.min(), d.max(), color=c, alpha=0.6, label=s)
    ax.set_ylabel("UVI proxy")
    ax2 = ax.twinx()
    dose = feat["dose_session"]
    ax2.plot(feat["timestamp"], dose / 1000, color="#444", lw=1.2, label="Cumulative dose")
    ax2.set_ylabel("Cumulative dose (kJ/m², sensor)")
    ax.set_title("SYNTHETIC data - pipeline test only (not sensor measurements)", fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_trace_dose_split.png"), dpi=200)
    plt.close(fig)

    report = {
        "WARNING": "Synthetic dataset - pipeline validation only, not experimental results",
        "audit": audit, "split_mode": split_mode,
        "eligible_rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "category_support": support, "absent_categories_unvalidated": absent,
        "label_set_for_macro_f1": [CATS[l] for l in labels],
        "chosen_hyperparameters": chosen,
        "confusion_matrix_full_RF_test": {"labels": [CATS[l] for l in labels],
                                          "matrix": cm.tolist()},
        "results": results, "validation_checks": tests,
    }
    with open(os.path.join(outdir, "aim2_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(json.dumps({k: report[k] for k in
                      ["audit", "split_mode", "eligible_rows", "category_support",
                       "absent_categories_unvalidated", "validation_checks"]},
                     indent=2, default=str))
    print(summary.to_string(index=False))
    print("Confusion matrix (full RF, test):", [CATS[l] for l in labels], cm.tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sunsense_synthetic_uv_dataset.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    main(a.data, a.out)
