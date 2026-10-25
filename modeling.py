#!/usr/bin/env python3
"""
modeling.py - Disease outbreak forecasting for the "Outsmarting Disease Outbreaks" challenge.

Enhanced Pipeline incorporating key techniques to bring predictions and MAE lower:
  1. Multi-Target Transformations:
     - 'org': Original target with L1 loss
     - 'log': Log1p transformation log(1 + y) to stabilize variance and compress right-skew
     - 'quantile': Sub-median quantile regression (alpha=0.42) for zero-inflated distributions
     - 'diff': Residual from historical lagged median
     - 'ratio': Log-ratio relative to historical lagged median
     - 'median': Historical lagged median baseline
  2. Multi-Model Tree Ensemble:
     - LightGBM (MAE & Quantile objectives)
     - XGBoost (reg:absoluteerror objective)
  3. Dynamic Temporal & Static Lag Features:
     - emaT: Exponential moving average with decay 0.8^i (recency emphasis)
     - pzeroT: Empirical zero-frequency across the lag window
     - minT, maxT, medianT, meanT, lastT, quantiles, semT, stdT
  4. Environmental Feature Selection:
     - Merges nearest environmental stations (toilets, waste management, water sources)
     - Retains features with |correlation| > 0.05 and eliminates multicollinearity
  5. Dual Cross-Validation:
     - KFold + TimeSeriesSplit cross-validation forward chaining
  6. Zero-Hurdle & Post-Processing Optimization:
     - Cholera prior: Enforces strictly 0.0 (100% zero in historical observations)
     - Inactivity prior: If maxT == 0 across the lag window, clamps prediction to 0.0
     - Zero-threshold hurdle: Clamps small predictions (< tau) on sparse series to 0.0
     - Shrinkage calibration: Searches optimal multiplier alpha in [0.85, 1.00] on validation MAE
  7. Time-stamped Submission:
     - Formats output identically to SampleSubmission.csv with non-negative bounded predictions.

Usage:
    python modeling.py                     # Full run with validation & calibration
    python modeling.py --skip-validation   # Fast run generating test submission directly
"""
import argparse
import os
import time
import warnings
from datetime import datetime
from itertools import combinations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# =============================================================================
# 1. Configuration
# =============================================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

target = "Total"
KEYS = ["Disease", "Location", "Year", "Month"]
cat_cols = ["Category_Health_Facility_UUID", "Disease"]
loc_cols = ["Transformed_Latitude", "Transformed_Longitude"]
key_cols = loc_cols + ["Year", "Month"]
grp_cols = KEYS + ["Category_Health_Facility_UUID"] + loc_cols

# Environmental variables (see VariableDefinitions.txt)
toilets_cols = [
    "10u", "10v", "2d", "2t", "evabs", "evaow", "evatc", "evavt", "albedo", "lshf",
    "lai_hv", "lai_lv", "pev", "ro", "src", "skt", "es", "stl1", "stl2", "stl3", "stl4",
    "ssro", "slhf", "ssr", "str", "sp", "sro", "sshf", "ssrd", "strd", "e", "tp",
    "swvl1", "swvl2", "swvl3", "swvl4",
]
ENV_SOURCES = {"toilets": "_toilets", "waste_management": "_wm", "water_sources": "_water"}

disease_cols = ["Diarrhea", "Malaria", "Intestinal Worms", "Typhoid", "Dysentery",
                "Schistosomiasis", "Cholera"]
dis_others = ["Dysentery", "Typhoid", "Cholera", "Schistosomiasis"]

blacklist_locations = []

N_LAGS = 36          # number of monthly lags created
LAG_WINDOW = 12      # number of lags (starting at horizon) used per model
CORR_THRESHOLD = 0.05
TEST_YEAR = 2023

corr_features = []   # populated in main()
CAT_LEVELS = {}      # categorical levels shared by train/test

# =============================================================================
# Model Hyperparameters (MAE / Quantile objectives)
# =============================================================================
_lgb_base = dict(
    n_estimators=450, learning_rate=0.04, num_leaves=18, max_depth=5,
    min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_alpha=0.2, reg_lambda=1.0, objective="mae", random_state=42, n_jobs=4, verbose=-1,
)
lgb_params_general = dict(_lgb_base)
lgb_params_malaria = dict(_lgb_base, num_leaves=25, max_depth=6, min_child_samples=25)
lgb_params_inst = dict(_lgb_base, num_leaves=20, learning_rate=0.035, n_estimators=500)
lgb_params_diarrhea = dict(_lgb_base, num_leaves=25, max_depth=6, colsample_bytree=0.65)
lgb_params_sum = dict(_lgb_base, num_leaves=12, max_depth=4, min_child_samples=15)

_xgb_base = dict(
    n_estimators=300, learning_rate=0.04, max_depth=4, subsample=0.8,
    colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=1.0,
    objective="reg:absoluteerror", random_state=42, n_jobs=4,
)
xgb_params_general = dict(_xgb_base)
xgb_params_malaria = dict(_xgb_base, max_depth=5, learning_rate=0.035)
xgb_params_diarrhea = dict(_xgb_base, max_depth=5, learning_rate=0.035)

model_parameter_sets = {
    "Others": {"lgb": lgb_params_general, "xgb": xgb_params_general},
    "Malaria": {"lgb": lgb_params_malaria, "xgb": xgb_params_malaria},
    "Intestinal Worms": {"lgb": lgb_params_inst, "xgb": xgb_params_general},
    "Diarrhea": {"lgb": lgb_params_diarrhea, "xgb": xgb_params_diarrhea},
    "Diarrhea_sum": {"lgb": lgb_params_sum, "xgb": xgb_params_general},
    "Malaria_sum": {"lgb": lgb_params_sum, "xgb": xgb_params_general},
}

_train_base = dict(
    year_valid=2022,
    use_org_target=True,
    use_log_target=True,
    use_quantile_target=True,
    use_diff_target=True,
    use_ratio_target=True,
    use_median_target=True,
    quantile_alpha=0.42,
    cv_types=("kfold", "timeseries"),
    n_splits=3,
    cv_shuffle=True,
    cv_rs=42,
    na_val=0.0,
)
training_parameter_sets = {
    "Others": dict(_train_base),
    "Malaria": dict(_train_base),
    "Intestinal Worms": dict(_train_base),
    "Diarrhea": dict(_train_base),
    "Diarrhea_sum": dict(_train_base),
    "Malaria_sum": dict(_train_base),
}

DISEASE_GROUPS = {
    "Others": dict(diseases=dis_others, levels=["row"]),
    "Malaria": dict(diseases=["Malaria"], levels=["row", "sum"]),
    "Intestinal Worms": dict(diseases=["Intestinal Worms"], levels=["row"]),
    "Diarrhea": dict(diseases=["Diarrhea"], levels=["row", "sum"]),
}

# Candidate transformation names for ensembling
ALL_TRANSFORMS = ["log", "quantile", "org", "diff", "ratio", "median"]
ENS_COLS_DEFAULT = {
    "Others": ["log", "quantile", "median"],
    "Malaria": ["log", "quantile", "median"],
    "Intestinal Worms": ["log", "quantile", "median"],
    "Diarrhea": ["log", "quantile", "median"],
}


# =============================================================================
# 2. Data Loading
# =============================================================================
def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))
    env = {name: pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv")) for name in ENV_SOURCES}

    train = train.dropna(subset=["Location", "Disease"]).copy()
    test[target] = np.nan
    test["Predicted_Total"] = np.nan
    train["is_test"] = 0
    test["is_test"] = 1
    return train, test, sample, env


# =============================================================================
# 3. Aggregation
# =============================================================================
def build_frames(train, test):
    """Row level frame (with tag_id) and summed frame (train_sum / test_sum)."""
    full = pd.concat([train, test], ignore_index=True, sort=False)
    coords = full.groupby("Location")[loc_cols].first()
    full = full.drop(columns=loc_cols).merge(coords, left_on="Location", right_index=True)
    full["tag_id"] = full.groupby(KEYS).cumcount() + 1

    full_sum = (full.groupby(grp_cols, as_index=False)
                .agg(Total=(target, lambda s: s.sum(min_count=1)), is_test=("is_test", "max")))
    full_sum["tag_id"] = 1
    return full, full_sum


# =============================================================================
# 4. Feature Engineering
# =============================================================================
def feature_engineering_and_preprocessing(df):
    df = df.copy()
    df["day"] = 1
    df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=df["day"]))
    df["tag"] = 1
    df["time_index"] = (df["date"] - pd.Timestamp("2019-01-01")).dt.days
    df = df.sort_values("date", kind="stable")
    df["start_date"] = df.groupby("Location")["date"].transform("min")
    df["diff_date"] = (df["date"] - df["start_date"]).dt.days
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    return df


# =============================================================================
# 5. Lag Features
# =============================================================================
def create_lag_features(df, n_lags=N_LAGS):
    on = ["Disease", "Location", "tag_id", "date"]
    base = df.loc[df[target].notna(), on + [target]]
    for i in range(1, n_lags + 1):
        temp_df = base.copy()
        temp_df["date"] = temp_df["date"] + pd.DateOffset(months=i)
        temp_df = temp_df.rename(columns={target: f"{target}_{i}"})
        df = df.merge(temp_df, on=on, how="left")
    return df


# =============================================================================
# 6. Environmental Data Merge
# =============================================================================
def merge_environmental_data(df, env):
    df = df.copy()
    df["hosp_id"] = df["Disease"].astype(str) + "_" + df["Location"].astype(str)
    locs = df.drop_duplicates("Location")[["Location"] + loc_cols].reset_index(drop=True)
    env_feature_cols = []
    for name, suffix in ENV_SOURCES.items():
        edf = env[name]
        cols = [c for c in toilets_cols if c in edf.columns]
        e = edf.groupby(key_cols, as_index=False)[cols].mean()
        sites = e[loc_cols].drop_duplicates().reset_index(drop=True)
        _, idx = cKDTree(sites.values).query(locs[loc_cols].values)
        site_lat, site_lon = f"site_lat{suffix}", f"site_lon{suffix}"
        mapping = pd.DataFrame({"Location": locs["Location"],
                                site_lat: sites.iloc[idx, 0].values,
                                site_lon: sites.iloc[idx, 1].values})
        e = e.rename(columns={"Transformed_Latitude": site_lat,
                              "Transformed_Longitude": site_lon,
                              **{c: f"{c}{suffix}" for c in cols}})
        df = df.merge(mapping, on="Location", how="left")
        df = df.merge(e, on=[site_lat, site_lon, "Year", "Month"], how="left")
        env_feature_cols += [f"{c}{suffix}" for c in cols]
    return df, env_feature_cols


# =============================================================================
# 7. Enhanced Static Features
# =============================================================================
def get_static_features(df, cols):
    v = df[cols]
    out = pd.DataFrame(index=df.index)
    out["medianT"] = v.median(axis=1)
    out["meanT"] = v.mean(axis=1)
    out["maxT"] = v.max(axis=1)
    out["minT"] = v.min(axis=1)
    out["sumT"] = v.sum(axis=1, min_count=1)
    out["prodT"] = np.log1p(v.clip(lower=0)).sum(axis=1, min_count=1)
    out["skewT"] = v.skew(axis=1)
    out["kurtT"] = v.kurt(axis=1)
    out["semT"] = v.sem(axis=1)
    out["stdT"] = v.std(axis=1)
    out["q75T"] = v.quantile(0.75, axis=1)
    out["q95T"] = v.quantile(0.95, axis=1)
    out["q99T"] = v.quantile(0.99, axis=1)
    out["nzT"] = (v > 0).sum(axis=1)
    out["cntT"] = v.notna().sum(axis=1)
    out["lastT"] = v.bfill(axis=1).iloc[:, 0]

    # Recency-weighted Exponential Moving Average (EMA)
    arr = v.values
    weights = np.array([0.8**i for i in range(len(cols))])
    weights = weights / weights.sum()
    out["emaT"] = np.nansum(np.nan_to_num(arr) * weights, axis=1)

    # Empirical zero probability across lag window
    out["pzeroT"] = (arr == 0).sum(axis=1) / np.maximum((~np.isnan(arr)).sum(axis=1), 1)

    return pd.concat([df.drop(columns=out.columns, errors="ignore"), out], axis=1)


STATIC_COLS = [
    "medianT", "meanT", "maxT", "minT", "sumT", "prodT", "skewT", "kurtT",
    "semT", "stdT", "q75T", "q95T", "q99T", "nzT", "cntT", "lastT", "emaT", "pzeroT"
]


# =============================================================================
# 8. Correlation Feature Selection
# =============================================================================
def select_correlated_features(df_sum, cols, threshold=CORR_THRESHOLD):
    tr = df_sum[(df_sum["Year"] < TEST_YEAR) & df_sum[target].notna()]
    corr = tr[cols + [target]].corr()[target].drop(target)
    selected = corr[corr.abs() > threshold]
    selected_cols = list(dict.fromkeys(selected.index))
    deduped = tr[selected_cols].T.drop_duplicates().index.tolist()
    return deduped


# =============================================================================
# 9. Generic Cross-Validation Trainer
# =============================================================================
def _prepare(df, feature_cols, is_lgb, na_val):
    d = df[feature_cols].copy()
    if is_lgb:
        for c in feature_cols:
            if c in CAT_LEVELS:
                d[c] = pd.Categorical(d[c], categories=CAT_LEVELS[c])
    else:
        d = d.replace([np.inf, -np.inf], np.nan).fillna(na_val).astype(float)
    return d


def generic_train_cv(X, y, valid_df, test_df, model, feature_cols, cv_type="kfold",
                     n_splits=3, shuffle=True, rs=42, na_val=0.0):
    """Returns (mean validation prediction, mean test prediction) across folds."""
    is_lgb = isinstance(model, lgb.LGBMRegressor)
    is_xgb = isinstance(model, xgb.XGBRegressor)

    if cv_type == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=rs if shuffle else None)
    elif cv_type == "timeseries":
        splitter = TimeSeriesSplit(n_splits=n_splits)
    else:
        raise ValueError(f"Unknown cv_type {cv_type}")

    Xf = _prepare(X, feature_cols, is_lgb, na_val)
    Vf = _prepare(valid_df, feature_cols, is_lgb, na_val) if valid_df is not None and len(valid_df) else None
    Tf = _prepare(test_df, feature_cols, is_lgb, na_val) if test_df is not None and len(test_df) else None
    y = np.asarray(y, dtype=float)

    valid_preds, test_preds = [], []
    for tr_idx, va_idx in splitter.split(Xf):
        m = clone(model)
        X_tr, y_tr = Xf.iloc[tr_idx], y[tr_idx]
        X_va, y_va = Xf.iloc[va_idx], y[va_idx]

        if is_lgb:
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
        elif is_xgb:
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            m.fit(X_tr, y_tr)

        if Vf is not None:
            valid_preds.append(m.predict(Vf))
        if Tf is not None:
            test_preds.append(m.predict(Tf))

    v = np.mean(valid_preds, axis=0) if valid_preds else None
    t = np.mean(test_preds, axis=0) if test_preds else None
    return v, t


# =============================================================================
# 10. Multi-Model Horizon Trainer
# =============================================================================
def _fit_predict_ensemble(X, y, V, T, feats_lgb, feats_num, model_params, tp,
                          override_objective=None, override_alpha=None):
    """Trains every model type across specified CV types and averages."""
    vps, tps = [], []
    for mname, params in model_params.items():
        p = dict(params)
        if override_objective is not None:
            p["objective"] = override_objective
        if override_alpha is not None and "alpha" in p:
            p["alpha"] = override_alpha

        if mname == "lgb":
            model, feats = lgb.LGBMRegressor(**p), feats_lgb
        elif mname == "xgb":
            # XGBoost only takes numerical features
            model, feats = xgb.XGBRegressor(**p), feats_num
        else:
            continue

        for cv_type in tp["cv_types"]:
            v, t = generic_train_cv(X, y, V, T, model, feats, cv_type=cv_type,
                                    n_splits=tp["n_splits"], shuffle=tp["cv_shuffle"],
                                    rs=tp["cv_rs"], na_val=tp["na_val"])
            if v is not None:
                vps.append(v)
            if t is not None:
                tps.append(t)
    v = np.mean(vps, axis=0) if vps else None
    t = np.mean(tps, axis=0) if tps else None
    return v, t


def generic_model_trainer(df, disease_name, model_params, train_params, env_feats,
                          do_validation=True):
    """
    Trains horizon-specific models for each forecast month.
    Evaluates org, log, quantile, diff, ratio, and median target transformations.
    """
    tp = train_params
    year_valid = tp["year_valid"]
    transforms = [
        t for t, flag in [
            ("log", tp.get("use_log_target", True)),
            ("quantile", tp.get("use_quantile_target", True)),
            ("org", tp.get("use_org_target", True)),
            ("diff", tp.get("use_diff_target", True)),
            ("ratio", tp.get("use_ratio_target", True)),
            ("median", tp.get("use_median_target", True)),
        ] if flag
    ]

    test_mask_all = df["is_test"] == 1
    test_out = pd.DataFrame(index=df.index[test_mask_all], columns=transforms, dtype=float)
    valid_out = []

    base_feats = ["Month", "month_sin", "month_cos", "diff_date", "tag_id"] + loc_cols
    trend_feats = ["Year", "time_index"]

    for m in sorted(df.loc[test_mask_all, "Month"].unique()):
        lag_cols = [f"{target}_{i}" for i in range(m, min(m + LAG_WINDOW, N_LAGS + 1))]
        extra = [f"{target}_{i}" for i in (24, 36) if i >= m + LAG_WINDOW and i <= N_LAGS]
        d = get_static_features(df, lag_cols)

        feats_num = base_feats + lag_cols + extra + STATIC_COLS + env_feats
        feats_lgb = feats_num + trend_feats + cat_cols

        hist = d[(d["is_test"] == 0) & d[target].notna() & (d["cntT"] > 0)]
        hist = hist.sort_values("date", kind="stable").reset_index()
        test_rows = d[(d["is_test"] == 1) & (d["Month"] == m)].reset_index()

        phases = []
        if do_validation:
            X_v = hist[hist["Year"] < year_valid].reset_index(drop=True)
            V_v = hist[(hist["Year"] == year_valid) & (hist["Month"] == m)
                       & (hist["tag_id"] == 1)].reset_index(drop=True)
            if len(X_v) > 50 and len(V_v):
                phases.append(("valid", X_v, V_v))
        phases.append(("test", hist, test_rows))

        for phase, X, E in phases:
            res = {}
            for tr in transforms:
                if tr == "median":
                    res[tr] = E["medianT"].values
                    continue
                elif tr == "log":
                    # log1p transformation: natural variance stabilization
                    Xt = X
                    y = np.log1p(np.clip(Xt[target].values, 0, None))
                    v_p, t_p = _fit_predict_ensemble(Xt, y, E if phase == "valid" else None,
                                                      E if phase == "test" else None,
                                                      feats_lgb, feats_num, model_params, tp)
                    p = v_p if phase == "valid" else t_p
                    p = np.expm1(np.clip(p, 0, None))
                elif tr == "quantile":
                    # Quantile regression targeting sub-median (alpha=0.42)
                    Xt = X
                    y = Xt[target].values
                    # For quantile, use LightGBM with quantile objective
                    p_q = dict(model_params.get("lgb", lgb_params_general))
                    p_q["objective"] = "quantile"
                    p_q["alpha"] = tp.get("quantile_alpha", 0.42)
                    m_q = lgb.LGBMRegressor(**p_q)
                    Xf = _prepare(Xt, feats_lgb, True, 0.0)
                    Ef = _prepare(E, feats_lgb, True, 0.0)
                    m_q.fit(Xf, y)
                    p = np.clip(m_q.predict(Ef), 0, None)
                elif tr == "org":
                    Xt, y = X, X[target].values
                    v_p, t_p = _fit_predict_ensemble(Xt, y, E if phase == "valid" else None,
                                                      E if phase == "test" else None,
                                                      feats_lgb, feats_num, model_params, tp)
                    p = v_p if phase == "valid" else t_p
                elif tr in ("diff", "ratio"):
                    Xt = X[X["medianT"].notna()].reset_index(drop=True)
                    if tr == "diff":
                        y = (Xt[target] - Xt["medianT"]).values
                    else:
                        y = np.log((Xt[target] + 1.0) / (Xt["medianT"] + 1.0)).values
                    v_p, t_p = _fit_predict_ensemble(Xt, y, E if phase == "valid" else None,
                                                      E if phase == "test" else None,
                                                      feats_lgb, feats_num, model_params, tp)
                    p = v_p if phase == "valid" else t_p
                    if tr == "diff":
                        p = p + E["medianT"].values
                    else:
                        p = np.exp(np.clip(p, -3, 3)) * (E["medianT"].values + 1.0) - 1.0

                res[tr] = np.clip(p, 0, None)

            if phase == "test":
                for tr, p in res.items():
                    test_out.loc[E["index"].values, tr] = p
            else:
                vo = pd.DataFrame(res)
                vo["y"] = E[target].values
                vo["Disease"] = E["Disease"].values
                vo["Month"] = m
                if "maxT" in E.columns:
                    vo["maxT"] = E["maxT"].values
                    vo["pzeroT"] = E["pzeroT"].values
                valid_out.append(vo)

        print(f"   [{disease_name}] month {m:2d} done", flush=True)

    valid_out = pd.concat(valid_out, ignore_index=True) if valid_out else pd.DataFrame()
    return test_out, valid_out


# =============================================================================
# 11. Ensembling & Zero-Hurdle Calibration
# =============================================================================
def ensemble_predictions_average(predictions_list):
    """Simple robust averaging ensemble."""
    return np.nanmean(np.vstack(predictions_list), axis=0)


def ensemble_frame(pred_df, ens_cols):
    cols = [c for c in ens_cols if c in pred_df.columns]
    if not cols:
        cols = [c for c in ("log", "quantile", "median", "org") if c in pred_df.columns]
    ens = ensemble_predictions_average([pred_df[c].values.astype(float) for c in cols])
    if "median" in pred_df.columns:
        ens = np.where(np.isnan(ens), pred_df["median"].values.astype(float), ens)
    elif "org" in pred_df.columns:
        ens = np.where(np.isnan(ens), pred_df["org"].values.astype(float), ens)
    return np.clip(np.nan_to_num(ens, nan=0.0), 0, None)


def apply_zero_hurdle(preds, max_lags=None, pzero_lags=None, threshold=0.6):
    """
    Hurdle post-processing to bring predictions and MAE lower:
      - If historical series had strictly 0 cases in the lag window (max_lags == 0), set 0
      - If prediction is below threshold and historical zero probability is high, set 0
    """
    p = np.asarray(preds, dtype=float).copy()
    if max_lags is not None:
        p[max_lags == 0] = 0.0
    if pzero_lags is not None:
        p[(p < threshold) & (pzero_lags > 0.6)] = 0.0
    else:
        p[p < threshold] = 0.0
    return np.clip(p, 0, None)


def calibrate_shrinkage(y_true, y_pred):
    """Finds optimal multiplier alpha in [0.75, 1.00] that minimizes MAE."""
    best_alpha = 1.0
    best_mae = mean_absolute_error(y_true, y_pred)
    for alpha in np.linspace(0.80, 1.00, 21):
        score = mean_absolute_error(y_true, y_pred * alpha)
        if score < best_mae:
            best_mae = score
            best_alpha = alpha
    return best_alpha, best_mae


def report_and_tune_validation(name, vdf, default_ens):
    """
    Evaluates all transformations on validation data, selects the best ensemble subset,
    optimizes the hurdle threshold and shrinkage factor.
    """
    if vdf.empty:
        return default_ens, 0.6, 1.0

    vdf = vdf.copy()
    cands = [c for c in ALL_TRANSFORMS if c in vdf.columns]
    scores = {}
    for r in range(1, min(4, len(cands) + 1)):
        for combo in combinations(cands, r):
            scores[combo] = mean_absolute_error(vdf["y"], ensemble_frame(vdf, list(combo)))

    best_combo = list(min(scores, key=scores.get))
    raw_ens = ensemble_frame(vdf, best_combo)
    raw_mae = mean_absolute_error(vdf["y"], raw_ens)

    # Tune hurdle & shrinkage
    max_lags = vdf["maxT"].values if "maxT" in vdf.columns else None
    pzero = vdf["pzeroT"].values if "pzeroT" in vdf.columns else None

    best_tau, best_tau_mae = 0.6, raw_mae
    for tau in [0.0, 0.4, 0.6, 0.8, 1.0]:
        h_pred = apply_zero_hurdle(raw_ens, max_lags, pzero, threshold=tau)
        score = mean_absolute_error(vdf["y"], h_pred)
        if score < best_tau_mae:
            best_tau_mae = score
            best_tau = tau

    h_pred = apply_zero_hurdle(raw_ens, max_lags, pzero, threshold=best_tau)
    best_alpha, final_mae = calibrate_shrinkage(vdf["y"], h_pred)

    msg = ", ".join(f"{c}={mean_absolute_error(vdf['y'], np.clip(vdf[c].fillna(0), 0, None)):.3f}"
                    for c in cands)
    print(f"   >> Validation MAE ({name}, {len(vdf)} rows): {msg}", flush=True)
    print(f"   >> Best Ensemble {best_combo}: MAE={raw_mae:.3f} -> Hurdle(tau={best_tau:.1f}) + Shrink({best_alpha:.2f}): MAE={final_mae:.3f}", flush=True)

    return best_combo, best_tau, best_alpha


# =============================================================================
# 12. Main Execution
# =============================================================================
def main():
    global corr_features, CAT_LEVELS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-validation", action="store_true", help="skip 2022 hold-out validation")
    ap.add_argument("--row-weight", type=float, default=0.5,
                    help="weight of row-level vs sum-level ensemble for Malaria/Diarrhea")
    ap.add_argument("--out-dir", default=DATA_DIR, help="where to write the submission")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("Disease Outbreak Prediction Modeling Pipeline (Enhanced MAE Reduction)")
    print("=" * 70)

    print("\n[1/5] Loading datasets ...")
    train, test, sample, env = load_data()
    full, full_sum = build_frames(train, test)
    CAT_LEVELS = {c: sorted(full[c].astype(str).unique()) for c in cat_cols}

    print("[2/5] Engineering temporal, lag & environmental features ...")
    frames = {}
    for lvl, df in [("row", full), ("sum", full_sum)]:
        df = feature_engineering_and_preprocessing(df)
        df = create_lag_features(df)
        df, env_cols = merge_environmental_data(df, env)
        frames[lvl] = df.reset_index(drop=True)

    corr_features = select_correlated_features(frames["sum"], env_cols)
    print(f"   Selected {len(corr_features)} correlated environmental features: {corr_features[:6]} ...")

    print("\n[3/5] Disease-Specific Modeling & Horizon Forecasting ...")
    test_preds = {}
    tuning_params = {}

    for group, cfg in DISEASE_GROUPS.items():
        for lvl in cfg["levels"]:
            pkey = f"{group}_sum" if (lvl == "sum" and f"{group}_sum" in model_parameter_sets) else group
            df = frames[lvl][frames[lvl]["Disease"].isin(cfg["diseases"])].reset_index(drop=True)
            print(f"\n--- Training {group} [{lvl}-level] ({int((df.is_test == 0).sum())} train rows) ---", flush=True)

            tp_df, v_df = generic_model_trainer(
                df, f"{group}/{lvl}", model_parameter_sets[pkey], training_parameter_sets[pkey],
                corr_features, do_validation=not args.skip_validation)

            best_cols, best_tau, best_alpha = report_and_tune_validation(
                f"{group}/{lvl}", v_df, ENS_COLS_DEFAULT.get(group, ["log", "quantile", "median"]))

            tuning_params[(group, lvl)] = (best_cols, best_tau, best_alpha)

            # Compute ensemble prediction on test set
            out = tp_df.copy()
            ens_vals = ensemble_frame(tp_df, best_cols)

            # Extract lag statistics for test rows to apply hurdle post-processing
            test_indices = tp_df.index
            d_lags = get_static_features(df.loc[test_indices], [f"{target}_{i}" for i in range(1, 13)])
            max_l = d_lags["maxT"].values if "maxT" in d_lags.columns else None
            pz_l = d_lags["pzeroT"].values if "pzeroT" in d_lags.columns else None

            # Apply Hurdle & Shrinkage
            h_vals = apply_zero_hurdle(ens_vals, max_l, pz_l, threshold=best_tau)
            out["Predicted_Total_ens"] = h_vals * best_alpha

            out = out.join(df.loc[test_indices, KEYS + (["ID"] if lvl == "row" else [])])
            test_preds[(group, lvl)] = out

    # ---------------------------------------------------------------- Final Assembly
    print("\n[4/5] Assembling final predictions across diseases ...")
    sub_test = test[["ID"] + KEYS].copy()
    sub_test["Predicted_Total"] = 0.0

    for group, cfg in DISEASE_GROUPS.items():
        row = test_preds[(group, "row")][["ID", "Predicted_Total_ens"]].rename(
            columns={"Predicted_Total_ens": "p_row"})
        tmp = sub_test[["ID"] + KEYS].merge(row, on="ID", how="inner")

        if "sum" in cfg["levels"]:
            s = test_preds[(group, "sum")][KEYS + ["Predicted_Total_ens"]].rename(
                columns={"Predicted_Total_ens": "p_sum"})
            tmp = tmp.merge(s, on=KEYS, how="left")
            tmp["p_sum"] = tmp["p_sum"].fillna(tmp["p_row"])
            w = args.row_weight
            tmp["p"] = w * tmp["p_row"] + (1 - w) * tmp["p_sum"]
            if group == "Diarrhea" and blacklist_locations:
                bl = tmp["Location"].isin(blacklist_locations)
                tmp.loc[bl, "p"] = tmp.loc[bl, "p_sum"]
        else:
            tmp["p"] = tmp["p_row"]

        sub_test.loc[sub_test["ID"].isin(tmp["ID"]), "Predicted_Total"] = (
            sub_test["ID"].map(tmp.set_index("ID")["p"]))

    # Domain Priors: Cholera has 100% historical zero cases
    sub_test.loc[sub_test["Disease"] == "Cholera", "Predicted_Total"] = 0.0

    # Ensure non-negative bounds
    sub_test["Predicted_Total"] = sub_test["Predicted_Total"].fillna(0.0).clip(lower=0.0)

    # ---------------------------------------------------------------- Submission
    print("\n[5/5] Generating submission file ...")
    pred_col = [c for c in sample.columns if c != "ID"][0]
    sub = sample[["ID"]].merge(sub_test[["ID", "Predicted_Total"]], on="ID", how="left")
    sub = sub.rename(columns={"Predicted_Total": pred_col}).fillna(0.0)

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"submission_{stamp}.csv")
    sub.to_csv(out_path, index=False)

    print(f"\n>>> Time-stamped Submission saved to: {out_path}")
    print("\nPrediction Summary Statistics:")
    print(sub[pred_col].describe())

    print("\nMean Predicted Outbreaks by Disease:")
    print(sub_test.groupby("Disease")["Predicted_Total"].mean().rename("mean_prediction"))

    print(f"\nExecution completed in {(time.time() - t0) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
