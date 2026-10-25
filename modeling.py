#!/usr/bin/env python3
"""
modeling.py - Disease outbreak forecasting for the "Outsmarting Disease Outbreaks" challenge.

Implements the approach described in README.md and in
"Mastering Disease Outbreak Prediction with Time Series Forecasting":

  1. Configuration
  2. Data loading (Train / Test / toilets / waste_management / water_sources)
  3. Aggregation (row level with `tag_id` + summed level `train_sum`)
  4. Temporal feature engineering (date, time_index, diff_date, ...)
  5. Lag features of the target (Total_1 ... Total_36)
  6. Environmental data merge (nearest environmental site, same Year/Month)
  7. Static (summary) features over the lag window (medianT, meanT, ...)
  8. Correlation based environmental feature selection (|corr| > 0.05)
  9. Generic CV trainer (KFold + TimeSeriesSplit, optional RobustScaler)
 10. Disease-specific model trainer (original / difference / ratio / median targets,
     LightGBM + LinearSVR), one model per forecast horizon (test month)
 11. Ensembling and a time-stamped submission file.

Usage:
    python modeling.py                     # validation + test predictions
    python modeling.py --skip-validation   # faster, test predictions only
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
from sklearn.svm import LinearSVR

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

# Diarrhea locations predicted with the aggregated (sum-level) model only.
blacklist_locations = []

N_LAGS = 36          # number of monthly lags created
LAG_WINDOW = 12      # number of lags (starting at the horizon) used per model
CORR_THRESHOLD = 0.05
TEST_YEAR = 2023

corr_features = []   # populated in main()
CAT_LEVELS = {}      # categorical levels shared by train/test (populated in main())

# =============================================================================
# Model / training parameters
# =============================================================================
_lgb_base = dict(
    n_estimators=500, learning_rate=0.05, num_leaves=15, max_depth=5,
    min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, objective="mae", random_state=42, n_jobs=4, verbose=-1,
)
lgb_params_general = dict(_lgb_base)
lgb_params_malaria = dict(_lgb_base, num_leaves=31, max_depth=6, min_child_samples=30)
lgb_params_inst = dict(_lgb_base, num_leaves=20, learning_rate=0.04, n_estimators=600)
lgb_params_diarrhea = dict(_lgb_base, num_leaves=31, max_depth=6, colsample_bytree=0.6)
lgb_params_bdr = dict(_lgb_base, num_leaves=10, max_depth=4, min_child_samples=10)
# primal solver (dual=False) is orders of magnitude faster than the dual L1-loss solver here
svr_params = dict(C=0.05, epsilon=0.0, loss="squared_epsilon_insensitive", dual=False,
                  max_iter=5000, tol=1e-4, random_state=42)

model_parameter_sets = {
    "Others": {"lgb": lgb_params_general, "svr": svr_params},
    "Malaria": {"lgb": lgb_params_malaria, "svr": svr_params},
    "Intestinal Worms": {"lgb": lgb_params_inst, "svr": svr_params},
    "Diarrhea": {"lgb": lgb_params_diarrhea, "svr": svr_params},
    "Diarrhea_sum": {"lgb": lgb_params_bdr, "svr": svr_params},
}

_train_base = dict(
    year_valid=2022, use_org_target=True, use_diff_target=True, use_ratio_target=True,
    use_median_target=True, cv_types=("kfold", "timeseries"), n_splits=3,
    cv_shuffle=True, cv_rs=42, use_scaler=True, na_val=0.0,
)
training_parameter_sets = {
    "Others": dict(_train_base),
    "Malaria": dict(_train_base),
    "Intestinal Worms": dict(_train_base),
    "Diarrhea": dict(_train_base),
    "Diarrhea_sum": dict(_train_base),
}

# Disease groups, the diseases they contain and data levels they are trained on.
DISEASE_GROUPS = {
    "Others": dict(diseases=dis_others, levels=["row"]),
    "Malaria": dict(diseases=["Malaria"], levels=["row", "sum"]),
    "Intestinal Worms": dict(diseases=["Intestinal Worms"], levels=["row"]),
    "Diarrhea": dict(diseases=["Diarrhea"], levels=["row", "sum"]),
}
# Which target transformations are averaged for the final per-level ensemble.
ENS_COLS = {
    "Others": ["org", "diff", "ratio", "median"],
    "Malaria": ["org", "diff", "ratio", "median"],
    "Intestinal Worms": ["org", "diff", "ratio", "median"],
    "Diarrhea": ["org", "diff", "ratio", "median"],
}


# =============================================================================
# 2. Data loading
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
    # Consistent coordinates per location
    coords = full.groupby("Location")[loc_cols].first()
    full = full.drop(columns=loc_cols).merge(coords, left_on="Location", right_index=True)
    full["tag_id"] = full.groupby(KEYS).cumcount() + 1

    full_sum = (full.groupby(grp_cols, as_index=False)
                .agg(Total=(target, lambda s: s.sum(min_count=1)), is_test=("is_test", "max")))
    full_sum["tag_id"] = 1
    return full, full_sum


# =============================================================================
# 4. Feature engineering
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
# 5. Lag features
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
# 6. Environmental data merge (nearest site, same Year/Month)
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
# 7. Static features over a list of (lag) columns
# =============================================================================
def get_static_features(df, cols):
    v = df[cols]
    out = pd.DataFrame(index=df.index)
    out["medianT"] = v.median(axis=1)
    out["meanT"] = v.mean(axis=1)
    out["maxT"] = v.max(axis=1)
    out["minT"] = v.min(axis=1)
    out["sumT"] = v.sum(axis=1, min_count=1)
    out["prodT"] = np.log1p(v.clip(lower=0)).sum(axis=1, min_count=1)  # log-product
    out["skewT"] = v.skew(axis=1)
    out["kurtT"] = v.kurt(axis=1)
    out["semT"] = v.sem(axis=1)
    out["stdT"] = v.std(axis=1)
    out["q75T"] = v.quantile(0.75, axis=1)
    out["q95T"] = v.quantile(0.95, axis=1)
    out["q99T"] = v.quantile(0.99, axis=1)
    out["nzT"] = (v > 0).sum(axis=1)
    out["cntT"] = v.notna().sum(axis=1)
    out["lastT"] = v.bfill(axis=1).iloc[:, 0]  # most recent available value
    return pd.concat([df.drop(columns=out.columns, errors="ignore"), out], axis=1)


STATIC_COLS = ["medianT", "meanT", "maxT", "minT", "sumT", "prodT", "skewT", "kurtT",
               "semT", "stdT", "q75T", "q95T", "q99T", "nzT", "cntT", "lastT"]


# =============================================================================
# 8. Correlation based feature selection
# =============================================================================
def select_correlated_features(df_sum, cols, threshold=CORR_THRESHOLD):
    tr = df_sum[(df_sum["Year"] < TEST_YEAR) & df_sum[target].notna()]
    corr = tr[cols + [target]].corr()[target].drop(target)
    selected = corr[corr.abs() > threshold]
    # drop exact duplicates (several sources share the same climate grid values)
    selected_cols = list(dict.fromkeys(selected.index))
    deduped = tr[selected_cols].T.drop_duplicates().index.tolist()
    return deduped


# =============================================================================
# 9. Generic cross-validation trainer
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
                     n_splits=5, shuffle=True, rs=42, use_scaler=False, na_val=0.0):
    """Returns (mean validation prediction, mean test prediction) across folds."""
    is_lgb = isinstance(model, lgb.LGBMRegressor)
    if cv_type == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=rs if shuffle else None)
    elif cv_type == "timeseries":
        splitter = TimeSeriesSplit(n_splits=n_splits)  # X must be sorted by date
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

        if use_scaler and not is_lgb:
            xs = RobustScaler().fit(X_tr)
            ys = RobustScaler().fit(y_tr.reshape(-1, 1))
            m.fit(np.clip(xs.transform(X_tr), -50, 50), ys.transform(y_tr.reshape(-1, 1)).ravel())

            def predict(D, m=m, xs=xs, ys=ys):
                p = m.predict(np.clip(xs.transform(D), -50, 50))
                return ys.inverse_transform(p.reshape(-1, 1)).ravel()
        else:
            if is_lgb:
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="l1",
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            else:
                m.fit(X_tr, y_tr)
            predict = m.predict

        if Vf is not None:
            valid_preds.append(predict(Vf))
        if Tf is not None:
            test_preds.append(predict(Tf))

    v = np.mean(valid_preds, axis=0) if valid_preds else None
    t = np.mean(test_preds, axis=0) if test_preds else None
    return v, t


# =============================================================================
# 10. Disease specific model trainer
# =============================================================================
def _fit_predict_ensemble(X, y, V, T, feats_lgb, feats_svr, model_params, tp):
    """Trains every model type x every CV type and averages the predictions."""
    vps, tps = [], []
    for mname, params in model_params.items():
        if mname == "lgb":
            model, feats = lgb.LGBMRegressor(**params), feats_lgb
        else:
            model, feats = LinearSVR(**params), feats_svr
        for cv_type in tp["cv_types"]:
            v, t = generic_train_cv(X, y, V, T, model, feats, cv_type=cv_type,
                                    n_splits=tp["n_splits"], shuffle=tp["cv_shuffle"],
                                    rs=tp["cv_rs"], use_scaler=tp["use_scaler"],
                                    na_val=tp["na_val"])
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
    Trains one set of models per forecast horizon (test month).
    For test month m only lags >= m are known (history ends Dec 2022), so the models
    for month m use lags Total_m ... Total_{m+LAG_WINDOW-1} for train and test alike.

    Returns (test_preds DataFrame indexed like df's test rows, valid_preds DataFrame).
    """
    tp = train_params
    year_valid = tp["year_valid"]
    transforms = [t for t, flag in [("org", tp["use_org_target"]), ("diff", tp["use_diff_target"]),
                                    ("ratio", tp["use_ratio_target"]),
                                    ("median", tp["use_median_target"])] if flag]

    test_mask_all = df["is_test"] == 1
    test_out = pd.DataFrame(index=df.index[test_mask_all], columns=transforms, dtype=float)
    valid_out = []

    base_feats = ["Month", "month_sin", "month_cos", "diff_date", "tag_id"] + loc_cols
    trend_feats = ["Year", "time_index"]

    for m in sorted(df.loc[test_mask_all, "Month"].unique()):
        lag_cols = [f"{target}_{i}" for i in range(m, min(m + LAG_WINDOW, N_LAGS + 1))]
        extra = [f"{target}_{i}" for i in (24, 36) if i >= m + LAG_WINDOW and i <= N_LAGS]
        d = get_static_features(df, lag_cols)

        feats_svr = base_feats + lag_cols + extra + STATIC_COLS + env_feats
        feats_lgb = feats_svr + trend_feats + cat_cols

        hist = d[(d["is_test"] == 0) & d[target].notna() & (d["cntT"] > 0)]
        hist = hist.sort_values("date", kind="stable").reset_index()  # keep orig index in 'index'
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
                if tr == "org":
                    Xt, y = X, X[target].values
                else:
                    Xt = X[X["medianT"].notna()].reset_index(drop=True)
                    # reset_index avoids "cannot reindex on an axis with duplicate labels"
                    if tr == "diff":
                        y = (Xt[target] - Xt["medianT"]).values
                    else:
                        # ratio target modelled in log space for numerical stability
                        y = np.log((Xt[target] + 1.0) / (Xt["medianT"] + 1.0)).values
                if phase == "valid":
                    p, _ = _fit_predict_ensemble(Xt, y, E, None, feats_lgb, feats_svr,
                                                 model_params, tp)
                else:
                    _, p = _fit_predict_ensemble(Xt, y, None, E, feats_lgb, feats_svr,
                                                 model_params, tp)
                if tr == "diff":
                    p = p + E["medianT"].values
                elif tr == "ratio":
                    p = np.exp(np.clip(p, -3, 3)) * (E["medianT"].values + 1.0) - 1.0
                res[tr] = p

            if phase == "test":
                for tr, p in res.items():
                    test_out.loc[E["index"].values, tr] = p
            else:
                vo = pd.DataFrame(res)
                vo["y"] = E[target].values
                vo["Disease"] = E["Disease"].values
                vo["Month"] = m
                valid_out.append(vo)

        print(f"   [{disease_name}] month {m:2d} done", flush=True)

    valid_out = pd.concat(valid_out, ignore_index=True) if valid_out else pd.DataFrame()
    return test_out, valid_out


# =============================================================================
# 11. Ensembling helpers
# =============================================================================
def ensemble_predictions_average(predictions_list):
    """Simple (NaN-aware) averaging ensemble."""
    return np.nanmean(np.vstack(predictions_list), axis=0)


def ensemble_frame(pred_df, ens_cols):
    cols = [c for c in ens_cols if c in pred_df.columns]
    ens = ensemble_predictions_average([pred_df[c].values.astype(float) for c in cols])
    # fallback (e.g. unseen location without lag history -> diff/ratio are NaN)
    if "org" in pred_df.columns:
        ens = np.where(np.isnan(ens), pred_df["org"].values.astype(float), ens)
    return np.clip(np.nan_to_num(ens, nan=0.0), 0, None)


def report_validation(name, vdf, ens_cols):
    """Prints validation MAE per transform and returns the best ensemble subset."""
    if vdf.empty:
        return ens_cols
    vdf = vdf.copy()
    cands = [c for c in ("org", "diff", "ratio", "median") if c in vdf.columns]
    scores = {}
    for r in range(1, len(cands) + 1):
        for combo in combinations(cands, r):
            scores[combo] = mean_absolute_error(vdf["y"], ensemble_frame(vdf, list(combo)))
    best = min(scores, key=scores.get)
    vdf["ens"] = ensemble_frame(vdf, ens_cols)
    msg = ", ".join(f"{c}={mean_absolute_error(vdf['y'], np.clip(vdf[c].fillna(0), 0, None)):.3f}"
                    for c in cands + ["ens"])
    print(f"   >> Validation MAE ({name}, {len(vdf)} rows): {msg}", flush=True)
    print(f"   >> Best ensemble {list(best)}: MAE={scores[best]:.3f}", flush=True)
    return list(best)


# =============================================================================
# 12. Main
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
    print("Loading data ...")
    train, test, sample, env = load_data()
    full, full_sum = build_frames(train, test)
    CAT_LEVELS = {c: sorted(full[c].astype(str).unique()) for c in cat_cols}

    print("Feature engineering ...")
    frames = {}
    for lvl, df in [("row", full), ("sum", full_sum)]:
        df = feature_engineering_and_preprocessing(df)
        df = create_lag_features(df)
        df, env_cols = merge_environmental_data(df, env)
        frames[lvl] = df.reset_index(drop=True)

    corr_features = select_correlated_features(frames["sum"], env_cols)
    print(f"Selected {len(corr_features)} correlated environmental features: {corr_features}")

    test_preds = {}  # (group, level) -> DataFrame with transform columns + KEYS + ID
    all_valid = {}
    for group, cfg in DISEASE_GROUPS.items():
        for lvl in cfg["levels"]:
            pkey = f"{group}_sum" if (lvl == "sum" and f"{group}_sum" in model_parameter_sets) else group
            df = frames[lvl][frames[lvl]["Disease"].isin(cfg["diseases"])].reset_index(drop=True)
            print(f"\n=== Training {group} [{lvl}-level] on {int((df.is_test == 0).sum())} rows ===", flush=True)
            tp_df, v_df = generic_model_trainer(
                df, f"{group}/{lvl}", model_parameter_sets[pkey], training_parameter_sets[pkey],
                corr_features, do_validation=not args.skip_validation)
            best_cols = report_validation(f"{group}/{lvl}", v_df, ENS_COLS[group])
            all_valid[(group, lvl)] = v_df
            out = tp_df.copy()
            out["Predicted_Total_ens"] = ensemble_frame(tp_df, best_cols)
            out = out.join(df.loc[tp_df.index, KEYS + (["ID"] if lvl == "row" else [])])
            test_preds[(group, lvl)] = out

    # ---------------------------------------------------------------- final assembly
    print("\nAssembling final predictions ...")
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

    sub_test["Predicted_Total"] = sub_test["Predicted_Total"].fillna(0).clip(lower=0)

    # Submission in SampleSubmission format / order
    pred_col = [c for c in sample.columns if c != "ID"][0]
    sub = sample[["ID"]].merge(sub_test[["ID", "Predicted_Total"]], on="ID", how="left")
    sub = sub.rename(columns={"Predicted_Total": pred_col}).fillna(0)

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"submission_{stamp}.csv")
    sub.to_csv(out_path, index=False)

    print(f"\nSubmission saved to: {out_path}")
    print(sub[pred_col].describe())
    print(sub_test.groupby("Disease")["Predicted_Total"].mean().rename("mean_pred"))
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
