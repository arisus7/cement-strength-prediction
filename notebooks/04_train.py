# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Training: 3 algorithms × 3 targets with MLflow
# MAGIC
# MAGIC Reads `gold_cement_features` and `gold_feature_spec` (no hard-coded feature list) and trains one model per
# MAGIC strength target on the rows where that target exists.
# MAGIC
# MAGIC | Stage | What it does |
# MAGIC |---|---|
# MAGIC | T1 · Screening | Quick grouped CV of 6 algorithms + a mean baseline on every target, to justify the final three |
# MAGIC | T2 · Final grid | Ridge, ExtraTrees and HistGradientBoosting, tuned with a random search under `GroupKFold` on `leakage_group`, evaluated once on the held-out test split |
# MAGIC | T2b · Robustness | Each tuned model re-scored with **repeated grouped CV** on all labelled rows: median and 5–95% range of R², MAE and MAPE, overall and for in-scope types, plus error by ProductType |
# MAGIC | T3 · Champion | Best model per target by **CV MAE** (the test split is never used for selection); registered in Unity Catalog with alias `champion` |
# MAGIC | T4 · Explain | Permutation importance for every model, SHAP values for each champion |
# MAGIC | T5 · Predictor | One multi-output model (`cement_strength_predictor`): raw quality inputs → R1, R7, R28 with difficulty-scaled 90% intervals, R1 ≤ R7 ≤ R28 enforced, and an out-of-scope flag for rare product types |
# MAGIC | T6 · Tables | `ml_model_screening`, `ml_model_comparison`, `ml_metric_ranges`, `ml_metrics_by_type`, `ml_test_predictions`, `ml_feature_importance`, `ml_shap_values` for the dashboard |
# MAGIC
# MAGIC **Why ranges, not one number.** The test split holds about 140 rows per target, and a single split's R² can move by
# MAGIC ±0.13. The headline metrics are the repeated-CV median and 5–95% range; the test split is a final confirmation.
# MAGIC
# MAGIC **Model scope.** Product types with fewer than `min_type_rows` rows in Gold (Types 4 and 5, about 5 rows, with
# MAGIC strengths far below the rest) are grouped as `Other`. They stay in training, but too few rows exist to learn them, so
# MAGIC "in-scope" metrics cover the kept types only and the predictor flags `Other` inputs as out of scope. Including those
# MAGIC rows would inflate overall R² because their strengths sit far from the rest.
# MAGIC
# MAGIC Every run logs the README's required params (`model_type`, `target`, `n_features`, `train_size`, `test_size`),
# MAGIC R² / MAE / MAPE, predicted-vs-actual and residual plots, feature importance, a signature and an input example.
# MAGIC MAPE is always reported next to `skill_vs_mean` (MAE reduction against predicting the train mean), because strength
# MAGIC varies little relative to its size and even the mean scores a low MAPE.

# COMMAND ----------

# MAGIC %pip install -q shap

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "datathon", "Catalog")
dbutils.widgets.text("schema", "gcc", "Schema")
dbutils.widgets.text("experiment", "", "MLflow experiment path (blank = ~/gcc_cement_strength)")
dbutils.widgets.text("n_iter", "10", "Random-search candidates per model")
dbutils.widgets.text("cv_folds", "5", "Grouped CV folds")
dbutils.widgets.text("cv_repeats", "5", "Repeats of grouped CV for metric ranges")
dbutils.widgets.text("seed", "42", "Random seed")
dbutils.widgets.dropdown("register_models", "true", ["true", "false"], "Register models in UC")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
N_ITER = int(dbutils.widgets.get("n_iter"))
CV_FOLDS = int(dbutils.widgets.get("cv_folds"))
CV_REPEATS = int(dbutils.widgets.get("cv_repeats"))
SEED = int(dbutils.widgets.get("seed"))
REGISTER = dbutils.widgets.get("register_models") == "true"

GOLD_TABLE = f"{CATALOG}.{SCHEMA}.gold_cement_features"
SPEC_TABLE = f"{CATALOG}.{SCHEMA}.gold_feature_spec"
PREDICTOR_NAME = f"{CATALOG}.{SCHEMA}.cement_strength_predictor"
EXPERIMENT = dbutils.widgets.get("experiment") or f"/Users/{spark.sql('SELECT current_user()').first()[0]}/gcc_cement_strength"


def ml_table(name):
    return f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

import re
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor

# Databricks turns on autologging by default; it would log every search candidate as its own run.
mlflow.autolog(disable=True)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()
MODEL_NAME_KWARG = "name" if int(mlflow.__version__.split(".")[0]) >= 3 else "artifact_path"
print(f"MLflow {mlflow.__version__} · experiment {EXPERIMENT}")

# COMMAND ----------

# Load Gold and the feature contract
spec = spark.table(SPEC_TABLE).toPandas().sort_values("position")
FEATURES = spec.loc[spec.role == "feature", "column"].tolist()
TARGETS = spec.loc[spec.role == "target", "column"].tolist()

gold = spark.table(GOLD_TABLE).toPandas()
gold[FEATURES] = gold[FEATURES].astype("float64")
train_df = gold[gold.split == "train"].reset_index(drop=True)
test_df = gold[gold.split == "test"].reset_index(drop=True)

# Kept product types (Gold's ProductType_group other than "Other") are the model's scope.
IN_SCOPE_TYPES = sorted(t for t in gold.ProductType.unique() if t in set(gold.ProductType_group) - {"Other"})

print(f"{len(FEATURES)} features: {FEATURES}")
print(f"train {len(train_df)} rows · test {len(test_df)} rows · in-scope types {IN_SCOPE_TYPES}")


def target_data(target):
    tr = train_df[train_df[target].notna()]
    te = test_df[test_df[target].notna()]
    return tr[FEATURES], tr[target], tr["leakage_group"], te[FEATURES], te[target], te


def labelled_data(target):
    """Train and test rows together, for repeated CV with already-tuned hyperparameters."""
    d = gold[gold[target].notna()].reset_index(drop=True)
    return d[FEATURES], d[target], d["leakage_group"], d


def metrics(y_true, y_pred, baseline_mae):
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mae,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_pct": 100 * mean_absolute_percentage_error(y_true, y_pred),
        "skill_vs_mean": 1 - mae / baseline_mae,
    }


CV = GroupKFold(n_splits=CV_FOLDS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## T1 · Screening
# MAGIC Default hyperparameters, grouped CV on the train split only. Logged as one parent run with a child per target × algorithm.

# COMMAND ----------

SCREEN_MODELS = {
    "mean_baseline": DummyRegressor(),
    "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
    "knn": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights="distance")),
    "svr": TransformedTargetRegressor(make_pipeline(StandardScaler(), SVR(C=3.0)), transformer=StandardScaler()),
    "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    "extra_trees": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    "hist_gradient_boosting": HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400, random_state=SEED),
}
SCORING = {"r2": "r2", "mae": "neg_mean_absolute_error", "mape": "neg_mean_absolute_percentage_error"}

screen_rows = []
with mlflow.start_run(run_name="screening"):
    mlflow.set_tag("stage", "screening")
    for target in TARGETS:
        X_tr, y_tr, g_tr, *_ = target_data(target)
        for name, model in SCREEN_MODELS.items():
            cv = cross_validate(model, X_tr, y_tr, groups=g_tr, cv=CV, scoring=SCORING, n_jobs=1)
            row = {
                "target": target,
                "model_type": name,
                "cv_r2": cv["test_r2"].mean(),
                "cv_mae": -cv["test_mae"].mean(),
                "cv_mape_pct": -100 * cv["test_mape"].mean(),
            }
            screen_rows.append(row)
            with mlflow.start_run(run_name=f"screen_{target}_{name}", nested=True):
                mlflow.log_params({"model_type": name, "target": target, "n_features": len(FEATURES),
                                   "train_size": len(y_tr), "cv_folds": CV_FOLDS})
                mlflow.log_metrics({k: v for k, v in row.items() if k.startswith("cv_")})

screening = pd.DataFrame(screen_rows)
display(screening.pivot(index="model_type", columns="target", values="cv_mae").round(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## T2–T4 · Final grid, champion, explanations
# MAGIC Three algorithm families that tell different stories: **Ridge** (linear, interpretable, where VIF matters),
# MAGIC **ExtraTrees** (bagging) and **HistGradientBoosting** (boosting). All ship with scikit-learn, so serverless needs
# MAGIC no extra installs.

# COMMAND ----------

FINAL_MODELS = {
    "ridge": (
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": loguniform(1e-3, 1e3)},
    ),
    "extra_trees": (
        ExtraTreesRegressor(random_state=SEED, n_jobs=-1),
        {
            "n_estimators": [300, 500],
            "max_features": [0.5, 0.7, 1.0],
            "min_samples_leaf": [1, 2, 4],
            "max_depth": [None, 12, 20],
        },
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingRegressor(random_state=SEED),
        {
            "learning_rate": loguniform(0.02, 0.2),
            "max_iter": [200, 400, 800],
            "max_leaf_nodes": [7, 15, 31],
            "min_samples_leaf": [5, 10, 20],
            "l2_regularization": loguniform(1e-3, 10),
            "loss": ["squared_error", "absolute_error"],
        },
    ),
}


def plot_pred_vs_actual(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, s=14, alpha=0.6)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set(xlabel="Actual (psi)", ylabel="Predicted (psi)", title=title)
    fig.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title):
    resid = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, resid, s=14, alpha=0.6)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="Predicted (psi)", ylabel="Residual (psi)", title=title)
    axes[1].hist(resid, bins=30)
    axes[1].set(xlabel="Residual (psi)", title="Residual distribution")
    fig.tight_layout()
    return fig


def plot_importance(imp, title):
    imp = imp.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(imp) + 1))
    ax.barh(imp.feature, imp.importance_mean, xerr=imp.importance_std)
    ax.set(xlabel="MAE increase when shuffled (psi)", title=title)
    fig.tight_layout()
    return fig


def log_fig(fig, path):
    mlflow.log_figure(fig, path)
    plt.close(fig)


def registered_version(model_info, name):
    version = getattr(model_info, "registered_model_version", None)
    if version is None:
        versions = client.search_model_versions(f"name='{name}'")
        version = max(int(v.version) for v in versions if v.run_id == model_info.run_id)
    return str(version)


def fold_metrics(y_true, y_pred):
    return {"r2": r2_score(y_true, y_pred), "mae": mean_absolute_error(y_true, y_pred),
            "mape_pct": 100 * mean_absolute_percentage_error(y_true, y_pred)}


def repeated_group_cv(model, X, y, groups, in_scope):
    """CV_REPEATS × CV_FOLDS grouped CV with a fresh group shuffle per repeat.

    Returns per-fold metrics (overall and in-scope rows) and the out-of-fold predictions of every repeat.
    """
    rng = np.random.default_rng(SEED)
    unique_groups = groups.unique()
    fold_rows, oof = [], np.empty((CV_REPEATS, len(y)))
    for r in range(CV_REPEATS):
        fold_of = pd.Series(rng.permutation(len(unique_groups)) % CV_FOLDS, index=unique_groups)
        fold = groups.map(fold_of).to_numpy()
        for k in range(CV_FOLDS):
            te = fold == k
            oof[r, te] = clone(model).fit(X[~te], y[~te]).predict(X[te])
            for scope, mask in (("all", np.ones(len(y), bool)), ("in_scope", in_scope)):
                sel = te & mask
                if sel.sum() >= 10:
                    fold_rows.append({"repeat": r, "fold": k, "scope": scope, **fold_metrics(y[sel], oof[r, sel])})
    return pd.DataFrame(fold_rows), oof


def summarise_ranges(folds):
    out = folds.groupby("scope")[["r2", "mae", "mape_pct"]].quantile([0.05, 0.5, 0.95]).unstack()
    out.columns = [f"{m}_{ {0.05: 'p05', 0.5: 'median', 0.95: 'p95'}[q]}" for m, q in out.columns]
    return out


def metrics_by_type(y, oof, product_types):
    """Error per ProductType across all repeats' out-of-fold predictions."""
    rows = []
    for ptype in sorted(product_types.unique()):
        sel = (product_types == ptype).to_numpy()
        yt, pt = y[sel].to_numpy(), oof[:, sel]
        err = yt - pt
        rows.append({
            "ProductType": ptype, "n_rows": int(sel.sum()), "in_scope": ptype in IN_SCOPE_TYPES,
            "mae": float(np.abs(err).mean()), "mape_pct": float(100 * (np.abs(err) / yt).mean()),
            "bias": float(err.mean()), "mean_strength": float(yt.mean()),
            "r2_median": float(np.median([r2_score(yt, p) for p in pt])) if sel.sum() >= 10 else None,
        })
    return pd.DataFrame(rows)


def difficulty(model, X):
    """Per-row spread across a bagging ensemble's trees; 1.0 for models without one (constant-width interval)."""
    trees = getattr(model, "estimators_", None)
    if trees is None or not hasattr(trees[0], "predict"):
        return np.ones(len(X))
    values = X.to_numpy() if hasattr(X, "to_numpy") else X
    return np.stack([t.predict(values) for t in trees]).std(axis=0)


def conformal_calibration(model, X, y, groups, level=0.9):
    """Normalised split-conformal on out-of-fold train residuals: interval = q × difficulty, floored."""
    scores, spreads = np.empty(len(y)), np.empty(len(y))
    for tr, te in CV.split(X, y, groups):
        fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
        spreads[te] = difficulty(fitted, X.iloc[te])
        scores[te] = np.abs(y.iloc[te].to_numpy() - fitted.predict(X.iloc[te]))
    floor = float(max(np.percentile(spreads, 5), 1e-6))
    ratios = np.sort(scores / np.maximum(spreads, floor))
    k = min(len(ratios) - 1, int(np.ceil((len(ratios) + 1) * level)) - 1)
    return {"q": float(ratios[k]), "floor": floor}


def interval_halfwidth(model, X, calibration):
    return calibration["q"] * np.maximum(difficulty(model, X), calibration["floor"])

# COMMAND ----------

comparison_rows, prediction_frames, importance_frames, range_frames, type_frames = [], [], [], [], []
champions = {}

for target in TARGETS:
    X_tr, y_tr, g_tr, X_te, y_te, te_rows = target_data(target)
    X_all, y_all, g_all, all_rows = labelled_data(target)
    in_scope_all = all_rows.ProductType.isin(IN_SCOPE_TYPES).to_numpy()
    baseline_mae = mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean()))
    uc_name = f"{CATALOG}.{SCHEMA}.{target.lower()}"
    target_results = {}

    with mlflow.start_run(run_name=f"final_{target}") as parent:
        mlflow.set_tags({"stage": "final", "target": target})
        mlflow.log_metric("test_mae_mean_baseline", baseline_mae)

        for name, (estimator, space) in FINAL_MODELS.items():
            search = RandomizedSearchCV(
                estimator, space, n_iter=N_ITER, cv=CV, scoring="neg_mean_absolute_error",
                random_state=SEED, n_jobs=1, refit=True,
            )
            search.fit(X_tr, y_tr, groups=g_tr)
            model = search.best_estimator_
            y_pred = model.predict(X_te)
            m = metrics(y_te, y_pred, baseline_mae)

            # T2b: repeated grouped CV with the tuned hyperparameters, on train + test rows
            folds, oof = repeated_group_cv(model, X_all, y_all, g_all, in_scope_all)
            ranges = summarise_ranges(folds)
            by_type = metrics_by_type(y_all, oof, all_rows.ProductType)
            range_frames.append(ranges.reset_index().assign(target=target, model_type=name))
            type_frames.append(by_type.assign(target=target, model_type=name))

            with mlflow.start_run(run_name=f"{target}_{name}", nested=True) as run:
                mlflow.set_tags({"stage": "final", "target": target, "model_type": name})
                mlflow.log_params({
                    "model_type": name, "target": target, "n_features": len(FEATURES),
                    "train_size": len(y_tr), "test_size": len(y_te), "cv_folds": CV_FOLDS,
                    "n_iter": N_ITER, "seed": SEED, "cv_repeats": CV_REPEATS,
                    "in_scope_types": ",".join(IN_SCOPE_TYPES),
                    **{f"best_{k}": v for k, v in search.best_params_.items()},
                })
                mlflow.log_metrics({"cv_mae": -search.best_score_, **{f"test_{k}": v for k, v in m.items()}})
                mlflow.log_metrics({f"rcv_{scope}_{col}": float(v) for scope, row in ranges.iterrows()
                                    for col, v in row.items()})
                mlflow.log_table(folds, "repeated_cv_folds.json")
                mlflow.log_table(by_type, "metrics_by_product_type.json")

                log_fig(plot_pred_vs_actual(y_te, y_pred, f"{target} · {name}"), "plots/pred_vs_actual.png")
                log_fig(plot_residuals(y_te, y_pred, f"{target} · {name}"), "plots/residuals.png")

                pi = permutation_importance(model, X_te, y_te, scoring="neg_mean_absolute_error",
                                            n_repeats=10, random_state=SEED, n_jobs=1)
                imp = pd.DataFrame({"feature": FEATURES, "importance_mean": pi.importances_mean,
                                    "importance_std": pi.importances_std})
                log_fig(plot_importance(imp, f"{target} · {name}"), "plots/permutation_importance.png")
                mlflow.log_table(imp, "feature_importance.json")

                model_info = mlflow.sklearn.log_model(
                    model,
                    **{MODEL_NAME_KWARG: "model"},
                    # Newer MLflow defaults to skops, which rejects tree models; cloudpickle works on every version.
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                    signature=infer_signature(X_tr, model.predict(X_tr)),
                    input_example=X_tr.head(5),
                    registered_model_name=uc_name if REGISTER else None,
                )
                version = registered_version(model_info, uc_name) if REGISTER else None
                if version:
                    client.set_registered_model_alias(uc_name, name, version)

            target_results[name] = {"model": model, "cv_mae": -search.best_score_, "version": version}
            comparison_rows.append({"target": target, "model_type": name, "cv_mae": -search.best_score_,
                                    **{f"test_{k}": v for k, v in m.items()},
                                    **{f"rcv_in_scope_{c}": float(v) for c, v in ranges.loc["in_scope"].items()},
                                    "run_id": run.info.run_id, "model_version": version or ""})
            prediction_frames.append(pd.DataFrame({
                "sample_id": te_rows.sample_id.values, "ProductType": te_rows.ProductType.values,
                "target": target, "model_type": name, "y_true": y_te.values, "y_pred": y_pred,
            }))
            importance_frames.append(imp.assign(target=target, model_type=name))

        # T3: champion by CV MAE, never by test score
        best = min(target_results, key=lambda k: target_results[k]["cv_mae"])
        champ = target_results[best]
        calibration = conformal_calibration(champ["model"], X_tr, y_tr, g_tr)
        halfwidths = interval_halfwidth(champ["model"], X_te, calibration)
        champ_pred = champ["model"].predict(X_te)
        coverage = float(np.mean(np.abs(y_te.values - champ_pred) <= halfwidths))
        champions[target] = {"model_type": best, "model": champ["model"], "calibration": calibration}

        mlflow.set_tag("champion", best)
        mlflow.log_metrics({"champion_interval90_median_halfwidth": float(np.median(halfwidths)),
                            "champion_interval90_test_coverage": coverage})
        if REGISTER and champ["version"]:
            client.set_registered_model_alias(uc_name, "champion", champ["version"])
        print(f"{target}: champion {best} (CV MAE {champ['cv_mae']:.0f}), 90% interval median ±{np.median(halfwidths):.0f} "
              f"psi (range {halfwidths.min():.0f}–{halfwidths.max():.0f}), test coverage {coverage:.0%}")

comparison = pd.DataFrame(comparison_rows)
comparison["is_champion"] = comparison.apply(lambda r: champions[r.target]["model_type"] == r.model_type, axis=1)
metric_ranges = pd.concat(range_frames, ignore_index=True)
by_type_all = pd.concat(type_frames, ignore_index=True)
display(comparison.drop(columns=["run_id"]).round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Headline metrics: repeated grouped CV, in-scope types (median [5%, 95%])

# COMMAND ----------

headline = metric_ranges[metric_ranges.scope == "in_scope"].copy()
for metric in ["r2", "mae", "mape_pct"]:
    fmt = "{:.2f}" if metric == "r2" else "{:.0f}" if metric == "mae" else "{:.1f}"
    headline[metric] = headline.apply(
        lambda r: f"{fmt.format(r[metric + '_median'])} [{fmt.format(r[metric + '_p05'])}, {fmt.format(r[metric + '_p95'])}]",
        axis=1)
display(headline.pivot(index="model_type", columns="target", values=["r2", "mae", "mape_pct"]))
display(by_type_all[by_type_all.model_type.isin([c["model_type"] for c in champions.values()])]
        .merge(pd.DataFrame([{"target": t, "model_type": c["model_type"]} for t, c in champions.items()]))
        .round(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ### SHAP values for the champions (test split)
# MAGIC Answers the README question *"which quality variables drove this prediction?"* per sample. Skipped with a message if
# MAGIC `shap` is unavailable, so the rest of the notebook still runs end to end.

# COMMAND ----------

shap_frames = []
try:
    import shap

    for target, champ in champions.items():
        X_tr, _, _, X_te, _, te_rows = target_data(target)
        try:
            explainer = shap.TreeExplainer(champ["model"])  # exact and fast for tree ensembles
        except Exception:
            explainer = shap.Explainer(champ["model"].predict, shap.maskers.Independent(X_tr, max_samples=50),
                                       seed=SEED)
        sv = explainer(X_te)
        long = pd.DataFrame(sv.values, columns=FEATURES).assign(sample_id=te_rows.sample_id.values)
        long = long.melt(id_vars="sample_id", var_name="feature", value_name="shap_value")
        values = X_te.assign(sample_id=te_rows.sample_id.values).melt(
            id_vars="sample_id", var_name="feature", value_name="feature_value")
        shap_frames.append(long.merge(values, on=["sample_id", "feature"]).assign(
            target=target, model_type=champ["model_type"], base_value=float(np.mean(sv.base_values))))

        with mlflow.start_run(run_name=f"shap_{target}"):
            mlflow.set_tags({"stage": "explain", "target": target, "model_type": champ["model_type"]})
            shap.plots.beeswarm(sv, show=False, max_display=len(FEATURES))
            log_fig(plt.gcf(), "plots/shap_beeswarm.png")
except ImportError:
    print("shap is not installed: skipping SHAP values (permutation importance is still logged).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T5 · Multi-output predictor
# MAGIC One registered model takes **raw quality inputs** (ProductType + the measured variables, nulls allowed) and applies
# MAGIC Gold's transformations itself using `gold_feature_spec`: train-median imputation, missing flags and ProductType
# MAGIC dummies. It returns R1, R7 and R28 with 90% intervals and enforces R1 ≤ R7 ≤ R28. The dashboard simulator calls it.
# MAGIC
# MAGIC Intervals are **difficulty-scaled**: for a tree-ensemble champion the width follows the spread of its trees on that
# MAGIC batch (narrow for typical batches, wide for unusual ones), calibrated on out-of-fold train residuals to 90%.
# MAGIC Inputs whose ProductType is outside the model's scope get `in_scope = false` and a `scope_note`.

# COMMAND ----------

def product_type_col(product_type):
    return "pt_" + re.sub(r"\W+", "_", str(product_type).strip().lower())


class CementStrengthPredictor(mlflow.pyfunc.PythonModel):
    def __init__(self, champions, spec, features, targets, in_scope_types):
        self.models = {t: c["model"] for t, c in champions.items()}
        self.calibrations = {t: c["calibration"] for t, c in champions.items()}
        self.spec = spec[["column", "role", "kind", "impute_value"]].to_dict("records")
        self.features = features
        self.targets = targets
        self.in_scope_types = list(in_scope_types)

    def _features(self, raw):
        X = pd.DataFrame(index=raw.index)
        for col in self.spec:
            name, kind = col["column"], col["kind"]
            if col["role"] != "feature":
                continue
            if kind == "numeric":
                values = pd.to_numeric(raw[name], errors="coerce") if name in raw else pd.Series(np.nan, index=raw.index)
                X[name] = values.fillna(col["impute_value"])
            elif kind == "flag" and name == "Physical_Metric_A_measured":
                X[name] = pd.to_numeric(raw.get("Physical_Metric_A"), errors="coerce").notna().astype(float) \
                    if "Physical_Metric_A" in raw else 0.0
            elif kind == "flag" and name.endswith("_missing"):
                base = name[: -len("_missing")]
                X[name] = pd.to_numeric(raw[base], errors="coerce").isna().astype(float) if base in raw else 1.0
            elif kind == "product_type":
                X[name] = (raw["ProductType"].map(product_type_col) == name).astype(float)
        return X[self.features]

    def predict(self, context, model_input, params=None):
        X = self._features(model_input)
        out = pd.DataFrame(index=model_input.index)
        for t in self.targets:
            out[t] = self.models[t].predict(X)
        # Strength cannot decrease with curing age
        out[self.targets] = np.maximum.accumulate(out[self.targets].to_numpy(), axis=1)
        for t in self.targets:
            half = interval_halfwidth(self.models[t], X, self.calibrations[t])
            out[f"{t}_lo"] = out[t] - half
            out[f"{t}_hi"] = out[t] + half
        out["in_scope"] = model_input["ProductType"].isin(self.in_scope_types).to_numpy()
        out["scope_note"] = np.where(
            out["in_scope"], "",
            f"ProductType outside the model's scope ({', '.join(self.in_scope_types)}): too few training rows, "
            "treat the prediction as unreliable",
        )
        return out


RAW_INPUTS = ["ProductType"] + spec.loc[spec.kind == "numeric", "column"].tolist()
if "Physical_Metric_A" not in RAW_INPUTS:
    RAW_INPUTS.append("Physical_Metric_A")
raw_example = test_df[RAW_INPUTS].head(5).copy()
raw_example[RAW_INPUTS[1:]] = raw_example[RAW_INPUTS[1:]].astype("float64")
raw_example.loc[raw_example.index[0], "Physical_Metric_A"] = np.nan  # show that nulls are accepted
out_of_scope = sorted(set(gold.ProductType) - set(IN_SCOPE_TYPES))
if out_of_scope:
    raw_example.loc[raw_example.index[-1], "ProductType"] = out_of_scope[0]  # show the out-of-scope flag

predictor = CementStrengthPredictor(champions, spec, FEATURES, TARGETS, IN_SCOPE_TYPES)
example_out = predictor.predict(None, raw_example)

with mlflow.start_run(run_name="cement_strength_predictor"):
    mlflow.set_tags({"stage": "predictor", **{f"champion_{t}": c["model_type"] for t, c in champions.items()}})
    mlflow.log_params({"targets": ",".join(TARGETS), "n_features": len(FEATURES),
                       "in_scope_types": ",".join(IN_SCOPE_TYPES),
                       **{f"interval90_q_{t}": round(c["calibration"]["q"], 3) for t, c in champions.items()}})
    predictor_info = mlflow.pyfunc.log_model(
        python_model=predictor,
        **{MODEL_NAME_KWARG: "predictor"},
        signature=infer_signature(raw_example, example_out),
        input_example=raw_example,
        registered_model_name=PREDICTOR_NAME if REGISTER else None,
    )
    if REGISTER:
        client.set_registered_model_alias(PREDICTOR_NAME, "champion", registered_version(predictor_info, PREDICTOR_NAME))

# Round-trip check: the logged model must reproduce the in-memory predictions
loaded = mlflow.pyfunc.load_model(predictor_info.model_uri)
assert np.allclose(loaded.predict(raw_example)[TARGETS].to_numpy(), example_out[TARGETS].to_numpy())
display(pd.concat([raw_example.reset_index(drop=True), example_out.reset_index(drop=True)], axis=1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## T6 · Tables for the dashboard

# COMMAND ----------

def save(pdf, name, comment):
    (
        spark.createDataFrame(pdf)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", True)
        .saveAsTable(ml_table(name))
    )
    spark.sql(f"COMMENT ON TABLE {ml_table(name)} IS '{comment}'")


predictions = pd.concat(prediction_frames, ignore_index=True)
predictions["residual"] = predictions.y_true - predictions.y_pred
predictions["is_champion"] = predictions.apply(lambda r: champions[r.target]["model_type"] == r.model_type, axis=1)

save(screening, "ml_model_screening", "T1 screening: grouped CV metrics on train for 7 algorithms x 3 targets")
save(comparison, "ml_model_comparison", "T2 final grid: CV MAE and test R2/MAE/MAPE for 3 algorithms x 3 targets")
save(metric_ranges, "ml_metric_ranges",
     "T2b repeated grouped CV: median and 5-95% range of R2/MAE/MAPE per target, model and scope")
save(by_type_all, "ml_metrics_by_type", "T2b repeated grouped CV error by ProductType; in_scope marks the model scope")
save(predictions, "ml_test_predictions", "Test-split predictions of every final model, psi")
save(pd.concat(importance_frames, ignore_index=True), "ml_feature_importance",
     "Permutation importance on the test split (MAE increase in psi)")
if shap_frames:
    save(pd.concat(shap_frames, ignore_index=True), "ml_shap_values", "SHAP values of each champion on the test split")

print(f"Experiment: {EXPERIMENT}")
print("Champions:", {t: c["model_type"] for t, c in champions.items()})
