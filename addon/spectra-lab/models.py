"""
models.py — scikit-learn model registry, cross-validated tuning,
external-set evaluation, and persistence for the soil-spectra app.

Every model is wrapped in an sklearn Pipeline so preprocessing (optional SNV,
optional autoscaling) travels with the estimator. That makes the saved joblib
artifact fully self-contained: load it and call .predict on raw band values.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

from utils import snv, regression_metrics


# --------------------------------------------------------------------------- #
#  SNV as a scikit-learn transformer (so it lives inside the pipeline)
# --------------------------------------------------------------------------- #
class SNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate scatter correction; stateless per-row transform."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return snv(np.asarray(X, dtype=float))


class PLSWrapper(BaseEstimator):
    """PLSRegression that exposes a 1-D predict (sklearn returns 2-D for PLS).

    Fitted state is stored with trailing-underscore names so sklearn's
    check_is_fitted recognises the estimator as trained after .fit().
    """

    def __init__(self, n_components=2, scale=True):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, y):
        self.pls_ = PLSRegression(n_components=self.n_components, scale=self.scale)
        self.pls_.fit(X, y)
        return self

    def predict(self, X):
        return self.pls_.predict(X).ravel()


class PCRegressor(BaseEstimator):
    """Principal Component Regression: PCA compression then OLS on the scores."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        self.pca_ = PCA(n_components=self.n_components)
        T = self.pca_.fit_transform(X)
        self.lr_ = LinearRegression().fit(T, y)
        return self

    def predict(self, X):
        return self.lr_.predict(self.pca_.transform(X))


# --------------------------------------------------------------------------- #
#  Model registry
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    key: str
    name: str
    desc: str
    estimator: object                      # the core regressor (post-preprocessing)
    param_grid: dict = field(default_factory=dict)
    # which preprocessing this model needs:
    #   'optional' -> caller decides autoscaling (PLS/PCR)
    #   'required' -> always autoscale (distance/kernel/penalty methods)
    scaling: str = "required"


def default_registry(cfg: dict) -> dict[str, ModelSpec]:
    """Build the model registry given a config dict of hyperparameter ranges.

    cfg keys (all optional, sensible defaults applied):
      pls_max, pcr_max, ridge_alphas, knn_ks, svr_C, svr_gamma,
      rf_trees, rf_depth, pls_autoscale
    """
    pls_max = int(cfg.get("pls_max", 10))
    pcr_max = int(cfg.get("pcr_max", 10))
    ridge_alphas = list(cfg.get("ridge_alphas", [0.01, 0.1, 1, 10, 100]))
    knn_ks = list(cfg.get("knn_ks", [3, 5, 7, 9, 15]))
    svr_C = float(cfg.get("svr_C", 10.0))
    svr_gamma = cfg.get("svr_gamma", "scale")
    rf_trees = int(cfg.get("rf_trees", 200))
    rf_depth = cfg.get("rf_depth", None)
    pls_autoscale = bool(cfg.get("pls_autoscale", True))

    reg: dict[str, ModelSpec] = {
        "pls": ModelSpec(
            "pls", "PLSR", "Latent-variable regression for collinear spectra.",
            PLSWrapper(scale=pls_autoscale),
            {"model__n_components": list(range(1, pls_max + 1))},
            scaling="optional",
        ),
        "pcr": ModelSpec(
            "pcr", "PCR", "PCA compression then OLS on the scores.",
            PCRegressor(),
            {"model__n_components": list(range(1, pcr_max + 1))},
            scaling="optional",
        ),
        "ridge": ModelSpec(
            "ridge", "Ridge", "L2-penalised linear regression.",
            Ridge(),
            {"model__alpha": ridge_alphas},
            scaling="required",
        ),
        "knn": ModelSpec(
            "knn", "k-NN", "Distance-weighted nearest-neighbour average.",
            KNeighborsRegressor(weights="distance"),
            {"model__n_neighbors": knn_ks},
            scaling="required",
        ),
        "svr": ModelSpec(
            "svr", "SVR (RBF)", "Kernel epsilon-insensitive regression.",
            SVR(kernel="rbf", C=svr_C, gamma=svr_gamma),
            {"model__C": [svr_C], "model__gamma": [svr_gamma]},
            scaling="required",
        ),
        "rf": ModelSpec(
            "rf", "Random Forest", "Bagged regression trees, feature subsampling.",
            RandomForestRegressor(
                n_estimators=rf_trees, max_depth=rf_depth,
                max_features="sqrt", random_state=42, n_jobs=-1,
            ),
            {},  # tree count / depth taken as given (tune via cfg, not grid)
            scaling="required",
        ),
    }
    return reg


# --------------------------------------------------------------------------- #
#  Pipeline assembly
# --------------------------------------------------------------------------- #
def build_pipeline(spec: ModelSpec, use_snv: bool, autoscale_optional: bool) -> Pipeline:
    """Wrap a model spec into a preprocessing + estimator pipeline.

    - SNV is prepended when use_snv is True.
    - StandardScaler is added when scaling is 'required', or when it's
      'optional' and the caller asked for autoscaling.
    """
    steps = []
    if use_snv:
        steps.append(("snv", SNV()))

    if spec.scaling == "required":
        steps.append(("scale", StandardScaler()))
    elif spec.scaling == "optional" and autoscale_optional:
        # PLS/PCR: centre only is handled by the estimator; add scaler for
        # unit-variance bands when requested.
        steps.append(("scale", StandardScaler()))
    else:
        # centre-only: PLS/PCR still need mean-centred X
        steps.append(("center", StandardScaler(with_std=False)))

    steps.append(("model", spec.estimator))
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
#  Tuning + evaluation
# --------------------------------------------------------------------------- #
def tune_and_evaluate(
    spec: ModelSpec,
    X_cal, y_cal,
    X_val, y_val,
    use_snv: bool,
    autoscale_optional: bool,
    folds: int,
):
    """Grid-search hyperparameters by CV on the calibration set, refit, and
    evaluate on the external set if one is provided.

    Returns a result dict with cv/ext metrics, OOF + external predictions,
    the best hyperparameters, and the fitted pipeline.
    """
    pipe = build_pipeline(spec, use_snv, autoscale_optional)
    folds = max(2, min(folds, len(y_cal)))
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)

    if spec.param_grid:
        gs = GridSearchCV(
            pipe, spec.param_grid, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        gs.fit(X_cal, y_cal)
        best_pipe = gs.best_estimator_
        best_params = gs.best_params_
    else:
        best_pipe = pipe.fit(X_cal, y_cal)
        best_params = {}

    # honest CV predictions at the chosen hyperparameters
    oof = cross_val_predict(best_pipe, X_cal, y_cal, cv=cv, n_jobs=-1)
    cv_metrics = regression_metrics(y_cal, oof)

    ext_metrics, ext_pred = None, None
    if X_val is not None and len(X_val) > 0:
        ext_pred = best_pipe.predict(X_val)
        ext_metrics = regression_metrics(y_val, ext_pred)

    return {
        "key": spec.key,
        "name": spec.name,
        "best_params": best_params,
        "tag": _format_tag(spec, best_params, autoscale_optional),
        "cv": cv_metrics,
        "cv_obs": np.asarray(y_cal, dtype=float),
        "cv_pred": np.asarray(oof, dtype=float),
        "ext": ext_metrics,
        "ext_obs": None if X_val is None or len(X_val) == 0 else np.asarray(y_val, float),
        "ext_pred": None if ext_pred is None else np.asarray(ext_pred, float),
        "pipeline": best_pipe,
    }


def _format_tag(spec, params, autoscale_optional):
    """Short human-readable summary of the tuned hyperparameters."""
    if spec.key in ("pls", "pcr"):
        a = params.get("model__n_components", "?")
        unit = "LV" if spec.key == "pls" else "PC"
        scl = "scaled" if autoscale_optional else "centered"
        return f"{a} {unit} · {scl}"
    if spec.key == "ridge":
        return f"alpha={params.get('model__alpha', '?')}"
    if spec.key == "knn":
        return f"k={params.get('model__n_neighbors', '?')}"
    if spec.key == "svr":
        return f"C={params.get('model__C', '?')}, gamma={params.get('model__gamma', '?')}"
    if spec.key == "rf":
        m = spec.estimator
        return f"{m.n_estimators} trees · depth {m.max_depth}"
    return ""


def rank_models(results: list[dict]) -> list[dict]:
    """Sort model results best-first by external RPD when available, else CV RPD."""
    def score(r):
        return r["ext"]["rpd"] if r["ext"] else r["cv"]["rpd"]
    return sorted(results, key=score, reverse=True)


# --------------------------------------------------------------------------- #
#  Persistence
# --------------------------------------------------------------------------- #
def export_bundle(results_by_target: dict, bands, meta: dict) -> dict:
    """Assemble a serialisable bundle of the best pipeline per target plus all
    candidates, ready for joblib.dump.
    """
    payload = {"bands": list(bands), "meta": dict(meta), "targets": {}}
    for target, ranked in results_by_target.items():
        payload["targets"][target] = {
            "best_key": ranked[0]["key"],
            "models": {
                r["key"]: {
                    "pipeline": r["pipeline"],
                    "tag": r["tag"],
                    "cv": r["cv"],
                    "ext": r["ext"],
                }
                for r in ranked
            },
        }
    return payload


def predict_with_bundle(payload: dict, X, target: str, model_key: str | None = None):
    """Predict a target from raw band array X using a loaded bundle.

    model_key=None uses the best model for that target.
    """
    tinfo = payload["targets"][target]
    key = model_key or tinfo["best_key"]
    pipe = tinfo["models"][key]["pipeline"]
    return pipe.predict(np.asarray(X, dtype=float))
