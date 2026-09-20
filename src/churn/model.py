"""Oversampling, model pipelines, cross-validation, voting ensemble and persistence."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import basic_preprocess
from churn.features import build_features, prepare_design_matrix
from churn.labels import build_labels_and_cutoff, last_event_cutoff

VOTING = "LR + KNN voting"
BASE_MODELS: tuple[str, ...] = ("Logistic Regression", "Extra Trees", "KNN")
MODEL_NAMES: tuple[str, ...] = (VOTING, *BASE_MODELS)  # voting = the submitted model
VOTING_MEMBERS: tuple[str, ...] = ("Logistic Regression", "KNN")
VOTE_WEIGHT_GRID: tuple[int, ...] = (1, 2, 3, 4)
RANDOM_STATE = 42  # StratifiedKFold seed

# Oversampling seeds copied from the notebook: (per-fold base -> base + fold, final fit)
SEEDS: dict[str, tuple[int, int]] = {
    "Logistic Regression": (42, 2025),
    "Extra Trees": (100, 303),
    "KNN": (200, 404),
}


def oversample_dataframe(
    X: pd.DataFrame, y: pd.Series | np.ndarray, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, np.ndarray]:
    """Replicate the minority class (label 1) ``int(n_major / n_minor)`` times, then shuffle.

    Row order and shuffling reproduce the notebook exactly
    (majority rows, then minority copies, then ``DataFrame.sample(frac=1)``).
    """
    y_arr = np.asarray(y)
    if len(X) != len(y_arr):
        raise ValueError("X and y must have the same length.")
    major = X[y_arr == 0]
    minor = X[y_arr == 1]
    if len(minor) == 0:
        raise ValueError("No positive samples, cannot oversample.")
    ratio = max(1, len(major) // len(minor))

    X_all = pd.concat([major] + [minor] * ratio, ignore_index=True)
    y_all = np.concatenate(
        [np.zeros(len(major), dtype=int), np.ones(len(minor) * ratio, dtype=int)]
    )
    order = X_all.sample(frac=1.0, random_state=random_state).index.to_numpy()
    return X_all.iloc[order].reset_index(drop=True), y_all[order]


def build_pipeline(model_name: str, cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    """Create the preprocessing + classifier pipeline used in the notebook."""
    if model_name not in BASE_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {BASE_MODELS}.")
    dense = model_name == "KNN"
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense), cat_cols),
        ]
    )
    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000, solver="liblinear")
    elif model_name == "Extra Trees":
        clf = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        clf = KNeighborsClassifier(n_neighbors=50, weights="distance", p=2, n_jobs=-1)
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def _fit(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    clf = pipe.named_steps["clf"]
    if isinstance(clf, KNeighborsClassifier):
        clf.set_params(n_neighbors=min(50, len(X)))
    return pipe.fit(X, y)


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cat_cols: list[str],
    num_cols: list[str],
    n_splits: int = 5,
    oversample: bool = True,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Stratified K-fold CV with oversampling inside each training fold only."""
    y_arr = np.asarray(y)
    seed_base = SEEDS[model_name][0]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X))
    folds: list[np.ndarray] = []
    for fold, (trn, val) in enumerate(skf.split(X, y_arr), start=1):
        X_trn, y_trn = X.iloc[trn], y_arr[trn]
        if oversample:
            X_trn, y_trn = oversample_dataframe(X_trn, y_trn, random_state=seed_base + fold)
        pipe = _fit(build_pipeline(model_name, cat_cols, num_cols), X_trn, y_trn)
        oof[val] = pipe.predict_proba(X.iloc[val])[:, 1]
        folds.append(val)
    return _cv_result(y_arr, oof, folds)


def _cv_result(y: np.ndarray, oof: np.ndarray, folds: list[np.ndarray]) -> dict:
    return {
        "fold_aucs": [float(roc_auc_score(y[val], oof[val])) for val in folds],
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof": oof,
        "folds": folds,
    }


# ------------------------------------------------------------------ voting
def weighted_average(probas: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights[name] for name in probas)
    return sum(weights[name] * p for name, p in probas.items()) / total


def search_vote_weights(
    oofs: dict[str, np.ndarray], y: np.ndarray, grid: tuple[int, ...] = VOTE_WEIGHT_GRID
) -> tuple[dict[str, int], float]:
    """Grid-search integer weights maximising OOF ROC AUC (first best kept, as in the notebook)."""
    names = list(oofs)
    best_weights: dict[str, int] = {}
    best_auc = -1.0
    for combo in itertools.product(grid, repeat=len(names)):
        weights = dict(zip(names, combo, strict=True))
        auc = roc_auc_score(y, weighted_average(oofs, weights))
        if auc > best_auc:
            best_auc, best_weights = float(auc), weights
    return best_weights, best_auc


@dataclass
class WeightedVote:
    """Soft-voting ensemble of fitted pipelines with fixed weights."""

    members: dict[str, Pipeline]
    weights: dict[str, float]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = {name: pipe.predict_proba(X)[:, 1] for name, pipe in self.members.items()}
        p1 = weighted_average(probas, self.weights)
        return np.column_stack([1 - p1, p1])


# ------------------------------------------------------------------ bundle
@dataclass
class ChurnModel:
    """Everything needed to score new logs: fitted estimator + feature contract."""

    model_name: str
    pipeline: Pipeline | WeightedVote
    pages_ref: list[str]
    columns: list[str]
    metrics: dict = field(default_factory=dict)

    def predict_proba_from_logs(
        self, raw_logs: pd.DataFrame, preprocessed: bool = False
    ) -> pd.Series:
        df = raw_logs if preprocessed else basic_preprocess(raw_logs)
        cutoff = last_event_cutoff(df)
        feats, _ = build_features(df, cutoff, pages_ref=self.pages_ref)
        X, _, _ = prepare_design_matrix(feats, columns=self.columns)
        return pd.Series(self.pipeline.predict_proba(X)[:, 1], index=X.index, name="churn_proba")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> ChurnModel:
        return joblib.load(path)


def _fit_final(
    model_name: str, X: pd.DataFrame, y: pd.Series, oversample: bool, cat: list, num: list
) -> Pipeline:
    if oversample:
        X_fit, y_fit = oversample_dataframe(X, y, random_state=SEEDS[model_name][1])
    else:
        X_fit, y_fit = X, y.to_numpy()
    return _fit(build_pipeline(model_name, cat, num), X_fit, y_fit)


def train_from_logs(
    raw_logs: pd.DataFrame,
    model_name: str = VOTING,
    n_splits: int = 5,
    oversample: bool = True,
    random_state: int = RANDOM_STATE,
    preprocessed: bool = False,
) -> ChurnModel:
    """Full training pipeline: preprocess -> labels -> features -> CV -> final fit.

    Pass ``preprocessed=True`` when ``raw_logs`` already went through ``basic_preprocess``.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {MODEL_NAMES}.")
    df = raw_logs if preprocessed else basic_preprocess(raw_logs)
    y, cutoff = build_labels_and_cutoff(df)
    feats, pages_ref = build_features(df, cutoff)
    X, cat, num = prepare_design_matrix(feats)
    y = y.loc[X.index]
    y_arr = y.to_numpy()

    if model_name == VOTING:
        member_cv = {
            name: cross_validate(X, y, name, cat, num, n_splits, oversample, random_state)
            for name in VOTING_MEMBERS
        }
        oofs = {name: cv["oof"] for name, cv in member_cv.items()}
        weights, _ = search_vote_weights(oofs, y_arr)
        cv = _cv_result(
            y_arr, weighted_average(oofs, weights), member_cv[VOTING_MEMBERS[0]]["folds"]
        )
        estimator: Pipeline | WeightedVote = WeightedVote(
            {name: _fit_final(name, X, y, oversample, cat, num) for name in VOTING_MEMBERS},
            weights,
        )
        extra = {
            "weights": weights,
            "member_oof_aucs": {name: c["oof_auc"] for name, c in member_cv.items()},
        }
    else:
        cv = cross_validate(X, y, model_name, cat, num, n_splits, oversample, random_state)
        estimator = _fit_final(model_name, X, y, oversample, cat, num)
        extra = {}

    metrics = {
        "fold_aucs": cv["fold_aucs"],
        "oof_auc": cv["oof_auc"],
        "n_users": int(len(X)),
        "positive_rate": float(y.mean()),
        **extra,
    }
    return ChurnModel(model_name, estimator, pages_ref, list(X.columns), metrics)


# ------------------------------------------------------------------ submission
def top_k_labels(proba: pd.Series, pos_rate: float = 0.5, n_total: int | None = None) -> pd.Series:
    """Flag the ``round(n_total * pos_rate)`` highest-probability users as churners.

    ``n_total`` defaults to ``len(proba)``; the notebook uses the submission template size.
    """
    if not 0 < pos_rate < 1:
        raise ValueError("pos_rate must be strictly between 0 and 1.")
    n = len(proba) if n_total is None else n_total
    k = max(1, min(round(n * pos_rate), n - 1))
    top = proba.sort_values(ascending=False).index[:k]
    labels = pd.Series(0, index=proba.index, dtype=int, name="target")
    labels.loc[top] = 1
    return labels


def make_submission(
    proba: pd.Series, ids: pd.Series | pd.Index | None = None, pos_rate: float = 0.5
) -> pd.DataFrame:
    """Build an ``id,target`` submission like the notebook's ``make_submission``.

    ``ids`` are the ids of ``example_submission.csv``; ids that were not scored get 0.
    """
    ids = pd.Series(proba.index if ids is None else ids, name="id").reset_index(drop=True)
    labels = top_k_labels(proba, pos_rate, n_total=len(ids))
    return pd.DataFrame({"id": ids, "target": ids.map(labels).fillna(0).astype(int)})
