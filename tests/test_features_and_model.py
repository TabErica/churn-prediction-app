import numpy as np
import pandas as pd
import pytest

from churn.data import basic_preprocess
from churn.features import build_features, prepare_design_matrix
from churn.labels import build_labels_and_cutoff, last_event_cutoff
from churn.model import (
    MODEL_NAMES,
    VOTING,
    ChurnModel,
    WeightedVote,
    build_pipeline,
    make_submission,
    oversample_dataframe,
    search_vote_weights,
    top_k_labels,
    train_from_logs,
)


@pytest.fixture
def logs(raw_logs):
    return basic_preprocess(raw_logs)


# ---------- labels ----------


def test_labels_and_cutoff(logs):
    y, cutoff = build_labels_and_cutoff(logs)
    assert y.to_dict() == {1: 1, 2: 0}
    assert cutoff[1] == pd.Timestamp("2018-10-03 09:11")  # first cancellation confirmation
    assert cutoff[2] == pd.Timestamp("2018-10-05 20:00")  # last event
    assert last_event_cutoff(logs)[1] == pd.Timestamp("2018-10-04 12:00")


# ---------- features ----------


def test_build_features_one_row_per_user_without_leakage(logs):
    _, cutoff = build_labels_and_cutoff(logs)
    feats, pages_ref = build_features(logs, cutoff)

    assert list(feats.index) == [1, 2]
    assert "Cancel" not in pages_ref and "Cancellation Confirmation" not in pages_ref
    assert not any("Cancel" in c for c in feats.columns)
    assert feats.select_dtypes(include=["datetime"]).empty
    # the NextSong after user 1's cancellation must be ignored
    assert feats.loc[1, "song_count"] == 2
    assert feats.loc[1, "n_level_change"] == 1
    assert feats.loc[2, "status_404"] == 1


def test_build_features_aligns_to_reference_pages(logs):
    cutoff = last_event_cutoff(logs)
    ref = ["Home", "NextSong", "Settings"]
    feats, returned = build_features(logs, cutoff, pages_ref=ref)
    assert returned == ref
    assert {f"cnt_page_{p}" for p in ref} <= set(feats.columns)
    assert "cnt_page_Error" not in feats.columns
    assert feats["cnt_page_Settings"].eq(0).all()


def test_prepare_design_matrix_is_clean(logs):
    _, cutoff = build_labels_and_cutoff(logs)
    feats, _ = build_features(logs, cutoff)
    feats.loc[2, "events_per_day"] = np.inf
    X, cat_cols, num_cols = prepare_design_matrix(feats)
    assert cat_cols == ["gender", "level"]
    assert not X[num_cols].isna().any().any()
    assert np.isfinite(X[num_cols].to_numpy()).all()

    X_aligned, _, _ = prepare_design_matrix(feats, columns=[*X.columns, "brand_new_col"])
    assert X_aligned["brand_new_col"].eq(0).all()


# ---------- model ----------


def test_oversample_balances_classes():
    X = pd.DataFrame({"a": range(10)})
    y = np.array([0] * 8 + [1] * 2)
    X_over, y_over = oversample_dataframe(X, y, random_state=0)
    assert (y_over == 1).sum() == 8 and (y_over == 0).sum() == 8
    assert len(X_over) == len(y_over)
    # rows still match their labels after shuffling
    assert set(X_over.loc[y_over == 1, "a"]) == {8, 9}


def _notebook_oversample(X_df, y_sr, random_state):
    """Verbatim logic of the notebook's oversample_dataframe (prints removed)."""
    df_xy = X_df.copy()
    df_xy["label"] = y_sr.values
    major_df = df_xy[df_xy["label"] == 0]
    minor_df = df_xy[df_xy["label"] == 1]
    ratio = max(1, int(len(major_df) / len(minor_df)))
    oversampled_minor = pd.concat([minor_df] * ratio, ignore_index=True)
    combined = pd.concat([major_df, oversampled_minor], axis=0)
    combined = combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return combined.drop(columns=["label"]), combined["label"].values


def test_oversample_matches_notebook_exactly():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=50), "b": rng.integers(0, 9, 50)}, index=range(100, 150))
    y = pd.Series((rng.random(50) < 0.25).astype(int), index=X.index)
    X_ours, y_ours = oversample_dataframe(X, y, random_state=2025)
    X_nb, y_nb = _notebook_oversample(X, y, random_state=2025)
    pd.testing.assert_frame_equal(X_ours, X_nb)
    np.testing.assert_array_equal(y_ours, y_nb)


def test_oversample_errors():
    X = pd.DataFrame({"a": range(3)})
    with pytest.raises(ValueError, match="No positive"):
        oversample_dataframe(X, [0, 0, 0])
    with pytest.raises(ValueError, match="same length"):
        oversample_dataframe(X, [0, 1])


def test_top_k_labels():
    proba = pd.Series([0.9, 0.1, 0.5, 0.7], index=[10, 11, 12, 13])
    labels = top_k_labels(proba, pos_rate=0.5)
    assert labels.to_dict() == {10: 1, 11: 0, 12: 0, 13: 1}
    with pytest.raises(ValueError):
        top_k_labels(proba, pos_rate=1.0)


def test_top_k_uses_template_size():
    proba = pd.Series([0.9, 0.1, 0.5, 0.7], index=[10, 11, 12, 13])
    assert top_k_labels(proba, pos_rate=0.5, n_total=2).sum() == 1


def test_make_submission_aligns_to_template_ids():
    proba = pd.Series([0.9, 0.2, 0.6], index=[1, 2, 3])
    template_ids = pd.Series([3, 1, 2, 99])  # 99 was never scored
    sub = make_submission(proba, template_ids, pos_rate=0.5)
    assert list(sub.columns) == ["id", "target"]
    assert sub["id"].tolist() == [3, 1, 2, 99]  # template order kept
    assert sub.set_index("id")["target"].to_dict() == {3: 1, 1: 1, 2: 0, 99: 0}


def test_submission_aligns_when_logs_store_user_ids_as_text(fake_event_logs):
    """Real logs carry userId as text while the template carries integers (regression)."""
    as_text = fake_event_logs.assign(userId=fake_event_logs["userId"].astype(str))
    model = train_from_logs(as_text, model_name="Logistic Regression", n_splits=2)
    proba = model.predict_proba_from_logs(as_text)
    sub = make_submission(proba, pd.Series(sorted(proba.index)), pos_rate=0.5)
    assert sub["target"].sum() == len(proba) // 2


def test_search_vote_weights_finds_best_combination():
    y = np.array([0, 0, 1, 1])
    good = np.array([0.1, 0.2, 0.8, 0.9])
    noisy = np.array([0.9, 0.1, 0.2, 0.8])
    weights, auc = search_vote_weights({"good": good, "noisy": noisy}, y)
    assert auc == 1.0
    assert weights["good"] > weights["noisy"]


def test_weighted_vote_averages_member_probabilities():
    class Const:
        def __init__(self, p):
            self.p = p

        def predict_proba(self, X):
            return np.column_stack([1 - np.full(len(X), self.p), np.full(len(X), self.p)])

    vote = WeightedVote({"lr": Const(0.8), "knn": Const(0.2)}, {"lr": 4, "knn": 1})
    proba = vote.predict_proba(pd.DataFrame({"a": [1, 2]}))
    np.testing.assert_allclose(proba[:, 1], (4 * 0.8 + 0.2) / 5)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_voting_is_default_and_reports_weights(fake_event_logs):
    model = train_from_logs(fake_event_logs, n_splits=3)
    assert model.model_name == VOTING
    assert set(model.metrics["weights"]) == {"Logistic Regression", "KNN"}
    assert all(w in (1, 2, 3, 4) for w in model.metrics["weights"].values())
    assert 0.0 <= model.metrics["oof_auc"] <= 1.0
    assert len(model.metrics["fold_aucs"]) == 3


def test_unknown_model_name(fake_event_logs):
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("XGBoost", [], ["a"])
    with pytest.raises(ValueError, match="Unknown model"):
        train_from_logs(fake_event_logs, model_name="XGBoost")


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_train_predict_roundtrip(tmp_path, fake_event_logs, model_name):
    model = train_from_logs(fake_event_logs, model_name=model_name, n_splits=3)
    assert 0.0 <= model.metrics["oof_auc"] <= 1.0
    assert len(model.metrics["fold_aucs"]) == 3

    path = tmp_path / "model.joblib"
    model.save(path)
    reloaded = ChurnModel.load(path)
    proba = reloaded.predict_proba_from_logs(fake_event_logs.head(3000))
    assert proba.between(0, 1).all()
    assert proba.index.is_unique


def test_training_is_reproducible(fake_event_logs):
    a = train_from_logs(fake_event_logs, "Extra Trees", n_splits=3).metrics["oof_auc"]
    b = train_from_logs(fake_event_logs, "Extra Trees", n_splits=3).metrics["oof_auc"]
    assert a == b
