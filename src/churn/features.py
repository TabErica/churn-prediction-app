"""User-level feature engineering (ported from the Edition 4 notebook)."""

from __future__ import annotations

import numpy as np
import pandas as pd

LEAK_PAGES: tuple[str, ...] = ("Cancel", "Cancellation Confirmation")
CAT_COLS: tuple[str, ...] = ("gender", "level")
EPS = 1e-6


def _count_level_change(s: pd.Series) -> int:
    s = s.dropna()
    if s.empty:
        return 0
    return int((s != s.shift(1)).sum() - 1)


def build_features(
    df: pd.DataFrame,
    cutoff_ts: pd.Series,
    pages_ref: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build one row of features per user from events up to their cutoff.

    Args:
        df: preprocessed logs (see ``churn.data.basic_preprocess``).
        cutoff_ts: index=userId, value=cutoff timestamp.
        pages_ref: page vocabulary. ``None`` when fitting (derived from ``df``);
            pass the training vocabulary at inference to align columns.

    Returns:
        ``(features, pages_ref)``
    """
    # Keep only events up to cutoff and remove leakage pages.
    # A single filtered copy is made; the (possibly multi-GB) input is never copied or mutated.
    row_cutoff = df["userId"].map(cutoff_ts)
    keep = (df["ts"] <= row_cutoff) & ~df["page"].isin(LEAK_PAGES)
    df_obs = df[keep].copy()
    df_obs["cutoff_ts"] = row_cutoff[keep]
    del row_cutoff, keep
    all_users = cutoff_ts.index

    # A. Basic attributes
    agg_basic = df_obs.groupby("userId").agg(
        gender=("gender", "first"),
        level=("level", "last"),
    )

    # B. Activity
    df_obs["date"] = df_obs["ts"].dt.date
    n_events = df_obs.groupby("userId").size().rename("n_events")
    n_active_days = df_obs.groupby("userId")["date"].nunique().rename("n_active_days")
    nextsong = df_obs[df_obs["page"] == "NextSong"]
    total_listen_time = nextsong.groupby("userId")["length"].sum().rename("total_listen_time")
    song_count = nextsong.groupby("userId").size().rename("song_count")

    activity = pd.concat([n_events, n_active_days, total_listen_time, song_count], axis=1)
    activity["avg_song_length"] = activity["total_listen_time"] / (activity["song_count"] + EPS)
    activity["events_per_day"] = activity["n_events"] / (activity["n_active_days"] + EPS)
    activity["listen_time_per_day"] = activity["total_listen_time"] / (
        activity["n_active_days"] + EPS
    )

    days_to_cutoff = (df_obs["cutoff_ts"] - df_obs["ts"]).dt.total_seconds() / 86400.0
    recent = days_to_cutoff <= 7
    events_last_7d = df_obs[recent].groupby("userId").size().rename("events_last_7d")
    songs_last_7d = (
        df_obs[recent & (df_obs["page"] == "NextSong")]
        .groupby("userId")
        .size()
        .rename("songs_last_7d")
    )
    activity = pd.concat([activity, events_last_7d, songs_last_7d], axis=1)

    # C. Page counts + ratios
    if pages_ref is None:
        pages_ref = sorted(df_obs["page"].unique().tolist())
    # Pages unseen in the reference vocabulary are ignored (same as the notebook)
    page_counts = (
        df_obs.groupby(["userId", "page"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=pages_ref, fill_value=0)
        .astype(float)
    )
    page_counts.columns = page_counts.columns.astype(str)
    page_counts.columns.name = None
    page_counts["total_events_from_pages"] = page_counts.sum(axis=1)
    ratio_feat = page_counts.div(page_counts["total_events_from_pages"] + EPS, axis=0)
    page_counts = page_counts.add_prefix("cnt_page_")
    ratio_feat = ratio_feat.add_prefix("ratio_page_")

    # D. Time
    first_obs_ts = df_obs.groupby("userId")["ts"].min().rename("first_obs_ts")
    last_obs_ts = df_obs.groupby("userId")["ts"].max().rename("last_obs_ts")
    registration_ts = df.groupby("userId")["registration"].first().rename("registration_ts")
    time_feat = pd.concat([first_obs_ts, last_obs_ts, registration_ts], axis=1)
    time_feat["days_since_registration"] = (
        time_feat["last_obs_ts"] - time_feat["registration_ts"]
    ).dt.days.clip(lower=0)
    time_feat["obs_window_days"] = (
        time_feat["last_obs_ts"] - time_feat["first_obs_ts"]
    ).dt.days.clip(lower=0)
    mid_ts = first_obs_ts + (last_obs_ts - first_obs_ts) / 2
    is_second_half = df_obs["ts"] > df_obs["userId"].map(mid_ts)
    events_first_half = df_obs[~is_second_half].groupby("userId").size().rename("events_first_half")
    events_second_half = (
        df_obs[is_second_half].groupby("userId").size().rename("events_second_half")
    )
    time_feat = pd.concat([time_feat, events_first_half, events_second_half], axis=1)
    time_feat["ratio_second_to_first"] = time_feat["events_second_half"] / (
        time_feat["events_first_half"] + EPS
    )

    # E. Subscription
    is_paid_last = (agg_basic["level"] == "paid").astype(int).rename("is_paid_last")
    ever_paid = (
        df_obs.groupby("userId")["level"]
        .apply(lambda s: int((s == "paid").any()))
        .rename("ever_paid")
    )
    n_level_change = (
        df_obs.groupby("userId")["level"].apply(_count_level_change).rename("n_level_change")
    )

    # F. HTTP status
    status_counts = df_obs.groupby(["userId", "status"]).size().unstack(fill_value=0)
    status_counts.columns = [f"status_{c}" for c in status_counts.columns]
    for code in (200, 307, 404):
        if f"status_{code}" not in status_counts.columns:
            status_counts[f"status_{code}"] = 0
    status_counts = status_counts[["status_200", "status_307", "status_404"]].copy()
    status_counts["frac_404"] = status_counts["status_404"] / (n_events + EPS)
    status_counts["frac_307"] = status_counts["status_307"] / (n_events + EPS)

    # G. Sessions
    session_stats = df_obs.groupby(["userId", "sessionId"]).agg(
        session_start=("ts", "min"),
        session_end=("ts", "max"),
        session_event_count=("ts", "count"),
    )
    session_stats["session_duration"] = (
        session_stats["session_end"] - session_stats["session_start"]
    ).dt.total_seconds()
    by_user = session_stats.groupby("userId")
    session_count = by_user.size().rename("session_count")
    duration_stats = by_user["session_duration"].agg(
        mean_session_duration="mean",
        max_session_duration="max",
        min_session_duration="min",
        std_session_duration="std",
    )
    event_stats = by_user["session_event_count"].agg(
        mean_event_count_per_session="mean",
        max_event_count_per_session="max",
        min_event_count_per_session="min",
        std_event_count_per_session="std",
    )

    sessions_sorted = session_stats.reset_index().sort_values(["userId", "session_start"])
    sessions_sorted["prev_end"] = sessions_sorted.groupby("userId")["session_end"].shift(1)
    sessions_sorted["idle_time"] = (
        (sessions_sorted["session_start"] - sessions_sorted["prev_end"])
        .dt.total_seconds()
        .fillna(0)
    )
    idle_stats = sessions_sorted.groupby("userId")["idle_time"].agg(
        mean_idle_time="mean",
        max_idle_time="max",
        min_idle_time="min",
        latest_idle_time="last",
    )
    idle_stats["is_big_idle"] = (idle_stats["max_idle_time"] >= 86_400).astype(int)
    idle_stats["is_very_big_idle"] = (idle_stats["max_idle_time"] >= 864_000).astype(int)

    song_per_session = (activity["song_count"] / (session_count + EPS)).rename("song_per_session")
    has_single_session = (session_count == 1).astype(int).rename("has_single_session")

    feats = pd.concat(
        [
            agg_basic,
            activity,
            page_counts,
            ratio_feat,
            time_feat,
            status_counts,
            session_count,
            duration_stats,
            event_stats,
            idle_stats,
            is_paid_last,
            ever_paid,
            n_level_change,
            song_per_session,
            has_single_session,
        ],
        axis=1,
    )
    feats = feats.reindex(all_users).sort_index()
    dt_cols = feats.select_dtypes(include=["datetime", "datetimetz"]).columns
    feats = feats.drop(columns=dt_cols)
    return feats, pages_ref


def prepare_design_matrix(
    feats: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Clean a feature table for scikit-learn.

    Aligns to ``columns`` (training columns) when given, replaces infinities,
    fills numeric NaNs with 0 and categorical NaNs with ``"missing"``.

    Returns:
        ``(X, cat_cols, num_cols)``
    """
    X = feats.copy() if columns is None else feats.reindex(columns=columns)
    X = X.replace([np.inf, -np.inf], np.nan)
    cat_cols = [c for c in CAT_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), "missing").astype(str)
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    return X, cat_cols, num_cols
