"""Churn labels and per-user observation cutoffs."""

from __future__ import annotations

import pandas as pd

CHURN_PAGE = "Cancellation Confirmation"


def build_labels_and_cutoff(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return ``(y, cutoff_ts)``, both indexed by userId.

    - User reached "Cancellation Confirmation": y=1, cutoff = first such event.
    - Otherwise: y=0, cutoff = last observed event.
    """
    first_cc = df.loc[df["page"] == CHURN_PAGE].groupby("userId")["ts"].min()
    last_ts = df.groupby("userId")["ts"].max()

    y = pd.Series(0, index=last_ts.index, dtype=int, name="target")
    y.loc[first_cc.index] = 1

    cutoff_ts = last_ts.copy()
    cutoff_ts.loc[first_cc.index] = first_cc
    cutoff_ts.name = "cutoff_ts"
    return y.sort_index(), cutoff_ts.sort_index()


def last_event_cutoff(df: pd.DataFrame) -> pd.Series:
    """Cutoff used at inference time (no labels): each user's last event."""
    return df.groupby("userId")["ts"].max().sort_index().rename("cutoff_ts")
