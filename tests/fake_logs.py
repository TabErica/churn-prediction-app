"""Fake event logs with the same schema as the real data — test-only helper.

The competition data cannot be committed, so the unit tests and the CI pipeline
run on these generated logs. The application itself only works on real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PAGES = [
    "NextSong", "Home", "Thumbs Up", "Thumbs Down", "Add to Playlist", "Add Friend",
    "Roll Advert", "Logout", "Settings", "Help", "Upgrade", "Downgrade", "Error",
]  # fmt: skip
# Page probabilities for "loyal" vs "at-risk" behaviour (same order as PAGES)
_BASE_P = np.array(
    [0.78, 0.04, 0.045, 0.01, 0.025, 0.015, 0.02, 0.012, 0.006, 0.005, 0.002, 0.004, 0.001]
)
_CHURN_P = np.array(
    [0.70, 0.05, 0.025, 0.035, 0.015, 0.008, 0.07, 0.02, 0.012, 0.012, 0.001, 0.012, 0.005]
)

START = pd.Timestamp("2018-10-01")


def generate_event_logs(
    n_users: int = 300,
    churn_rate: float = 0.25,
    seed: int = 0,
    include_anonymous: bool = True,
) -> pd.DataFrame:
    """Generate raw logs (``ts`` in epoch milliseconds, like the source parquet)."""
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    session_id = 0
    for user_id in range(1, n_users + 1):
        churner = rng.random() < churn_rate
        # Behaviour matches the label only 75% of the time -> realistic, imperfect signal
        risky = churner if rng.random() < 0.75 else not churner
        probs = _CHURN_P if risky else _BASE_P
        probs = probs / probs.sum()
        n_sessions = int(rng.integers(1, 9 if risky else 14))
        registration = START - pd.Timedelta(days=int(rng.integers(1, 400)))
        gender = rng.choice(["M", "F"])
        paid = rng.random() < (0.35 if risky else 0.5)

        t = START + pd.Timedelta(hours=float(rng.uniform(0, 24 * 20)))
        for _ in range(n_sessions):
            session_id += 1
            n_events = int(rng.integers(3, 50 if risky else 80))
            gaps = rng.exponential(200, n_events).cumsum()
            ts = t + pd.to_timedelta(gaps, unit="s")
            pages = rng.choice(PAGES, size=n_events, p=probs)
            redirect = np.isin(pages, ["Logout", "Upgrade", "Downgrade"])
            status = np.where(pages == "Error", 404, np.where(redirect, 307, 200))
            song_len = rng.normal(240, 60, n_events).clip(30)
            rows.append(
                pd.DataFrame(
                    {
                        "userId": user_id,
                        "sessionId": session_id,
                        "ts": ts,
                        "page": pages,
                        "gender": gender,
                        "level": "paid" if paid else "free",
                        "length": np.where(pages == "NextSong", song_len, np.nan),
                        "status": status,
                        "registration": registration,
                    }
                )
            )
            t = ts[-1] + pd.Timedelta(hours=float(rng.exponential(50 if risky else 30)))

        if churner:
            end = t + pd.Timedelta(minutes=1)
            rows.append(
                pd.DataFrame(
                    {
                        "userId": user_id,
                        "sessionId": session_id,
                        "ts": [end, end + pd.Timedelta(seconds=5)],
                        "page": ["Cancel", "Cancellation Confirmation"],
                        "gender": gender,
                        "level": "paid" if paid else "free",
                        "length": np.nan,
                        "status": [307, 200],
                        "registration": registration,
                    }
                )
            )

    df = pd.concat(rows, ignore_index=True)
    if include_anonymous:
        anon = df.sample(n=min(20, len(df)), random_state=seed).assign(userId=0, gender=None)
        df = pd.concat([df, anon], ignore_index=True)

    df["time"] = df["ts"]
    df["ts"] = (df["ts"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
