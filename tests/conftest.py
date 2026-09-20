import pandas as pd
import pytest
from fake_logs import generate_event_logs


def _ms(s: str) -> int:
    return int(pd.Timestamp(s).value // 1_000_000)


@pytest.fixture
def raw_logs() -> pd.DataFrame:
    """Tiny hand-written log: user 1 churns, user 2 stays, user 0 is anonymous."""
    rows = [
        # userId, sessionId, ts, page, gender, level, length, status
        (1, 10, "2018-10-01 10:00", "NextSong", "M", "free", 200.0, 200),
        (1, 10, "2018-10-01 10:05", "Thumbs Down", "M", "free", None, 307),
        (1, 11, "2018-10-03 09:00", " NextSong ", "M", "paid", 180.0, 200),
        (1, 11, "2018-10-03 09:10", "Cancel", "M", "paid", None, 307),
        (1, 11, "2018-10-03 09:11", "Cancellation Confirmation", "M", "paid", None, 200),
        (1, 12, "2018-10-04 12:00", "NextSong", "M", "paid", 150.0, 200),  # after churn
        (2, 20, "2018-10-02 08:00", "Home", "F", "paid", None, 200),
        (2, 20, "2018-10-02 08:03", "NextSong", "F", "paid", 240.0, 200),
        (2, 21, "2018-10-05 20:00", "Error", "F", "paid", None, 404),
        (0, 99, "2018-10-02 11:00", "Home", None, "free", None, 200),
    ]
    df = pd.DataFrame(
        rows, columns=["userId", "sessionId", "ts", "page", "gender", "level", "length", "status"]
    )
    df["time"] = df["ts"]
    df["ts"] = df["ts"].map(_ms)
    df["registration"] = "2018-09-01"
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)  # unsorted on purpose


@pytest.fixture(scope="session")
def fake_event_logs() -> pd.DataFrame:
    return generate_event_logs(n_users=120, seed=123)
