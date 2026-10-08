"""Load the two replication files, check them, and derive the per-offer outcomes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import AER_FILE, ECTA_FILE, STANDARD_RATE, ARM_EDGES, ARM_LABELS


class DataError(RuntimeError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise DataError(msg)


def load_aer(path: Path = AER_FILE) -> pd.DataFrame:
    """53,810 direct-mail offers with randomised monthly interest rates (Karlan & Zinman 2008)."""
    df = pd.read_stata(path, convert_categoricals=False)
    _check(len(df) == 53_810, f"expected 53,810 offers, found {len(df)}")
    _check(set(df["risk"]) == set(STANDARD_RATE), "unexpected risk classes")
    _check(df["offer4"].between(3.25, 14.75).all(), "offer rate outside the experiment's range")
    _check((df["final4"] <= df["offer4"] + 1e-9).all(), "contract rate above offer rate")
    _check(((df["tookup"] == 0) <= (df["grossinterest"] == 0)).all(), "interest revenue without a loan")
    _check(((df["tookup"] == 0) <= (df["loansize"] == 0)).all(), "loan size without a loan")
    _check((df.loc[df["tookup"] == 1, "pstdue_average"].notna()).all(), "borrower without repayment record")

    df["branch"] = df["branchuse"].astype(str)
    df["standard_rate"] = df["risk"].map(STANDARD_RATE)
    df["gap"] = df["offer4"] - df["standard_rate"]
    df["arm"] = pd.cut(df["gap"], ARM_EDGES, labels=False, right=True)   # NaN above the standard rate
    df["arm_label"] = df["arm"].map(dict(enumerate(ARM_LABELS)))

    # per-offer outcome components (zero when the offer was not taken up)
    df["revenue"] = df["grossinterest"].astype(float)
    df["past_due"] = df["pstdue_average"].fillna(0.0).astype(float)
    df["balance_months"] = (df["loansize"] * df["term"]).astype(float)
    df["loans"] = df["tookup"].astype(float)
    return df


def pricing_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Offers usable for evaluating a rate card.

    * contract rate equal to the offer rate: a separate random draw cut the contract rate below the
      offer for some clients after they applied. Those offers mix two prices, so revenue cannot be
      attributed to the offer. Because that draw was random, the remaining offers are still a random
      sample (the same restriction Karlan & Zinman use for revenue and loan size).
    * offer at or below the standard rate: rates above it were only tested in the first mailer wave.
    """
    s = df[(df["offer4"] == df["final4"]) & df["arm"].notna()].copy()
    s["arm"] = s["arm"].astype(int)
    return s.reset_index(drop=True)


def load_ecta(path: Path = ECTA_FILE) -> pd.DataFrame:
    """57,533 offers with separately randomised offer rate, contract rate and repayment incentive (Karlan & Zinman 2009)."""
    df = pd.read_stata(path, convert_categoricals=False)
    _check(len(df) == 57_533, f"expected 57,533 offers, found {len(df)}")
    _check((df["final4"] <= df["offer4"] + 1e-9).all(), "contract rate above offer rate")
    df["borrowed"] = df["Sindex"].notna().astype(int)
    return df
