"""Unit economics by risk class and price arm: what each price actually produced in the experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import policy as P
from .config import RISKS, ECON, SEED, ARM_LABELS


def arm_means(sample: pd.DataFrame, econ=ECON, B: int = 300, seed: int = SEED) -> pd.DataFrame:
    """Inverse-propensity-weighted (Hajek) means per risk x arm, with a branch bootstrap.

    Within a risk class the arm mix differs slightly by mailer wave, so plain means could mix wave
    effects into the price comparison; weighting by 1/e(a | risk, wave) removes that.
    """
    rng = np.random.default_rng(seed)
    e = P.propensities(sample)
    w = P.profit_weights(econ)
    Y = sample[list(P.COMPONENTS)].to_numpy(dtype=float)
    Y = np.column_stack([Y, Y @ w, sample["tookup"].to_numpy(dtype=float), sample["loansize"].to_numpy(dtype=float)])
    names = list(P.COMPONENTS) + ["profit", "take_up", "loan_size"]
    A = sample["arm"].to_numpy()
    wt = 1.0 / e[np.arange(len(sample)), A]
    codes, uniq = pd.factorize(sample["branch"])
    rows = []
    allowed = P.allowed_arms(sample)
    for r in RISKS:
        mr = (sample["risk"] == r).to_numpy()
        for a in allowed[r]:
            m = mr & (A == a)
            num = np.zeros((len(uniq), Y.shape[1]))
            den = np.zeros(len(uniq))
            np.add.at(num, codes[m], Y[m] * wt[m, None])
            np.add.at(den, codes[m], wt[m])
            point = num.sum(0) / den.sum()
            bs = np.empty((B, Y.shape[1]))
            for b in range(B):
                pick = rng.integers(0, len(uniq), len(uniq))
                bs[b] = num[pick].sum(0) / den[pick].sum()
            lo, hi = np.quantile(bs, [0.05, 0.95], axis=0)
            rate = float(np.average(sample.loc[m, "offer4"], weights=wt[m]))
            for k, n in enumerate(names):
                rows.append({"risk": r, "arm": int(a), "arm_label": ARM_LABELS[a], "mean_rate": rate,
                             "n_offers": int(m.sum()), "metric": n, "value": float(point[k]),
                             "p05": float(lo[k]), "p95": float(hi[k])})
    return pd.DataFrame(rows)


def above_standard(df: pd.DataFrame) -> pd.DataFrame:
    """Wave 1 only (the only wave that tested rates above the standard): take-up below vs above standard."""
    w1 = df[df["wave"] == 1].copy()
    w1["side"] = np.where(w1["normrate_less"] == 1, "at or below standard", "above standard")
    return (w1.groupby(["risk", "side"]).agg(offers=("applied", "size"), applied=("applied", "mean"),
                                             took_up=("tookup", "mean"), mean_rate=("offer4", "mean"))
              .reset_index())
