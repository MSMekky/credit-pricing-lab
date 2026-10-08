"""Campaign P&L under uncertainty: statistical (policy value) and economic (assumptions) together."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import policy as P
from .config import ECON, Economics, SEED

ASSUMPTIONS = ("loss_per_rand_past_due", "cost_of_funds_monthly", "avg_balance_factor", "cost_per_loan")


def draw_weights(n: int, rng: np.random.Generator, econ: Economics = ECON) -> tuple[np.ndarray, pd.DataFrame]:
    """n x components profit weights, with each assumption drawn uniformly from its range."""
    th = pd.DataFrame({k: rng.uniform(*econ.ranges[k], size=n) for k in ASSUMPTIONS})
    W = np.column_stack([np.ones(n), -th["loss_per_rand_past_due"],
                         -th["cost_of_funds_monthly"] * th["avg_balance_factor"], -th["cost_per_loan"]])
    return W, th


def campaign(draws: np.ndarray, names: tuple, offers: int, n: int = 20000, seed: int = SEED,
             econ: Economics = ECON) -> pd.DataFrame:
    """Profit of a mailing of `offers` offers, one row per Monte Carlo draw and policy.

    draws: (bootstrap draws) x policies x components, per-offer means from evaluate.Evaluation.draws().
    Each Monte Carlo draw pairs one bootstrap draw (same for every policy, so differences are paired)
    with one set of economic assumptions.
    """
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(draws), n)
    W, th = draw_weights(n, rng, econ)
    per_offer = np.einsum("npc,nc->np", draws[k], W)                 # n x policies
    out = pd.DataFrame(per_offer * offers, columns=list(names))
    return pd.concat([out, th], axis=1)


def summarise(mc: pd.DataFrame, names: tuple, baseline: str = "standard rate card") -> pd.DataFrame:
    rows = []
    for p in names:
        d = mc[p] - mc[baseline]
        rows.append({"policy": p, "profit_p50": mc[p].median(), "profit_p05": mc[p].quantile(0.05),
                     "profit_p95": mc[p].quantile(0.95), "p_loss": float((mc[p] < 0).mean()),
                     "vs_standard_p50": d.median(), "vs_standard_p05": d.quantile(0.05),
                     "vs_standard_p95": d.quantile(0.95),
                     "p_beats_standard": float((d > 0).mean()) if p != baseline else np.nan})
    return pd.DataFrame(rows)


def tornado(draws: np.ndarray, names: tuple, policy: str, offers: int, econ: Economics = ECON,
            baseline: str = "standard rate card") -> pd.DataFrame:
    """Change in a policy's advantage over the baseline when each assumption moves to the ends of its range."""
    mean = draws.mean(0)                                             # policies x components
    j, j0 = names.index(policy), names.index(baseline)

    def adv(e: Economics) -> float:
        w = P.profit_weights(e)
        return float((mean[j] - mean[j0]) @ w * offers)

    base = adv(econ)
    rows = []
    for k in ASSUMPTIONS:
        lo, hi = econ.ranges[k]
        vals = {}
        for side, v in (("low", lo), ("high", hi)):
            kw = {a: getattr(econ, a) for a in ASSUMPTIONS}
            kw[k] = v
            vals[side] = adv(Economics(**kw))
        rows.append({"assumption": k, "low_value": lo, "high_value": hi, "advantage_at_low": vals["low"],
                     "advantage_at_high": vals["high"], "advantage_at_default": base})
    return pd.DataFrame(rows)
