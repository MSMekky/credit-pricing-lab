"""Honest evaluation: learn rate cards on half the branches, score them on the other half, repeat.

Each repetition draws a fresh random split of branches. Inside the learning half the DR scores are
cross-fitted again (two folds), so the policy never sees outcome-model predictions fitted on its own
offers. The evaluation half is scored with outcome models fitted only on the learning half. Uncertainty
inside each repetition comes from a bootstrap over evaluation branches (offers within a branch are
not independent); stacking the repetitions adds the variation from the choice of split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import policy as P
from .config import SEED, ECON, RISKS


@dataclass
class Evaluation:
    names: tuple
    estimates: np.ndarray      # splits x policies x components  (per-offer means)
    boot: np.ndarray           # splits x B x policies x components
    chosen_uniform: list       # per split: {risk: arm}
    trees: list                # per split: {risk: Rule}

    def draws(self) -> np.ndarray:
        """Stacked bootstrap draws: (splits*B) x policies x components."""
        return self.boot.reshape(-1, *self.boot.shape[2:])


def _branch_bootstrap(branch: np.ndarray, comps: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """B x policies x components bootstrap means, resampling whole branches. comps: N x policies x components."""
    codes, uniq = pd.factorize(branch)
    nb = len(uniq)
    sums = np.zeros((nb, *comps.shape[1:]))
    np.add.at(sums, codes, comps)
    counts = np.bincount(codes, minlength=nb).astype(float)
    out = np.empty((B, *comps.shape[1:]))
    for b in range(B):
        pick = rng.integers(0, nb, nb)
        out[b] = sums[pick].sum(0) / counts[pick].sum()
    return out


def run(sample: pd.DataFrame, n_splits: int = 20, B: int = 200, econ=ECON, seed: int = SEED) -> Evaluation:
    rng = np.random.default_rng(seed)
    w = P.profit_weights(econ)
    arms = P.allowed_arms(sample)
    e_full = P.propensities(sample)
    Y = sample[list(P.COMPONENTS)].to_numpy(dtype=float)
    branches = sample["branch"].unique()
    est, boot, uni, trees = [], [], [], []
    for s in range(n_splits):
        learn_b = set(rng.choice(branches, len(branches) // 2, replace=False))
        is_learn = sample["branch"].isin(learn_b).to_numpy()
        learn, ev = sample[is_learn].reset_index(drop=True), sample[~is_learn].reset_index(drop=True)

        G_learn = P.crossfit_scores(learn, 2, rng)
        pol = P.learn(learn, G_learn, w, arms)
        models = P.fit_outcomes(learn)
        G_ev = P.dr_scores(ev, P.predict_all_arms(models, ev), e_full[~is_learn])

        comps = np.empty((len(ev), len(P.POLICY_NAMES), len(P.COMPONENTS)))
        for j, name in enumerate(P.POLICY_NAMES):
            if name == "experiment (random rates)":
                comps[:, j, :] = Y[~is_learn]
            else:
                comps[:, j, :] = P.value_components(G_ev, ev, P.assign(name, ev, pol))
        est.append(comps.mean(0))
        boot.append(_branch_bootstrap(ev["branch"].to_numpy(), comps, B, rng))
        uni.append(pol.uniform)
        trees.append(pol.trees)
    return Evaluation(P.POLICY_NAMES, np.array(est), np.array(boot), uni, trees)


def summary(ev: Evaluation, econ=ECON, baseline: str = "standard rate card") -> pd.DataFrame:
    w = P.profit_weights(econ)
    d = ev.draws() @ w                                   # draws x policies
    e = ev.estimates @ w                                 # splits x policies
    j0 = ev.names.index(baseline)
    rows = []
    for j, name in enumerate(ev.names):
        diff = d[:, j] - d[:, j0]
        rows.append({"policy": name,
                     "profit_per_offer": float(np.median(e[:, j])),
                     "p05": float(np.quantile(d[:, j], 0.05)), "p95": float(np.quantile(d[:, j], 0.95)),
                     "vs_standard": float(np.median(e[:, j] - e[:, j0])),
                     "vs_standard_p05": float(np.quantile(diff, 0.05)), "vs_standard_p95": float(np.quantile(diff, 0.95)),
                     "p_beats_standard": float((diff > 0).mean()) if j != j0 else np.nan})
    return pd.DataFrame(rows)


def components_table(ev: Evaluation) -> pd.DataFrame:
    m = np.median(ev.estimates, axis=0)
    return pd.DataFrame(m, index=list(ev.names), columns=list(P.COMPONENTS))


def stability(ev: Evaluation) -> pd.DataFrame:
    """How often each arm was chosen as the best single rate per risk class across splits."""
    rows = []
    for r in RISKS:
        picks = pd.Series([u[r] for u in ev.chosen_uniform]).value_counts(normalize=True)
        for a, share in picks.items():
            rows.append({"risk": r, "arm": int(a), "share_of_splits": float(share)})
    return pd.DataFrame(rows)
