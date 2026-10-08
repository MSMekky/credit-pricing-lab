"""Rate cards learned and evaluated off-policy from the randomised experiment.

Notation: an offer i has features x_i, risk class r_i, mailer wave w_i and a randomly assigned price arm A_i.
Each offer has four outcome components Y_ic (revenue, past due, balance-months, loans); profit is a linear
combination of them with the assumptions in config.Economics, so every estimate is kept per component and
profit can be recomputed under any assumption without refitting.

Doubly robust score for arm a (Dudik et al. 2011; Athey & Wager 2021):
    G_ic(a) = m_c(x_i, a) + 1[A_i = a] / e(a | r_i, w_i) * (Y_ic - m_c(x_i, A_i))
e is known by design (rates were randomised within risk class and wave), so it is taken as the empirical
assignment share in each risk x wave cell. m_c is a gradient-boosting outcome model, always fitted on
branches other than the ones being scored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import (ARM_LABELS, STATUS_QUO_ARM, MIN_ARM_SHARE, POLICY_FEATURES, OUTCOME_FEATURES,
                     RISKS, SEED, Economics, ECON)

COMPONENTS = ("revenue", "past_due", "balance_months", "loans")
N_ARMS = len(ARM_LABELS)


def profit_weights(econ: Economics = ECON) -> np.ndarray:
    """Profit = w . components."""
    return np.array([1.0, -econ.loss_per_rand_past_due,
                     -econ.cost_of_funds_monthly * econ.avg_balance_factor, -econ.cost_per_loan])


def allowed_arms(sample: pd.DataFrame) -> dict[str, np.ndarray]:
    """Arms with enough offers in every wave of a risk class (positivity)."""
    out = {}
    for r in RISKS:
        d = sample[sample["risk"] == r]
        shares = d.groupby("wave")["arm"].value_counts(normalize=True).unstack(fill_value=0.0)
        out[r] = np.array([a for a in range(N_ARMS) if a in shares.columns and (shares[a] >= MIN_ARM_SHARE).all()])
    return out


def propensities(sample: pd.DataFrame) -> np.ndarray:
    """N x arms matrix of assignment probabilities by risk class and wave."""
    cell = sample["risk"] + "_" + sample["wave"].astype(str)
    shares = pd.crosstab(cell, sample["arm"], normalize="index").reindex(columns=range(N_ARMS), fill_value=0.0)
    return shares.loc[cell].to_numpy()


def _design(sample: pd.DataFrame, arm: np.ndarray) -> np.ndarray:
    X = sample[list(OUTCOME_FEATURES)].to_numpy(dtype=float)
    risk = sample["risk"].map({r: i for i, r in enumerate(RISKS)}).to_numpy(dtype=float)
    return np.column_stack([X, risk, arm])


def fit_outcomes(train: pd.DataFrame) -> list[HistGradientBoostingRegressor]:
    models = []
    X = _design(train, train["arm"].to_numpy(dtype=float))
    risk_col = X.shape[1] - 2
    for c in COMPONENTS:
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
                                          l2_regularization=1.0, categorical_features=[risk_col], random_state=SEED)
        models.append(m.fit(X, train[c].to_numpy(dtype=float)))
    return models


def predict_all_arms(models, sample: pd.DataFrame) -> np.ndarray:
    """N x arms x components predictions."""
    out = np.empty((len(sample), N_ARMS, len(COMPONENTS)))
    for a in range(N_ARMS):
        X = _design(sample, np.full(len(sample), float(a)))
        for c, m in enumerate(models):
            out[:, a, c] = m.predict(X)
    return out


def dr_scores(sample: pd.DataFrame, mu: np.ndarray, e: np.ndarray) -> np.ndarray:
    """N x arms x components doubly robust scores."""
    Y = sample[list(COMPONENTS)].to_numpy(dtype=float)
    A = sample["arm"].to_numpy()
    idx = np.arange(len(sample))
    resid = Y - mu[idx, A, :]                                    # N x C
    G = mu.copy()
    G[idx, A, :] += resid / e[idx, A][:, None]
    return G


def crossfit_scores(sample: pd.DataFrame, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """DR scores with the outcome model fitted on other branches (folds are whole branches)."""
    branches = sample["branch"].unique()
    fold_of = dict(zip(branches, rng.permutation(len(branches)) % n_folds))
    fold = sample["branch"].map(fold_of).to_numpy()
    e = propensities(sample)
    G = np.empty((len(sample), N_ARMS, len(COMPONENTS)))
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        models = fit_outcomes(sample[tr])
        sub = sample[te].reset_index(drop=True)
        G[te] = dr_scores(sub, predict_all_arms(models, sub), e[te])
    return G


# ---------------------------------------------------------------- policies
@dataclass
class Rule:
    """Depth-2 policy tree for one risk class: root split, then one split per child, arm per leaf."""
    risk: str
    root: tuple            # (feature, threshold)
    left: tuple            # (feature, threshold, arm_if_below, arm_if_above) or (None, None, arm, arm)
    right: tuple

    def assign(self, X: pd.DataFrame) -> np.ndarray:
        def leaf(node, x):
            f, t, a_lo, a_hi = node
            if f is None:
                return np.full(len(x), a_lo)
            v = x[f].to_numpy(dtype=float)
            return np.where(np.nan_to_num(v, nan=-np.inf) <= t, a_lo, a_hi)
        f, t = self.root
        v = np.nan_to_num(X[f].to_numpy(dtype=float), nan=-np.inf)
        out = np.empty(len(X), dtype=int)
        lo = v <= t
        out[lo] = leaf(self.left, X[lo])
        out[~lo] = leaf(self.right, X[~lo])
        return out

    def describe(self) -> str:
        def node(n):
            f, t, a, b = n
            return ARM_LABELS[a] if f is None else f"if {f} <= {t:g}: {ARM_LABELS[a]}, else {ARM_LABELS[b]}"
        f, t = self.root
        return f"{self.risk}: {f} <= {t:g} -> [{node(self.left)}]; above -> [{node(self.right)}]"


def _best_split(x: np.ndarray, gp: np.ndarray, arms: np.ndarray, min_leaf: int, n_q: int = 9):
    """Best single split of one feature: returns (gain_total, threshold, arm_lo, arm_hi)."""
    x = np.nan_to_num(x, nan=-np.inf)
    best = (gp[:, arms].sum(0).max(), None, int(arms[gp[:, arms].sum(0).argmax()]), int(arms[gp[:, arms].sum(0).argmax()]))
    qs = np.unique(np.quantile(x[np.isfinite(x)], np.linspace(0.1, 0.9, n_q))) if np.isfinite(x).any() else []
    for t in qs:
        lo = x <= t
        if lo.sum() < min_leaf or (~lo).sum() < min_leaf:
            continue
        sl, sh = gp[lo][:, arms].sum(0), gp[~lo][:, arms].sum(0)
        tot = sl.max() + sh.max()
        if tot > best[0]:
            best = (tot, float(t), int(arms[sl.argmax()]), int(arms[sh.argmax()]))
    return best


def learn_tree(X: pd.DataFrame, gp: np.ndarray, arms: np.ndarray, risk: str, min_leaf: int = 400) -> Rule:
    """Exhaustive depth-2 policy tree maximising the summed DR profit score (Athey & Wager 2021 style)."""
    best_val, best_rule = -np.inf, None
    feats = list(POLICY_FEATURES)
    for f in feats:
        x = np.nan_to_num(X[f].to_numpy(dtype=float), nan=-np.inf)
        qs = np.unique(np.quantile(x[np.isfinite(x)], np.linspace(0.1, 0.9, 9))) if np.isfinite(x).any() else []
        for t in qs:
            lo = x <= t
            if lo.sum() < 2 * min_leaf or (~lo).sum() < 2 * min_leaf:
                continue
            children = []
            val = 0.0
            for m in (lo, ~lo):
                cb = (-np.inf, None)
                for g in feats:
                    tot, th, a, b = _best_split(X[g].to_numpy(dtype=float)[m], gp[m], arms, min_leaf)
                    if tot > cb[0]:
                        cb = (tot, (g if th is not None else None, th, a, b))
                val += cb[0]
                children.append(cb[1])
            if val > best_val:
                best_val, best_rule = val, Rule(risk, (f, float(t)), children[0], children[1])
    if best_rule is None:   # too few offers to split: one arm for the whole class
        a = int(arms[gp[:, arms].sum(0).argmax()])
        best_rule = Rule(risk, (feats[0], np.inf), (None, None, a, a), (None, None, a, a))
    return best_rule


@dataclass
class Policies:
    uniform: dict          # risk -> arm
    trees: dict            # risk -> Rule


def learn(sample: pd.DataFrame, G: np.ndarray, w: np.ndarray, arms: dict) -> Policies:
    gp = G @ w                                            # N x arms, profit scores
    uniform, trees = {}, {}
    for r in RISKS:
        m = (sample["risk"] == r).to_numpy()
        a = arms[r]
        uniform[r] = int(a[gp[m][:, a].mean(0).argmax()])
        trees[r] = learn_tree(sample[m].reset_index(drop=True), gp[m], a, r)
    return Policies(uniform, trees)


def assign(policy: str, sample: pd.DataFrame, pol: Policies | None) -> np.ndarray:
    if policy == "standard rate card":
        return np.full(len(sample), STATUS_QUO_ARM)
    if policy == "experiment (random rates)":
        return sample["arm"].to_numpy()
    out = np.empty(len(sample), dtype=int)
    for r in RISKS:
        m = (sample["risk"] == r).to_numpy()
        if policy == "best rate per risk class":
            out[m] = pol.uniform[r]
        elif policy == "segment rules (policy tree)":
            out[m] = pol.trees[r].assign(sample[m].reset_index(drop=True))
        else:
            raise ValueError(policy)
    return out


POLICY_NAMES = ("standard rate card", "experiment (random rates)", "best rate per risk class", "segment rules (policy tree)")


def value_components(G: np.ndarray, sample: pd.DataFrame, a: np.ndarray) -> np.ndarray:
    """N x components scores of a policy (for the experiment itself, the observed outcomes)."""
    return G[np.arange(len(sample)), a, :]
