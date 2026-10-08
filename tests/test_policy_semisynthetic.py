"""The estimators, tested where the right answer is known.

Real offers, real covariates and the real randomised price arms are kept; only the outcomes are
replaced by ones generated from a known model. If the off-policy machinery is right it must recover
the true value of a rate card, and the policy tree must find a planted segment rule.
"""
import numpy as np
import pandas as pd
import pytest

from creditlab import data, policy as P, evaluate as E
from creditlab.config import RAW

pytestmark = pytest.mark.skipif(not (RAW / "kz_demandelasts_aer08.dta").exists(), reason="data not present")


@pytest.fixture(scope="module")
def synth():
    s = data.pricing_sample(data.load_aer())
    rng = np.random.default_rng(0)
    s = s.sample(12000, random_state=1).reset_index(drop=True)
    high_score = (s["appscore"] > s["appscore"].median()).to_numpy()
    A = s["arm"].to_numpy()
    # true expected revenue: base level from the client, plus an arm effect whose direction depends on the segment
    base = 20 + 0.02 * s["lastamount"].to_numpy()
    slope = np.where(high_score, -4.0, 4.0)            # high scores: cheaper arms pay; low scores: dearer arms pay
    def truth(a):
        return base + slope * (a - 2)
    s["revenue"] = truth(A) + rng.normal(0, 25, len(s))
    s["past_due"] = 0.0
    s["balance_months"] = 0.0
    s["loans"] = 0.0
    return s, truth, high_score


def test_dr_recovers_the_true_value_of_fixed_rate_cards(synth):
    s, truth, _ = synth
    G = P.crossfit_scores(s, 5, np.random.default_rng(2))
    w = np.array([1.0, 0, 0, 0])
    for a in range(5):
        est = (G[:, a, :] @ w).mean()
        true = truth(np.full(len(s), a)).mean()
        assert abs(est - true) < 1.5, (a, est, true)


def test_policy_tree_finds_the_planted_segment(synth):
    s, truth, high_score = synth
    G = P.crossfit_scores(s, 5, np.random.default_rng(3))
    gp = G @ np.array([1.0, 0, 0, 0])
    m = (s["risk"] == "HIGH").to_numpy()
    sub = s[m].reset_index(drop=True)
    rule = P.learn_tree(sub, gp[m], np.array([0, 1, 2, 3, 4]), "HIGH")
    got = rule.assign(sub)
    best = np.where(high_score[m], 0, 4)
    assert (got == best).mean() > 0.8


def test_honest_evaluation_shows_the_gain_out_of_sample(synth):
    s, truth, high_score = synth
    ev = E.run(s, n_splits=2, B=50, econ=P.ECON.__class__(loss_per_rand_past_due=0, cost_of_funds_monthly=0,
                                                          avg_balance_factor=0, cost_per_loan=0))
    w = np.array([1.0, 0, 0, 0])
    v = (ev.estimates @ w).mean(0)
    names = list(ev.names)
    # segment rules beat any single rate when the right rate depends on the segment
    assert v[names.index("segment rules (policy tree)")] > v[names.index("best rate per risk class")] + 3
