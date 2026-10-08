import numpy as np
import pandas as pd
import pytest

from creditlab import data, policy as P, pnl, replicate
from creditlab.config import RAW, ECON, Economics, STATUS_QUO_ARM

HAVE_AER = (RAW / "kz_demandelasts_aer08.dta").exists()
HAVE_ECTA = (RAW / "kz_observing_unobservables_ecma09.dta").exists()
needs_aer = pytest.mark.skipif(not HAVE_AER, reason="AER data not present")


@pytest.fixture(scope="module")
def aer():
    return data.load_aer()


# ---------------------------------------------------------------- data and replication
@needs_aer
def test_replication_matches_every_published_number(aer):
    rep = replicate.aer(aer)
    assert rep["match"].all(), rep.loc[~rep["match"], ["estimate", "ours", "published"]]


@needs_aer
def test_pricing_sample_is_the_papers_revenue_sample(aer):
    s = data.pricing_sample(aer)
    assert len(s) == 31_231                                   # N of Table 5, column 1
    assert (s["offer4"] == s["final4"]).all() and (s["gap"] <= 0).all()


@needs_aer
def test_loader_rejects_corrupted_data(aer, tmp_path):
    bad = pd.read_stata(RAW / "kz_demandelasts_aer08.dta", convert_categoricals=False)
    bad.loc[0, "final4"] = bad.loc[0, "offer4"] + 1           # contract rate above the offer
    p = tmp_path / "bad.dta"
    bad.to_stata(p, write_index=False)
    with pytest.raises(data.DataError, match="contract rate above offer rate"):
        data.load_aer(p)


@pytest.mark.skipif(not HAVE_ECTA, reason="Econometrica data not present")
def test_ecta_table_has_three_randomised_regressors():
    t = replicate.ecta(data.load_ecta())
    assert set(t["regressor"]) == {"offer rate (selection)", "contract rate (hazard)", "dynamic incentive"}
    assert (t["n"] == 4348).all()


# ---------------------------------------------------------------- estimator mechanics
def _toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    s = pd.DataFrame({"risk": rng.choice(["LOW", "MEDIUM", "HIGH"], n), "wave": rng.integers(1, 4, n),
                      "arm": rng.integers(0, 5, n), "branch": rng.choice([f"b{i}" for i in range(30)], n)})
    for c in P.COMPONENTS:
        s[c] = rng.gamma(2.0, 10.0, n)
    return s


def test_propensities_are_probabilities_and_match_the_design():
    s = _toy()
    e = P.propensities(s)
    assert np.allclose(e.sum(1), 1.0)
    cell = (s["risk"] == "LOW") & (s["wave"] == 1)
    assert np.allclose(e[cell.to_numpy(), 2], (s.loc[cell, "arm"] == 2).mean())


def test_dr_with_zero_outcome_model_is_ipw():
    s = _toy()
    e = P.propensities(s)
    G = P.dr_scores(s, np.zeros((len(s), 5, len(P.COMPONENTS))), e)
    a = 3
    ipw = ((s["arm"] == a).to_numpy()[:, None] * s[list(P.COMPONENTS)].to_numpy() / e[:, a][:, None]).mean(0)
    assert np.allclose(G[:, a, :].mean(0), ipw)


def test_dr_with_perfect_outcome_model_has_no_correction():
    s = _toy()
    e = P.propensities(s)
    Y = s[list(P.COMPONENTS)].to_numpy()
    mu = np.repeat(Y[:, None, :], 5, axis=1)                 # predicts the observed value for every arm
    G = P.dr_scores(s, mu, e)
    assert np.allclose(G, mu)


def test_profit_weights_follow_the_assumptions():
    w = P.profit_weights(Economics(loss_per_rand_past_due=0.8, cost_of_funds_monthly=0.01,
                                   avg_balance_factor=0.5, cost_per_loan=40.0))
    assert np.allclose(w, [1.0, -0.8, -0.005, -40.0])


def test_rule_assigns_by_thresholds_and_handles_missing_values():
    r = P.Rule("HIGH", ("appscore", 30.0), (None, None, 1, 1), ("trcount", 3.0, 2, 4))
    X = pd.DataFrame({"appscore": [10.0, 40.0, 40.0, np.nan], "trcount": [9.0, 1.0, 9.0, 1.0]})
    assert list(r.assign(X)) == [1, 2, 4, 1]                  # missing appscore goes left


def test_standard_card_assigns_the_status_quo_arm():
    s = _toy()
    assert (P.assign("standard rate card", s, None) == STATUS_QUO_ARM).all()


# ---------------------------------------------------------------- campaign P&L
def test_campaign_is_paired_and_scales_with_offers():
    rng = np.random.default_rng(0)
    draws = rng.gamma(2.0, 10.0, size=(500, 2, len(P.COMPONENTS)))
    draws[:, 1] = draws[:, 0]                                  # second policy identical to the first
    names = ("standard rate card", "copy")
    mc = pnl.campaign(draws, names, offers=1000, n=2000)
    assert np.allclose(mc["copy"], mc["standard rate card"])  # same draw for both policies
    mc2 = pnl.campaign(draws, names, offers=2000, n=2000)
    assert np.allclose(mc2["copy"], 2 * mc["copy"])


def test_assumption_draws_stay_in_their_ranges():
    W, th = pnl.draw_weights(5000, np.random.default_rng(1))
    for k, (lo, hi) in ECON.ranges.items():
        assert th[k].between(lo, hi).all()
    assert (W[:, 0] == 1).all() and (W[:, 1:] <= 0).all()
