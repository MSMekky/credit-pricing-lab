"""End to end: checks -> replication -> unit economics -> rate cards -> off-policy evaluation -> campaign P&L.

    python -m creditlab.pipeline            # full run (about 3 minutes)
    python -m creditlab.pipeline --quick    # 4 splits, for a fast check
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import curves, data, evaluate, pnl, policy as P, replicate
from .config import REPORTS, ECON, ARM_LABELS, RISKS, SEED

CAMPAIGN_OFFERS = 50_000


def final_rate_card(sample: pd.DataFrame, rng) -> tuple[dict, np.ndarray]:
    """The rate card recommended from all the data (5-fold cross-fitted scores), and those scores."""
    G = P.crossfit_scores(sample, 5, rng)
    pol = P.learn(sample, G, P.profit_weights(ECON), P.allowed_arms(sample))
    card = {"uniform": {r: {"arm": a, "label": ARM_LABELS[a]} for r, a in pol.uniform.items()},
            "trees": {r: t.describe() for r, t in pol.trees.items()}}
    return card, G


def main(n_splits: int = 20, B: int = 200) -> dict:
    t0 = time.time()
    REPORTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    aer = data.load_aer()
    sample = data.pricing_sample(aer)

    rep = replicate.aer(aer)
    rep.to_csv(REPORTS / "replication_aer.csv", index=False)
    try:
        ecta = replicate.ecta(data.load_ecta())
        ecta.to_csv(REPORTS / "replication_ecta.csv", index=False)
    except FileNotFoundError:
        ecta = None

    arms = curves.arm_means(sample)
    arms.to_csv(REPORTS / "unit_economics.csv", index=False)
    above = curves.above_standard(aer)
    above.to_csv(REPORTS / "above_standard.csv", index=False)

    ev = evaluate.run(sample, n_splits=n_splits, B=B)
    summ = evaluate.summary(ev)
    summ.to_csv(REPORTS / "policy_values.csv", index=False)
    evaluate.components_table(ev).to_csv(REPORTS / "policy_components.csv")
    stab = evaluate.stability(ev)
    stab.to_csv(REPORTS / "rate_choice_stability.csv", index=False)
    np.save(REPORTS / "policy_draws.npy", ev.draws())

    mc = pnl.campaign(ev.draws(), ev.names, CAMPAIGN_OFFERS)
    mc_sum = pnl.summarise(mc, ev.names)
    mc_sum.to_csv(REPORTS / "campaign_pnl.csv", index=False)
    mc.sample(5000, random_state=SEED).to_parquet(REPORTS / "campaign_draws.parquet", index=False)
    torn = pnl.tornado(ev.draws(), ev.names, "best rate per risk class", CAMPAIGN_OFFERS)
    torn.to_csv(REPORTS / "tornado.csv", index=False)

    card, G = final_rate_card(sample, rng)
    allowed = P.allowed_arms(sample)
    np.savez_compressed(REPORTS / "dr_scores.npz", G=G.astype(np.float32), risk=sample["risk"].to_numpy(),
                        branch=sample["branch"].to_numpy(), arm=sample["arm"].to_numpy(),
                        mean_rate=np.array([[sample.loc[(sample.risk == r) & (sample.arm == a), "offer4"].mean()
                                             if a in allowed[r] else np.nan for a in range(len(ARM_LABELS))] for r in RISKS]),
                        allowed=np.array([[a in allowed[r] for a in range(len(ARM_LABELS))] for r in RISKS]))
    results = {
        "sample": {"offers": int(len(aer)), "pricing_sample": int(len(sample)),
                   "branches": int(sample["branch"].nunique()), "risk_mix": sample["risk"].value_counts(normalize=True).to_dict()},
        "replication_all_match": bool(rep["match"].all()),
        "replication": rep.to_dict("records"),
        "ecta": None if ecta is None else ecta.to_dict("records"),
        "policy_values": summ.to_dict("records"),
        "campaign": {"offers": CAMPAIGN_OFFERS, "summary": mc_sum.to_dict("records")},
        "tornado": torn.to_dict("records"),
        "rate_choice_stability": stab.to_dict("records"),
        "recommended_rate_card": card,
        "assumptions": {k: getattr(ECON, k) for k in pnl.ASSUMPTIONS} | {"ranges": ECON.ranges},
        "settings": {"splits": n_splits, "bootstrap": B, "seed": SEED},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    r = main(n_splits=4, B=100) if a.quick else main()
    print(pd.DataFrame(r["policy_values"]).round(2).to_string(index=False))
    print(pd.DataFrame(r["campaign"]["summary"]).round(0).to_string(index=False))
    print(json.dumps(r["recommended_rate_card"], indent=1))
    print("runtime", r["runtime_seconds"], "s")
