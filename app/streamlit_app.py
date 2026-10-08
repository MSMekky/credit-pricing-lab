"""Rate-card simulator.  Run:  streamlit run app/streamlit_app.py"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from creditlab.config import ARM_LABELS, RISKS, STANDARD_RATE, STATUS_QUO_ARM, ECON  # noqa: E402
from creditlab import pnl  # noqa: E402

st.set_page_config(page_title="Credit rate-card simulator", layout="wide")


@st.cache_data
def load():
    z = np.load(ROOT / "reports/dr_scores.npz", allow_pickle=True)
    rec = json.loads((ROOT / "reports/results.json").read_text())["recommended_rate_card"]["uniform"]
    return {k: z[k] for k in z.files}, {r: v["arm"] for r, v in rec.items()}


z, recommended = load()
G, risk, branch = z["G"].astype(float), z["risk"], z["branch"]
mean_rate, allowed = z["mean_rate"], z["allowed"]
codes, uniq = pd.factorize(branch)

st.title("Credit rate-card simulator")
st.caption("Built on a randomised interest-rate experiment: 31,231 consumer-loan offers from a South African lender "
           "(Karlan & Zinman 2008). Every value is a doubly robust off-policy estimate against the lender's standard rates.")

with st.sidebar:
    st.header("Rate card")
    st.caption("Starts at the card the analysis recommends. Set every class to 'std -1 to 0' for the standard card.")
    choice = {}
    for i, r in enumerate(RISKS):
        opts = [a for a in range(len(ARM_LABELS)) if allowed[i, a]]
        labels = {a: f"{ARM_LABELS[a]} (avg {mean_rate[i, a]:.2f}%)" for a in opts}
        choice[r] = st.selectbox(f"{r.title()} risk (standard {STANDARD_RATE[r]}% a month)", opts,
                                 index=opts.index(recommended.get(r, STATUS_QUO_ARM)), format_func=labels.get)
    st.header("Assumptions")
    loss = st.slider("Share of past-due balance lost", 0.4, 1.6, ECON.loss_per_rand_past_due, 0.05)
    cof = st.slider("Cost of funds, % a year", 6.0, 24.0, ECON.cost_of_funds_monthly * 1200, 0.5)
    fee = st.slider("Cost per loan, rand", 0.0, 200.0, ECON.cost_per_loan, 5.0)
    offers = st.number_input("Offers in the mailing", 1000, 1_000_000, 50_000, 1000)
    B = st.select_slider("Bootstrap draws", [200, 500, 1000], 500)

w = np.array([1.0, -loss, -(cof / 1200) * ECON.avg_balance_factor, -fee])
gp = G @ w                                                   # offers x arms
arm_new = np.array([choice[r] for r in risk])
idx = np.arange(len(gp))
v_new, v_std = gp[idx, arm_new], gp[:, STATUS_QUO_ARM]

nb = len(uniq)
s_new, s_std = np.zeros(nb), np.zeros(nb)
np.add.at(s_new, codes, v_new)
np.add.at(s_std, codes, v_std)
cnt = np.bincount(codes, minlength=nb).astype(float)
rng = np.random.default_rng(0)
picks = rng.integers(0, nb, size=(B, nb))
b_new = s_new[picks].sum(1) / cnt[picks].sum(1)
b_std = s_std[picks].sum(1) / cnt[picks].sum(1)
diff = (b_new - b_std) * offers

k = lambda x: f"R{x/1e3:,.0f}k"
c1, c2, c3 = st.columns(3)
c1.metric("Campaign profit, your card", k(v_new.mean() * offers))
c1.caption(f"90% interval {k(np.quantile(b_new, .05) * offers)} to {k(np.quantile(b_new, .95) * offers)}")
c2.metric("Difference vs standard card", k(diff.mean()))
c2.caption(f"90% interval {k(np.quantile(diff, .05))} to {k(np.quantile(diff, .95))}")
p_better = (diff > 0).mean()
c3.metric("Chance your card is better", "<1%" if 0 < p_better < 0.005 else (">99%" if 0.995 < p_better < 1 else f"{p_better:.0%}"))

st.subheader("By risk class")
rows = []
for i, r in enumerate(RISKS):
    m = risk == r
    rows.append({"risk class": r, "offers in data": int(m.sum()), "your arm": ARM_LABELS[choice[r]],
                 "avg rate": f"{mean_rate[i, choice[r]]:.2f}%",
                 "profit per offer": f"R{v_new[m].mean():.2f}", "standard card": f"R{v_std[m].mean():.2f}"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

if all(choice[r] == STATUS_QUO_ARM for r in RISKS):
    st.info("This is the standard rate card, so there is nothing to compare against.")
else:
    counts, edges = np.histogram(diff / 1e3, bins=40)
    st.subheader("Difference vs standard card, rand thousand (bootstrap over branches)")
    st.bar_chart(pd.DataFrame({"draws": counts}, index=np.round((edges[:-1] + edges[1:]) / 2, 0)))
    st.caption("In-sample view on all branches. The held-out evaluation in the README is the honest test of a learned card.")

with st.expander("Method and limits"):
    st.markdown((ROOT / "reports/results.md").read_text())
