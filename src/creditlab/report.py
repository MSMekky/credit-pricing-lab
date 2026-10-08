"""Figures and reports/results.md from the pipeline outputs."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import REPORTS, FIGURES, STANDARD_RATE, RISKS

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})
RISK_NAME = {"LOW": "Low risk", "MEDIUM": "Medium risk", "HIGH": "High risk"}


def fig_unit_economics() -> None:
    u = pd.read_csv(REPORTS / "unit_economics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=False)
    for j, r in enumerate(RISKS):
        for i, (metric, lab, scale) in enumerate([("take_up", "take-up rate", 100), ("profit", "profit per offer (rand)", 1)]):
            ax = axes[i, j]
            d = u[(u.risk == r) & (u.metric == metric)].sort_values("mean_rate")
            ax.fill_between(d.mean_rate, d.p05 * scale, d.p95 * scale, color=BLUE, alpha=0.15, lw=0)
            ax.plot(d.mean_rate, d.value * scale, "-o", color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1.5)
            ax.axvline(STANDARD_RATE[r], color=ORANGE, lw=1.4, ls=(0, (4, 3)))
            if metric == "profit":
                ax.axhline(0, color=MUTED, lw=0.8)
            if i == 0:
                ax.set_title(RISK_NAME[r], loc="left", fontsize=10.5)
                ax.text(STANDARD_RATE[r], ax.get_ylim()[1], " standard rate", color=MUTED, fontsize=8, va="top")
            if j == 0:
                ax.set_ylabel(lab + (" (%)" if metric == "take_up" else ""))
            if i == 1:
                ax.set_xlabel("offered monthly interest rate (%)")
    fig.suptitle("Deep discounts bring few extra borrowers and lose money; close to the standard rate the profit curve flattens",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.text(0.01, 0.005, "Means weighted by the randomisation design; bands are 90% intervals from a branch bootstrap. "
             "Offers whose contract rate equalled the offer rate.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIGURES / "unit_economics.png", dpi=160)
    plt.close(fig)


def fig_policies(res: dict) -> None:
    pv = pd.DataFrame(res["policy_values"])
    pv = pv[pv.policy != "standard rate card"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    y = np.arange(len(pv))
    for i, (_, r) in enumerate(pv.iterrows()):
        c = ORANGE if r.vs_standard < 0 else BLUE
        ax.plot([r.vs_standard_p05, r.vs_standard_p95], [i, i], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(r.vs_standard, i, "o", color=c, ms=8, mec=SURFACE, mew=2)
        ax.text(r.vs_standard_p95 + 0.4, i, f"{r.vs_standard:+.1f}  (P better = {r.p_beats_standard:.0%})",
                va="center", fontsize=9, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, pv.policy)
    ax.set_xlabel("profit per offer vs the standard rate card, rand (median and 90% interval, held-out branches)")
    ax.set_title("Nothing tested beats the standard rate card out of sample", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=ax.get_xlim()[1] + 9)
    fig.tight_layout()
    fig.savefig(FIGURES / "policy_values.png", dpi=160)
    plt.close(fig)


def fig_campaign() -> None:
    mc = pd.read_parquet(REPORTS / "campaign_draws.parquet")
    names = ["standard rate card", "best rate per risk class", "segment rules (policy tree)", "experiment (random rates)"]
    data = [mc[n].to_numpy() / 1e6 for n in names][::-1]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    parts = ax.violinplot(data, orientation="horizontal", showextrema=False, widths=0.8)
    for pc, n in zip(parts["bodies"], names[::-1]):
        pc.set_facecolor(BLUE if n == "standard rate card" else MUTED)
        pc.set_edgecolor(SURFACE)
        pc.set_alpha(0.8)
    for i, x in enumerate(data, start=1):
        q05, q50, q95 = np.quantile(x, [0.05, 0.5, 0.95])
        ax.plot([q05, q95], [i, i], color=INK, lw=1)
        ax.plot(q50, i, "o", color=INK, ms=5)
    ax.set_yticks(range(1, len(names) + 1), names[::-1])
    ax.set_xlabel("profit of a 50,000-offer mailing, million rand (statistical and assumption uncertainty)")
    ax.set_title("Campaign P&L under each rate card", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "campaign.png", dpi=160)
    plt.close(fig)


def fig_selection_hazard() -> None:
    p = REPORTS / "replication_ecta.csv"
    if not p.exists():
        return
    t = pd.read_csv(p)
    t = t[t.outcome.isin(["account in collection", "monthly average proportion past due"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.9), sharey=True)
    for ax, (o, d) in zip(axes, t.groupby("outcome", sort=False)):
        d = d.iloc[::-1]
        for i, (_, r) in enumerate(d.iterrows()):
            c = BLUE if r.regressor == "dynamic incentive" else MUTED
            ax.plot([r.coef - 1.645 * r.se, r.coef + 1.645 * r.se], [i, i], color=c, lw=2.2, solid_capstyle="round")
            ax.plot(r.coef, i, "o", color=c, ms=7, mec=SURFACE, mew=1.5)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(d)), d.regressor)
        ax.set_title(o, loc="left", fontsize=10)
        ax.set_xlabel("effect (rate effects per 1 pp a month; 90% CI)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Within the tested range, higher rates did not visibly worsen repayment; the repayment incentive did improve it",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURES / "selection_hazard.png", dpi=160)
    plt.close(fig)


def r(x, d=0):
    return f"{x:,.{d}f}"


def results_md(res: dict) -> str:
    L = ["# Results", "", f"Generated from `results.json` ({res['settings']['splits']} branch splits x "
         f"{res['settings']['bootstrap']} bootstrap draws, seed {res['settings']['seed']}). Money in 2003 South African rand.", ""]
    L += ["## 1. Replication (Karlan & Zinman 2008, AER)", "", "| estimate | ours | published | N | match |", "|---|---|---|---|---|"]
    for x in res["replication"]:
        L.append(f"| {x['estimate']} | {x['ours']:.5g} ({x['ours_se']:.3g}) | {x['published']:.5g} ({x['published_se']:.3g}) | {x['ours_n']:,} | {'yes' if x['match'] else 'NO'} |")
    if res["ecta"]:
        L += ["", "## 2. Selection and hazard (Karlan & Zinman 2009, Econometrica, Table 1 specification)", "",
              "Reproduced with the authors' specification; the printed table was not available to compare against.", "",
              "| outcome | regressor | coefficient | s.e. | p |", "|---|---|---|---|---|"]
        for x in res["ecta"]:
            L.append(f"| {x['outcome']} | {x['regressor']} | {x['coef']:.4f} | {x['se']:.4f} | {x['p']:.3f} |")
    L += ["", "## 3. Unit economics by price arm", "", "See `unit_economics.csv` (take-up, loan size, revenue, past due, profit per offer, with 90% intervals).", ""]
    a = pd.read_csv(REPORTS / "above_standard.csv")
    L += ["Wave 1, the only wave that tested rates above the standard:", "", "| risk | side | offers | applied | took up | mean rate |", "|---|---|---|---|---|---|"]
    for _, x in a.iterrows():
        L.append(f"| {x.risk} | {x.side} | {x.offers:,} | {x.applied:.1%} | {x.took_up:.1%} | {x.mean_rate:.2f}% |")
    L += ["", "## 4. Rate cards, evaluated on held-out branches", "",
          "| policy | profit per offer | 90% interval | vs standard | 90% interval | P(beats standard) |", "|---|---|---|---|---|---|"]
    for x in res["policy_values"]:
        pb = "" if x["p_beats_standard"] != x["p_beats_standard"] else f"{x['p_beats_standard']:.0%}"
        L.append(f"| {x['policy']} | R{x['profit_per_offer']:.2f} | R{x['p05']:.2f} to R{x['p95']:.2f} | {x['vs_standard']:+.2f} | "
                 f"{x['vs_standard_p05']:+.2f} to {x['vs_standard_p95']:+.2f} | {pb} |")
    L += ["", "Best single arm per class, learned from all the data (out of sample it is not distinguishable from the standard card, see above):", ""]
    for k, v in res["recommended_rate_card"]["uniform"].items():
        L.append(f"- {k}: {v['label']}")
    L += ["", "Policy trees learned from all the data (shown for transparency; they did not hold up out of sample):", ""]
    for k, v in res["recommended_rate_card"]["trees"].items():
        L.append(f"- {v}")
    c = res["campaign"]
    L += ["", f"## 5. Campaign P&L ({c['offers']:,} offers)", "",
          "| policy | median profit | 90% interval | vs standard (median) | 90% interval | P(beats standard) |", "|---|---|---|---|---|---|"]
    for x in c["summary"]:
        pb = "" if x["p_beats_standard"] != x["p_beats_standard"] else f"{x['p_beats_standard']:.0%}"
        L.append(f"| {x['policy']} | R{r(x['profit_p50'])} | R{r(x['profit_p05'])} to R{r(x['profit_p95'])} | {x['vs_standard_p50']:+,.0f} | "
                 f"{x['vs_standard_p05']:+,.0f} to {x['vs_standard_p95']:+,.0f} | {pb} |")
    L += ["", "Assumptions (defaults; ranges used in the Monte Carlo):", ""]
    A = res["assumptions"]
    for k, (lo, hi) in A["ranges"].items():
        L.append(f"- {k}: {A[k]} ({lo} to {hi})")
    L += ["", "Sensitivity of the best-rate-per-class advantage to each assumption (campaign, rand):", "",
          "| assumption | at low end | at default | at high end |", "|---|---|---|---|"]
    for t in res["tornado"]:
        L.append(f"| {t['assumption']} | {t['advantage_at_low']:+,.0f} | {t['advantage_at_default']:+,.0f} | {t['advantage_at_high']:+,.0f} |")
    L += ["", "How often each arm was chosen as the single best rate, across splits:", "", "| risk | arm | share of splits |", "|---|---|---|"]
    for x in res["rate_choice_stability"]:
        L.append(f"| {x['risk']} | {x['arm']} | {x['share_of_splits']:.0%} |")
    return "\n".join(L) + "\n"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    res = json.loads((REPORTS / "results.json").read_text())
    fig_unit_economics()
    fig_policies(res)
    fig_campaign()
    fig_selection_hazard()
    (REPORTS / "results.md").write_text(results_md(res))


if __name__ == "__main__":
    main()
