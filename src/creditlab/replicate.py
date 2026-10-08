"""Reproduce the published headline estimates before building anything on the data.

AER 2008 numbers are compared with the printed tables. The Econometrica 2009 table is reproduced from
the authors' own Stata code (same specification); its printed values were not available to check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CONTROLS = "low + med + waved2 + waved3"

# printed values: Karlan & Zinman (2008), AER 98(3), Tables 3 to 5
PUBLISHED = {
    "T3 c1 applied, offers at or below standard (probit ME)": (-0.00289, 0.00047, 53178),
    "T4 c1 loan size, unconditional (OLS)": (-4.368, 1.093, 31231),
    "T4 c3 loan size, borrowers (OLS)": (-25.876, 12.994, 2325),
    "T5 c1 gross interest revenue (OLS)": (2.553, 0.438, 31231),
    "T5 c2 average past due, borrowers (OLS)": (12.161, 3.523, 2325),
    "T3 c3 applied, offers above standard (probit ME)": (-0.01723, 0.00160, 632),
}


def _groups(d: pd.DataFrame) -> np.ndarray:
    return pd.factorize(d["branchuse"])[0]


def _probit_me(df: pd.DataFrame, y: str, mask, controls: str = CONTROLS):
    d = df[mask]
    m = smf.probit(f"{y} ~ offer4 + {controls}", d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": _groups(d)})
    me = m.get_margeff(at="mean")
    i = list(me.summary_frame().index).index("offer4")
    return float(me.margeff[i]), float(me.margeff_se[i]), int(m.nobs)


def _ols(df: pd.DataFrame, y: str, mask, rhs: str = f"offer4 + {CONTROLS}", coef: str = "offer4"):
    d = df[mask]
    m = smf.ols(f"{y} ~ {rhs}", d).fit(cov_type="cluster", cov_kwds={"groups": _groups(d)})
    return float(m.params[coef]), float(m.bse[coef]), int(m.nobs)


def aer(df: pd.DataFrame) -> pd.DataFrame:
    below = df["normrate_less"] == 1
    same = df["offer4"] == df["final4"]
    took = df["tookup"] == 1
    ours = {
        "T3 c1 applied, offers at or below standard (probit ME)": _probit_me(df, "applied", below),
        "T4 c1 loan size, unconditional (OLS)": _ols(df, "loansize", below & same),
        "T4 c3 loan size, borrowers (OLS)": _ols(df, "loansize", below & same & took),
        "T5 c1 gross interest revenue (OLS)": _ols(df, "grossinterest", below & same),
        "T5 c2 average past due, borrowers (OLS)": _ols(df, "pstdue_average", below & same & took),
    }
    rows = []
    for k, (b, se, n) in ours.items():
        pb, pse, pn = PUBLISHED[k]
        rows.append({"estimate": k, "ours": b, "ours_se": se, "ours_n": n, "published": pb, "published_se": pse,
                     "published_n": pn, "match": bool(np.isclose(b, pb, atol=5e-4 * max(1, abs(pb))) and n == pn)})
    # above the standard rate (first wave only, so the wave dummies drop out)
    k = "T3 c3 applied, offers above standard (probit ME)"
    b, se, n = _probit_me(df, "applied", df["normrate_less"] == 0, controls="low + med")
    pb, pse, pn = PUBLISHED[k]
    rows.append({"estimate": k, "ours": b, "ours_se": se, "ours_n": n, "published": pb, "published_se": pse,
                 "published_n": pn, "match": bool(np.isclose(b, pb, atol=5e-4 * max(1, abs(pb))) and n == pn)})
    return pd.DataFrame(rows)


def ecta(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1: repayment on offer rate (selection), contract rate (hazard) and the dynamic incentive."""
    b = df[df["borrowed"] == 1].copy()
    b["g"] = _groups(b)
    rows = []
    labels = {"pstdue_perc_average": "monthly average proportion past due", "cdl_pos_average": "share of months in arrears",
              "badacct_last": "account in collection", "Sindex": "summary index"}
    for y, lab in labels.items():
        m = smf.ols(f"{y} ~ offer4 + final4 + yearlong + C(risk) + C(wave) + C(branchuse)", b).fit(
            cov_type="cluster", cov_kwds={"groups": b["g"]})
        for c, name in (("offer4", "offer rate (selection)"), ("final4", "contract rate (hazard)"), ("yearlong", "dynamic incentive")):
            rows.append({"outcome": lab, "regressor": name, "coef": float(m.params[c]), "se": float(m.bse[c]),
                         "p": float(m.pvalues[c]), "n": int(m.nobs), "mean_y": float(b[y].mean())})
    return pd.DataFrame(rows)
