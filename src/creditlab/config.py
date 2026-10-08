"""Paths, design constants and economic assumptions in one place."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

AER_FILE = RAW / "kz_demandelasts_aer08.dta"                 # openICPSR 113240, CC BY 4.0
ECTA_FILE = RAW / "kz_observing_unobservables_ecma09.dta"    # Econometrica supplement 5781, fetched by the user

# the lender's standard four-month monthly rates by risk class (Karlan & Zinman 2008, Table 1)
STANDARD_RATE = {"LOW": 7.75, "MEDIUM": 9.75, "HIGH": 11.75}
RISKS = ("LOW", "MEDIUM", "HIGH")

# price arms: distance of the offered monthly rate from the risk class's standard rate (percentage points)
ARM_EDGES = (-99.0, -5.0, -3.5, -2.0, -1.0, 0.001)
ARM_LABELS = ("std -5 or more", "std -5 to -3.5", "std -3.5 to -2", "std -2 to -1", "std -1 to 0")
STATUS_QUO_ARM = 4          # the arm containing the standard rate
MIN_ARM_SHARE = 0.05        # an arm is in the policy space for a risk class only if it holds 5%+ of its offers

# features a pricing policy may use: credit and relationship variables only.
# Gender, marital status, location type and education are deliberately excluded.
POLICY_FEATURES = ("itcscore", "itczero", "appscore", "appscore0", "dormancy", "trcount",
                   "lastamount", "lastterm", "grossincome", "age", "dependants")
OUTCOME_FEATURES = POLICY_FEATURES + ("wave",)

SEED = 11


@dataclass(frozen=True)
class Economics:
    """Assumptions the data cannot supply. Every one is varied in the Monte Carlo."""
    loss_per_rand_past_due: float = 1.0   # share of average past-due balance that is ultimately lost
    cost_of_funds_monthly: float = 0.0125 # lender's monthly funding cost (about 16% a year)
    avg_balance_factor: float = 0.55      # average outstanding balance over the term, as a share of principal
    cost_per_loan: float = 50.0           # origination and servicing cost per loan, rand

    # ranges for the Monte Carlo (uniform)
    ranges: dict = field(default_factory=lambda: {
        "loss_per_rand_past_due": (0.6, 1.4),
        "cost_of_funds_monthly": (0.0083, 0.0167),   # 10% to 20% a year
        "avg_balance_factor": (0.50, 0.60),
        "cost_per_loan": (0.0, 120.0),
    })


ECON = Economics()
