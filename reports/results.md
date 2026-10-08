# Results

Generated from `results.json` (20 branch splits x 200 bootstrap draws, seed 11). Money in 2003 South African rand.

## 1. Replication (Karlan & Zinman 2008, AER)

| estimate | ours | published | N | match |
|---|---|---|---|---|
| T3 c1 applied, offers at or below standard (probit ME) | -0.0028884 (0.000467) | -0.00289 (0.00047) | 53,178 | yes |
| T4 c1 loan size, unconditional (OLS) | -4.368 (1.09) | -4.368 (1.09) | 31,231 | yes |
| T4 c3 loan size, borrowers (OLS) | -25.876 (13) | -25.876 (13) | 2,325 | yes |
| T5 c1 gross interest revenue (OLS) | 2.5527 (0.438) | 2.553 (0.438) | 31,231 | yes |
| T5 c2 average past due, borrowers (OLS) | 12.161 (3.52) | 12.161 (3.52) | 2,325 | yes |
| T3 c3 applied, offers above standard (probit ME) | -0.017232 (0.0016) | -0.01723 (0.0016) | 632 | yes |

## 2. Selection and hazard (Karlan & Zinman 2009, Econometrica, Table 1 specification)

Reproduced with the authors' specification; the printed table was not available to compare against.

| outcome | regressor | coefficient | s.e. | p |
|---|---|---|---|---|
| monthly average proportion past due | offer rate (selection) | 0.0045 | 0.0031 | 0.149 |
| monthly average proportion past due | contract rate (hazard) | 0.0048 | 0.0033 | 0.148 |
| monthly average proportion past due | dynamic incentive | -0.0189 | 0.0105 | 0.070 |
| share of months in arrears | offer rate (selection) | 0.0017 | 0.0035 | 0.619 |
| share of months in arrears | contract rate (hazard) | 0.0063 | 0.0034 | 0.066 |
| share of months in arrears | dynamic incentive | -0.0277 | 0.0110 | 0.011 |
| account in collection | offer rate (selection) | 0.0073 | 0.0047 | 0.120 |
| account in collection | contract rate (hazard) | 0.0012 | 0.0047 | 0.795 |
| account in collection | dynamic incentive | -0.0245 | 0.0118 | 0.037 |
| summary index | offer rate (selection) | 0.0150 | 0.0114 | 0.190 |
| summary index | contract rate (hazard) | 0.0143 | 0.0112 | 0.201 |
| summary index | dynamic incentive | -0.0797 | 0.0320 | 0.013 |

## 3. Unit economics by price arm

See `unit_economics.csv` (take-up, loan size, revenue, past due, profit per offer, with 90% intervals).

Wave 1, the only wave that tested rates above the standard:

| risk | side | offers | applied | took up | mean rate |
|---|---|---|---|---|---|
| HIGH | above standard | 457 | 4.4% | 3.3% | 13.51% |
| HIGH | at or below standard | 2,686 | 6.6% | 5.0% | 7.92% |
| LOW | above standard | 111 | 14.4% | 11.7% | 9.98% |
| LOW | at or below standard | 938 | 17.6% | 14.8% | 5.79% |
| MEDIUM | above standard | 64 | 9.4% | 6.2% | 12.36% |
| MEDIUM | at or below standard | 702 | 22.8% | 18.2% | 6.94% |

## 4. Rate cards, evaluated on held-out branches

| policy | profit per offer | 90% interval | vs standard | 90% interval | P(beats standard) |
|---|---|---|---|---|---|
| standard rate card | R26.29 | R14.68 to R36.78 | +0.00 | +0.00 to +0.00 |  |
| experiment (random rates) | R18.08 | R12.95 to R22.62 | -8.16 | -15.63 to -0.29 | 4% |
| best rate per risk class | R26.82 | R17.80 to R36.52 | +2.44 | -9.21 to +9.34 | 61% |
| segment rules (policy tree) | R21.36 | R12.75 to R32.95 | -3.48 | -13.41 to +5.45 | 30% |

Best single arm per class, learned from all the data (out of sample it is not distinguishable from the standard card, see above):

- LOW: std -1 to 0
- MEDIUM: std -3.5 to -2
- HIGH: std -2 to -1

Policy trees learned from all the data (shown for transparency; they did not hold up out of sample):

- LOW: appscore <= 33 -> [if age <= 33.7467: std -2 to -1, else std -1 to 0]; above -> [if lastamount <= 1400: std -1 to 0, else std -5 to -3.5]
- MEDIUM: itcscore <= 572 -> [if grossincome <= 2.8188: std -5 to -3.5, else std -3.5 to -2]; above -> [if dependants <= 2: std -3.5 to -2, else std -2 to -1]
- HIGH: lastamount <= 1000 -> [if grossincome <= 2.94935: std -1 to 0, else std -2 to -1]; above -> [if appscore <= 27: std -2 to -1, else std -3.5 to -2]

## 5. Campaign P&L (50,000 offers)

| policy | median profit | 90% interval | vs standard (median) | 90% interval | P(beats standard) |
|---|---|---|---|---|---|
| standard rate card | R1,245,666 | R610,528 to R1,887,822 | +0 | +0 to +0 |  |
| experiment (random rates) | R846,320 | R442,089 to R1,250,748 | -393,640 | -788,949 to -23,849 | 4% |
| best rate per risk class | R1,271,678 | R754,987 to R1,869,517 | +65,831 | -454,921 to +467,151 | 60% |
| segment rules (policy tree) | R1,066,658 | R481,327 to R1,687,115 | -157,536 | -668,348 to +269,357 | 29% |

Assumptions (defaults; ranges used in the Monte Carlo):

- loss_per_rand_past_due: 1.0 (0.6 to 1.4)
- cost_of_funds_monthly: 0.0125 (0.0083 to 0.0167)
- avg_balance_factor: 0.55 (0.5 to 0.6)
- cost_per_loan: 50.0 (0.0 to 120.0)

Sensitivity of the best-rate-per-class advantage to each assumption (campaign, rand):

| assumption | at low end | at default | at high end |
|---|---|---|---|
| loss_per_rand_past_due | -16,297 | +45,284 | +106,865 |
| cost_of_funds_monthly | +53,023 | +45,284 | +37,546 |
| avg_balance_factor | +47,378 | +45,284 | +43,190 |
| cost_per_loan | +54,855 | +45,284 | +31,886 |

How often each arm was chosen as the single best rate, across splits:

| risk | arm | share of splits |
|---|---|---|
| LOW | 4 | 95% |
| LOW | 3 | 5% |
| MEDIUM | 2 | 95% |
| MEDIUM | 3 | 5% |
| HIGH | 3 | 80% |
| HIGH | 2 | 20% |
