# Data

## Karlan & Zinman (2008), AER: included

`raw/kz_demandelasts_aer08.dta`, 53,810 direct-mail loan offers with randomised monthly interest rates.

Karlan, Dean S., and Jonathan Zinman. 2008. Replication data for: Credit Elasticities in Less-Developed Economies: Implications for Microfinance. American Economic Association, Inter-university Consortium for Political and Social Research. https://doi.org/10.3886/E113240V1

Licence: data under CC BY 4.0, code under the Modified BSD licence (`raw/LICENSE_AEA_openICPSR_113240.txt`). File unchanged except for its name.

## Karlan & Zinman (2009), Econometrica: download it yourself

Used for the selection-versus-hazard analysis. Its licence does not clearly allow redistribution, so it is not in this repository.

1. Download the supplement zip from https://www.econometricsociety.org/publications/econometrica/2009/11/01/observing-unobservables-identifying-information-asymmetries (Supplemental Material, "data and programs").
2. Save `Karlan&Zinman_OU_ecma_replication.dta` as `data/raw/kz_observing_unobservables_ecma09.dta`.

Without it, the pipeline skips that section and its tests.
