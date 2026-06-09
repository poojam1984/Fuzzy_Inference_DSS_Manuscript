# Extended Methodology Notes

This document complements the published manuscript with implementation-level
details that did not fit in the journal page budget.

## 1. Membership function calibration

The (a, b, c) triplets in `src/membership_functions.py` were anchored on
the 10th, 50th and 90th percentiles of the 421-respondent survey reported in
the companion paper (Mishra et al., 2026, ECAM submission). Linguistic-term
labels and overlap zones follow standard practice for triangular MFs in
construction-management fuzzy DSS applications (e.g. Hanna and Senouci, 1995;
Pham et al., 2024).

The pilot calibration and any expert review of MF shapes documented in the
first author's PhD thesis (Mishra, 2025) supersede this code-level
approximation; the values shipped here are the manuscript-version values
intended for reproducibility, not for direct industrial deployment.

## 2. Rule weights

R1 and R2 are encoded as veto rules with negative weight. In scikit-fuzzy
this is implemented by mapping a negative weight to the `unsuitable`
consequent — the effect is that whenever safety is poor or height is
insufficient, the output is dominated by the unsuitable membership
regardless of the other inputs.

Positive weights (R3-R8) follow Saaty's 1-9 scale, normalised to [0.4, 1.0]
for compatibility with scikit-fuzzy's rule-weighting API.

## 3. Defuzzification

Centroid (centre-of-gravity) defuzzification is used throughout:

\[
y^* = \frac{\int y \cdot \mu_{out}(y)\,dy}{\int \mu_{out}(y)\,dy}
\]

The output universe is discretised at 0.5-unit resolution over [0, 100],
which is well below the smallest membership-function support width.

## 4. Sensitivity analysis

`src/sensitivity.py` perturbs each rule weight by +/-20% one at a time
(local one-at-a-time, OAT) while holding the others at baseline. The
per-rule swing reported in the manuscript Table 4 is the mean across the
three case studies. A global Sobol' sensitivity (1 000 Monte-Carlo runs)
gives the same rank ordering but requires the `SALib` package.

## 5. Validation status

The FDSS is presented in the manuscript as a **methodological proof-of-
concept**. The case-study inputs in `data/case_studies.csv` are illustrative
parameter vectors used to demonstrate the inference engine end-to-end —
they are not field-measured project data. No independent expert panel has
been elicited to provide ground-truth rankings for these cases in this
reproducibility package.

Formal empirical validation — against either an independent expert-panel
consensus or against outcomes from completed high-rise projects — is
identified in the paper as the most important direction for follow-up work.

Face validity is supported by:

1. MF anchoring on the 421-respondent industry survey (Mishra et al., 2026)
2. Rule-base structure consistent with established formwork-selection
   literature (Hanna and Senouci, 1995; Jarkas, 2012; Pham et al., 2024)
3. Sensitivity analysis (Section 4) showing that the ranking is stable under
   ±20% rule-weight perturbation

## 6. AHP / TOPSIS comparison (illustrative)

Implemented in `src/benchmark_ahp.py` and `src/benchmark_topsis.py`. These
scripts apply AHP (with an illustrative Saaty pairwise matrix, CR < 0.10)
and TOPSIS (with entropy weights derived from the decision matrix itself)
to the same six formwork alternatives. The purpose is to highlight
methodological differences in how the three methods handle linguistic
inputs — not to provide a head-to-head empirical benchmark against ground
truth, which the repository does not contain.

## 7. Reproducibility

- Python 3.10+ (tested on 3.10, 3.11, 3.12)
- `numpy==1.26`, `scipy==1.11`, `scikit-fuzzy==0.4.2`, `pandas==2.1`
- All random seeds fixed (`np.random.seed(42)`)
- Tested on Linux (Ubuntu 22.04), macOS (Sonoma) and Windows 11
