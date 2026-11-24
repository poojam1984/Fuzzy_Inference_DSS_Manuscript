# FDSS-Formwork

A Mamdani fuzzy inference decision-support system (FDSS) for formwork and equipment selection in high-rise construction.

This repository contains the Python implementation of the FDSS described in:

> Mishra, P., Rajak, F. and Kumar, R. A Mamdani fuzzy inference decision-support system for formwork and equipment selection in high-rise construction. Civil Engineering and Architecture (Horizon Research Publishing), manuscript ID 14846492, under review.

Version 2.0 (revised submission). This version reproduces every case score, figure and sensitivity value reported in the revised manuscript.

## What the code does

- Five inputs: safety performance, ease of installation, height capability, reusability and operational cost.
- Height enters as the capability ratio rho = H_cap / H_req (system height capability divided by the project's required height), capped at 3.
- In rules R3, R4 and R7, "Height = Sufficient" is read as "at least Sufficient", and in R3 and R4 "(Ease/Reuse/Cost = Moderate)" is read as "at least Moderate".
- Triangular membership functions with the (a, b, c) parameters in Table 3 of the paper (`src/membership_functions.py`).
- Eight weighted IF-THEN rules from Table 4 (`src/rule_base.py`). R1 and R2 are veto rules with negative weights.
- Weighted Mamdani max-min inference and centroid defuzzification, following Equations (4)-(8) of the paper (`src/fdss.py`).
- One-at-a-time ±20% rule-weight sensitivity analysis, computed by the model rather than entered by hand (`src/sensitivity.py`).

The case-study inputs in `data/case_studies.csv` are illustrative values, not field-measured project data. No expert panel has ranked these cases. Reuse counts follow published typical values (see Appendix Table A1 of the paper); safety, ease and cost ratings and system height capabilities are the authors' illustrative assumptions. Inputs outside a variable's range are clipped to the range limit (e.g. 475 reuses is read as 260).

## Run

```bash
pip install -r requirements.txt
python -m src.fdss          # case-study scores
python -m src.sensitivity   # ±20% sensitivity table
cd figures && python make_figures.py && cd ..   # Figures 1, 5 and 6
pytest tests -q             # unit tests
```

Every result printed by these scripts is computed by the code. No result values are hard-coded.

## Data

`data/survey_top5.csv` lists the five top-ranked factors and their Relative Importance Index values from the authors' questionnaire survey (Mishra, 2025, Ph.D. thesis, NIT Patna, http://hdl.handle.net/10603/619616).

## License

MIT License. See `LICENSE`.

## Contact

Pooja Mishra (corresponding author), Department of Architecture and Planning, National Institute of Technology Patna, Bihar 800005, India. poojam.ph21@nitp.ac.in
