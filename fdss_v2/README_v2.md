# FDSS v2 — reproducibility package for the revised manuscript

```
pip install numpy scipy pandas matplotlib scikit-fuzzy pytest
python fdss_v2/run_all.py     # writes fdss_v2/results/* (all numbers in the manuscript; seed 42, ~90 s)
python fdss_v2/figures.py     # writes fdss_v2/figures/*.png (300 dpi)
pytest fdss_v2/test_fdss.py
```

* `fdss.py` – hierarchical Mamdani-type FDSS (20 + 9 + 9 rules), Ruspini membership partitions, Larsen product implication,
  max aggregation, centroid defuzzification, optional rule weights, "don't-care" handling of missing inputs.
* `cases.py` / `case_studies.csv` – **illustrative** scenarios (not field data). Safety, ease, reuse and cost ratings are those of the
  original CSV; height capabilities per system and required heights are illustrative assumptions.
* `benchmarks.py` – AHP (absolute measurement) and TOPSIS on the same decision matrix (illustrative Saaty matrix, not elicited).
* `run_all.py` – baseline scores, verification checks, decision surface, Monte-Carlo sensitivity, AHP/TOPSIS comparison, missing-data experiment.
Licence: to be set by the authors before release.
