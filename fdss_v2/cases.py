"""Illustrative scenarios (NOT field data). Ratings (safety, ease, reuse, cost) are carried over
from the original data/case_studies.csv; height capability per system and required heights are
illustrative assumptions introduced in v2 so that height adequacy = capability / required height."""
import pandas as pd
from pathlib import Path

REQUIRED_HEIGHT_M = {"A_10_storey": 35, "B_45_storey": 160, "C_90_storey": 350}
CAPABILITY_M = {"Fabric": 60, "Timber": 80, "Plastic": 90, "Aluminium": 420, "Steel": 600, "Tunnel": 200}
CASE_LABEL = {"A_10_storey": "10-storey (35 m)", "B_45_storey": "45-storey (160 m)", "C_90_storey": "90-storey (350 m)"}

def load():
    df = pd.read_csv(Path(__file__).with_name("case_studies.csv")).drop(columns=["height"])
    df["required_m"] = df["case"].map(REQUIRED_HEIGHT_M)
    df["capability_m"] = df["system"].map(CAPABILITY_M)
    df["height"] = (df["capability_m"] / df["required_m"]).clip(upper=3.0)
    df["height_raw"] = df["capability_m"] / df["required_m"]
    df["feasible"] = df["capability_m"] >= df["required_m"]
    return df
