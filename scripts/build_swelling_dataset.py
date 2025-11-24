"""

Computes approximate lattice swelling, density and Upper cutoff voltage features for LMFP compositions.

- uses fixed MP IDs for olivine endmembers (LFP, LMP, FePO4, MnPO4)

"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from mp_api.client import MPRester

# --- Constants ---
OLIVINE_SG = "Pnma"
OLIVINE_FIXED_IDS = {
    "LFP": "mp-19017",
    "LMP": "mp-18981",
    "FePO4": "mp-18990",
    "MnPO4": "mp-18974"
}


def fetch_endmembers(mpr, sg_symbol=OLIVINE_SG):
    
    """Fetch the four olivine endmembers"""
    print(f"Searching for olivine endmembers ({sg_symbol})...")
    tags = ["LFP", "LMP", "FePO4", "MnPO4"]
    formulas = ["LiFePO4", "LiMnPO4", "FePO4", "MnPO4"]

    results, missing = {}, []
    for tag, formula in zip(tags, formulas):
        q = mpr.summary.search(formula=formula, fields=["material_id", "formula_pretty", "structure", "symmetry"])
        q = [r for r in q if (r.symmetry and getattr(r.symmetry, "symbol", None) == sg_symbol)]
        if not q:
            missing.append(tag)
        else:
            results[tag] = q[0]

    if missing:
        print(f"Missing {len(missing)} endmembers: {', '.join(missing)}")
        print("Using fixed MP IDs instead...")
        for tag in missing:
            mpid = OLIVINE_FIXED_IDS.get(tag)
            if not mpid:
                raise RuntimeError(f"No fixed MP ID for {tag}")
            try:
                res = mpr.summary.search(material_ids=[mpid], fields=["material_id", "formula_pretty", "structure", "symmetry"])
                if res:
                    results[tag] = res[0]
                    print(f"Loaded {tag} from {mpid}")
                else:
                    raise RuntimeError(f"MP ID {mpid} not found.")
            except Exception as e:
                raise RuntimeError(f"Failed fetching {tag} ({mpid}): {e}")

    print("All 4 endmembers available.")
    return results


def compute_swelling(V_lithiated, V_delithiated):
    """Percent volume change between lithiated & delithiated phases."""
    return ((V_lithiated - V_delithiated) / V_delithiated) * 100.0


def estimate_upper_cutoff_voltage(x_Fe):
    
    """Approximate U_cut as function of Fe content (simple linear heuristic)."""
    return 4.3 - 0.2 * x_Fe  # 4.3V for pure Mn and ~4.1V for pure Fe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="CSV with formulas (column: formula, x_Fe optional)")
    ap.add_argument("--out", required=True, help="Output CSV file")
    args = ap.parse_args()

    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        raise SystemExit("Error: MP_API_KEY not set")

    df = pd.read_csv(args.inp)
    if "x_Fe" not in df.columns:
        # infer from formula 
        df["x_Fe"] = df["formula"].str.extract(r"Fe(\d\.\d+)").astype(float)

    with MPRester(api_key) as mpr:
        endm = fetch_endmembers(mpr, sg_symbol=OLIVINE_SG)

        # Get reference volumes and densities
        ref_data = {}
        for tag, rec in endm.items():
            try:
                ref_data[tag] = {
                    "V": rec.structure.volume,
                    "density": rec.structure.density,
                }
            except Exception:
                ref_data[tag] = {"V": np.nan, "density": np.nan}

        V_lith_Fe, V_lith_Mn = ref_data["LFP"]["V"], ref_data["LMP"]["V"]
        V_delith_Fe, V_delith_Mn = ref_data["FePO4"]["V"], ref_data["MnPO4"]["V"]

        # Interpolate along composition axis
        df["V_lithiated"] = df["x_Fe"] * V_lith_Fe + (1 - df["x_Fe"]) * V_lith_Mn
        df["V_delithiated"] = df["x_Fe"] * V_delith_Fe + (1 - df["x_Fe"]) * V_delith_Mn
        df["dV_percent"] = compute_swelling(df["V_lithiated"], df["V_delithiated"])
        df["density_g_cm3"] = df["x_Fe"] * ref_data["LFP"]["density"] + (1 - df["x_Fe"]) * ref_data["LMP"]["density"]
        df["U_cut_V"] = df["x_Fe"].apply(estimate_upper_cutoff_voltage)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved swelling dataset to {out_path}")


if __name__ == "__main__":
    main()
