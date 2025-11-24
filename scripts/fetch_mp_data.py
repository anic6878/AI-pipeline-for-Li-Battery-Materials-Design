"""
----------------
Fetches LMFP-relevant data (LiFePO4–LiMnPO4 solid solution) from Materials Project database.


- Falls back to fixed MP IDs for olivine endmembers if the search fails somehow
- Avoids RuntimeError on missing entries.
"""

import os
import argparse
from pathlib import Path
from mp_api.client import MPRester
import pandas as pd
import numpy as np

# Constant
OLIVINE_SG = "Pnma"

# --- PATCHING FOR OLIVINE END-MEMBERS ---
OLIVINE_FIXED_IDS = {
    "LFP": "mp-19017",   # LiFePO4
    "LMP": "mp-18981",   # LiMnPO4
    "FePO4": "mp-18990", # Delithiated Fe
    "MnPO4": "mp-18974"  # Delithiated Mn
}


def fetch_endmembers(mpr: MPRester, sg_symbol=OLIVINE_SG):
    
    """Fetching the four olivine endmembers from MP dataset or use fixed IDs."""
    
    tags = ["LFP", "LMP", "FePO4", "MnPO4"]
    formulas = ["LiFePO4", "LiMnPO4", "FePO4", "MnPO4"]

    results = {}
    missing = []

    for tag, formula in zip(tags, formulas):
        q = mpr.summary.search(formula=formula, fields=["material_id", "formula_pretty", "structure", "symmetry"])
        
        # Filter to desired spacegroup if possible
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
                    raise RuntimeError(f"MP ID {mpid} not found in Materials Project.")
            except Exception as e:
                raise RuntimeError(f"Failed fetching {tag} ({mpid}): {e}")

    print(f"Successfully fetched all 4 endmembers.")
    return results


def make_lmfp_vegard_grid(x_grid):
    """Generates simple Vegard interpolation grid for LMFP compositions"""
    grid = []
    for x in x_grid:
        formula = f"LiFe{x:.2f}Mn{1 - x:.2f}PO4"
        grid.append({"formula": formula, "x_Fe": round(x, 3)})
    return pd.DataFrame(grid)


def fetch_mp_summary(mpr: MPRester):
    """Fetching general summary data for LMFP-like compositions from MP"""
    print("Fetching summary data for LMFP system...")
    q = mpr.summary.search(
        chemsys=["Li-Fe-Mn-P-O"],
        fields=["material_id", "formula_pretty", "energy_above_hull", "band_gap", "density", "symmetry"]
    )
    df = pd.DataFrame([{
        "material_id": r.material_id,
        "formula": r.formula_pretty,
        "E_hull_meV": getattr(r, "energy_above_hull", None) * 1000 if getattr(r, "energy_above_hull", None) else None,
        "band_gap": getattr(r, "band_gap", None),
        "density": getattr(r, "density", None),
        "spacegroup": getattr(r.symmetry, "symbol", None) if getattr(r, "symmetry", None) else None
    } for r in q])
    print(f"Retrieved {len(df)} entries from Materials Project.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output CSV file for MP summary data")
    ap.add_argument("--make-lmfp-grid", action="store_true", help="Generate a Vegard grid of LMFP compositions")
    ap.add_argument("--x-grid", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0], help="x_Fe grid points")
    args = ap.parse_args()

    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        raise SystemExit("Error: MP_API_KEY not set")

    with MPRester(api_key) as mpr:
        # Try to fetch endmembers first 
        endmembers = fetch_endmembers(mpr, sg_symbol=OLIVINE_SG)

        # Fetch MP summary
        df = fetch_mp_summary(mpr)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved MP summary to {out_path}")

        # build Vegard grid
        if args.make_lmfp_grid:
            vg = make_lmfp_vegard_grid(args.x_grid)
            grid_path = out_path.parent / "lmfp_vegard.csv"
            vg.to_csv(grid_path, index=False)
            print(f"Saved LMFP Vegard grid to {grid_path}")

if __name__ == "__main__":
    main()
