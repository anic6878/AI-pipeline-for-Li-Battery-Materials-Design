import pandas as pd
from mp_api.client import MPRester
from sklearn.model_selection import train_test_split
import re


# Configuration

API_KEY = "FT2N5KnHU0T45Yao26o1PxHDvmGVMzvE"
CHEMSYS_LIST = [
    "Li-Fe-Mn-P-O",
    "Li-Fe-Mn-Co-Ni-P-O",
    "Li-Fe-Co-Ni-P-O",
    "Li-Fe-P-O"
]
FIELDS = [
    "material_id",
    "formula_pretty",
    "composition_reduced",
    "energy_above_hull",
    "band_gap",
    "density",
    "volume",
    "nsites",
    "energy_per_atom",
]
OUT_TRAIN = "data/train.csv"
OUT_VAL = "data/val.csv"



# MAIN SCRIPT

def safe_float(s):
    """Converting a string to float"""
    try:
        s = s.replace('"', '').replace("'", "")
        return float(s)
    except Exception:
        return 0.0


def compute_x_Fe(formula):
    
    """Estimate Fe fraction in Li-based olivines."""
    fe = re.findall(r"Fe([0-9.]*)", formula)
    mn = re.findall(r"Mn([0-9.]*)", formula)
    fe_val = safe_float(fe[0]) if fe else 0.0
    mn_val = safe_float(mn[0]) if mn else 0.0
    total = fe_val + mn_val if (fe_val + mn_val) > 0 else 1
    return fe_val / total


def main():
    all_entries = []

    print("Fetching Materials Project entries using API...")
    with MPRester(API_KEY) as mpr:
        for chemsys in CHEMSYS_LIST:
            print(f"Querying system: {chemsys}")
            try:
                docs = mpr.materials.summary.search(
                    chemsys=chemsys,
                    fields=FIELDS
                )
                if docs:
                    flat = []
                    for d in docs:
                        entry = {f: getattr(d, f, None) for f in FIELDS}
                        flat.append(entry)
                    print(f"Retrieved {len(flat)} entries from {chemsys}")
                    all_entries.extend(flat)
            except Exception as e:
                print(f"Skipped {chemsys} due to: {e}")

    if not all_entries:
        raise RuntimeError("No entries returned")

    # Convert to DataFrame safely
    df = pd.DataFrame(all_entries)
    print(f"Total combined entries: {len(df)}")

    # Ensure material_id exists
    if "material_id" not in df.columns:
        raise KeyError("'material_id' missing in API response")

    # Clean data
    df = df.drop_duplicates(subset="material_id")
    df = df.dropna(subset=["formula_pretty", "energy_above_hull", "band_gap", "density"])

    # Compute Fe fraction robustly
    df["x_Fe"] = df["formula_pretty"].apply(compute_x_Fe)

    # Split into train/val
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save
    train_df.to_csv(OUT_TRAIN, index=False)
    val_df.to_csv(OUT_VAL, index=False)

    print(f"Saved training data: {OUT_TRAIN} ({len(train_df)} rows)")
    print(f"Saved validation data: {OUT_VAL} ({len(val_df)} rows)")


if __name__ == "__main__":
    main()
