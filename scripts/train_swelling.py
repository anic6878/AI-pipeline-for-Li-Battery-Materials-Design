
"""
This script is a PHYSICS-BASED LABEL BUILDER
Computes ΔV% physically using Materials Project endmembers (Pnma)

1. Loads candidate compositions with a formula column.
2. Uses the same physics logic as build_swelling_dataset.py:
   - Enforce olivine endmembers (LiFePO4, LiMnPO4, FePO4, MnPO4)
   - Compute per formula unit volumes using Vegard’s law
   - Compute signed ΔV% = 100*(V_delith - V_lith)/V_lith
   - Compute crystal density (g/cm3)
   - Compute thermodynamic upper plateau voltage
   - Compute practical upper cutoff (electrolyte cap)
3. Saves final swelling/density/voltage labels to CSV.


USAGE:
MP_API_KEY=... python train_swelling.py \
    --input data/candidates.csv \
    --output data/swelling_labels.csv \
    --vlfp 3.45 \
    --vlmp 4.10 \
    --electrolyte-cap 4.25 \
    --safety-margin 0.05
"""

import os, argparse, math
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition

OLIVINE_SG = "Pnma"

def per_fu(volume, formula_pretty, ref):
    comp = Composition(formula_pretty)
    ref_c = Composition(ref)
    n_fu = min(comp.get(el.symbol, 0) / amt for el, amt in ref_c.items())
    if n_fu <= 0:
        n_fu = 4.0
    return volume / n_fu

def pick_olivine(docs, elems):
    cand = []
    for d in docs:
        try:
            if d.symmetry["symbol"] != OLIVINE_SG:
                continue
        except:
            continue
        if not set(elems).issubset(set(d.elements)):
            continue
        cand.append(d)
    if not cand:
        return None
    cand.sort(key=lambda x: x.energy_per_atom)
    return cand[0]

def fetch_endmembers(mpr):
    f = ["material_id","formula_pretty","elements","symmetry","energy_per_atom","volume"]
    q = mpr.materials.summary
    lfp = pick_olivine(q.search(formula="LiFePO4", fields=f), ["Li","Fe","P","O"])
    lmp = pick_olivine(q.search(formula="LiMnPO4", fields=f), ["Li","Mn","P","O"])
    fep = pick_olivine(q.search(formula="FePO4",   fields=f), ["Fe","P","O"])
    mnp = pick_olivine(q.search(formula="MnPO4",   fields=f), ["Mn","P","O"])
    if None in [lfp, lmp, fep, mnp]:
        raise RuntimeError("Missing olivine endmembers in MP")

    V_LFP = per_fu(lfp.volume, lfp.formula_pretty, "LiFePO4")
    V_LMP = per_fu(lmp.volume, lmp.formula_pretty, "LiFePO4")
    V_FEP = per_fu(fep.volume, fep.formula_pretty, "FePO4")
    V_MNP = per_fu(mnp.volume, mnp.formula_pretty, "FePO4")

    return {
        "lfp_id": lfp.material_id, "lmp_id": lmp.material_id,
        "fepo4_id": fep.material_id, "mnpo4_id": mnp.material_id,
        "V_LFP": V_LFP, "V_LMP": V_LMP,
        "V_FePO4": V_FEP, "V_MnPO4": V_MNP
    }

def vegard(x, V_LFP, V_LMP, V_FEP, V_MNP):
    V_lith = x*V_LFP + (1-x)*V_LMP
    V_del  = x*V_FEP + (1-x)*V_MNP
    dV_pct = 100.0*(V_del - V_lith)/V_lith
    return V_lith, V_del, dV_pct

def density(formula, V_fU):
    comp = Composition(formula)
    mass_amu = comp.weight
    return (mass_amu/6.02214076e23)/(V_fU*1e-24)

def parse_x_fe(formula):
    c = Composition(formula)
    fe = c.get_el_amt_dict().get("Fe",0)
    mn = c.get_el_amt_dict().get("Mn",0)
    if fe+mn <= 0:
        raise ValueError(f"No Fe/Mn in {formula}")
    return fe/(fe+mn)

def upper_thermo(x, vlfp, vlmp):
    return x*vlfp + (1-x)*vlmp

def practical_ucut(vmax, cap, margin):
    return min(max(vmax - margin, 0), cap)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with 'formula' column")
    ap.add_argument("--output", required=True, help="CSV to write swelling/density/voltage labels")
    ap.add_argument("--vlfp", type=float, default=3.45, help="Avg LFP plateau V")
    ap.add_argument("--vlmp", type=float, default=4.10, help="Avg LMP plateau V")
    ap.add_argument("--electrolyte-cap", type=float, default=4.25, help="Practical voltage cap")
    ap.add_argument("--safety-margin", type=float, default=0.05, help="Subtract from Vmax_thermo before cap")
    args = ap.parse_args()

    if "MP_API_KEY" not in os.environ:
        raise SystemExit("MP_API_KEY not set.")

    df = pd.read_csv(args.input)
    if "formula" not in df.columns:
        raise ValueError("Input must have 'formula' column.")

    with MPRester(os.environ["MP_API_KEY"]) as mpr:
        meta = fetch_endmembers(mpr)

    rows = []
    for _, row in df.iterrows():
        formula = str(row["formula"]).strip()
        try:
            x = parse_x_fe(formula)
        except:
            continue
        V_lith, V_del, dV_pct = vegard(x, meta["V_LFP"], meta["V_LMP"], meta["V_FePO4"], meta["V_MnPO4"])
        rho = density(formula, V_lith)
        vmax = upper_thermo(x, args.vlfp, args.vlmp)
        ucut = practical_ucut(vmax, args.electrolyte_cap, args.safety_margin)

        rec = {
            "formula": formula,
            "x_Fe": x,
            "V_lithiated_fU_A3": V_lith,
            "V_delithiated_fU_A3": V_del,
            "dV_percent": dV_pct,
            "density_g_cm3": rho,
            "V_upper_thermo": vmax,
            "U_cut_V": ucut,
            **meta
        }
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    if len(out) == 0:
        print("Noo LMFP-like formulas found.")

if __name__ == "__main__":
    main()
