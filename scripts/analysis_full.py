
"""

Create BO-diagnostics plots & a small summary CSV from multi-cycle data.

Assumes a structure like:

runs/run1/
    cycle_01/
        predictions_calibrated.csv
        next_batch_for_labeling.csv
        generated_swelling.csv
    cycle_02/
        ...
    cycle_03/
        ...
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------------

OUTDIR = Path("analysis_outputs_bo")
OUTDIR.mkdir(parents=True, exist_ok=True)

CYCLES = [
    {
        "cycle": 1,
        "pred": "runs/run1/cycle_01/predictions_calibrated.csv",
        "sel": "runs/run1/cycle_01/next_batch_for_labeling.csv",
        "swel": "runs/run1/cycle_01/generated_swelling.csv",
    },
    {
        "cycle": 2,
        "pred": "runs/run1/cycle_02/predictions_calibrated.csv",
        "sel": "runs/run1/cycle_02/next_batch_for_labeling.csv",
        "swel": "runs/run1/cycle_02/generated_swelling.csv",
    },
    {
        "cycle": 3,
        "pred": "runs/run1/cycle_03/predictions_calibrated.csv",
        "sel": "runs/run1/cycle_03/next_batch_for_labeling.csv",
        "swel": "runs/run1/cycle_03/generated_swelling.csv",
    },
]


def load_cycle(cfg):
    """Load predictions, selected batch and swelling dataset for one cycle."""
    cyc = cfg["cycle"]

    preds = pd.read_csv(cfg["pred"])
    preds["cycle"] = cyc

    sel = pd.read_csv(cfg["sel"])
    sel["cycle"] = cyc

    swel = pd.read_csv(cfg["swel"])
    swel["cycle"] = cyc

    return preds, sel, swel


def ensure_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in dataframe.")
    return df


def main():
    # --------------------------------------------------------------
    # 1) Load / concatenate across cycles
    # --------------------------------------------------------------
    pred_all_list, sel_all_list, swel_all_list = [], [], []

    for cfg in CYCLES:
        preds, sel, swel = load_cycle(cfg)
        pred_all_list.append(preds)
        sel_all_list.append(sel)
        swel_all_list.append(swel)

    pred_all = pd.concat(pred_all_list, ignore_index=True)
    sel_all = pd.concat(sel_all_list, ignore_index=True)
    swel_all = pd.concat(swel_all_list, ignore_index=True)

    # sanity
    ensure_cols(pred_all, ["Vavg_V", "E_hull_meV"])
    ensure_cols(sel_all, ["Vavg_V", "E_hull_meV"])
    # swelling file columns might differ slightly; we use these if present:
    for c in ["dV_percent", "density_g_cm3", "x_Fe", "x_Mn"]:
        if c not in swel_all.columns:
            print(f"{c} not in swelling CSVs.")

    cycles = sorted(pred_all["cycle"].unique())

    # --------------------------------------------------------------
    # 2) Cycle summary CSV (best & median metrics)
    # --------------------------------------------------------------
    rows = []
    for cyc in cycles:
        p = pred_all[pred_all["cycle"] == cyc]
        s = sel_all[sel_all["cycle"] == cyc]

        row = {
            "cycle": cyc,
            "best_Vavg_all": p["Vavg_V"].max(),
            "best_Vavg_selected": s["Vavg_V"].max(),
            "median_Ehull_all": p["E_hull_meV"].median(),
            "median_Ehull_selected": s["E_hull_meV"].median(),
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("cycle")
    summary_path = OUTDIR / "bo_cycle_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    

    # Best Vavg vs cycle
    plt.figure(figsize=(6,5))
    plt.plot(summary_df["cycle"], summary_df["best_Vavg_all"],
             marker="o", label="Best Vavg (all candidates)")
    plt.plot(summary_df["cycle"], summary_df["best_Vavg_selected"],
             marker="s", label="Best Vavg (BO-selected)")
    plt.xlabel("Cycle")
    plt.ylabel("Best predicted Vavg_V (V)")
    plt.title("Best voltage vs BO cycle")
    plt.tight_layout()
    plt.legend()
    out = OUTDIR / "bo_best_Vavg_vs_cycle.png"
    plt.savefig(out, dpi=300)
    plt.close()

    # Median E_hull vs cycle
    plt.figure(figsize=(6,5))
    plt.plot(summary_df["cycle"], summary_df["median_Ehull_all"],
             marker="o", label="Median E_hull (all candidates)")
    plt.plot(summary_df["cycle"], summary_df["median_Ehull_selected"],
             marker="s", label="Median E_hull (BO-selected)")
    plt.xlabel("Cycle")
    plt.ylabel("Median E_hull_meV")
    plt.title("Stability (E_hull) vs BO cycle")
    plt.tight_layout()
    plt.legend()
    out = OUTDIR / "bo_median_Ehull_vs_cycle.png"
    plt.savefig(out, dpi=300)
    plt.close()


    # Composition of BO-selected across cycles
    if set(["x_Fe", "x_Mn"]).issubset(sel_all.columns):
        plt.figure(figsize=(6,5))
        for cyc in cycles:
            sub = sel_all[sel_all["cycle"] == cyc]
            plt.scatter(sub["x_Fe"], sub["x_Mn"], alpha=0.8, label=f"Cycle {cyc}")
        plt.xlabel("x_Fe")
        plt.ylabel("x_Mn")
        plt.title("BO-selected batches in Fe–Mn composition space")
        plt.tight_layout()
        plt.legend()
        out = OUTDIR / "bo_selected_composition_by_cycle.png"
        plt.savefig(out, dpi=300)
        plt.close()
        
    else:
        print("x_Fe/x_Mn not found in selected batches; "
              "skipping composition-by-cycle plot.")

 
        


if __name__ == "__main__":
    main()
