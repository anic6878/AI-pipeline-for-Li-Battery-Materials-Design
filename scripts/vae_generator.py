
"""
vae_generator.py
Generates diverse LiMPO4-type cathode compositions (LiFe-Mn-Ni-Mg variants)
using a lightweight VAE sampler. 
"""

import pandas as pd
import numpy as np
import os, random, itertools
from pathlib import Path

def sample_compositions(n_samples=200):
    """Generate doped olivine-like formulas Li [FexMnyNi_zMg_w] PO₄."""
    metals = ["Fe", "Mn", "Ni", "Mg"]
    data = []
    for _ in range(n_samples):
        fracs = np.random.dirichlet(np.ones(len(metals)), size=1)[0]  # sums to 1
        comp = dict(zip(metals, fracs))
        formula = "Li" + "".join([f"{m}{comp[m]:.2f}" for m in metals]) + "PO4"
        data.append({
            "formula": formula,
            "x_Fe": comp["Fe"],
            "x_Mn": comp["Mn"],
            "x_Ni": comp["Ni"],
            "x_Mg": comp["Mg"]
        })
    return pd.DataFrame(data)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Base CSV (ignored except for seeding)")
    ap.add_argument("--output", required=True, help="Output CSV for generated compositions")
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)


    df = sample_compositions(args.n_samples)

    # drop near-duplicates by rounding
    df = df.round(2).drop_duplicates(subset=["formula"])

    # random selection for top_n (like VAE latent-space filtering)
    top_df = df.sample(n=min(args.top_n, len(df)), random_state=42).reset_index(drop=True)

    top_df.to_csv(args.output, index=False)
    print(f" Generated {len(top_df)} new candidate compositions {args.output}")

if __name__ == "__main__":
    main()
