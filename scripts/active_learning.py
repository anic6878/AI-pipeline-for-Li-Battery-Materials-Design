"""
Active Learning for LMFP Materials Discovery

"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.stats import norm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="CSV from predict_new.py")
    ap.add_argument(
        "--output",
        default="next_batch_for_labeling.csv",
        help="Output CSV",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="How many candidates to select",
    )
    ap.add_argument(
        "--kappa",
        type=float,
        default=1.0,
        help="Exploration weight for UCB",
    )
    ap.add_argument(
        "--ehull-cut",
        type=float,
        default=50.0,
        help="Max allowed Ehull (meV/atom)",
    )
    ap.add_argument(
        "--dv-limit",
        type=float,
        default=12.0,
        help="Max allowed |ΔV%| if abs_dV_percent is present",
    )
    ap.add_argument(
        "--acq-fn",
        type=str,
        default="ucb",
        choices=["ucb", "ei", "kg"],
        help="Acquisition: ucb (μ+κσ), ei (Expected Improvement), kg (EI-based proxy)",
    )
    return ap.parse_args()


# ----------------- Physics filter -----------------

def filter_physics(df, ehull_cut, dv_limit):
    """
    Applying realistic literature derived physics constraints.
    """
    cond = pd.Series(True, index=df.index)

    if "E_hull_meV" in df.columns:
        cond &= (df["E_hull_meV"] <= ehull_cut)

    if "abs_dV_percent" in df.columns:
        cond &= (df["abs_dV_percent"] <= dv_limit)

    # Voltage safety window
    v_col = None
    for c in ["Vavg_V_mean", "Vavg_V"]:
        if c in df.columns:
            v_col = c
            break

    if v_col is not None and "U_cut_V" in df.columns:
        cond &= (df[v_col] <= df["U_cut_V"])

    return df[cond]


# ----------------- Objective + uncertainty -----------------

def _get_score_and_unc(df):
    """
    Extract a scalar objective mean and its std dev from the prediction columns.

    Priority for mean:
       1) score
       2) Vavg_V_mean
       3) Vavg_V

    Priority for std:
       1) score_std / score_unc
       2) Vavg_V_std / Vavg_V_unc
       else: small constant (almost no exploration)
    """
    # Mean 
    if "score" in df.columns:
        mu = df["score"].astype(float).values
    elif "Vavg_V_mean" in df.columns:
        mu = df["Vavg_V_mean"].astype(float).values
    elif "Vavg_V" in df.columns:
        mu = df["Vavg_V"].astype(float).values
    else:
        raise KeyError(
            "No suitable objective column found. Expected one of: "
            "['score', 'Vavg_V_mean', 'Vavg_V']."
        )

    # Std 
    if "score_std" in df.columns:
        sigma = df["score_std"].astype(float).values
    elif "score_unc" in df.columns:
        sigma = df["score_unc"].astype(float).values
    elif "Vavg_V_std" in df.columns:
        sigma = df["Vavg_V_std"].astype(float).values
    elif "Vavg_V_unc" in df.columns:
        sigma = df["Vavg_V_unc"].astype(float).values
    else:
        # No uncertainty - almost pure exploitation
        sigma = np.full_like(mu, 1e-8, dtype=float)

    # Avoid exactly zero for EI 
    sigma = np.where(sigma <= 1e-10, 1e-10, sigma)
    return mu, sigma


def compute_acquisition(df, kappa, acq_fn):
    """
    Bayesian-style acquisition on prediction distribution:

        UCB:  mean + κ*std
        EI:   E[max(0, mean - best - XI)]
        
    """
    mu, sigma = _get_score_and_unc(df)
    best = np.max(mu)          # current best predicted objective
    xi = 0.01                  # small improvement margin

    if acq_fn == "ucb":
        return mu + kappa * sigma

    # EI share the same EI formula
    z = (mu - best - xi) / sigma
    ei = (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(sigma <= 0, 0.0, ei)



# ----------------- Diversity selection -----------------

def select_diverse_top(df, features, top_n):
    """
    Greedy max–min diversity in feature space.
    Only indices are chosen; df values are left unchanged.
    """
    if len(df) <= top_n:
        return df

    X = df[features].to_numpy(dtype=float)
    selected = []
    remaining = list(range(len(df)))

    # start from best acquisition point
    first = int(np.argmax(df["acq"].values))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < top_n and remaining:
        # distances from remaining points to current selected set
        dists = pairwise_distances(X[remaining], X[selected], metric="euclidean")
        min_dist = dists.min(axis=1)
        next_idx = remaining[int(np.argmax(min_dist))]
        selected.append(next_idx)
        remaining.remove(next_idx)

    return df.iloc[selected]


# ----------------- Main -----------------

def main():
    args = parse_args()

    df = pd.read_csv(args.pred)
    print(f"Loaded predictions: {args.pred} (n={len(df)})")

    # 1) Physics filter
    df_f = filter_physics(df, args.ehull_cut, args.dv_limit)
    if len(df_f) == 0:
        print("[No candidates passed physics filters → using full set.")
        df_f = df.copy()

    df_f = df_f.reset_index(drop=True)

    # 2) Acquisition
    df_f["acq"] = compute_acquisition(df_f, args.kappa, args.acq_fn)
    df_sorted = df_f.sort_values("acq", ascending=False).reset_index(drop=True)

    # 3) Diversity in numeric feature space
    feat_candidates = [
        c
        for c in df_sorted.columns
        if c not in [
            "score", "score_std", "score_unc",
            "acq", "flags", "material_id", "formula"
        ]
        and np.issubdtype(df_sorted[c].dtype, np.number)
    ]
    if not feat_candidates:
        feat_candidates = ["acq"]

    next_batch = select_diverse_top(df_sorted, feat_candidates, args.top_n)
    next_batch = next_batch.reset_index(drop=True)

    # 4) Save
    next_batch.to_csv(args.output, index=False)
    print(
        f"Saved {len(next_batch)} candidates → {args.output} "
        f"(acq_fn={args.acq_fn}, kappa={args.kappa})"
    )


if __name__ == "__main__":
    main()
