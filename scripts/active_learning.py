"""
Active Learning + Bayesian Optimization for LMFP Materials Discovery

"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel


# Argument Parser

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="CSV from predict_new.py")
    ap.add_argument("--output", default="next_batch_for_labeling.csv")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--ehull-cut", type=float, default=50.0)
    ap.add_argument("--dv-limit", type=float, default=12.0)
    ap.add_argument("--acq-fn", type=str, default="ei", choices=["ei"])
    return ap.parse_args()


# ---------------------------------------------------------------------
# Apply physics constraints

def filter_physics(df, ehull_cut, dv_limit):
    cond = pd.Series(True, index=df.index)

    if "E_hull_meV" in df.columns:
        cond &= (df["E_hull_meV"] <= ehull_cut)

    if "abs_dV_percent" in df.columns:
        cond &= (df["abs_dV_percent"] <= dv_limit)

    if "Vavg_V_mean" in df.columns and "U_cut_V" in df.columns:
        cond &= (df["Vavg_V_mean"] <= df["U_cut_V"])

    return df[cond]


# ---------------------------------------------------------------------
# Extract BO features + target

def extract_features_and_target(df):
    """
    Inputs:
        df = filtered prediction dataframe

    Returns:
        X    = tensor [N × d] of dopant fractions
        y    = tensor [N × 1] objective = Vavg_V_mean
    """

    # Features = numeric composition features (dopant fractions)
    feat_cols = [
        c for c in df.columns
        if ("x_" in c) and np.issubdtype(df[c].dtype, np.number)
    ]

    if len(feat_cols) == 0:
        raise ValueError("No dopant fraction columns found (expected x_Fe, x_Mn, ...)")

    if "Vavg_V_mean" not in df.columns:
        raise ValueError("Vavg_V_mean missing from predictions CSV")

    X = torch.tensor(df[feat_cols].to_numpy(), dtype=torch.float64)
    y = torch.tensor(df["Vavg_V_mean"].to_numpy().reshape(-1, 1), dtype=torch.float64)

    return X, y, feat_cols


# ---------------------------------------------------------------------
# Fit GP model (Matern 2.5 kernel)

def fit_gp_model(X, y):
    gp = SingleTaskGP(
        X, y,
        covar_module=ScaleKernel(MaternKernel(nu=2.5))
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


# ---------------------------------------------------------------------
# Acquisition via EI


def compute_ei_acq(gp, X):
    gp.eval()
    best_f = gp.train_targets.max()

    ei = ExpectedImprovement(model=gp, best_f=best_f)
    acq_vals = ei(X).detach().numpy().reshape(-1)
    return acq_vals


# ---------------------------------------------------------------------
# Max–min diversity selection


def select_diverse(df, acq_vals, feat_cols, top_n):
    df2 = df.copy()
    df2["acq"] = acq_vals

    df2 = df2.sort_values("acq", ascending=False).reset_index(drop=True)

    if len(df2) <= top_n:
        return df2

    X = df2[feat_cols].to_numpy()
    selected = []
    remaining = list(range(len(df2)))

    # pick best EI point
    first = 0
    selected.append(first)
    remaining.remove(first)

    while len(selected) < top_n and remaining:
        dists = pairwise_distances(X[remaining], X[selected])
        min_dist = dists.min(axis=1)
        next_idx = remaining[int(np.argmax(min_dist))]
        selected.append(next_idx)
        remaining.remove(next_idx)

    return df2.iloc[selected].reset_index(drop=True)


# ---------------------------------------------------------------------
# MAIN


def main():
    args = parse_args()

    df = pd.read_csv(args.pred)
    print(f"Loaded predictions: {args.pred} (n={len(df)})")

    # 1) Physics filter
    df_f = filter_physics(df, args.ehull_cut, args.dv_limit)
    if len(df_f) == 0:
        print(" No candidates passed physics filters thus using full list.")
        df_f = df.copy()
    df_f = df_f.reset_index(drop=True)

    # 2) Extract features + target
    X, y, feat_cols = extract_features_and_target(df_f)

    # 3) Fit GP model
    
    gp = fit_gp_model(X, y)

    # 4) Compute EI acquisition
    acq_vals = compute_ei_acq(gp, X)
    df_f["acq"] = acq_vals

    # 5) Diversity selection
    next_batch = select_diverse(df_f, acq_vals, feat_cols, args.top_n)

    # 6) Save
    next_batch.to_csv(args.output, index=False)
    print(f"Saved {len(next_batch)} candidates to {args.output}")


if __name__ == "__main__":
    main()
