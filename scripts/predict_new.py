

import os, json, joblib
import numpy as np
import pandas as pd
from glob import glob


# LOAD MODELS + METADATA
# ============================================================
def load_artifacts(art_dir):
    meta_path = os.path.join(art_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata.json in {art_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # ---- Determine targets ----
    if "targets" in meta:
        targets = meta["targets"]
    else:
        targets = [k for k in meta.keys() if isinstance(meta[k], dict)]

    models = {}
    scalers = {}

    for target in targets:
        # Load ensemble members
        files = sorted(glob(os.path.join(art_dir, f"{target}_member*.joblib")))
        if not files:
            raise FileNotFoundError(f"No model files found for target {target}")

        models[target] = [joblib.load(f) for f in files]

        # Load scaler for this target
        scaler_path = os.path.join(art_dir, f"{target}_scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler: {scaler_path}")

        scalers[target] = joblib.load(scaler_path)

    print(f"Loaded targets: {targets}")
    return models, scalers, meta, targets


# ============================================================
# ENSEMBLE PREDICTION
# ============================================================
def predict_ensemble(model_list, scaler, X):
    X_s = scaler.transform(X)
    preds = np.stack([m.predict(X_s) for m in model_list])
    mu = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mu, std


# ============================================================
# MAIN
# ============================================================
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, help="Folder containing trained models")
    ap.add_argument("--input", required=True, help="Candidate compositions CSV")
    ap.add_argument("--physics", required=False, help="Optional swelling/physics CSV")
    ap.add_argument("--output", required=True, help="Output CSV")
    args = ap.parse_args()

    # --------------------------------------------------------
    # Load candidates
    # --------------------------------------------------------
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"{args.input} not found.")

    df = pd.read_csv(args.input)

    # Merge physics if provided
    if args.physics and os.path.exists(args.physics):
        phys = pd.read_csv(args.physics)
        df = df.merge(phys, on="formula", how="left")
    

    # ---- FIX: normalize dopant fraction column names after merge ----
    # If merge created x_Fe_x/x_Fe_y etc, map back to plain x_Fe, x_Mn, x_Ni, x_Mg.
    for el in ["Fe", "Mn", "Ni", "Mg"]:
        base = f"x_{el}"
        xcol = f"{base}_x"
        ycol = f"{base}_y"
        if base not in df.columns:
            if xcol in df.columns:
                df[base] = df[xcol]
            elif ycol in df.columns:
                df[base] = df[ycol]    

    # --------------------------------------------------------
    # Load models + metadata
    # --------------------------------------------------------
    models, scalers, meta, targets = load_artifacts(args.artifacts)

    results = df.copy()

    # --------------------------------------------------------
    # Predict each target using EXACT training features
    # --------------------------------------------------------
    for target in targets:

        # Features used during training
        feats = meta[target]["features"]

        # Ensure missing columns become zero 
        for col in feats:
            if col not in results.columns:
                print(f"Missing feature '{col}' in input therefore filling with zeros.")
                results[col] = 0.0

        # Prepare feature matrix
        X = results[feats].astype(float).values

        mu, std = predict_ensemble(models[target], scalers[target], X)

        # Save predictions
        results[target] = mu
        results[f"{target}_unc"] = std

    # --------------------------------------------------------
    # Save final predictions
    # --------------------------------------------------------
    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results.to_csv(out_path, index=False)

    print(f"\n Predictions written to: {out_path}")


if __name__ == "__main__":
    main()
