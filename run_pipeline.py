
"""
run_pipeline.py  — LMFP PIPELINE (ML model ensemble + Bayesian acquisition)

Pipeline:
    1) Fetch/base MP data           (fetch_mp_data.py)
    2) Build swelling/physics data  (build_swelling_dataset.py)
    3) Predict properties           (predict_new.py)
    4) Bayesian-style selection     (active_learning.py, UCB/EI/KG over μ, σ)
    5) Optional VAE augmentation    (vae_generator.py)

Usage example:
    python run_pipeline.py \
        --outdir runs/vae_expt1 \
        --candidates data/base_candidates.csv \
        --cycles 1 \
        --acq-fn ei \
        --use-vae
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


# ---------------------------------------------------------
# Helper: run shell commands
# ---------------------------------------------------------
def run(cmd, cwd=None):
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True,
                    help="Output root folder for entire pipeline (e.g. runs/vae_expt1)")
    ap.add_argument("--candidates", required=True,
                    help="Initial candidate CSV for cycle 1 (must have 'formula' column)")
    ap.add_argument("--cycles", type=int, default=1,
                    help="Number of active-learning cycles")

    # Bayesian acquisition options
    ap.add_argument("--acq-fn", type=str, default="ei",
                    choices=["ei", "ucb", "kg"],
                    help="Acquisition function for selection (ucb / ei)")
    ap.add_argument("--kappa", type=float, default=1.0,
                    help="Exploration weight for UCB")
    ap.add_argument("--ehull-cut", type=float, default=50.0,
                    help="Max allowed E_hull (meV/atom) in physics filter")
    ap.add_argument("--dv-limit", type=float, default=12.0,
                    help="Max allowed |ΔV%| if abs_dV_percent present")

    # VAE options
    ap.add_argument("--use-vae", action="store_true",
                    help="Generate new candidates each cycle via vae_generator.py")
    ap.add_argument("--vae-samples", type=int, default=500,
                    help="Number of random samples in latent composition space")
    ap.add_argument("--vae-topn", type=int, default=100,
                    help="Top-N distinct compositions to keep from VAE sampling")

    args = ap.parse_args()

    # Check MP API key
    if "MP_API_KEY" not in os.environ:
        raise SystemExit("MP_API_KEY not set. Use: export MP_API_KEY=YOURKEY")

    # Folders
    OUT = Path(args.outdir)
    DATA = OUT / "data"
    ensure_dir(OUT)
    ensure_dir(DATA)

    # Script paths (repo root + Scripts/)
    S = Path("Scripts")
    FETCH = S / "fetch_mp_data.py"
    SWELL = S / "build_swelling_dataset.py"
    PRED  = S / "predict_new.py"
    AL    = S / "active_learning.py"
    VAE   = S / "vae_generator.py"

    # Pretrained model artifacts 
    ART_DIR = Path("runs") / "final_models" / "artifacts"

    if not ART_DIR.exists():
        raise SystemExit(f"Artifacts directory not found: {ART_DIR}")

    # -----------------------------------------------------
    # Step 1: Fetch MP dataset (mp_summary + LMFP Vegard grid)
    # -----------------------------------------------------
    run([
        sys.executable, str(FETCH),
        "--out", str(DATA / "mp_summary.csv"),
        "--make-lmfp-grid",
        "--x-grid", "0.0", "0.25", "0.5", "0.75", "1.0"
    ])

    # -----------------------------------------------------
    # Step 2: Build swelling/physics labels for base candidates
    # -----------------------------------------------------
    base_swelling = DATA / "swelling_labels.csv"
    run([
        sys.executable, str(SWELL),
        "--in", args.candidates,
        "--out", str(base_swelling)
    ])

    # -----------------------------------------------------
    # Step 3–N: Active-learning / BO cycles
    # -----------------------------------------------------
    # This is the candidate list conceptually in hand
    candidates_csv = args.candidates

    for cyc in range(1, args.cycles + 1):
        print(f"\n==============================")
        print(f"  ACTIVE LEARNING CYCLE {cyc}")
        print(f"==============================\n")

        cyc_dir = OUT / f"cycle_{cyc:02d}"
        ensure_dir(cyc_dir)

        # ---------------------------------------------
        # VAE candidate generation
        # ---------------------------------------------
        if args.use_vae:
            gen_csv = cyc_dir / "generated_candidates.csv"
            run([
                sys.executable, str(VAE),
                "--input", candidates_csv,
                "--output", str(gen_csv),
                "--n-samples", str(args.vae_samples),
                "--top-n", str(args.vae_topn),
            ])

            # Build physics labels for new candidates
            gen_swelling = cyc_dir / "generated_swelling.csv"
            run([
                sys.executable, str(SWELL),
                "--in", str(gen_csv),
                "--out", str(gen_swelling),
            ])

            candidates_for_pred = str(gen_csv)
            physics_for_pred = str(gen_swelling)
        else:
            # No VAE i.e., use the current candidate list and base physics
            candidates_for_pred = candidates_csv
            physics_for_pred = str(base_swelling)

        # ---------------------------------------------
        # Predict using ensemble models + physics merge
        # ---------------------------------------------
        preds_csv = cyc_dir / "predictions_calibrated.csv"
        run([
            sys.executable, str(PRED),
            "--artifacts", str(ART_DIR),
            "--input", str(candidates_for_pred),
            "--physics", physics_for_pred,
            "--output", str(preds_csv),
        ])

        # ---------------------------------------------
        # Bayesian acquisition
        # ---------------------------------------------
        next_batch_csv = cyc_dir / "next_batch_for_labeling.csv"
        run([
            sys.executable, str(AL),
            "--pred", str(preds_csv),
            "--output", str(next_batch_csv),
            "--top-n", "50",
            "--kappa", str(args.kappa),
            "--acq-fn", args.acq_fn,
            "--ehull-cut", str(args.ehull_cut),
            "--dv-limit", str(args.dv_limit),
        ])

        print(f"[CYCLE {cyc}] Selected → {next_batch_csv}")

        # Update candidate list for next iteration
        # (in a actual loop DFT/experiment data needs to be run and retrain)
        candidates_csv = str(next_batch_csv)

    print(f"\n Pipeline completed successfully under: {OUT}\n")


if __name__ == "__main__":
    main()
