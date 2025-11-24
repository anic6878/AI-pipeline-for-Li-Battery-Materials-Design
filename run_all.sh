
set -e

echo "========================================"
echo "     LMFP – FULL PIPELINE EXECUTION"
echo "========================================"

# go to repo root 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# activate environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Ensure MP_API_KEY is set
if [ -z "$MP_API_KEY" ]; then
    echo "ERROR: MP_API_KEY is not set!"
    exit 1
fi

# OUTPUT DIRECTORY
OUTDIR="runs/run1"

# Seed candidates
CANDIDATES="$OUTDIR/cycle_01/generated_candidates.csv"

echo "Output directory = $OUTDIR"
echo "Seed candidates  = $CANDIDATES"

# Sanity checks
if [ ! -f "$CANDIDATES" ]; then
    echo "ERROR: Seed candidates file '$CANDIDATES' not found."
    exit 1
fi

echo "[STEP] Running full pipeline through run_pipeline.py"
python run_pipeline.py \
    --outdir "$OUTDIR" \
    --candidates "$CANDIDATES" \
    --cycles 3 \
    --acq-fn ei \
    --ehull-cut 50.0 \
    --dv-limit 12.0 \
    --use-vae

echo "========================================"
echo "        FULL PIPELINE COMPLETE!"
echo "========================================"