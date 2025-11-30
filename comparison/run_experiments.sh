#!/bin/bash
# Run classification experiments for DRL vs EA comparison
# This script launches multiple seeds for both methods and aggregates results.
#
# Usage:
#   ./comparison/run_experiments.sh [--method drl|ea|both] [--seeds "42 43 44"]
#
# Prerequisites:
#   - Set up entrypoint commands in configs/drl_classification.yaml and ea_classification.yaml
#   - Ensure dependencies are installed (pip install -r comparison/requirements.txt)

set -e

# Default settings
METHOD="${METHOD:-both}"
SEEDS="${SEEDS:-42 43}"
DRL_CONFIG="comparison/experiments/configs/drl_classification.yaml"
EA_CONFIG="comparison/experiments/configs/ea_classification.yaml"
LOG_DIR="comparison/logs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--method drl|ea|both] [--seeds \"42 43 44\"] [--log-dir path]"
            exit 1
            ;;
    esac
done

echo "=== Classification Comparison Experiments ==="
echo "Method: $METHOD"
echo "Seeds: $SEEDS"
echo "Log directory: $LOG_DIR"
echo ""

# Create log directories
mkdir -p "$LOG_DIR/drl"
mkdir -p "$LOG_DIR/ea"

run_drl() {
    local seed=$1
    echo "Running DRL experiment with seed $seed..."
    
    # TODO: Replace with actual DRL runner command
    # Example:
    # python tools/run_drl_agent.py \
    #     --config "$DRL_CONFIG" \
    #     --seed "$seed" \
    #     --output "$LOG_DIR/drl/drl_classif_seed${seed}.jsonl"
    
    echo "TODO: Set entrypoint_command in $DRL_CONFIG"
    echo "  Placeholder: DRL seed $seed would run here"
}

run_ea() {
    local seed=$1
    echo "Running EA experiment with seed $seed..."
    
    # TODO: Replace with actual EA runner command
    # Example:
    # python tools/run_ea_agent.py \
    #     --config "$EA_CONFIG" \
    #     --seed "$seed" \
    #     --output "$LOG_DIR/ea/ea_classif_seed${seed}.jsonl"
    
    echo "TODO: Set entrypoint_command in $EA_CONFIG"
    echo "  Placeholder: EA seed $seed would run here"
}

# Run experiments
for seed in $SEEDS; do
    if [[ "$METHOD" == "drl" ]] || [[ "$METHOD" == "both" ]]; then
        run_drl "$seed"
    fi
    
    if [[ "$METHOD" == "ea" ]] || [[ "$METHOD" == "both" ]]; then
        run_ea "$seed"
    fi
done

echo ""
echo "=== Experiments Complete ==="
echo "Logs saved to: $LOG_DIR"
echo ""
echo "To analyze results:"
echo "  python -m comparison.analysis.compute_classif_metrics \\"
echo "      --input \"$LOG_DIR/**/*.jsonl\" \\"
echo "      --out $LOG_DIR/classif_analysis"
echo ""
echo "To generate plots:"
echo "  python -m comparison.analysis.generate_classif_plots \\"
echo "      --input $LOG_DIR/classif_analysis \\"
echo "      --out $LOG_DIR/plots"
