#!/usr/bin/env bash
# ============================================================================
# run_experiments.sh — Run classification experiments for DRL vs EA comparison
# ============================================================================
#
# This script launches multiple seeds for both DRL and EA methods by reading
# the entrypoint_command from YAML configuration files. It substitutes
# placeholders and logs output per-seed.
#
# Usage:
#   ./comparison/run_experiments.sh [--method drl|ea|both] [--seeds "42 43 44"] [--log-dir path] [--dry-run]
#
# Arguments:
#   --method    Method to run: "drl", "ea", or "both" (default: both)
#   --seeds     Space-separated list of seeds, e.g. "42 43 44" (default: "42 43")
#   --log-dir   Directory for log files (default: comparison/logs)
#   --dry-run   Print commands without executing them
#
# Prerequisites:
#   - Set entrypoint_command in configs/drl_classification.yaml and/or ea_classification.yaml
#   - Ensure dependencies are installed (pip install -r comparison/requirements.txt)
#
# Placeholders in entrypoint_command:
#   {config}  — Path to the YAML config file
#   {seed}    — Current seed value
#   {output}  — Output log file path (auto-generated: LOG_DIR/METHOD/METHOD_classif_seed{seed}.log)
#   $seed / ${seed} — Shell-style seed substitution (also supported)
#   $output / ${output} — Shell-style output substitution (also supported)
#   $config / ${config} — Shell-style config substitution (also supported)
#
# Example entrypoint_command in YAML:
#   entrypoint_command: "python tools/run_drl_agent.py --config {config} --seed {seed} --output {output}"
#
# Exit codes:
#   0 — All experiments completed successfully
#   1 — One or more experiments failed or entrypoint_command was missing
#
# ============================================================================

set -euo pipefail

# Default settings
METHOD="${METHOD:-both}"
SEEDS="${SEEDS:-42 43}"
DRL_CONFIG="comparison/experiments/configs/drl_classification.yaml"
EA_CONFIG="comparison/experiments/configs/ea_classification.yaml"
LOG_DIR="comparison/logs"
DRY_RUN=false

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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            head -n 40 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--method drl|ea|both] [--seeds \"42 43 44\"] [--log-dir path] [--dry-run]"
            echo "Run with --help for more information."
            exit 1
            ;;
    esac
done

echo "=== Classification Comparison Experiments ==="
echo "Method: $METHOD"
echo "Seeds: $SEEDS"
echo "Log directory: $LOG_DIR"
if [[ "$DRY_RUN" == true ]]; then
    echo "Mode: DRY RUN (commands will be shown but not executed)"
fi
echo ""

# Create log directories
mkdir -p "$LOG_DIR/drl"
mkdir -p "$LOG_DIR/ea"

# Track overall success
EXIT_CODE=0

# Extract entrypoint_command from a YAML config file (line-based extraction)
# Args: $1 = config file path
# Returns: the entrypoint_command value (single line expected)
# Note: Supports both double and single quotes, e.g.:
#   entrypoint_command: "python foo.py --seed {seed}"
#   entrypoint_command: 'python foo.py --seed {seed}'
extract_entrypoint_command() {
    local config_file="$1"
    local entrypoint_line=""
    local extracted_cmd=""
    
    if [[ ! -f "$config_file" ]]; then
        echo ""
        return
    fi
    
    # Extract the line containing entrypoint_command that has an actual command
    # Look for lines like: entrypoint_command: "python ..." or entrypoint_command: 'python ...'
    # Note: Using character class ["'] to match both double and single quotes
    entrypoint_line=$(grep -E "^entrypoint_command:\s*[\"']" "$config_file" 2>/dev/null | head -n 1 || true)
    
    if [[ -n "$entrypoint_line" ]]; then
        # Extract the command from within quotes (double or single)
        # First remove the key and any whitespace
        extracted_cmd="${entrypoint_line#entrypoint_command:}"
        # Trim leading whitespace
        extracted_cmd="${extracted_cmd#"${extracted_cmd%%[![:space:]]*}"}"
        # Remove surrounding quotes (double or single) and any trailing whitespace/comments
        if [[ "$extracted_cmd" == \"* ]]; then
            # Double quotes
            extracted_cmd="${extracted_cmd#\"}"
            extracted_cmd="${extracted_cmd%%\"*}"
        elif [[ "$extracted_cmd" == \'* ]]; then
            # Single quotes
            extracted_cmd="${extracted_cmd#\'}"
            extracted_cmd="${extracted_cmd%%\'*}"
        fi
        echo "$extracted_cmd"
        return
    fi
    
    echo ""
}

# Substitute placeholders in a command string
# Args: $1 = command, $2 = config, $3 = seed, $4 = output
substitute_placeholders() {
    local cmd="$1"
    local config="$2"
    local seed="$3"
    local output="$4"
    
    # Substitute {config}, {seed}, {output} placeholders
    cmd="${cmd//\{config\}/$config}"
    cmd="${cmd//\{seed\}/$seed}"
    cmd="${cmd//\{output\}/$output}"
    
    # Substitute shell-style ${seed}, ${output}, ${config} (braced form)
    cmd="${cmd//\$\{seed\}/$seed}"
    cmd="${cmd//\$\{output\}/$output}"
    cmd="${cmd//\$\{config\}/$config}"
    
    # Substitute shell-style $seed, $output, $config (unbraced form)
    cmd="${cmd//\$seed/$seed}"
    cmd="${cmd//\$output/$output}"
    cmd="${cmd//\$config/$config}"
    
    echo "$cmd"
}

# Print guidance when entrypoint_command is missing
print_missing_entrypoint_guidance() {
    local method="$1"
    local config_file="$2"
    
    echo ""
    echo "ERROR: entrypoint_command not found or not properly configured in $config_file"
    echo ""
    echo "To fix this, add an entrypoint_command line to $config_file:"
    echo ""
    if [[ "$method" == "drl" ]]; then
        echo '  entrypoint_command: "python tools/run_drl_agent.py --config {config} --seed {seed} --output {output}"'
    else
        echo '  entrypoint_command: "python tools/run_ea_agent.py --config {config} --seed {seed} --output {output}"'
    fi
    echo ""
    echo "Placeholders:"
    echo "  {config} — Path to the YAML config file"
    echo "  {seed}   — Current seed value"
    echo "  {output} — Output log file path"
    echo ""
}

# Check if a command is a simple Python command
# A simple Python command starts with 'python' or 'python3' and doesn't contain
# shell operators like pipes, redirects, or chained commands
# Args: $1 = command
# Returns: 0 (true) if simple, 1 (false) if complex
is_simple_python_command() {
    local cmd="$1"
    
    # Check if it starts with python or python3
    if [[ ! "$cmd" =~ ^[[:space:]]*(python3?|python[0-9.]+)[[:space:]] ]]; then
        return 1
    fi
    
    # Check for shell operators (pipes, redirects, chained commands, subshells)
    # This is a heuristic - a complex command would use |, ||, &&, ;, >, <, $( )
    # shellcheck disable=SC2016
    if [[ "$cmd" == *"|"* ]] || [[ "$cmd" == *";"* ]] || [[ "$cmd" == *">"* ]] || [[ "$cmd" == *"<"* ]] || [[ "$cmd" == *"&&"* ]] || [[ "$cmd" == *'$('* ]]; then
        return 1
    fi
    
    return 0
}

# Run an experiment for a specific method and seed
# Args: $1 = method (drl|ea), $2 = seed, $3 = config file
run_experiment() {
    local method="$1"
    local seed="$2"
    local config_file="$3"
    local output_file="$LOG_DIR/$method/${method}_classif_seed${seed}.log"
    
    echo "Running $method experiment with seed $seed..."
    
    # Extract entrypoint_command from config
    local entrypoint_cmd
    entrypoint_cmd=$(extract_entrypoint_command "$config_file")
    
    if [[ -z "$entrypoint_cmd" ]]; then
        print_missing_entrypoint_guidance "$method" "$config_file"
        return 1
    fi
    
    # Substitute placeholders
    local cmd
    cmd=$(substitute_placeholders "$entrypoint_cmd" "$config_file" "$seed" "$output_file")
    
    echo "  Config: $config_file"
    echo "  Output: $output_file"
    echo "  Command: $cmd"
    
    # Detect command type and inform user
    if is_simple_python_command "$cmd"; then
        echo "  Type: Simple Python command"
    else
        echo "  Type: Complex shell command (will use eval)"
    fi
    echo ""
    
    # If dry-run mode, don't execute
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would execute: $cmd"
        echo ""
        return 0
    fi
    
    # Execute the command and tee output to log file
    # For simple Python commands, we can run directly without eval
    # For complex commands with shell operators, we need eval
    if is_simple_python_command "$cmd"; then
        # Simple Python command - can run directly with proper word splitting
        if ! $cmd 2>&1 | tee "$output_file"; then
            echo "ERROR: $method experiment with seed $seed failed!"
            return 1
        fi
    else
        # Complex shell command - use eval
        if ! eval "$cmd" 2>&1 | tee "$output_file"; then
            echo "ERROR: $method experiment with seed $seed failed!"
            return 1
        fi
    fi
    
    echo "$method experiment with seed $seed completed successfully."
    echo ""
    return 0
}

# Run experiments
for seed in $SEEDS; do
    if [[ "$METHOD" == "drl" ]] || [[ "$METHOD" == "both" ]]; then
        if ! run_experiment "drl" "$seed" "$DRL_CONFIG"; then
            EXIT_CODE=1
        fi
    fi
    
    if [[ "$METHOD" == "ea" ]] || [[ "$METHOD" == "both" ]]; then
        if ! run_experiment "ea" "$seed" "$EA_CONFIG"; then
            EXIT_CODE=1
        fi
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

exit $EXIT_CODE
