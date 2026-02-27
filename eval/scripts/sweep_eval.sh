#!/bin/bash
# =============================================================================
# sweep_eval.sh — Hyperparameter sweep for eval generation parameters
#
# Usage:
#   bash scripts/sweep_eval.sh [round]
#     round=1  gen_length/block_length sweep (default)
#     round=2  temperature/cfg_scale sweep (edit BEST_GL/BEST_BL first)
#     round=3  steps sweep (edit BEST_* first)
#     round=all  run all rounds sequentially
#     round=summary  only print summary of existing results
#
# Set EVAL_LIMIT=500 for fast screening, unset for full dataset (3858 samples).
# Results are saved per-experiment under eval/exp/ and summarized at the end.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."  # cd to eval/

ROUND=${1:-1}
EVAL_LIMIT=${EVAL_LIMIT:-500}
export EVAL_LIMIT

# ---- Best params from previous rounds (update after each round) ----
BEST_GL=${BEST_GL:-160}
BEST_BL=${BEST_BL:-160}
BEST_STEPS=${BEST_STEPS:-${BEST_GL}}
BEST_TEMP=${BEST_TEMP:-0}
BEST_CFG=${BEST_CFG:-0}

SUMMARY_FILE="exp/sweep_summary.txt"
mkdir -p exp

# =============================================================================
# Helper: run one experiment
# =============================================================================
run_experiment() {
    local gl=$1 bl=$2 steps=$3 temp=$4 cfg=$5
    local tag="gl${gl}_bl${bl}_s${steps}_t${temp}_cfg${cfg}"
    echo ""
    echo "================================================================"
    echo "  Experiment: $tag  (limit=$EVAL_LIMIT)"
    echo "================================================================"

    GEN_LENGTH=$gl BLOCK_LENGTH=$bl STEPS=$steps \
    TEMPERATURE=$temp CFG_SCALE=$cfg \
    bash scripts/evaluate.sh

    echo "  >> Finished: $tag"
}

# =============================================================================
# Helper: extract metrics from results.json files
# =============================================================================
collect_results() {
    echo ""
    echo "================================================================"
    echo "  SWEEP RESULTS SUMMARY"
    echo "================================================================"
    printf "%-45s  %8s  %8s  %8s  %8s\n" "Experiment" "ROUGE-L" "METEOR" "BLEU-1" "BLEU-4"
    printf "%-45s  %8s  %8s  %8s  %8s\n" "---------" "-------" "------" "------" "------"

    # Find all results.json under exp/, pick the latest per output dir
    for results_dir in exp/llava_v_core_eval_*/; do
        [ -d "$results_dir" ] || continue
        # Find the latest results.json
        latest=$(find "$results_dir" -name "*_results.json" -type f 2>/dev/null | sort | tail -1)
        [ -z "$latest" ] && continue

        # Extract metrics using python
        metrics=$(python3 -c "
import json, sys
with open('$latest') as f:
    d = json.load(f)
r = d.get('results', {}).get('mimic_cxr', {})
rl = r.get('rouge_l,none', 'N/A')
mt = r.get('meteor,none', 'N/A')
b1 = r.get('bleu_1,none', 'N/A')
b4 = r.get('bleu_4,none', 'N/A')
gl = d.get('config', {}).get('gen_kwargs', {}).get('gen_length', '?')
bl = d.get('config', {}).get('gen_kwargs', {}).get('block_length', '?')
st = d.get('config', {}).get('gen_kwargs', {}).get('steps', '?')
tp = d.get('config', {}).get('gen_kwargs', {}).get('temperature', '?')
cf = d.get('config', {}).get('gen_kwargs', {}).get('cfg_scale', '?')
lim = d.get('config', {}).get('limit', 'full')
if lim is None: lim = 'full'
tag = f'gl{gl}_bl{bl}_s{st}_t{tp}_cfg{cf} (n={lim})'
print(f'{tag}|{rl}|{mt}|{b1}|{b4}')
" 2>/dev/null) || continue

        IFS='|' read -r tag rl mt b1 b4 <<< "$metrics"
        if [ "$rl" != "N/A" ]; then
            printf "%-45s  %8.4f  %8.4f  %8.4f  %8.4f\n" "$tag" "$rl" "$mt" "$b1" "$b4"
        fi
    done | sort -t'(' -k1,1

    echo ""
    echo "(Results also saved to $SUMMARY_FILE)"
}

# =============================================================================
# Round 1: gen_length + block_length joint search
# =============================================================================
run_round1() {
    echo "===== Round 1: gen_length + block_length sweep ====="
    echo "Fixed: temperature=0, cfg_scale=0, remasking=low_confidence, steps=gen_length"
    echo ""

    # (gen_length, block_length) combinations
    run_experiment 100 100 100 0 0
    run_experiment 120 120 120 0 0
    run_experiment 120  60 120 0 0
    run_experiment 120  40 120 0 0
    run_experiment 160 160 160 0 0
    run_experiment 160  80 160 0 0
    run_experiment 160  40 160 0 0
    run_experiment 200 200 200 0 0
    run_experiment 200 100 200 0 0
    run_experiment 200  40 200 0 0
}

# =============================================================================
# Round 2: temperature + cfg_scale search
# =============================================================================
run_round2() {
    echo "===== Round 2: temperature + cfg_scale sweep ====="
    echo "Using best from Round 1: gl=$BEST_GL, bl=$BEST_BL, steps=$BEST_STEPS"
    echo ""

    local gl=$BEST_GL bl=$BEST_BL s=$BEST_STEPS

    # baseline (0, 0) likely already run in round 1, skip if exists
    run_experiment $gl $bl $s 0   0
    run_experiment $gl $bl $s 0.2 0
    run_experiment $gl $bl $s 0.5 0
    run_experiment $gl $bl $s 0   0.5
    run_experiment $gl $bl $s 0   1.0
    run_experiment $gl $bl $s 0   2.0
    run_experiment $gl $bl $s 0.2 1.0
}

# =============================================================================
# Round 3: steps optimization
# =============================================================================
run_round3() {
    echo "===== Round 3: steps optimization ====="
    echo "Using best: gl=$BEST_GL, bl=$BEST_BL, temp=$BEST_TEMP, cfg=$BEST_CFG"
    echo ""

    local gl=$BEST_GL bl=$BEST_BL

    # steps = gen_length (baseline, already run)
    run_experiment $gl $bl $gl          $BEST_TEMP $BEST_CFG
    # steps = gen_length / 2
    run_experiment $gl $bl $((gl / 2))  $BEST_TEMP $BEST_CFG
    # steps = gen_length / 4
    run_experiment $gl $bl $((gl / 4))  $BEST_TEMP $BEST_CFG
}

# =============================================================================
# Main dispatch
# =============================================================================
case "$ROUND" in
    1)
        run_round1
        collect_results | tee "$SUMMARY_FILE"
        ;;
    2)
        run_round2
        collect_results | tee "$SUMMARY_FILE"
        ;;
    3)
        run_round3
        collect_results | tee "$SUMMARY_FILE"
        ;;
    all)
        run_round1
        echo ""
        echo ">> Round 1 complete. Review results and set BEST_GL/BEST_BL before Round 2."
        collect_results | tee "$SUMMARY_FILE"
        echo ""
        echo ">> Continuing to Round 2 with BEST_GL=$BEST_GL, BEST_BL=$BEST_BL ..."
        run_round2
        collect_results | tee "$SUMMARY_FILE"
        echo ""
        echo ">> Continuing to Round 3 with BEST_TEMP=$BEST_TEMP, BEST_CFG=$BEST_CFG ..."
        run_round3
        collect_results | tee "$SUMMARY_FILE"
        ;;
    summary)
        collect_results | tee "$SUMMARY_FILE"
        ;;
    *)
        echo "Usage: bash scripts/sweep_eval.sh [1|2|3|all|summary]"
        exit 1
        ;;
esac

echo ""
echo "Sweep done."
