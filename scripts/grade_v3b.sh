#!/usr/bin/env bash
# Grade the v3b decoding-variant predictions against the DTW-aligned real heldout,
# then print the comparison table (v3b variants vs v3 / curr2 / curr / coder7b).
#
# v3b tests two decoders on the SAME v3 adapters to break the phrase-looping that
# capped v3 (variety 0.11, just under the 0.15 anti-drone gate):
#   ngram : --repetition-penalty 1.2 --no-repeat-ngram-size 6  (block long verbatim loops only)
#   temp  : --repetition-penalty 1.3 --temperature 0.5         (mild sampling)
#
# Bring back these files from Colab into runs/ (names produced by colab_curriculum_v3.md Cell 5):
#   runs/v3b_n2w_ngram_preds.jsonl   runs/v3b_w2n_ngram_preds.jsonl
#   runs/v3b_n2w_temp_preds.jsonl    runs/v3b_w2n_temp_preds.jsonl
#
# Usage: bash scripts/grade_v3b.sh
set -euo pipefail
cd "$(dirname "$0")/.."

N2W_EVAL="data/byzantine/sft_aligned_n2w_heldout.jsonl"
W2N_EVAL="data/byzantine/sft_aligned_w2n_heldout.jsonl"

grade() {  # <eval> <pred> <out-tag>
  local eval="$1" pred="$2" out="runs/$3_realscore.json"
  if [[ ! -f "$pred" ]]; then
    echo "SKIP: missing $pred" >&2
    return
  fi
  echo ">>> grading $pred -> $out"
  python3 scripts/score_real_musical.py --eval "$eval" --pred "$pred" --out "$out"
}

# ngram variant
grade "$N2W_EVAL" runs/v3b_n2w_ngram_preds.jsonl v3b_ngram_n2w
grade "$W2N_EVAL" runs/v3b_w2n_ngram_preds.jsonl v3b_ngram_w2n
# temp variant
grade "$N2W_EVAL" runs/v3b_n2w_temp_preds.jsonl  v3b_temp_n2w
grade "$W2N_EVAL" runs/v3b_w2n_temp_preds.jsonl  v3b_temp_w2n
# ngram8 variant (looser long-loop block)
grade "$N2W_EVAL" runs/v3b_n2w_ngram8_preds.jsonl v3b_ngram8_n2w
grade "$W2N_EVAL" runs/v3b_w2n_ngram8_preds.jsonl v3b_ngram8_w2n

echo
echo "=================== comparison ==================="
python3 scripts/compare_realscores.py \
  --order coder7b curr curr2 v3 v3b_ngram v3b_ngram8 v3b_temp

cat <<'EOF'

Read the tables like this:
- WINNER = highest above_gate_music with above_gate_rows climbing well past v3's 67/501,
  and variety landing in a healthy ~0.3-0.6 band (NOT ~0 drone, NOT curr2's ~0.8 hallucination).
- Guard against reward-hacking: if variety jumps but set_f1 / hist_sim / interval_hist_sim
  FALL vs v3, the decoder is manufacturing novel-but-wrong tokens (v2's failure) -- reject it.
- n2w leads; w2n ceiling ~1.2 (oligon/petaste both +1), judge w2n by set_f1 / hist_sim.
EOF
