python weigh_by_pos.py \
    --models-all-but ADTPOS my_adt KNNLM_v2 pos_adaptive \
    --weigh-by-pos xpos \
    --pos-prob-path ../pos_lm_v2/checkpoints/sweep/0501_232107/results/ \
    --train-with-predictions \
    --train-val-split 0.9 \
    >> weigh_by_pos_results.txt