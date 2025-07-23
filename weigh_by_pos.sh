python weigh_by_pos.py \
    --models AdaptiveInputsMine my_adt KNNLM_v2 \
    --weigh-by-pos xpos \
    --pos-prob-path ../pos_lm_v2/checkpoints/0501_232107/results/ \
        ../pos_lm_v2/checkpoints/adaptive_lm_wiki103.v2/results/xpos \
        ../pos_lm_v2/checkpoints/my_adt/results/xpos \
    --train-with-predictions \
    --train-val-split 0.9 \
