python weigh_by_pos.py \
    --models AdaptiveInputsMine my_adt KNNLM_v2 \
    --weigh-by-pos xpos \
    --pos-prob-path ../pos_lm_v2/checkpoints/0501_232107/results/ \
    --train-with-predictions \
    --train-val-split 0.9 \
