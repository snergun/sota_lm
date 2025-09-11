python weigh_by_pos.py \
    --models KNN AdaptiveInputsMine \
    --weigh-by-pos xpos \
    --pos-clusters-path /home/jovyan/pos_lm_v2/data/wikitext-103-stanza/upos_clusters \
    --pos-prob-path /home/jovyan/pos_lm_v2/checkpoints/0602_182449/results \
    --train-val-split 0.9 \
    --data-dir /home/jovyan/pos_lm_v2/data/wikitext-103-stanza
