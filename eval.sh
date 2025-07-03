python main.py --models my_adt pos_adaptive >> results.txt
python main.py --models my_adt pos_adaptive AdaptiveInputs >> results.txt
python main.py --models my_adt pos_adaptive AdaptiveInputs KNNLM_v2 >> results.txt
python main.py --models-all-but KNNLM_v2 >> results.txt
python main.py --models-all-but KNNLM >> results.txt