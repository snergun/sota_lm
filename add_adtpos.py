import torch 
import numpy as np
adtpos_probs = {"val" : np.exp(np.memmap("/work/hdd/bcyi/eergun/pos_lm_v2/checkpoints/my_adt/results/validation_prob.npy", dtype='float32', mode='r', shape=(217646,))),
                "test" : np.exp(np.memmap("/work/hdd/bcyi/eergun/pos_lm_v2/checkpoints/my_adt/results/test_prob.npy", dtype='float32', mode='r', shape=(245569,)))}
# adtpos_probs = {k :v["modified_logits"].exp().cpu().numpy() for k,v in adtpos_results.items()}

#Write probs to file, each prob in new line:
with open("WikiText-103/valid/my_adt.txt", "w") as f:
    for prob in adtpos_probs["val"]:
        f.write(str(prob) + "\n")
with open("WikiText-103/test/my_adt.txt", "w") as f:
    for prob in adtpos_probs["test"]:
        f.write(str(prob) + "\n")
