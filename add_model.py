import torch 
import numpy as np
import os
model_name = "AdaptiveInputsMine"
results_path = "../pos_lm_v2/checkpoints/adaptive_lm_wiki103.v2/results/xpos"
data_lens = {"validation": 217646, "test": 245569}
splits = ["validation", "test"]
probs = {
    split : np.exp(
        np.memmap(
                os.path.join(results_path,f"{split}_prob.npy"),
                dtype='float32',
                mode='r',
                shape=(data_lens[split],)
                           )
        ) for split in splits
    }

#Write probs to file, each prob in new line:
with open(f"WikiText-103/valid/{model_name}.txt", "w") as f:
    for prob in probs["validation"]:
        f.write(str(prob) + "\n")
with open(f"WikiText-103/test/{model_name}.txt", "w") as f:
    for prob in probs["test"]:
        f.write(str(prob) + "\n")
