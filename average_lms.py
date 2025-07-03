import math 
model_1 = AdaptiveInputs
for split in ["valid", "test"]:
    with open(f"WikiText-103/{split}/AdaptiveInputs.txt", "r") as f:
        lines = f.readlines()
    
    probs1 = [float(line.strip()) for line in lines]
    perplexity_1 = math.exp(sum(-1 * math.log(prob) for prob in probs1) / len(probs1))
    print(f"Long context {split} perplexity")
    print(perplexity_1)
    with open(f"WikiText-103/{split}/my_adt.txt", "r") as f:
        lines2 = f.readlines()
    probs2 = [float(line.strip()) for line in lines2]
    perplexity_2 = math.exp(sum(-1 * math.log(prob) for prob in probs2) / len(probs2))
    print(f"Short context {split} perplexity")
    print(perplexity_2)

    #perplexity of average
    averag_probs = [(p1 + p2) / 2 for p1, p2 in zip(probs1, probs2)]
    perplexity_avg = math.exp(sum(-1 * math.log(prob) for prob in averag_probs) / len(averag_probs))
    print(perplexity_avg)

