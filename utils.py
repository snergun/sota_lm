import numpy as np
import torch
import os
from tabulate import tabulate

class Perplexity_loss(torch.nn.Module):
    def __init__(self, num_models: int):
        """
        We instantiate weights of the ensemble model.
        """
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones((num_models, 1)))

    def forward(self, probabilities):
        """
        We calculate directly the cross entropy loss used in Perplexity.
        """
        weights_softmax = torch.softmax(self.weights, dim=0)
        lin_comb = weights_softmax * probabilities
        lin_comb_sum = torch.sum(lin_comb, dim=0)
        loss = -1 * torch.mean(torch.log(lin_comb_sum))
        return loss

class Perplexity_loss_per_POS(torch.nn.Module):
    def __init__(self, num_models: int, num_pos: int):
        """
        We instantiate weights of the ensemble model.
        """
        super().__init__()
        self.weights_by_pos = torch.nn.Parameter(torch.ones((num_pos, num_models )))

    def forward(self, probabilities, pos_tokens):
        """
        We calculate directly the cross entropy loss used in Perplexity.
        """
        weights_softmax = torch.softmax(self.weights_by_pos, dim=1)
        weighted_probs = torch.zeros_like(probabilities)
        for pos_id in range(weights_softmax.shape[0]):
            pos_indices = np.isin(pos_tokens, pos_id)
            weighted_probs[:, pos_indices] = weights_softmax[pos_id] * probabilities[:, pos_indices]
        weighted_probs = torch.sum(weighted_probs, dim=0)
        loss = -1 * torch.mean(torch.log(weighted_probs))
        return loss

def optimise_ensemble_weights(probabilities: np.ndarray, num_steps: int = 5000, lr: float=0.05):
    """
    Find optimal linear weights for the ensemble model given the word probabilities for each model.
    """
    probabilities = torch.from_numpy(probabilities)
    model = Perplexity_loss(probabilities.shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scaled_weights = None
    for t in range(num_steps):
        # Forward pass
        loss = model(probabilities)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 0:
            ppl = torch.exp(loss)
            scaled_weights = torch.softmax(model.weights, dim=0).detach().numpy()[:, 0]
            #print('Loss:', loss.item(), 'Perplexity:', ppl.item(), 'Weights:', scaled_weights)

    return scaled_weights


def softmax(x):
    x = np.exp(x)
    f_x = x / np.sum(x)
    return f_x


def combine_prob_text(txt_file):
    arr = []
    with open(txt_file, encoding="utf-8") as f:
        for line in f:
            temp = float(line)
            arr.append(temp)
    arr = np.array(arr)
    return arr


# given sequence of predicted and actual tokens - calculate CE loss
def calculate_sequence_loss(x):
    L = -1 * np.mean(np.log(x))
    ppl = np.exp(L)
    return L, ppl

def calculate_sequence_loss_per_pos(x, pos_tokens, pos_dict):
    """
    Calculate the sequence loss for each POS tag.
    :param x: predicted probabilities
    :param pos_tokens: list of POS tokens
    :param pos_dict: dictionary mapping POS tags to indices
    :return: dictionary with POS tags as keys and their corresponding losses
    """
    pos_losses = {}
    for pos_id in range(len(pos_dict)):
        pos_word = pos_dict.symbols[pos_id]
        pos_indices = np.isin(pos_tokens, pos_id)
        if np.sum(pos_indices) == 0:
            continue
        pos_probs = x[pos_indices]
        loss, ppl = calculate_sequence_loss(pos_probs)
        pos_losses[pos_word] = (loss, ppl)
    return pos_losses

def optimize_ensemble_weights_by_pos(probabilities, pos_tokens, pos_dict):
    """
    Optimize ensemble weights by POS tags.
    :param probabilities: array of probabilities for each model
    :param pos_tokens: list of POS tokens
    :param pos_dict: dictionary mapping POS tags to indices
    :param default_weight: default weight for the ensemble
    :return: dictionary with POS tags as keys and their corresponding weights
    """
    weights_by_pos = {}
    for pos_id in range(len(pos_dict)):
        pos_word = pos_dict.symbols[pos_id]
        pos_indices = np.isin(pos_tokens, pos_id)
        if np.sum(pos_indices) == 0:
            continue
        pos_probs = probabilities[:,pos_indices]
        weights_by_pos[pos_word] = optimise_ensemble_weights(pos_probs, num_steps=100)    
    return weights_by_pos

def tabulate_data(table, model_names, emphasize=None):
    headers = ["POS"] + model_names
    table_data = []
    for pos in sorted(table.keys()):
        row = [pos]
        data = table.get(pos, None)
        if data is None:
            continue
        # Find min loss value (ignoring missing)
        emph_data = None
        if emphasize:
            emph_data = min(data.values()) if emphasize == "min" else max(data.values())

        for model in model_names:
            value = data.get(model, None)
            if value is None:
                row.append("NA")
            elif value == emph_data:
                # Emphasize lowest loss — bold or asterisk-style depending on output
                row.append(f"*{value:.2f}*")  # or f"**{loss:.2f}**" for markdown, or ANSI color if supported
            else:
                row.append(f"{value:.2f}")
        table_data.append(row)
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

def weigh_by_pos(probs, pos_tokens, pos_dict, weights_by_pos):
    loss = 0
    for pos_id in range(len(pos_dict)):
        pos_word = pos_dict.symbols[pos_id]
        pos_indices = np.isin(pos_tokens, pos_id)
        if np.sum(pos_indices) == 0:
            continue
        pos_probs = probs[:,pos_indices]
        weights = weights_by_pos.get(pos_word, None)
        weighted_probs = (weights[:, np.newaxis] * pos_probs).sum(axis=0)
        loss += np.sum(-np.log(weighted_probs))
    return loss / probs.shape[1]  # Average loss over all tokens
#
# if __name__ == '__main__':
#     np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#     # start a new wandb run to track this script
#     # wandb.init(
#     #     # set the wandb project where this run will be logged
#     #     project="ensemble_hyperparams",
#     #     entity="hereldav"
#     # )
#
#     rel_paths = ['ptb-ppl', 'wt2-ppl', 'wt103-ppl']
#     for path in rel_paths:
#         print('\n'+'-'*50)
#         print('\n'+path)
#
#         path = os.path.relpath(path)
#         val_files = sorted(os.listdir(os.path.join(path, 'valid')))
#         test_files = sorted(os.listdir(os.path.join(path, 'test')))
#
#
#         val_files_parsed = [f.replace('valid','') for f in val_files]
#         test_files_parsed = [f.replace('test','') for f in test_files]
#         assert val_files_parsed == test_files_parsed, 'Different names for validation and test files'
#
#         val_probabilities = np.vstack([combine_prob_text(os.path.join(path,'valid', file_name)) for file_name in val_files])
#         test_probabilities = np.vstack([combine_prob_text(os.path.join(path, 'test', file_name)) for file_name in test_files])
#
#         print("\nIndividual valid ppl of models")
#         for name,i in zip(val_files,val_probabilities):
#             #skip unigram cache
#             if "unigram" in name:
#                 continue
#             print(name +": "+str(round(calculate_sequence_loss(i)[1],2)))
#
#         print("\nIndividual test ppl of models")
#         for name, i in zip(test_files, test_probabilities):
#             # skip unigram cache
#             if "unigram" in name:
#                 continue
#             print(name + ": " + str(round(calculate_sequence_loss(i)[1], 2)))
#
#         weights = optimise_ensemble_weights(val_probabilities)
#         # print('\nModel path', path)
#         # print('Final weights', weights)
#
#         val_file_prob = (weights[:, np.newaxis] * val_probabilities).sum(axis=0)
#         test_file_prob = (weights[:, np.newaxis] * test_probabilities).sum(axis=0)
#
#         val_loss, val_ppl = calculate_sequence_loss(val_file_prob)
#         test_loss, test_ppl = calculate_sequence_loss(test_file_prob)
#         print('\nValidation Perplexity: ', val_ppl)
#         print('Test Perplexity: ', test_ppl)
#
#         print("\nName of files with weights")
#         for name, w in zip(test_files_parsed, weights):
#             print(name+': '+str(round(w, 2)))

