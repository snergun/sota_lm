import numpy as np
import torch
import os
from tabulate import tabulate
import torch.nn.functional as F
from tqdm import tqdm
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
        Instantiate weights of the ensemble model.
        """
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones((num_models, num_pos)))

    def forward(self, probabilities, pos_tokens=None, pos_predictions=None):
        """
        Compute cross-entropy loss used in perplexity, using vectorized operations.
        
        Arguments:
        - probabilities: Tensor of shape (num_models, num_tokens)
        - pos_tokens: LongTensor of shape (num_tokens,) with POS tag IDs (0 <= ID < num_pos)
        """
        assert pos_tokens is not None or pos_predictions is not None, "Either pos_tokens or pos_predictions must be provided"
        # Apply softmax to weights over the model axis
        weights_softmax = torch.softmax(self.weights, dim=0)  # shape: (num_models, num_pos)

        # Gather the softmax weights corresponding to the POS of each token
        # pos_tokens: (num_tokens,) → weights_selected: (num_models, num_tokens)
        if pos_predictions is None:
            weights_selected = weights_softmax.gather(dim=1, index=pos_tokens.unsqueeze(0).expand(probabilities.size(0), -1))
        else:
            # pos_predictions: (num_tokens, num_pos) → (num_pos, num_tokens)
            pos_predictions_t = pos_predictions.t()
            # Compute weighted sum of weights by predicted POS probabilities
            # Resulting shape: (num_models, num_tokens)
            weights_selected = torch.matmul(weights_softmax, pos_predictions_t)
        # Compute weighted probabilities (elementwise multiply and sum across models)
        weighted_probs = torch.sum(probabilities * weights_selected, dim=0)  # shape: (num_tokens,)

        # Compute negative log-likelihood loss (mean over tokens)
        loss = -torch.mean(torch.log(weighted_probs + 1e-12))  # add epsilon to avoid log(0)
        return loss

def optimise_ensemble_weights(probabilities: np.ndarray, num_steps: int = 5000, lr: float=0.05):
    """
    Find optimal linear weights for the ensemble model given the word probabilities for each model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probabilities = torch.from_numpy(probabilities).to(device)
    model = Perplexity_loss(probabilities.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scaled_weights = None
    for t in tqdm(range(num_steps)):
        # Forward pass
        loss = model(probabilities)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 0:
            ppl = torch.exp(loss)
            scaled_weights = torch.softmax(model.weights, dim=0).detach().cpu().numpy()[:, 0]
            # print('Loss:', loss.item(), 'Perplexity:', ppl.item(), 'Weights:', scaled_weights)

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

def optimize_ensemble_weights_by_pos(probabilities, pos_dict, lr: float=0.05, num_steps: int = 5000, pos_tokens = None, pos_predictions=None, train_val_split=None, use_correct_pos=False):
    """    
    Find optimal linear weights for the ensemble model given the word probabilities for each model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probabilities = torch.from_numpy(probabilities).to(device)
    pos_predictions = pos_predictions.to(device) if pos_predictions is not None else None
    pos_tokens = torch.tensor(pos_tokens, dtype=torch.long).to(device) if pos_tokens is not None else None
    model = Perplexity_loss_per_POS(probabilities.shape[0], len(pos_dict)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if train_val_split is not None:
        print(f"Using first {train_val_split*100}% of data for training, rest for validation")
    cut_index = int(probabilities.shape[1] * train_val_split) if train_val_split is not None else probabilities.shape[1]
    scaled_weights = None
    best_val_loss = float('inf')
    iter_without_improvement = 0
    using_pos_predictions = pos_predictions is not None and not use_correct_pos
    if using_pos_predictions:
        print("Using predicted POS tags for training")
    else:
        print("Using correct POS tags for training")
    for t in range(num_steps):
        # Forward pass
        loss = model(probabilities[:,:cut_index], pos_tokens, pos_predictions[:cut_index] if using_pos_predictions else None)
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 0:
            ppl = torch.exp(loss)
            scaled_weights = torch.softmax(model.weights, dim=0).detach().cpu().numpy()
            if train_val_split is not None:
                val_loss = model(probabilities[:,cut_index:], pos_tokens, pos_predictions[cut_index:] if using_pos_predictions else None)
                if best_val_loss - val_loss > 1e-4:
                    best_val_loss = val_loss
                    iter_without_improvement = 0
                else:
                    iter_without_improvement += 1
                val_ppl = torch.exp(val_loss)
                print(f'Step {t}: Train Loss: {loss.item():.4f}, Train Perplexity: {ppl.item():.4f} | Val Loss: {val_loss.item():.4f}, Val Perplexity: {val_ppl.item():.4f}')
            else:
                print('Loss:', loss.item(), 'Perplexity:', ppl.item())
        if iter_without_improvement >= 5:
            print("Early stopping due to no improvement in validation loss")
            break
    return model, scaled_weights

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

