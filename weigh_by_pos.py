import plotly.express as px
import pandas as pd
import numpy as np
import torch
import os
import argparse
from utils import *
from pos_lm.data.dictionary import Dictionary
from pos_lm.data.utils import load_tokens_from_lines
from tabulate import tabulate
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble model for language modeling")
    parser.add_argument("--dataset", type=str, default="WikiText-103", help="Dataset name")
    parser.add_argument("--models", nargs='+', default=[], help="List of model names to use for ensemble")
    parser.add_argument("--models-all-but", nargs='+', default=[], help="List of model names to exclude from ensemble")
    parser.add_argument("--weigh-by-pos", type=str, default=None, choices=["xpos", "upos"], help="Use POS tags to weigh the models")
    parser.add_argument("--pos-prob-path", type=str, default=None, help="Path to the POS probabilities")
    parser.add_argument("--train-with-predictions", action='store_true', help="Train with predicted POS tags instead of gold POS tags")
    parser.add_argument("--train-val-split", type=float, default=None, help="Fraction of data to use for training when optimizing weights by POS tags")
    return parser.parse_args()

def plot_cumulative_state(df_line: pd.DataFrame, df_pie: pd.DataFrame, outfile: str, dataset_name: str):
    fig_line_chart = px.line(
        df_line,
        x="Model",
        y="Perplexity",
        title="Perplexity of best models on " + dataset_name+ " dataset",
    )

    fig_pie_chart = px.pie(
        df_pie,
        values='Weight',
        names='Model',
        title='Weights of a models in ensemble on ' + dataset_name + ' dataset',
    )

    with open(outfile, 'a') as f:
        f.write(fig_line_chart.to_html(full_html=False, include_plotlyjs='cdn', default_height="70%", default_width="70%"))
        f.write(fig_pie_chart.to_html(full_html=False, include_plotlyjs='cdn',  default_height="70%", default_width="70%"))

def load_pos_tags(pos_name: str = None):
    # Load pos dictionary
    pos_dict = Dictionary(path = f"../pos_lm_v2/data/wikitext-103-stanza/{pos_name}_vocab.json")

    # Load pos_tokens
    pos_tokens = {split: load_tokens_from_lines(f"../pos_lm_v2/data/wikitext-103-stanza/{split}/{pos_name}", pos_dict.eos_index)[1:] \
        for split in ["validation", "test"]}

    return pos_dict, pos_tokens

def load_pos_predictions(num_pos, pos_prob_path: str = None):
    pos_predictions = None
    dataset_sizes = {
    "validation": 217646,  # Example size for validation set
    "test": 245569         # Example size for test set
}
    if pos_prob_path is not None:
        # Load pos predictions
        pos_predictions = {split: 
            torch.from_numpy(
                np.memmap(f"{pos_prob_path}/{split}_full_prob.npy", shape=(dataset_sizes[split], num_pos), dtype=np.float32)
                ).exp()
                for split in ["validation", "test"]
            }
    return pos_predictions

def load_model_probabilities(args, path):
    """Load model probabilities from the specified path."""
    print('\n' + '-' * 50)
    print('\n' + path)

    path = os.path.relpath(path)
    val_files = sorted(os.listdir(os.path.join(path, 'valid')))
    test_files = sorted(os.listdir(os.path.join(path, 'test')))

    #Take only models specified in args if any
    if len(args.models) > 0:
        val_files = [f for f in val_files if f.replace('.txt', '') in args.models]
        test_files = [f for f in test_files if f.replace('.txt', '') in args.models]
    if len(args.models_all_but) > 0:
        val_files = [f for f in val_files if not f.replace('.txt', '') in args.models_all_but]
        test_files = [f for f in test_files if not f.replace('.txt', '') in args.models_all_but]

    val_files_parsed = [f.replace('.txt', '') for f in val_files]
    test_files_parsed = [f.replace('.txt', '') for f in test_files]
    assert val_files_parsed == test_files_parsed, 'Different names for validation and test files'

    val_probabilities = np.vstack(
        [combine_prob_text(os.path.join(path, 'valid', file_name)) for file_name in val_files])
    test_probabilities = np.vstack(
        [combine_prob_text(os.path.join(path, 'test', file_name)) for file_name in test_files])
    #list for storing individual loss on test set
    test_los_individual = []
    print("\nIndividual test ppl of models")
    for name, i in zip(test_files, test_probabilities):
        # skip unigram cache
        if "unigram" in name:
            continue
        test_los_individual.append(calculate_sequence_loss(i)[1])
        print(name + ": " + str(round(calculate_sequence_loss(i)[1], 2)))

    return val_files, val_probabilities, test_files, test_probabilities, test_los_individual

if __name__ == "__main__":
    args = parse_args()
    if os.path.exists("index.html"):
        os.remove("index.html")
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Get dataset paths
    rel_paths = args.dataset
    if isinstance(rel_paths, str):
        rel_paths = [rel_paths]

    if args.weigh_by_pos:
        # Load POS tags and dictionary
        pos_dict, pos_tokens = load_pos_tags(args.weigh_by_pos)
        # Load POS predictions if path is provided
        pos_predictions = load_pos_predictions(len(pos_dict), args.pos_prob_path)

    for path in rel_paths:
        val_files, val_probabilities, test_files, test_probabilities, test_los_individual = load_model_probabilities(args, path)
        print("\nIndividual valid ppl of models")
        for name, i in zip(val_files, val_probabilities):
            # skip unigram cache
            if "unigram" in name:
                continue
            print(name + ": " + str(round(calculate_sequence_loss(i)[1], 2)))
        
        if args.weigh_by_pos:
            print("\nIndividual valid ppl of models for each POS tag")
            pos_loss_table = {}  
            for name, i in zip(val_files, val_probabilities):
                loss_by_pos = calculate_sequence_loss_per_pos(i, pos_tokens['validation'], pos_dict)
                for pos, (_,ppl) in loss_by_pos.items():
                    if pos not in pos_loss_table:
                        pos_loss_table[pos] = {}
                    pos_loss_table[pos][name] = ppl
            # Build the table: rows are POS tags, columns are model names
            tabulate_data(pos_loss_table, val_files, emphasize="min")
           
        # Optimize ensemble weights using POS tags
        if args.weigh_by_pos:
            model, weights_by_pos = optimize_ensemble_weights_by_pos(
                                                                val_probabilities,
                                                                pos_dict,
                                                                pos_tokens=pos_tokens['validation'],
                                                                pos_predictions=pos_predictions['validation'] if pos_predictions is not None else None,
                                                                train_val_split=args.train_val_split,
                                                                )
            weights_by_pos_dict = {pos: weights_by_pos[:,i] for i, pos in enumerate(pos_dict.symbols)}
            weights_by_pos_table = {pos: {name: weight for name, weight in zip(val_files, weights_by_pos_dict[pos])} for pos in weights_by_pos_dict}
            tabulate_data(weights_by_pos_table, val_files, emphasize="max")
            val_pos_weighted_loss = weigh_by_pos(val_probabilities, pos_tokens['validation'], pos_dict, weights_by_pos_dict)
            test_pos_weighted_loss = weigh_by_pos(test_probabilities, pos_tokens['test'], pos_dict, weights_by_pos_dict)
            print('\nPOS Weighted Validation Perplexity using Correct POS: ', math.exp(val_pos_weighted_loss))
            print('POS Weighted Test Perplexity using Correct POS: ', math.exp(test_pos_weighted_loss))
            val_pos_weighted_loss_pred = model(
                                                torch.from_numpy(val_probabilities),
                                                pos_predictions=pos_predictions["validation"])
            test_pos_weighted_loss_pred = model(
                                                torch.from_numpy(test_probabilities),
                                                pos_predictions=pos_predictions["test"])
            print('POS Weighted Validation Perplexity using Predicted POS: ', math.exp(val_pos_weighted_loss_pred))
            print('POS Weighted Test Perplexity using Predicted POS: ', math.exp(test_pos_weighted_loss_pred))
        else:
            weights = optimise_ensemble_weights(val_probabilities)

            val_file_prob = (weights[:, np.newaxis] * val_probabilities).sum(axis=0)
            test_file_prob = (weights[:, np.newaxis] * test_probabilities).sum(axis=0)

            val_loss, val_ppl = calculate_sequence_loss(val_file_prob)
            test_loss, test_ppl = calculate_sequence_loss(test_file_prob)
            test_los_individual.append(test_ppl)

            print('\nValidation Perplexity: ', val_ppl)
            print('Test Perplexity: ', test_ppl)

            print("\nName of files with weights")
            for name, w in zip(test_files_parsed, weights):
                print(name + ': ' + str(round(w, 2)))

            df_line = pd.DataFrame(list(zip(test_files_parsed+['Ensemble of All'], test_los_individual)),
                                columns=['Model', 'Perplexity'])
            df_line = df_line.sort_values(by=['Perplexity'], ascending=False)

            df_pie = pd.DataFrame(list(zip(test_files_parsed, weights)),
                            columns=['Model', 'Weight'])

            plot_cumulative_state(df_line, df_pie, "index.html", path)