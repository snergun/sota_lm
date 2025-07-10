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
    parser.add_argument("--weigh-by-pos", action='store_true', help="Use POS tags to weigh the models")
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

def load_pos_tags():
    # Load pos dictionary
    pos_dict = Dictionary(path = "../pos_lm_v2/data/wikitext-103-stanza/upos_vocab.json")

    # Load pos_tokens
    pos_tokens = {split: load_tokens_from_lines(f"../pos_lm_v2/data/wikitext-103-stanza/{split}/upos", pos_dict.eos_index)[1:] \
        for split in ["validation", "test"]}

    return pos_dict, pos_tokens

if __name__ == "__main__":
    args = parse_args()
    if os.path.exists("index.html"):
        os.remove("index.html")
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    rel_paths = args.dataset
    if isinstance(rel_paths, str):
        rel_paths = [rel_paths]

    # Load POS tags and dictionary
    pos_dict, pos_tokens = load_pos_tags()

    for path in rel_paths:
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
           
        #list for storing individual loss on test set
        test_los_individual = []
        print("\nIndividual test ppl of models")
        for name, i in zip(test_files, test_probabilities):
            # skip unigram cache
            if "unigram" in name:
                continue
            test_los_individual.append(calculate_sequence_loss(i)[1])
            print(name + ": " + str(round(calculate_sequence_loss(i)[1], 2)))


        weights = optimise_ensemble_weights(val_probabilities)
        if args.weigh_by_pos:
            weights_by_pos = optimize_ensemble_weights_by_pos(val_probabilities, pos_tokens['validation'], pos_dict)
            weights_by_pos_table = {pos: {name: weight for name, weight in zip(val_files, weights_by_pos[pos])} for pos in weights_by_pos}
            tabulate_data(weights_by_pos_table, val_files, emphasize="max")
            val_pos_weighted_loss = weigh_by_pos(val_probabilities, pos_tokens['validation'], pos_dict, weights_by_pos)
            test_pos_weighted_loss = weigh_by_pos(test_probabilities, pos_tokens['test'], pos_dict, weights_by_pos)
            print('\nPOS Weighted Validation Perplexity: ', math.exp(val_pos_weighted_loss))
            print('POS Weighted Test Perplexity: ', math.exp(test_pos_weighted_loss))

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