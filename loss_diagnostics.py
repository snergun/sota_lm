import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import combine_prob_text, optimise_ensemble_weights

def load_probs(namelist, dirpath):
    return [combine_prob_text(os.path.join(dirpath, f"{n}.txt")) for n in namelist]

def plot_loss_gap(loss_i, loss_j, names, outpath):
    delta = loss_i - loss_j
    plt.figure(figsize=(12, 3))
    plt.plot(delta, linewidth=0.5)
    plt.axhline(0, linewidth=1, color='black')
    plt.xlabel('Token index')
    plt.ylabel(f'Loss({names[0]}) − Loss({names[1]})')
    plt.title(f'Per-token loss gap ({names[0]} vs {names[1]})')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_run_lengths(loss_i, loss_j, names, outpath):
    delta = loss_i - loss_j
    wins_i = delta <= 0  # True when model i wins
    runs, cur = [], 1
    for k in range(1, len(wins_i)):
        if wins_i[k] == wins_i[k-1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    plt.figure(figsize=(6, 4))
    plt.hist(runs, bins=np.arange(1, 51)-0.5, log=True, edgecolor='black')
    plt.xlabel('Run length')
    plt.ylabel('Frequency')
    plt.title(f'Run-lengths of the better model ({names[0]} vs {names[1]})')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_loss_corr(losses, namelist, outpath):
    corr = np.corrcoef(losses, rowvar=False)
    M = corr.shape[0]
    plt.figure(figsize=(M, M))
    im = plt.imshow(corr, cmap='winter', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(M), namelist, rotation=90)
    plt.yticks(range(M), namelist)
    plt.title('Loss correlation matrix')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_run_lengths_either(losses, names, model1, model0, outpath):
    """
    Plot empirical run-length distribution for runs of either model,
    overlaid with theoretical P(run length=k) = p^k(1-p) + (1-p)^k p.
    
    losses: np.ndarray, shape (N, M) of per-token -log probs.
    names: list of model names, length M.
    model1, model0: names of the two models to compare.
    outpath: path to save PNG.
    """
    i = names.index(model1)
    j = names.index(model0)

    # Boolean: True if model1 wins, False if model0 wins
    wins1 = losses[:, i] < losses[:, j]

    # Extract run lengths for both True and False runs
    runs = []
    cur = 1
    for prev, curr in zip(wins1, wins1[1:]):
        if curr == prev:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)

    # Empirical PMF
    max_k = max(runs)
    k_vals = np.arange(1, max_k + 1)
    counts = np.array([runs.count(k) for k in k_vals])
    empirical = counts / counts.sum()

    # Probability p of model1 winning any token
    p = wins1.mean()

    # Theoretical PMF for runs of either model:
    # P1-run(k) = p^k (1-p), P0-run(k) = (1-p)^k p
    theoretical = (p**k_vals) * (1 - p) + ((1 - p)**k_vals) * p

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(k_vals, empirical, width=0.8, alpha=0.6, label='Empirical')
    plt.plot(k_vals, theoretical, 'r-o', label=f'Theoretical (p={p:.3f})')
    plt.xlabel('Run length of consecutive wins (either model)')
    plt.ylabel('Probability')
    plt.title(f'Empirical vs. Theoretical Run-Lengths: {model1} vs {model0}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def run_diagnostics(probs_list, weights, namelist, prefix, compare_idx):
    # Build loss matrix
    losses = np.stack([-np.log(p) for p in probs_list], axis=1)  # shape (N, M)
    # 1) Loss gap for two selected models
    i, j = compare_idx
    plot_loss_gap(losses[:, i], losses[:, j], [namelist[i], namelist[j]], f"{prefix}_loss_gap.png")
    # 2) Run-lengths for same two
    plot_run_lengths(losses[:, i], losses[:, j], [namelist[i], namelist[j]], f"{prefix}_run_lengths.png")
    # 2.1) Run-lengths for either model with independent asssumption as baseline
    plot_run_lengths_either(losses, namelist, namelist[i], namelist[j], f"{prefix}_run_lengths_either.png")
    # 2.2) Run-length
    # 3) Correlation for all
    plot_loss_corr(losses, namelist, f"{prefix}_loss_corr.png")
    # 4) Oracle vs static gap
    stacked = np.vstack(probs_list)  # shape (M, N)
    oracle_loss = -np.mean(np.log(np.max(stacked, axis=0)))
    static_loss = -np.mean(np.log(weights.dot(stacked)))
    gap = (static_loss - oracle_loss) / np.log(2)
    print(f"{prefix}: Oracle PPL = {np.exp(oracle_loss):.3f}, Static PPL = {np.exp(static_loss):.3f}, Max gain = {gap:.3f} bits/token\n")

def main():
    parser = argparse.ArgumentParser(description="Run ensemble diagnostics on text-formatted probs")
    parser.add_argument("--dataset", default="WikiText-103",
                        help="Root path with 'valid/' and 'test/' dirs")
    parser.add_argument("--models", nargs='+', default=[],
                        help="List of model names to include")
    parser.add_argument("--models-all-but", nargs='+', default=[],
                        help="List of model names to exclude")
    parser.add_argument("--compare", nargs=2, metavar=('M1', 'M2'), default=None,
                        help="Two model names for loss-gap/run-length diagnostics")
    parser.add_argument("--prefix", default="diagnostic",
                        help="Filename prefix for plots")
    parser.add_argument("--save-dir", default="diagnostics",
                        help="Directory to save plots")
    args = parser.parse_args()

    # Gather file names
    val_dir, test_dir = os.path.join(args.dataset, "valid"), os.path.join(args.dataset, "test")
    val_names = sorted(f[:-4] for f in os.listdir(val_dir) if f.endswith('.txt'))
    # Filter
    if args.models:
        val_names = [n for n in val_names if n in args.models]
    if args.models_all_but:
        val_names = [n for n in val_names if n not in args.models_all_but]
    test_names = sorted(f[:-4] for f in os.listdir(test_dir) if f.endswith('.txt'))
    names = [n for n in val_names if n in test_names]
    assert set(names) == set(val_names), "Validation/Test mismatch after filtering"

    # Load probabilities
    val_probs = load_probs(names, val_dir)
    test_probs= load_probs(names, test_dir)

    # Fit static weights on validation
    weights = optimise_ensemble_weights(np.vstack(val_probs))
    print("Models:", names)
    print("Static weights:", dict(zip(names, weights.round(3))))

    # Determine which two models to compare
    if args.compare:
        assert args.compare[0] in names and args.compare[1] in names, "Compare models must be in selected set"
        compare_idx = [names.index(args.compare[0]), names.index(args.compare[1])]
    else:
        # pick top-2 by static weight
        top2 = np.argsort(weights)[-2:][::-1]  # indices of largest two
        compare_idx = [int(top2[0]), int(top2[1])]
    os.makedirs(args.save_dir, exist_ok=True)
    # Run diagnostics
    run_diagnostics(val_probs, weights, names, os.path.join(args.save_dir, args.prefix + "_val"), compare_idx)
    run_diagnostics(test_probs, weights, names, os.path.join(args.save_dir, args.prefix + "_test"), compare_idx)

if __name__ == "__main__":
    main()
