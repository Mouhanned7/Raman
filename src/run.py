"""Quantify adenine from SERS spectra, with two evaluation protocols.

    python src/run.py            # full run
    python src/run.py --quick    # fast check (PLS + Ridge only)

Protocol A, "official split": the train/test labels given with the dataset
(same laboratories in both sets, different samples).
Protocol B, "leave labs out": 5 folds grouped by laboratory, so every test
laboratory (instrument, operator, substrate) is unseen during training. This is
the realistic protocol: a model has to work in a new lab.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from data import load, preprocess
from models import fit_cnn, fit_pls, fit_ridge


def metrics(y, p):
    err = p - y
    ss_res, ss_tot = np.sum(err ** 2), np.sum((y - y.mean()) ** 2)
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - ss_res / ss_tot)}


def make_predictors(Xtr, ytr, use_cnn, seed):
    pls = fit_pls(Xtr, ytr, seed=seed)
    ridge = fit_ridge(Xtr, ytr)
    preds = {"PLS (%d LV)" % pls.n_components_chosen: lambda X: pls.predict(X).ravel(),
             "Ridge": ridge}
    if use_cnn:
        preds["1D CNN"] = fit_cnn(Xtr, ytr, seed=seed,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
    return preds


def scatter(ax, y, p, title):
    ax.scatter(y, p, s=6, alpha=0.4)
    lim = [0, max(y.max(), p.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(min(p.min(), 0) - 0.5, lim[1])
    ax.set_xlabel("True concentration")
    ax.set_ylabel("Predicted")
    ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--norm", default="area", choices=["none", "snv", "area"])
    ap.add_argument("--baseline", action="store_true", help="remove the baseline (AsLS) first; off by default, see README ablation")
    ap.add_argument("--quick", action="store_true", help="skip the CNN")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta, X, wn = load(args.data)
    y = meta["conc"].to_numpy()
    labs = meta["labcode"].to_numpy()
    print("Spectra: %d, points: %d, labs: %d" % (len(X), X.shape[1], len(set(labs))))
    print("Preprocessing (baseline=%s, norm=%s)..." % (args.baseline, args.norm))
    Xp = preprocess(X, baseline=args.baseline, norm=args.norm)
    use_cnn = not args.quick

    # Protocol A: official split (blanks are added to the training set)
    typ = meta["type"].to_numpy()
    tr, te = np.isin(typ, ["train", "blank"]), typ == "test"
    preds = make_predictors(Xp[tr], y[tr], use_cnn, args.seed)
    resA = {n: (f(Xp[te]), metrics(y[te], f(Xp[te]))) for n, f in preds.items()}

    # Protocol B: leave labs out
    gkf = GroupKFold(5)
    names, store = None, {}
    for k, (a, b) in enumerate(gkf.split(Xp, y, groups=labs)):
        print("Fold %d: test labs %s" % (k + 1, sorted(set(labs[b]))))
        preds = make_predictors(Xp[a], y[a], use_cnn, args.seed)
        for n, f in preds.items():
            key = n.split(" (")[0]
            store.setdefault(key, np.zeros(len(y)))[b] = f(Xp[b])
    resB = {n: (p, metrics(y, p)) for n, p in store.items()}

    lines = ["| Protocol | Model | RMSE | MAE | R² |", "|---|---|---|---|---|"]
    for proto, res in [("A: official split", resA), ("B: leave labs out", resB)]:
        for n, (_, m) in res.items():
            lines.append("| %s | %s | %.3f | %.3f | %.3f |" % (proto, n, m["rmse"], m["mae"], m["r2"]))
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(args.out, "results.md"), "w") as fh:
        fh.write("Preprocessing: baseline=%s, norm=%s\n\n" % (args.baseline, args.norm) + table + "\n")

    best = min(resB, key=lambda n: resB[n][1]["rmse"])
    bestA = list(resA)[min(range(len(resA)), key=lambda i: list(resA.values())[i][1]["rmse"])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    scatter(axes[0], y[te], resA[bestA][0], "A: official split, %s" % bestA)
    scatter(axes[1], y, resB[best][0], "B: leave labs out, %s" % best)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "predictions.png"), dpi=130)
    plt.close(fig)

    # Example of preprocessing
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    i = int(np.argmax(y))
    ax[0].plot(wn, X[i]); ax[0].set_title("Raw spectrum (highest concentration)", fontsize=10)
    ax[1].plot(wn, Xp[i]); ax[1].set_title("Preprocessed (baseline removal: %s, norm: %s)" % (args.baseline, args.norm), fontsize=10)
    for a in ax:
        a.set_xlabel("Raman shift (cm$^{-1}$)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "preprocessing.png"), dpi=130)
    plt.close(fig)
    print("\nFiles written to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
