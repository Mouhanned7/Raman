"""SERS interlaboratory dataset (adenine): download, load, preprocess.

Source: Fornasaro et al., Anal. Chem. 2020, Zenodo record 3572359 (open access).
3516 spectra, 15 laboratories, 4 substrates, 2 lasers, 400-1999 cm-1 (534 points).
Target: adenine concentration (the 'conc' column, in the dataset's units).
"""
import os
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

URL = "https://zenodo.org/record/3572359/files/ILSdata.csv?download=1"
META_COLS = ["labcode", "substrate", "laser", "method", "sample", "type", "conc",
             "batch", "replica"]


def download(data_dir, tries=6):
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, "ILSdata.csv")
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    for k in range(tries):
        try:
            print("Downloading the SERS dataset from Zenodo (10 MB)...")
            with urllib.request.urlopen(URL, timeout=120) as r, open(dest, "wb") as fh:
                fh.write(r.read())
            return dest
        except Exception as e:
            if k == tries - 1:
                raise RuntimeError("Download failed: %s" % e)
            time.sleep(3 * (k + 1))


def load(data_dir):
    """Returns (meta DataFrame, spectra array (N, B), wavenumbers (B,))."""
    df = pd.read_csv(download(data_dir))
    meta = df[META_COLS].copy()
    spec_cols = [c for c in df.columns if c not in META_COLS]
    X = df[spec_cols].to_numpy(dtype=np.float64)
    wn = np.array([float(c) for c in spec_cols])
    # Two laboratories measured a narrower spectral range: keep the common range
    # (columns without any missing value) so that every spectrum is complete.
    ok = ~np.isnan(X).any(axis=0)
    idx = np.where(ok)[0]
    assert np.all(np.diff(idx) == 1), "missing values in the middle of the range"
    return meta, X[:, ok], wn[ok]


def asls_baseline(y, lam=1e5, p=0.01, n_iter=10):
    """Asymmetric least squares baseline (Eilers and Boelens, 2005)."""
    n = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n, n - 2))
    H = lam * D.dot(D.T)
    w = np.ones(n)
    z = y
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve((W + H).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def preprocess(X, baseline=True, norm="snv"):
    """Baseline removal then per-spectrum normalisation.

    norm: 'none', 'snv' (zero mean, unit variance) or 'area' (unit area).
    """
    out = np.empty_like(X)
    for i in range(len(X)):
        s = X[i]
        if baseline:
            s = s - asls_baseline(s)
        out[i] = s
    if norm == "snv":
        out = (out - out.mean(1, keepdims=True)) / (out.std(1, keepdims=True) + 1e-9)
    elif norm == "area":
        out = out / (np.abs(out).sum(1, keepdims=True) + 1e-9)
    elif norm != "none":
        raise ValueError(norm)
    return out
