"""
utils.py — data handling, preprocessing, sample splitting and metrics
for the soil-spectra PLSR / ML prediction app.

Pure functions, no Streamlit imports, so this module stays testable on its own.
"""
from __future__ import annotations

import io
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# --------------------------------------------------------------------------- #
#  I/O
# --------------------------------------------------------------------------- #
def read_table(file) -> pd.DataFrame:
    """Read a CSV/TSV/TXT upload into a DataFrame, auto-detecting the separator.

    `file` may be a path, a file-like object, or raw bytes/str.
    """
    if hasattr(file, "read"):
        raw = file.read()
    elif isinstance(file, (bytes, bytearray)):
        raw = file
    else:  # path
        with open(file, "rb") as fh:
            raw = fh.read()

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig", errors="replace")
    raw = raw.replace("\r", "")

    first = raw.split("\n", 1)[0]
    sep = "\t" if first.count("\t") > first.count(",") else ","
    return pd.read_csv(io.StringIO(raw), sep=sep)


def guess_columns(df: pd.DataFrame):
    """Heuristically classify columns into (id_col, spectral_bands, target_candidates).

    - Spectral bands: numeric columns whose *header* parses to a number
      (e.g. '400', '1350.5', 'w350nm', '2.5kev').
    - Targets: numeric columns whose header is a named property (not purely numeric).
    - ID: first column matching id/sample/name/code, else the first column.
    """
    cols = list(df.columns)

    id_guess = next(
        (c for c in cols if re.search(r"^id$|sample|name|code", str(c), re.I)),
        cols[0],
    )

    def numeric_frac(col):
        s = pd.to_numeric(df[col], errors="coerce")
        return s.notna().mean()

    def header_is_numberish(col):
        # strip common unit letters then test for a bare number
        stripped = re.sub(r"[a-z_ ]", "", str(col), flags=re.I)
        return bool(re.fullmatch(r"-?\d*\.?\d+", stripped))

    bands, targets = [], []
    for c in cols:
        if c == id_guess:
            continue
        nf = numeric_frac(c)
        if nf > 0.8 and header_is_numberish(c):
            bands.append(c)
        elif nf > 0.6 and not re.fullmatch(r"[\d.\s]+", str(c)):
            targets.append(c)

    # fallback: if no header-numeric bands found, treat all high-numeric cols as bands
    if len(bands) < 2:
        bands = [c for c in cols if c != id_guess and numeric_frac(c) > 0.8]
        targets = [c for c in targets if c not in bands]

    return id_guess, bands, targets


def band_axis(bands) -> np.ndarray:
    """Return a numeric x-axis parsed from band labels, or an index fallback."""
    nums = []
    for b in bands:
        m = re.search(r"-?\d+\.?\d*", str(b))
        nums.append(float(m.group()) if m else None)
    if all(n is not None for n in nums):
        return np.asarray(nums, dtype=float)
    return np.arange(len(bands), dtype=float)


def build_xy(df_spectra, df_ref, id_col, bands, targets):
    """Assemble aligned X (bands) and Y (targets) arrays, dropping incomplete rows.

    If df_ref is None, targets are read from df_spectra itself. Otherwise the two
    frames are joined on id_col.
    """
    if df_ref is not None:
        merged = df_spectra.merge(df_ref, on=id_col, how="inner", suffixes=("", "_ref"))
    else:
        merged = df_spectra

    X = merged[bands].apply(pd.to_numeric, errors="coerce")
    Y = merged[targets].apply(pd.to_numeric, errors="coerce")
    ids = merged[id_col].astype(str).values

    mask = X.notna().all(axis=1) & Y.notna().all(axis=1)
    return (
        X[mask].to_numpy(dtype=float),
        Y[mask].to_numpy(dtype=float),
        ids[mask.to_numpy()],
    )


# --------------------------------------------------------------------------- #
#  Preprocessing
# --------------------------------------------------------------------------- #
def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: per-row mean-centre and scale to unit SD."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, ddof=1, keepdims=True)
    sd[sd == 0] = 1e-9
    return (X - mu) / sd


# --------------------------------------------------------------------------- #
#  Sample splitting
# --------------------------------------------------------------------------- #
def kennard_stone_split(X: np.ndarray, n_cal: int):
    """Kennard-Stone selection on autoscaled features.

    Returns (cal_idx, val_idx). Calibration samples are chosen to span the
    feature space, so the held-out validation set is interpolated, not
    extrapolated.
    """
    n = X.shape[0]
    n_cal = max(2, min(n_cal, n))

    # autoscale so every band weighs equally in the distance metric
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1e-9
    Z = (X - mu) / sd

    D = cdist(Z, Z)  # full pairwise distance matrix

    # seed with the two most distant points
    i, j = np.unravel_index(np.argmax(D), D.shape)
    selected = [int(i), int(j)]
    remaining = set(range(n)) - set(selected)

    while len(selected) < n_cal and remaining:
        rem = np.fromiter(remaining, dtype=int)
        # for each remaining point, distance to its nearest selected point
        min_to_sel = D[np.ix_(rem, selected)].min(axis=1)
        chosen = int(rem[np.argmax(min_to_sel)])  # maximise that minimum
        selected.append(chosen)
        remaining.discard(chosen)

    cal = np.array(sorted(selected))
    val = np.array(sorted(remaining))
    return cal, val


def random_split(n: int, n_cal: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_cal = max(2, min(n_cal, n))
    return np.sort(idx[:n_cal]), np.sort(idx[n_cal:])


def make_split(X, method: str, cal_frac: float):
    """Dispatch to the chosen split method. method in {'ks','random','none'}."""
    n = X.shape[0]
    if method == "none":
        return np.arange(n), np.array([], dtype=int)
    n_cal = max(6, round(n * cal_frac))
    if method == "ks":
        return kennard_stone_split(X, n_cal)
    return random_split(n, n_cal)


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """R2, RMSE, RPD, RPIQ and bias for a set of predictions."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = obs.size
    resid = obs - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2)) or 1e-12
    rmse = float(np.sqrt(ss_res / n))
    sd = float(np.std(obs, ddof=1)) or 1e-9
    q1, q3 = np.percentile(obs, [25, 75])
    return {
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": rmse,
        "rpd": sd / rmse if rmse else np.inf,
        "rpiq": (q3 - q1) / rmse if rmse else np.inf,
        "bias": float(np.mean(pred - obs)),
        "n": int(n),
    }


def rpd_class(rpd: float) -> str:
    """Qualitative band for an RPD value (used for colour cues in the UI)."""
    if rpd >= 2.5:
        return "good"
    if rpd >= 2.0:
        return "ok"
    return "poor"
