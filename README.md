# Spectra-Lab 🌱

**Soil property prediction from spectra (pXRF / vis-NIR / MIR).**
Calibrate and compare regression models, validate against a held-out external
set, and export reproducible, citable models — built on scikit-learn.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why

Spectroscopy-based soil analysis needs more than a single fitted model: you want
to compare methods, prove predictive skill on an **independent** validation set,
and keep the workflow reproducible for publication. Spectra-Lab does this in a
browser UI with no coding required, while using validated scikit-learn
implementations under the hood.

## Features

- **Six models, compared side-by-side** — PLSR, PCR, Ridge, k-NN, SVR (RBF),
  Random Forest. Each is tuned by cross-validation and ranked by external RPD.
- **Honest external validation** — Kennard-Stone sample selection holds out a
  set that spans the feature space (interpolated, not extrapolated), plus random
  and CV-only options.
- **Preprocessing** — optional SNV scatter correction; optional band autoscaling
  for PLSR/PCR (required for the distance/kernel/penalty models).
- **Interactive spectra view** — raw or SNV-previewed traces, colourable by a
  target value to see whether spectral shape tracks the property.
- **Multi-target** — model several soil properties (e.g. ECe, SOC, pH, CaCO3) in
  one run.
- **Model export/import** — download fitted pipelines as `.joblib` and reload
  them later to predict without retraining. Preprocessing travels inside the
  pipeline, so the artifact is self-contained.
- **Manual & batch prediction** — type a spectrum or upload a CSV; export results.

## Metrics

Per property and model: **R2cv / RMSECV / RPDcv** (cross-validation) and, when a
split is used, **R2p / RMSEP / RPDp / bias** (external).
RPD > 2 usable, > 2.5 good, > 3 excellent for screening.

---

## Quick start

```bash
git clone https://github.com/marocmatrix/spectra-lab.git
cd spectra-lab
pip install -r requirements.txt
streamlit run app.py
```

Open the printed URL (default `http://localhost:8501`).

### Input format

A CSV/TSV where each row is a sample. Columns: a sample ID, the spectral bands
(numeric-named, e.g. `400nm`, `450nm`), and one or more measured properties
(e.g. `ECe`, `SOC`). Bands and targets are auto-detected and adjustable in the
UI. A separate reference file can be joined on the ID column.

---

## Project structure

```
app.py                 Streamlit UI (load -> spectra -> compare -> predict)
models.py              scikit-learn registry, CV tuning, evaluation, save/load
utils.py               I/O, preprocessing, Kennard-Stone split, metrics
requirements.txt       dependencies
setup.sh               one-shot venv installer (Raspberry Pi / Linux)
spectra-lab.service    systemd unit for background hosting
.streamlit/config.toml headless server settings
DEPLOY.md              Raspberry Pi + Cloudflare deployment guide
```

## Deployment

Run it as a service on a Raspberry Pi and expose it through a Cloudflare tunnel —
see **[DEPLOY.md](DEPLOY.md)** for the full walkthrough (venv + systemd +
`cloudflared` ingress + optional Cloudflare Access login).

---

## Notes & limitations

- Kennard-Stone builds a full N x N distance matrix; fine to a few thousand
  samples, switch to the random split for very large libraries.
- Random Forest / SVR are the heaviest fits — lower tree counts on low-power
  hosts like a Pi.

## License

MIT — see [LICENSE](LICENSE).
