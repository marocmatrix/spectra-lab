"""
app.py — Spectra-Lab: soil property prediction from spectra.

A Streamlit application that calibrates and compares regression models
(PLSR, PCR, Ridge, k-NN, SVR, Random Forest) on user-supplied spectra and
reference data, with Kennard-Stone external validation and model export.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import utils
import models as M


# --------------------------------------------------------------------------- #
#  Page config & light theming
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Spectra-Lab · Soil Prediction", layout="wide")

LEAF, SOIL, MUTED = "#7fb069", "#c98a3e", "#8fa291"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      h1 {letter-spacing:-0.02em;}
      .stTabs [data-baseweb="tab-list"] {gap: 4px;}
      div[data-testid="stMetricValue"] {font-size: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Spectra-Lab")
st.caption(
    "Predict soil properties from spectra (pXRF / vis-NIR / MIR). "
    "Calibrate, validate against a held-out set, compare models, and export."
)

# session state holders
ss = st.session_state
ss.setdefault("df_cal", None)
ss.setdefault("df_ref", None)
ss.setdefault("results", None)     # dict: target -> ranked list of model results
ss.setdefault("bundle", None)      # exportable payload
ss.setdefault("bands", None)
ss.setdefault("targets", None)


# --------------------------------------------------------------------------- #
#  Plot helpers
# --------------------------------------------------------------------------- #
def spectra_figure(X, bands, color_vals=None, color_label=None, snv_preview=False):
    Xp = utils.snv(X) if snv_preview else X
    xaxis = utils.band_axis(bands)
    fig = go.Figure()

    # cap drawn traces for responsiveness
    step = max(1, len(Xp) // 400)
    if color_vals is not None:
        cmin, cmax = float(np.min(color_vals)), float(np.max(color_vals))
    for i in range(0, len(Xp), step):
        if color_vals is not None:
            t = (color_vals[i] - cmin) / ((cmax - cmin) or 1)
            col = f"rgb({int(154+(127-154)*t)},{int(95+(176-95)*t)},{int(36+(105-36)*t)})"
            fig.add_trace(go.Scatter(x=xaxis, y=Xp[i], mode="lines",
                                     line=dict(width=0.8, color=col),
                                     opacity=0.55, showlegend=False,
                                     hoverinfo="skip"))
        else:
            fig.add_trace(go.Scatter(x=xaxis, y=Xp[i], mode="lines",
                                     line=dict(width=0.8, color=LEAF),
                                     opacity=0.35, showlegend=False,
                                     hoverinfo="skip"))

    is_index = np.allclose(xaxis, np.arange(len(bands)))
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Band index" if is_index else "Wavelength / channel",
        yaxis_title="SNV response" if snv_preview else "Response",
        template="plotly_white",
    )
    return fig


def scatter_figure(res, target):
    fig = go.Figure()
    obs, pred = res["cv_obs"], res["cv_pred"]
    lo = min(obs.min(), pred.min())
    hi = max(obs.max(), pred.max())
    fig.add_trace(go.Scatter(x=obs, y=pred, mode="markers",
                             marker=dict(size=7, color=LEAF, opacity=0.55),
                             name="CV (calibration)"))
    if res["ext"] is not None:
        eo, ep = res["ext_obs"], res["ext_pred"]
        lo = min(lo, eo.min(), ep.min())
        hi = max(hi, eo.max(), ep.max())
        fig.add_trace(go.Scatter(x=eo, y=ep, mode="markers",
                                 marker=dict(size=9, color=SOIL, symbol="circle-open",
                                             line=dict(width=2)),
                                 name="External"))
    pad = (hi - lo) * 0.05 or 1
    fig.add_trace(go.Scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad],
                             mode="lines", line=dict(color=SOIL, dash="dash"),
                             name="1:1", hoverinfo="skip"))
    fig.update_layout(
        height=420, template="plotly_white",
        xaxis_title=f"Measured {target}", yaxis_title=f"Predicted {target}",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


# --------------------------------------------------------------------------- #
#  STEP 1 — load data
# --------------------------------------------------------------------------- #
st.header("1 · Calibration data")

c1, c2 = st.columns(2)
with c1:
    up_cal = st.file_uploader(
        "Spectra + reference (one table)", type=["csv", "tsv", "txt"], key="up_cal"
    )
    if up_cal is not None:
        ss.df_cal = utils.read_table(up_cal)
with c2:
    up_ref = st.file_uploader(
        "Optional — separate reference file (joined on ID)",
        type=["csv", "tsv", "txt"], key="up_ref",
    )
    ss.df_ref = utils.read_table(up_ref) if up_ref is not None else None

if ss.df_cal is None:
    st.info("Upload a calibration table to begin. "
            "Rows are samples; columns hold a sample ID, spectral bands, and measured properties.")
    st.stop()

df = ss.df_cal
st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns.")

id_guess, band_guess, target_guess = utils.guess_columns(df)
# allow targets to come from the ref file when present
target_pool = (
    [c for c in ss.df_ref.columns if c != id_guess] if ss.df_ref is not None else target_guess
)

cc1, cc2, cc3 = st.columns([1, 2, 2])
with cc1:
    id_col = st.selectbox("Sample ID column", df.columns,
                          index=list(df.columns).index(id_guess))
with cc2:
    targets = st.multiselect("Target properties", target_pool,
                             default=[t for t in target_guess if t in target_pool] or target_pool[:1])
with cc3:
    band_pool = [c for c in df.columns if c != id_col]
    bands = st.multiselect("Spectral bands", band_pool,
                           default=[b for b in band_guess if b in band_pool])

if len(bands) < 2 or not targets:
    st.warning("Select at least two spectral bands and one target property.")
    st.stop()

X_all, Y_all, ids_all = utils.build_xy(df, ss.df_ref, id_col, bands, targets)
if X_all.shape[0] < 8:
    st.error(f"Only {X_all.shape[0]} complete samples after dropping rows with missing values. "
             "Need at least ~8 to calibrate.")
    st.stop()
st.caption(f"{X_all.shape[0]} complete samples · {len(bands)} bands · {len(targets)} target(s).")


# --------------------------------------------------------------------------- #
#  Spectral overview
# --------------------------------------------------------------------------- #
st.header("Spectral overview")
sc1, sc2 = st.columns([1, 1])
with sc1:
    snv_prev = st.checkbox("Preview SNV scatter correction", value=False)
with sc2:
    color_by = st.selectbox("Colour spectra by", ["Uniform"] + targets)

color_vals = None
clabel = None
if color_by != "Uniform":
    color_vals = Y_all[:, targets.index(color_by)]
    clabel = color_by
st.plotly_chart(spectra_figure(X_all, bands, color_vals, clabel, snv_prev),
                use_container_width=True)


# --------------------------------------------------------------------------- #
#  STEP 2 — configure & compare
# --------------------------------------------------------------------------- #
st.header("2 · Models & validation")

MODEL_CHOICES = {
    "pls": "PLSR", "pcr": "PCR", "ridge": "Ridge",
    "knn": "k-NN", "svr": "SVR (RBF)", "rf": "Random Forest",
}
chosen = st.multiselect(
    "Models to compare", list(MODEL_CHOICES.keys()),
    default=["pls", "rf", "svr"],
    format_func=lambda k: MODEL_CHOICES[k],
)

with st.expander("Hyperparameters & preprocessing", expanded=False):
    h1, h2, h3 = st.columns(3)
    with h1:
        pls_max = st.number_input("PLSR max components", 1, 30, 10)
        pcr_max = st.number_input("PCR max components", 1, 30, 10)
        pls_autoscale = st.checkbox("PLSR/PCR autoscale bands (unit variance)", value=True)
    with h2:
        ridge_alphas = st.text_input("Ridge alpha grid", "0.01,0.1,1,10,100")
        knn_ks = st.text_input("k-NN neighbour grid", "3,5,7,9,15")
    with h3:
        svr_C = st.number_input("SVR C", 0.01, 1e4, 10.0)
        svr_gamma = st.text_input("SVR gamma", "scale")
        rf_trees = st.number_input("RF trees", 10, 1000, 200)
        rf_depth = st.number_input("RF max depth (0 = unlimited)", 0, 100, 0)

    use_snv = st.checkbox("Apply SNV before modelling", value=False)
    folds = st.number_input("Cross-validation folds", 2, 20, 5)

st.subheader("Calibration / external validation split")
s1, s2 = st.columns([1, 1])
with s1:
    split_method = st.selectbox(
        "Split method",
        ["ks", "random", "none"],
        format_func=lambda k: {"ks": "Kennard-Stone (spectral coverage)",
                               "random": "Random",
                               "none": "No split (CV only)"}[k],
    )
with s2:
    cal_pct = st.slider("Calibration share (%)", 50, 90, 75)

st.caption(
    "Kennard-Stone picks calibration samples spanning the full feature space, "
    "so the held-out set is interpolated rather than extrapolated. Models are "
    "tuned by CV on calibration only; external metrics are the honest test."
)

if st.button("Calibrate & compare", type="primary"):
    cfg = {
        "pls_max": pls_max, "pcr_max": pcr_max,
        "ridge_alphas": [float(x) for x in ridge_alphas.split(",") if x.strip()],
        "knn_ks": [int(float(x)) for x in knn_ks.split(",") if x.strip()],
        "svr_C": svr_C,
        "svr_gamma": float(svr_gamma) if svr_gamma.replace(".", "", 1).isdigit() else svr_gamma,
        "rf_trees": rf_trees, "rf_depth": (None if rf_depth == 0 else rf_depth),
        "pls_autoscale": pls_autoscale,
    }
    registry = M.default_registry(cfg)

    cal_idx, val_idx = utils.make_split(X_all, split_method, cal_pct / 100)
    if split_method != "none" and len(val_idx) < 2:
        st.error("Validation set too small. Lower the calibration share.")
        st.stop()

    Xc, Yc = X_all[cal_idx], Y_all[cal_idx]
    Xv, Yv = (X_all[val_idx], Y_all[val_idx]) if len(val_idx) else (None, None)

    results_by_target = {}
    prog = st.progress(0.0, text="Fitting models…")
    total = len(targets) * len(chosen)
    done = 0
    for ti, target in enumerate(targets):
        yc = Yc[:, ti]
        yv = Yv[:, ti] if Yv is not None else None
        per_model = []
        for key in chosen:
            spec = registry[key]
            res = M.tune_and_evaluate(
                spec, Xc, yc, Xv, yv,
                use_snv=use_snv,
                autoscale_optional=pls_autoscale,
                folds=folds,
            )
            per_model.append(res)
            done += 1
            prog.progress(done / total, text=f"{target} · {spec.name}")
        results_by_target[target] = M.rank_models(per_model)
    prog.empty()

    ss.results = results_by_target
    ss.bands = bands
    ss.targets = targets
    ss.bundle = M.export_bundle(
        results_by_target, bands,
        meta={"split": split_method, "n_cal": int(len(cal_idx)),
              "n_val": int(len(val_idx)), "use_snv": use_snv},
    )


# --------------------------------------------------------------------------- #
#  STEP 3 — results
# --------------------------------------------------------------------------- #
if ss.results:
    st.header("3 · Comparison")
    meta = ss.bundle["meta"]
    if meta["n_val"] > 0:
        st.info(f"{ {'ks':'Kennard-Stone','random':'Random','none':'No split'}[meta['split']] } "
                f"split — {meta['n_cal']} calibration / {meta['n_val']} external · "
                f"{len(ss.results[ss.targets[0]])} models compared.")
    else:
        st.info("No external split — ranked by cross-validation RPD.")

    for target, ranked in ss.results.items():
        st.subheader(f"{target}  ·  best: {ranked[0]['name']} ({ranked[0]['tag']})")

        rows = []
        for i, r in enumerate(ranked):
            row = {"#": "★" if i == 0 else i + 1, "Model": r["name"], "Tuning": r["tag"],
                   "R²cv": r["cv"]["r2"], "RMSECV": r["cv"]["rmse"], "RPDcv": r["cv"]["rpd"]}
            if r["ext"] is not None:
                row |= {"R²p": r["ext"]["r2"], "RMSEP": r["ext"]["rmse"],
                        "RPDp": r["ext"]["rpd"], "Bias": r["ext"]["bias"]}
            rows.append(row)
        tbl = pd.DataFrame(rows).set_index("#")
        st.dataframe(
            tbl.style.format({c: "{:.3f}" for c in tbl.columns if tbl[c].dtype != object}),
            use_container_width=True,
        )
        st.plotly_chart(scatter_figure(ranked[0], target), use_container_width=True)

    # export
    buf = io.BytesIO()
    joblib.dump(ss.bundle, buf)
    st.download_button(
        "⬇ Download fitted models (.joblib)", buf.getvalue(),
        file_name="soil_models.joblib", mime="application/octet-stream",
    )


# --------------------------------------------------------------------------- #
#  STEP 4 — predict
# --------------------------------------------------------------------------- #
st.header("4 · Predict new samples")

pred_src = st.radio("Model source", ["Use models from this session", "Upload a saved .joblib"],
                    horizontal=True)

payload = None
if pred_src == "Use models from this session":
    if ss.bundle is None:
        st.info("Calibrate models above first, or switch to uploading a saved bundle.")
    else:
        payload = ss.bundle
else:
    up_model = st.file_uploader("Saved model bundle", type=["joblib"], key="up_model")
    if up_model is not None:
        payload = joblib.load(io.BytesIO(up_model.read()))
        st.success(f"Loaded bundle · {len(payload['bands'])} bands · "
                   f"targets: {', '.join(payload['targets'].keys())}")

if payload is not None:
    p_bands = payload["bands"]
    p_targets = list(payload["targets"].keys())

    model_pick = st.selectbox(
        "Predict with",
        ["Best per property"] + sorted({k for t in p_targets for k in payload["targets"][t]["models"]}),
    )
    key_override = None if model_pick == "Best per property" else model_pick

    ptab1, ptab2 = st.tabs(["Manual entry", "Batch upload"])

    with ptab1:
        st.caption(f"Enter values for all {len(p_bands)} bands.")
        cols = st.columns(4)
        vals = []
        for i, b in enumerate(p_bands):
            with cols[i % 4]:
                vals.append(st.number_input(str(b), value=0.0, format="%.4f", key=f"mb_{i}"))
        if st.button("Predict sample"):
            x = np.array(vals, dtype=float).reshape(1, -1)
            out = {t: float(M.predict_with_bundle(payload, x, t, key_override)[0]) for t in p_targets}
            st.dataframe(pd.DataFrame([out]).style.format("{:.3f}"), use_container_width=True)

    with ptab2:
        up_pred = st.file_uploader("Prediction spectra (same band columns)",
                                   type=["csv", "tsv", "txt"], key="up_pred")
        if up_pred is not None:
            dfp = utils.read_table(up_pred)
            missing = [b for b in p_bands if b not in dfp.columns]
            if missing:
                st.error(f"File is missing {len(missing)} band column(s), e.g. {missing[:5]}")
            else:
                Xp = dfp[p_bands].apply(pd.to_numeric, errors="coerce")
                ok = Xp.notna().all(axis=1)
                Xp = Xp[ok].to_numpy(float)
                idcol = next((c for c in dfp.columns if c.lower() in ("id", "sample", "name", "code")), None)
                ids = dfp[idcol][ok].astype(str).values if idcol else np.arange(1, len(Xp) + 1)
                res = {"id": ids}
                for t in p_targets:
                    res[t] = M.predict_with_bundle(payload, Xp, t, key_override)
                out_df = pd.DataFrame(res)
                st.dataframe(out_df.style.format({t: "{:.3f}" for t in p_targets}),
                             use_container_width=True)
                st.download_button("⬇ Download predictions CSV",
                                   out_df.to_csv(index=False).encode(),
                                   "soil_predictions.csv", "text/csv")
