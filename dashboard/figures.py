"""Plotly figures for the dashboard, built from the tables in results/.

Every builder returns a dict of variants: {key: plotly figure as a dict}.
A figure with one view has a single key, "default". Figures with a
dropdown have one variant per option (keys joined with "|" when there are
two dropdowns), and app.js swaps between them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .theme import (
    ALGO_COLORS, ALGO_LABELS, ALGO_SYMBOLS, ALGOS, AQUA, AXIS, BLUE, CLIN, INK, INK2, MUTED,
    NEUTRAL, ORANGE, RAMP, SHADE, SURFACE, layout,
)
STEPS = ["L1", "L2", "L3", "L4", "L5"]
OFFSETS = dict(zip(ALGOS, [-0.18, -0.06, 0.06, 0.18]))
STUDY_COLORS = {"A": BLUE, "B": ORANGE, "C": AQUA}


def as_dict(fig: go.Figure) -> dict:
    d = json.loads(pio.to_json(fig, validate=True))
    d["layout"].pop("template", None)
    return d


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _algo_marker(algo, size=9, open_=False):
    c = ALGO_COLORS[algo]
    return {
        "color": SURFACE if open_ else c,
        "symbol": ALGO_SYMBOLS[algo],
        "size": size,
        "line": {"color": c if open_ else SURFACE, "width": 2 if open_ else 1.5},
    }


# ------------------------------------------------------------------ ladder
def ladder_timeline(ladder: pd.DataFrame, design: dict) -> dict:
    fig = go.Figure()
    lo, hi = design["outcome_window"]
    fig.add_shape(type="rect", x0=lo, x1=hi, y0=0, y1=1, yref="paper", fillcolor=SHADE, line={"width": 0}, layer="below")
    fig.add_annotation(x=(lo + hi) / 2, y=1, yref="paper", yanchor="bottom", text="outcome window", showarrow=False, font={"size": 11, "color": INK2})
    for day in design["visit_days"].values():
        fig.add_shape(type="line", x0=day, x1=day, y0=0, y1=1, yref="paper", line={"color": AXIS, "width": 1, "dash": "dot"}, layer="below")
    steps = ladder.step.tolist()[::-1]
    for _, r in ladder.iterrows():
        color = BLUE if r.kind == "ladder" else MUTED
        fig.add_trace(go.Scatter(
            x=[0, r.known_day], y=[r.step, r.step], mode="lines", line={"color": color, "width": 2},
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[r.known_day], y=[r.step], mode="markers+text", showlegend=False,
            marker={"color": color, "size": 11, "line": {"color": SURFACE, "width": 2}},
            text=[f"day {r.known_day}, {r.n_features} features"], textposition="middle right", textfont={"size": 11, "color": INK2},
            customdata=[[r.label, int(r.n_features), r.blocks]],
            hovertemplate="<b>%{y}</b> %{customdata[0]}<br>%{customdata[1]} features, all known by day %{x}<br>%{customdata[2]}<extra></extra>",
        ))
    lay = layout("Day since the first infusion (dotted lines: infusions 1 to 4)", "", legend=False)
    lay["yaxis"].update(categoryorder="array", categoryarray=steps, showgrid=False)
    lay["xaxis"].update(range=[-12, hi + 12], showgrid=False, tickvals=[0, 100, 200, 300, 400])
    lay["margin"] = {"l": 8, "r": 8, "t": 22, "b": 8}
    fig.update_layout(**lay)
    return {"default": as_dict(fig)}


def log_ticks(lo: float, hi: float) -> list[float]:
    """1-2-5 ticks inside [lo, hi]; only powers of ten if that gives too many."""
    k0, k1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    ticks = [m * 10.0**k for k in range(k0, k1 + 1) for m in (1, 2, 5) if lo <= m * 10.0**k <= hi]
    if len(ticks) > 8:
        ticks = [t for t in ticks if np.isclose(np.log10(t) % 1, 0)]
    return ticks


# ------------------------------------------------------------------ discrimination
def auroc_ladder(summary: pd.DataFrame, ceiling: float) -> dict:
    main = summary[(summary.imputer == "median") & summary.tag.isin(STEPS + ["L4r"])]
    xpos = {s: i for i, s in enumerate(STEPS)} | {"L4r": len(STEPS) + 0.4}
    out = {}
    for key, col, label in [("repeats", "auc_run_sd", "SD over repeats"), ("folds", "fold_sd", "SD over outer folds")]:
        fig = go.Figure()
        for algo in ALGOS:
            d = main[main.algo == algo].set_index("tag")
            for part, steps, mode in [("ladder", STEPS, "lines+markers"), ("check", ["L4r"], "markers")]:
                dd = d.loc[steps]
                fig.add_trace(go.Scatter(
                    x=[xpos[s] + OFFSETS[algo] for s in steps], y=dd.auc_mean,
                    error_y={"type": "data", "array": dd[col], "color": ALGO_COLORS[algo], "thickness": 1.2, "width": 0},
                    mode=mode, name=ALGO_LABELS[algo], legendgroup=algo, showlegend=part == "ladder",
                    line={"color": ALGO_COLORS[algo], "width": 2}, marker=_algo_marker(algo, open_=part == "check"),
                    customdata=np.column_stack([steps, dd[col]]),
                    hovertemplate=f"<b>{ALGO_LABELS[algo]}</b> %{{customdata[0]}}<br>AUROC %{{y:.3f}} ({label} %{{customdata[1]:.3f}})<extra></extra>",
                ))
        fig.add_hline(y=ceiling, line={"color": MUTED, "width": 1.2, "dash": "dash"})
        fig.add_annotation(x=0, xanchor="left", y=ceiling, yanchor="bottom", text=f"true probability {ceiling:.3f}", showarrow=False, font={"size": 11, "color": INK2})
        lay = layout("Feature group", "Out-of-fold AUROC")
        lay["xaxis"].update(tickvals=list(xpos.values()), ticktext=STEPS + ["L4r (check)"], showgrid=False, range=[-0.5, len(STEPS) + 0.9])
        lay["yaxis"].update(range=[0.7, 1.0] if key == "repeats" else [0.65, 1.02])
        fig.update_layout(**lay)
        out[key] = as_dict(fig)
    return out


def ladder_deltas(deltas: pd.DataFrame, contrasts: pd.DataFrame) -> dict:
    rng = np.random.default_rng(3)
    names = list(dict.fromkeys(deltas.contrast))
    fig = go.Figure()
    for algo in ALGOS:
        d = deltas[deltas.algo == algo]
        x = [names.index(c) + OFFSETS[algo] + rng.uniform(-0.04, 0.04) for c in d.contrast]
        fig.add_trace(go.Scatter(
            x=x, y=d.delta, mode="markers", name=ALGO_LABELS[algo], legendgroup=algo, showlegend=False,
            marker={"color": ALGO_COLORS[algo], "size": 5, "opacity": 0.45, "symbol": ALGO_SYMBOLS[algo]},
            customdata=np.column_stack([d.contrast.str.replace("->", " to "), d.repeat + 1, d.fold + 1]),
            hovertemplate=f"{ALGO_LABELS[algo]}, %{{customdata[0]}}<br>repeat %{{customdata[1]}}, fold %{{customdata[2]}}: %{{y:+.3f}}<extra></extra>",
        ))
        c = contrasts[contrasts.algo == algo].copy()
        c["contrast"] = c["from"] + "->" + c["to"]
        c = c.set_index("contrast").loc[names]
        fig.add_trace(go.Scatter(
            x=[i + OFFSETS[algo] for i in range(len(names))], y=c.delta_mean, mode="markers",
            name=ALGO_LABELS[algo], legendgroup=algo, marker=_algo_marker(algo, size=11),
            customdata=np.column_stack([[n.replace("->", " to ") for n in names], [_fmt_p(p) for p in c.p_corrected], [_fmt_p(p) for p in c.p_naive]]),
            hovertemplate=f"<b>{ALGO_LABELS[algo]}</b>, %{{customdata[0]}}<br>mean change %{{y:+.3f}}<br>corrected p %{{customdata[1]}}, naive p %{{customdata[2]}}<extra></extra>",
        ))
    fig.add_hline(y=0, line={"color": AXIS, "width": 1.2})
    lay = layout("Planned contrast", "Change in AUROC per outer fold")
    lay["xaxis"].update(tickvals=list(range(len(names))), ticktext=[n.replace("->", " to ") for n in names], showgrid=False)
    fig.update_layout(**lay)
    return {"default": as_dict(fig)}


def raw_check(summary: pd.DataFrame, contrasts: pd.DataFrame) -> dict:
    s = summary[summary.imputer == "median"].set_index(["tag", "algo"])
    c = contrasts[(contrasts["from"] == "L4r") & (contrasts["to"] == "L4")].set_index("algo")
    labels = [ALGO_LABELS[a] for a in ALGOS]
    fig = go.Figure()
    for a, lab in zip(ALGOS, labels):
        fig.add_trace(go.Scatter(
            x=[s.loc[("L4r", a), "auc_mean"], s.loc[("L4", a), "auc_mean"]], y=[lab, lab], mode="lines",
            line={"color": AXIS, "width": 3}, hoverinfo="skip", showlegend=False,
        ))
    for step, name, color, sym in [("L4r", "L4r: day-98 data as raw values", ORANGE, "diamond"), ("L4", "L4: day-98 data through the PK/PD model", BLUE, "circle")]:
        vals = [s.loc[(step, a), "auc_mean"] for a in ALGOS]
        sds = [s.loc[(step, a), "auc_run_sd"] for a in ALGOS]
        fig.add_trace(go.Scatter(
            x=vals, y=labels, mode="markers", name=name,
            error_x={"type": "data", "array": sds, "color": color, "thickness": 1.2, "width": 0},
            marker={"color": color, "size": 12, "symbol": sym, "line": {"color": SURFACE, "width": 2}},
            customdata=np.column_stack([sds, [c.loc[a, "delta_mean"] for a in ALGOS], [_fmt_p(c.loc[a, "p_corrected"]) for a in ALGOS]]),
            hovertemplate=f"<b>%{{y}}</b>, {step}<br>AUROC %{{x:.3f}} (SD over repeats %{{customdata[0]:.3f}})<br>L4 minus L4r %{{customdata[1]:+.3f}}, corrected p %{{customdata[2]}}<extra></extra>",
        ))
    lay = layout("Out-of-fold AUROC", "")
    lay["yaxis"].update(categoryorder="array", categoryarray=labels[::-1], showgrid=False)
    lay["xaxis"].update(range=[0.87, 0.95])
    fig.update_layout(**lay)
    return {"default": as_dict(fig)}


# ------------------------------------------------------------------ checks
def loso(lo: pd.DataFrame, cohort: pd.DataFrame, summary: pd.DataFrame) -> dict:
    coh = cohort.set_index("study")
    studies = sorted(lo.held_out.unique())
    ticks = [f"Study {s}<br>{coh.loc[s, 'at_target']:.0%} at target<br>n = {int(coh.loc[s, 'n'])}" for s in studies]
    cv = summary[summary.imputer == "median"].set_index(["tag", "algo"])
    out = {}
    for step in STEPS:
        fig = go.Figure()
        for algo in ALGOS:
            d = lo[(lo.tag == step) & (lo.algo == algo)].set_index("held_out").loc[studies]
            fig.add_trace(go.Scatter(
                x=[i + OFFSETS[algo] for i in range(len(studies))], y=d.auc, mode="markers", name=ALGO_LABELS[algo],
                marker=_algo_marker(algo, size=11),
                customdata=np.column_stack([studies, [cv.loc[(step, algo), "auc_mean"]] * len(studies)]),
                hovertemplate=f"<b>{ALGO_LABELS[algo]}</b>, {step}<br>held out study %{{customdata[0]}}: AUROC %{{y:.3f}}<br>nested CV on all studies: %{{customdata[1]:.3f}}<extra></extra>",
            ))
        lay = layout("Held-out study", "AUROC in the held-out study")
        lay["xaxis"].update(tickvals=list(range(len(studies))), ticktext=ticks, showgrid=False, range=[-0.5, len(studies) - 0.5])
        lay["yaxis"].update(range=[0.6, 1.0])
        fig.update_layout(**lay)
        out[step] = as_dict(fig)
    return out


def checks(summary: pd.DataFrame) -> dict:
    s = summary[summary.imputer == "median"].set_index(["tag", "algo"])
    cats = [("missing_only", "Missing-value<br>indicators only"), ("study_only", "Study<br>membership only"), ("L1", "L1 baseline<br>clinical")]
    fig = go.Figure()
    algos = [a for a in ALGOS if ("missing_only", a) in s.index]
    for algo in algos:
        rows = [s.loc[(t, algo)] for t, _ in cats]
        fig.add_trace(go.Scatter(
            x=[i + OFFSETS[algo] * 1.5 for i in range(len(cats))], y=[r.auc_mean for r in rows], mode="markers", name=ALGO_LABELS[algo],
            error_y={"type": "data", "array": [r.fold_sd for r in rows], "color": ALGO_COLORS[algo], "thickness": 1.2, "width": 0},
            marker=_algo_marker(algo, size=11),
            customdata=np.column_stack([[c[1].replace("<br>", " ") for c in cats], [r.auc_run_sd for r in rows], [r.fold_sd for r in rows]]),
            hovertemplate=f"<b>{ALGO_LABELS[algo]}</b><br>%{{customdata[0]}}<br>AUROC %{{y:.3f}}<br>SD over repeats %{{customdata[1]:.3f}}, over folds %{{customdata[2]:.3f}}<extra></extra>",
        ))
    fig.add_hline(y=0.5, line={"color": MUTED, "width": 1.2, "dash": "dash"})
    fig.add_annotation(x=len(cats) - 0.5, xanchor="right", y=0.5, yanchor="bottom", text="chance", showarrow=False, font={"size": 11, "color": INK2})
    lay = layout("", "Out-of-fold AUROC")
    lay["xaxis"].update(tickvals=list(range(len(cats))), ticktext=[c[1] for c in cats], showgrid=False, range=[-0.5, len(cats) - 0.5])
    lay["yaxis"].update(range=[0.4, 0.95])
    fig.update_layout(**lay)
    return {"default": as_dict(fig)}


def missingness(miss: pd.DataFrame, finfo: pd.DataFrame) -> dict:
    m = miss.set_index("feature")
    m = m[m["All"] > 0]
    labels = finfo.set_index("feature").loc[m.index, "label"].tolist()
    cols = [c for c in m.columns if c != "All"] + ["All"]
    colnames = [f"Study {c}" if c != "All" else "All" for c in cols]
    z = m[cols].to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=colnames, y=labels, zmin=0, zmax=1, xgap=2, ygap=2,
        colorscale=[[i / (len(RAMP) - 1), c] for i, c in enumerate(RAMP)],
        colorbar={"title": {"text": "missing", "font": {"size": 11, "color": INK2}}, "tickformat": ".0%", "tickfont": {"size": 10.5, "color": INK2}, "outlinewidth": 0, "thickness": 10, "len": 0.8},
        hovertemplate="%{y}, %{x}<br>%{z:.0%} missing<extra></extra>",
    ))
    lay = layout("", "", legend=False)
    lay["yaxis"].update(autorange="reversed", showgrid=False, showline=False, tickfont={"size": 10.5, "color": INK2})
    lay["xaxis"].update(side="top", showgrid=False, showline=False)
    lay["height"] = 18 * len(labels) + 70
    fig.update_layout(**lay)
    return {"default": as_dict(fig)}


def imputation(sens: pd.DataFrame, labels: dict) -> dict:
    order = [k for k in labels if k in set(sens.imputer)]
    fig = go.Figure()
    for algo in ALGOS:
        d = sens[sens.algo == algo].set_index("imputer").loc[order]
        fig.add_trace(go.Scatter(
            x=[i + OFFSETS[algo] for i in range(len(order))], y=d["mean"], mode="markers", name=ALGO_LABELS[algo],
            error_y={"type": "data", "array": d["std"], "color": ALGO_COLORS[algo], "thickness": 1.2, "width": 0},
            marker=_algo_marker(algo, size=10),
            customdata=np.column_stack([[labels[i] for i in order], d["std"]]),
            hovertemplate=f"<b>{ALGO_LABELS[algo]}</b>, %{{customdata[0]}}<br>AUROC %{{y:.3f}} (SD over folds %{{customdata[1]:.3f}})<extra></extra>",
        ))
    lay = layout("Imputation", "AUROC on L5, first repeat")
    lay["xaxis"].update(tickvals=list(range(len(order))), ticktext=[labels[i].replace(" ", "<br>", 1) for i in order], showgrid=False)
    lay["yaxis"].update(range=[0.75, 1.02])
    fig.update_layout(**lay)
    return {"default": as_dict(fig)}


# ------------------------------------------------------------------ calibration and decisions
def calibration_curve(cal: pd.DataFrame, title: str) -> dict:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration", line={"color": MUTED, "width": 1.2, "dash": "dash"}, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=cal.mean_pred, y=cal.obs, mode="lines+markers", name=title,
        line={"color": BLUE, "width": 2}, marker={"color": BLUE, "size": 10, "line": {"color": SURFACE, "width": 2}},
        customdata=cal.n, hovertemplate="predicted %{x:.2f}, observed %{y:.2f}<br>%{customdata} patients in this fifth<extra></extra>",
    ))
    lay = layout("Mean predicted probability of being at target", "Observed share at target")
    lay["xaxis"].update(range=[0, 1.02])
    lay["yaxis"].update(range=[0, 1.05])
    fig.update_layout(**lay)
    return {"default": as_dict(fig)}


def calibration_stats(stats: pd.DataFrame) -> dict:
    out = {}
    for key, ref, title in [("slope", 1.0, "Calibration slope"), ("intercept", 0.0, "Calibration intercept")]:
        fig = go.Figure()
        for algo in ALGOS:
            d = stats[stats.algo == algo].set_index("tag").loc[STEPS]
            fig.add_trace(go.Scatter(
                x=[i + OFFSETS[algo] for i in range(len(STEPS))], y=d[key], mode="lines+markers", name=ALGO_LABELS[algo],
                line={"color": ALGO_COLORS[algo], "width": 2}, marker=_algo_marker(algo),
                customdata=STEPS, hovertemplate=f"<b>{ALGO_LABELS[algo]}</b> %{{customdata}}<br>{title.lower()} %{{y:.2f}}<extra></extra>",
            ))
        fig.add_hline(y=ref, line={"color": MUTED, "width": 1.2, "dash": "dash"})
        lay = layout("Feature group", title)
        lay["xaxis"].update(tickvals=list(range(len(STEPS))), ticktext=STEPS, showgrid=False)
        fig.update_layout(**lay)
        out[key] = as_dict(fig)
    return out


def pr_curve(pr: pd.DataFrame, ladder_labels: dict) -> dict:
    fig = go.Figure()
    for tag, color in [("L1", ORANGE), ("L4", BLUE)]:
        d = pr[pr.tag == tag]
        algo = d.algo.iloc[0]
        fig.add_trace(go.Scatter(
            x=d.recall, y=d.precision, mode="lines", line={"color": color, "width": 2, "shape": "hv"},
            name=f"{tag} {ALGO_LABELS[algo].lower()}, AUPRC {d.auprc.iloc[0]:.3f}",
            hovertemplate=f"{tag}: recall %{{x:.2f}}, precision %{{y:.2f}}<extra></extra>",
        ))
    prev = pr.prevalence.iloc[0]
    fig.add_hline(y=prev, line={"color": MUTED, "width": 1.2, "dash": "dash"})
    fig.add_annotation(x=0, xanchor="left", y=prev, yanchor="top", text=f"prevalence {prev:.2f}", showarrow=False, font={"size": 11, "color": INK2})
    lay = layout("Recall (at target)", "Precision")
    lay["xaxis"].update(range=[0, 1.02])
    lay["yaxis"].update(range=[0.6, 1.02])
    fig.update_layout(**lay)
    return {"default": as_dict(fig)}


def decision_curve(dc: pd.DataFrame) -> dict:
    fig = go.Figure()
    first = dc[dc.tag == dc.tag.iloc[0]]
    fig.add_trace(go.Scatter(x=first.threshold, y=first.treat_none, mode="lines", name="intensify no one", line={"color": INK2, "width": 1.5, "dash": "dot"}, hovertemplate="no one: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=first.threshold, y=first.treat_all, mode="lines", name="intensify everyone", line={"color": MUTED, "width": 2, "dash": "dash"}, hovertemplate="threshold %{x:.2f}<br>everyone: %{y:.3f}<extra></extra>"))
    for tag, color in [("L1", ORANGE), ("L4", BLUE)]:
        d = dc[dc.tag == tag]
        fig.add_trace(go.Scatter(x=d.threshold, y=d.model, mode="lines", name=f"{tag} model", line={"color": color, "width": 2}, hovertemplate=f"threshold %{{x:.2f}}<br>{tag} model: %{{y:.3f}}<extra></extra>"))
    lay = layout("Threshold probability of missing target", "Net benefit")
    lay["yaxis"].update(range=[-0.06, max(dc.model.max(), dc.treat_all.max()) + 0.03])
    lay["hovermode"] = "x unified"
    fig.update_layout(**lay)
    return {"default": as_dict(fig)}


# ------------------------------------------------------------------ SHAP
def shap_importance(imp: pd.DataFrame, finfo: pd.DataFrame, top: int = 15) -> dict:
    lab = finfo.set_index("feature")
    out = {}
    for algo in ALGOS:
        d = imp[(imp.algo == algo) & (imp.mean_abs_shap > 0)].sort_values("mean_abs_shap", ascending=False).head(top)
        d = d.assign(label=lab.loc[d.feature, "label"].to_numpy(), pkpd=lab.loc[d.feature, "pkpd"].to_numpy())
        fig = go.Figure()
        for pk, name, color in [(True, "PK/PD estimate", BLUE), (False, "clinical", CLIN)]:
            dd = d[d.pkpd == pk]
            fig.add_trace(go.Bar(
                x=dd.mean_abs_shap, y=dd.label, orientation="h", name=name, marker={"color": color, "line": {"width": 0}},
                customdata=dd.share * 100, hovertemplate="<b>%{y}</b><br>mean |SHAP| %{x:.3f}, %{customdata:.1f}% of the total<extra></extra>",
            ))
        scale = d.scale.iloc[0]
        lay = layout(f"Mean |SHAP| ({scale} scale)", "")
        lay["yaxis"].update(categoryorder="array", categoryarray=d.label.tolist()[::-1], showgrid=False)
        lay["xaxis"].update(rangemode="tozero")
        lay["bargap"] = 0.35
        lay["height"] = 22 * len(d) + 90
        fig.update_layout(**lay)
        out[algo] = as_dict(fig)
    return out


def shap_dots(values: dict, feats: pd.DataFrame, finfo: pd.DataFrame, top: int = 10) -> dict:
    lab = finfo.set_index("feature")["label"]
    rng = np.random.default_rng(11)
    out = {}
    for algo in ALGOS:
        sv = values[algo]
        order = sv.abs().mean().sort_values(ascending=False)
        order = order[order > 0].head(top).index.tolist()
        fig = go.Figure()
        xs, ys, cs, texts = [], [], [], []
        mx, my, mt = [], [], []
        for i, f in enumerate(order):
            v = feats.loc[sv.index, f]
            rank = v.rank(pct=True)
            jitter = rng.uniform(-0.28, 0.28, len(v))
            ok = v.notna().to_numpy()
            for j, pid in enumerate(sv.index):
                target = (xs, ys, texts) if ok[j] else (mx, my, mt)
                target[0].append(float(sv.loc[pid, f]))
                target[1].append(i + jitter[j])
                target[2].append(f"patient {pid}<br>{lab[f]}: " + (f"{v.loc[pid]:.2f}" if ok[j] else "missing, imputed"))
                if ok[j]:
                    cs.append(float(rank.loc[pid]))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name="observed value", showlegend=False,
            marker={"color": cs, "cmin": 0, "cmax": 1, "size": 6, "opacity": 0.85,
                    "colorscale": [[0, BLUE], [0.5, NEUTRAL], [1, ORANGE]],
                    "colorbar": {"title": {"text": "feature value", "font": {"size": 11, "color": INK2}, "side": "right"},
                                 "tickvals": [0, 1], "ticktext": ["low", "high"], "tickfont": {"size": 10.5, "color": INK2},
                                 "outlinewidth": 0, "thickness": 10, "len": 0.7}},
            text=texts, hovertemplate="%{text}<br>SHAP %{x:+.3f}<extra></extra>",
        ))
        if mx:
            fig.add_trace(go.Scatter(
                x=mx, y=my, mode="markers", name="missing (imputed)", marker={"color": SURFACE, "size": 6, "line": {"color": MUTED, "width": 1}},
                text=mt, hovertemplate="%{text}<br>SHAP %{x:+.3f}<extra></extra>",
            ))
        fig.add_vline(x=0, line={"color": AXIS, "width": 1.2})
        scale = "probability" if algo == "rf" else "log-odds"
        lay = layout(f"SHAP value ({scale} scale)", "", legend=bool(mx))
        lay["yaxis"].update(tickvals=list(range(len(order))), ticktext=[lab[f] for f in order], autorange="reversed", showgrid=False, zeroline=False)
        lay["height"] = 34 * len(order) + 90
        fig.update_layout(**lay)
        out[algo] = as_dict(fig)
    return out


def shap_consensus(cons: pd.DataFrame, finfo: pd.DataFrame, top: int = 10) -> dict:
    lab = finfo.set_index("feature")["label"]
    d = cons.sort_values("mean_share", ascending=False).head(top)
    labels = [lab[f] for f in d.feature]
    fig = go.Figure()
    for algo in ALGOS:
        fig.add_trace(go.Bar(
            x=d[algo] * 100, y=labels, orientation="h", name=ALGO_LABELS[algo], marker={"color": ALGO_COLORS[algo], "line": {"color": SURFACE, "width": 1}},
            hovertemplate=f"<b>%{{y}}</b><br>{ALGO_LABELS[algo]}: %{{x:.1f}}% of total |SHAP|<extra></extra>",
        ))
    lay = layout("Share of total mean |SHAP| (%)", "")
    lay["yaxis"].update(categoryorder="array", categoryarray=labels[::-1], showgrid=False)
    lay["barmode"] = "group"
    lay["bargap"] = 0.25
    lay["height"] = 44 * len(labels) + 90
    fig.update_layout(**lay)
    return {"default": as_dict(fig)}


GROUPS = ["exposure", "baseline level", "drug sensitivity"]
TRUE_COL = {"exposure": "phi_exposure", "baseline level": "phi_base", "drug sensitivity": "phi_ic50"}


def shap_vs_truth_shares(svt: pd.DataFrame) -> dict:
    fig = go.Figure()
    truth = svt[svt.algo == "enet"].set_index("group").loc[GROUPS, "true_share"]
    fig.add_trace(go.Bar(x=GROUPS, y=truth * 100, name="true split", marker={"color": INK2, "line": {"color": SURFACE, "width": 1}},
                         hovertemplate="<b>%{x}</b><br>true share %{y:.0f}%<extra></extra>"))
    for algo in ALGOS:
        d = svt[svt.algo == algo].set_index("group").loc[GROUPS]
        rho = ["none kept" if np.isnan(r) else f"{r:.2f}" for r in d.spearman]
        # shares here are within the three mechanism groups, to compare with the truth
        fig.add_trace(go.Bar(
            x=GROUPS, y=d.shap_share * 100, name=ALGO_LABELS[algo], marker={"color": ALGO_COLORS[algo], "line": {"color": SURFACE, "width": 1}},
            customdata=rho, hovertemplate=f"<b>%{{x}}</b>, {ALGO_LABELS[algo]}<br>share of |SHAP| %{{y:.0f}}%<br>Spearman with the truth %{{customdata}}<extra></extra>",
        ))
    lay = layout("Mechanism group", "Share of the total (%)")
    lay["barmode"] = "group"
    lay["bargap"] = 0.2
    lay["xaxis"].update(showgrid=False)
    fig.update_layout(**lay)
    return {"default": as_dict(fig)}


def shap_vs_truth_scatter(grouped: dict, truth: pd.DataFrame, svt: pd.DataFrame) -> dict:
    out = {}
    for algo in ALGOS:
        g = grouped[algo]
        t = truth.loc[g.index]
        for grp in GROUPS:
            row = svt[(svt.algo == algo) & (svt.group == grp)].iloc[0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t[TRUE_COL[grp]], y=g[grp], mode="markers", name=ALGO_LABELS[algo], showlegend=False,
                marker={"color": ALGO_COLORS[algo], "symbol": ALGO_SYMBOLS[algo], "size": 7, "opacity": 0.8, "line": {"color": SURFACE, "width": 1}},
                customdata=g.index, hovertemplate="patient %{customdata}<br>true part %{x:.2f}, summed SHAP %{y:.3f}<extra></extra>",
            ))
            note = "no features from this group were kept" if np.isnan(row.spearman) else f"Spearman {row.spearman:.2f}"
            fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top", text=note, showarrow=False, font={"size": 12, "color": INK})
            lay = layout(f"True Shapley part for {grp} (on -log biomarker)", "Summed SHAP for the group", legend=False)
            fig.update_layout(**lay)
            out[f"{algo}|{grp}"] = as_dict(fig)
    return out


def selection(freq: pd.DataFrame, finfo: pd.DataFrame, ladder: pd.DataFrame) -> dict:
    lab = finfo.set_index("feature")["label"]
    out = {}
    for step in ladder.step:
        d = freq[freq.tag == step].pivot(index="feature", columns="algo", values="selected")[ALGOS]
        dropped = freq[freq.tag == step].groupby("feature").dropped.max()
        d = d.loc[d.mean(axis=1).sort_values(ascending=False).index]
        text = [[("dropped for missingness in every fold" if dropped[f] >= 1 else f"kept in {d.loc[f, a]:.0%} of folds") for a in ALGOS] for f in d.index]
        fig = go.Figure(go.Heatmap(
            z=d.to_numpy(), x=[ALGO_LABELS[a] for a in ALGOS], y=[lab[f] for f in d.index], zmin=0, zmax=1, xgap=2, ygap=2,
            colorscale=[[i / (len(RAMP) - 1), c] for i, c in enumerate(RAMP)], text=text,
            colorbar={"title": {"text": "kept", "font": {"size": 11, "color": INK2}}, "tickformat": ".0%", "tickfont": {"size": 10.5, "color": INK2}, "outlinewidth": 0, "thickness": 10, "len": 0.6},
            hovertemplate="%{y}, %{x}<br>%{text}<extra></extra>",
        ))
        lay = layout("", "", legend=False)
        lay["yaxis"].update(autorange="reversed", showgrid=False, showline=False, tickfont={"size": 10.5, "color": INK2})
        lay["xaxis"].update(side="top", showgrid=False, showline=False)
        lay["height"] = 19 * len(d) + 70
        fig.update_layout(**lay)
        out[step] = as_dict(fig)
    return out


# ------------------------------------------------------------------ cohort and PK/PD
def trajectories(bio: pd.DataFrame, design: dict) -> dict:
    lo, hi = design["outcome_window"]
    target = design["target"]
    out = {}
    for study in ["All"] + sorted(bio.study.unique()):
        d = bio if study == "All" else bio[bio.study == study]
        fig = go.Figure()
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=0, y1=1, yref="paper", fillcolor=SHADE, line={"width": 0}, layer="below")
        for at, name, color in [(1, "at target", BLUE), (0, "not at target", ORANGE)]:
            dd = d[d.at_target == at].sort_values(["id", "time"])
            xs, ys, ids = [], [], []
            for pid, g in dd.groupby("id", sort=False):
                xs += g.time.tolist() + [None]
                ys += g.value.tolist() + [None]
                ids += [f"patient {pid}, study {g.study.iloc[0]}"] * len(g) + [None]
            n = dd.id.nunique()
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=f"{name} (n = {n})", line={"color": color, "width": 1}, opacity=0.45,
                marker={"size": 3.5, "color": color}, text=ids, connectgaps=False,
                hovertemplate="%{text}<br>day %{x:.0f}: %{y:.0f} units<extra></extra>",
            ))
        fig.add_hline(y=target, line={"color": INK, "width": 1.2, "dash": "dash"})
        fig.add_annotation(x=1, xref="paper", xanchor="right", y=np.log10(target), yanchor="top", text=f"target {target:.0f}", showarrow=False, font={"size": 11, "color": INK2})
        fig.add_annotation(x=(lo + hi) / 2, y=1, yref="paper", yanchor="bottom", text="outcome window", showarrow=False, font={"size": 11, "color": INK2})
        lay = layout("Day", "Observed biomarker (units, log scale)")
        lay["yaxis"].update(type="log", tickvals=log_ticks(bio.value.min(), bio.value.max()))
        lay["legend"].update(y=1.08)
        lay["margin"]["t"] = 44
        fig.update_layout(**lay)
        out[study] = as_dict(fig)
    return out


def map_vs_truth(est: pd.DataFrame, truth: pd.DataFrame, mvt: pd.DataFrame, study: pd.Series, block_labels: dict) -> dict:
    out = {}
    units = {"CL": "L/day", "V1": "L", "BASE": "units", "IC50": "mg/L"}
    for block in mvt.block.unique():
        e = est[est.block == block].set_index("id")
        for p in ["CL", "V1", "BASE", "IC50"]:
            row = mvt[(mvt.block == block) & (mvt.parameter == p)].iloc[0]
            ids = e.index.intersection(truth.index)
            fig = go.Figure()
            lo = float(min(truth.loc[ids, p].min(), e.loc[ids, p].min())) * 0.9
            hi = float(max(truth.loc[ids, p].max(), e.loc[ids, p].max())) * 1.1
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"color": MUTED, "width": 1.2, "dash": "dash"}, hoverinfo="skip", showlegend=False))
            for s, color in STUDY_COLORS.items():
                sid = [i for i in ids if study.get(i) == s]
                fig.add_trace(go.Scatter(
                    x=truth.loc[sid, p], y=e.loc[sid, p], mode="markers", name=f"study {s}",
                    marker={"color": color, "size": 7, "opacity": 0.8, "line": {"color": SURFACE, "width": 1}},
                    customdata=sid, hovertemplate=f"patient %{{customdata}}, study {s}<br>true {p} %{{x:.3g}}, MAP %{{y:.3g}} {units[p]}<extra></extra>",
                ))
            fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False, font={"size": 12, "color": INK},
                               text=f"r (log) {row.r_log:.2f}, shrinkage {row.shrinkage:.2f}")
            lay = layout(f"True {p} ({units[p]})", f"MAP estimate, {block_labels[block]}")
            lay["xaxis"].update(type="log", tickvals=log_ticks(lo, hi))
            lay["yaxis"].update(type="log", tickvals=log_ticks(lo, hi))
            fig.update_layout(**lay)
            out[f"{block}|{p}"] = as_dict(fig)
    return out
