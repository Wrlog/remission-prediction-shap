"""Draw every figure in figures/ from the tables in results/.

    python scripts/make_figures.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from remission import config  # noqa: E402
from remission.features import LADDER, LADDER_LABELS, is_pkpd, pretty  # noqa: E402
from remission.models import ALGO_LABELS, ALGOS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"enet": "#2a78d6", "rf": "#eb6834", "xgb": "#1baf7a", "cat": "#eda100"}
MARKERS = {"enet": "o", "rf": "s", "xgb": "D", "cat": "^"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PKPD_C, CLIN_C = "#2a78d6", "#a8a69f"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 9.5,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    }
)


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_cohort(res, figs):
    bio = pd.read_csv(res / "biomarker_records.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    lo, hi = config.OUTCOME_WINDOW
    for ax, (study, g) in zip(axes, bio.groupby("study")):
        for at, color, label in [(1.0, "#2a78d6", "at target"), (0.0, "#eb6834", "not at target")]:
            sub = g[g.at_target == at]
            for _, p in sub.groupby("id"):
                ax.plot(p.time, p.value, color=color, alpha=0.28, lw=0.9)
            ax.plot([], [], color=color, label=f"{label} ({sub.id.nunique()})")
        ax.axhline(config.TARGET, color=INK, lw=1, ls="--")
        ax.axvspan(lo, hi, color=GRID, alpha=0.7, lw=0)
        ax.set_yscale("log")
        ax.set_title(f"Study {study}")
        ax.set_xlabel("Day")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Biomarker (units, observed)")
    axes[0].text(lo - 4, 2.2, "outcome window", fontsize=8, color=INK2, ha="right")
    save(fig, figs / "cohort.png")


def fig_ladder(res, figs, info):
    s = pd.read_csv(res / "cv_summary.csv")
    main = s[s.tag.isin(list(LADDER)) & (s.imputer == "median")]
    steps = list(LADDER)
    x = np.arange(len(steps))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, algo in enumerate(ALGOS):
        g = main[main.algo == algo].set_index("tag").loc[steps]
        off = (i - 1.5) * 0.06
        ax.errorbar(x + off, g.auc_mean, yerr=g.auc_run_sd, color=COLORS[algo], marker=MARKERS[algo], ms=6, capsize=0, lw=2, label=ALGO_LABELS[algo])
    ceil = info["champion"]["ceiling_auc_true_probability"]
    ax.axhline(ceil, color=INK, ls="--", lw=1)
    ax.text(len(steps) - 0.6, ceil + 0.004, "true probability", ha="right", va="bottom", fontsize=8, color=INK2)
    study_only = s[(s.tag == "study_only") & (s.algo == "enet")].auc_mean.iloc[0]
    miss_only = s[(s.tag == "missing_only")].auc_mean.max()
    ax.axhline(study_only, color=INK2, ls=":", lw=1)
    ax.text(-0.3, study_only + 0.004, "study only", fontsize=8, color=INK2, va="bottom")
    ax.axhline(miss_only, color=INK2, ls="-.", lw=1)
    ax.text(-0.3, miss_only - 0.006, "missing indicators only (best)", fontsize=8, color=INK2, va="top")
    short = {"L1": "baseline\nclinical", "L2": "+ baseline\nPK/PD", "L3": "+ clinical\nto inf 3", "L4": "PK/PD updated\nto inf 4", "L5": "everything"}
    ax.set_xticks(x, [f"{k}\n{short[k]}" for k in steps], fontsize=8)
    ax.set_ylabel("AUROC (mean over repeats, bar = SD)")
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.set_title("Out-of-fold AUROC by feature group and algorithm")
    save(fig, figs / "auroc_ladder.png")


def fig_deltas(res, figs):
    d = pd.read_csv(res / "ladder_deltas.csv")
    contrasts = list(dict.fromkeys(d.contrast))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    rng = np.random.default_rng(0)
    for ci, c in enumerate(contrasts):
        for i, algo in enumerate(ALGOS):
            g = d[(d.contrast == c) & (d.algo == algo)]
            xc = ci + (i - 1.5) * 0.18
            ax.scatter(xc + rng.uniform(-0.05, 0.05, len(g)), g.delta, s=10, color=COLORS[algo], alpha=0.45, lw=0)
            ax.scatter([xc], [g.delta.mean()], s=60, color=COLORS[algo], marker=MARKERS[algo], edgecolor=SURFACE, lw=1.5, zorder=3, label=ALGO_LABELS[algo] if ci == 0 else None)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(contrasts)), [c.replace("->", " to ") for c in contrasts])
    ax.set_ylabel("Change in AUROC per outer fold")
    ax.set_title("Per-step change (small dots: folds, large: mean)")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    save(fig, figs / "ladder_deltas.png")


def fig_shap_bars(res, figs):
    imp = pd.read_csv(res / "shap_importance.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, algo in zip(axes.ravel(), ALGOS):
        g = imp[(imp.algo == algo) & (imp.mean_abs_shap > 0)].sort_values("mean_abs_shap", ascending=False).head(12)[::-1]
        colors = [PKPD_C if is_pkpd(f) else CLIN_C for f in g.feature]
        ax.barh([pretty(f) for f in g.feature], g.mean_abs_shap, color=colors, height=0.7)
        ax.set_title(f"{ALGO_LABELS[algo]} (SHAP on {g.scale.iloc[0]} scale)")
        ax.set_xlabel("mean |SHAP|")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PKPD_C), plt.Rectangle((0, 0), 1, 1, color=CLIN_C)]
    fig.legend(handles, ["PK/PD estimate", "clinical"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, figs / "shap_bar.png")


def fig_shap_dots(res, figs):
    X = pd.read_csv(res / "features.csv", index_col="id")
    for algo in ALGOS:
        sv = pd.read_csv(res / f"shap_values_{algo}.csv", index_col="id").loc[X.index]
        Xa = X[sv.columns]
        plt.figure()
        shap.summary_plot(sv.to_numpy(), Xa.to_numpy(), feature_names=[pretty(c) for c in Xa.columns], max_display=12, show=False, plot_size=(7.5, 5))
        plt.title(f"{ALGO_LABELS[algo]}: SHAP per patient (gray = missing)", fontsize=10)
        save(plt.gcf(), figs / f"shap_dot_{algo}.png")


def fig_consensus(res, figs):
    c = pd.read_csv(res / "shap_consensus.csv").head(15)
    mat = c[ALGOS].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 6))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(ALGOS)), [ALGO_LABELS[a] for a in ALGOS], rotation=20)
    ax.set_yticks(range(len(c)), [pretty(f) for f in c.feature], fontsize=8)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if mat[i, j] > 0.6 * mat.max() else INK)
    fig.colorbar(im, ax=ax, label="share of total mean |SHAP|", shrink=0.7)
    ax.set_title("Consensus: top 15 features by mean share")
    save(fig, figs / "shap_consensus.png")


def fig_shap_truth(res, figs, algo):
    g = pd.read_csv(res / f"shap_grouped_{algo}.csv", index_col="id")
    t = pd.read_csv(res / "truth_components.csv", index_col="id").loc[g.index]
    tab = pd.read_csv(res / "shap_vs_truth.csv")
    tab = tab[tab.algo == algo].set_index("group")
    pairs = [("exposure", "phi_exposure", "Exposure"), ("baseline level", "phi_base", "Baseline level (BASE)"), ("drug sensitivity", "phi_ic50", "Drug sensitivity (IC50)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (grp, col, title) in zip(axes, pairs):
        ax.scatter(t[col], g[grp], s=14, color="#2a78d6", alpha=0.7, lw=0)
        ax.set_title(f"{title}\nSpearman {tab.loc[grp, 'spearman']:.2f}")
        ax.set_xlabel("true Shapley part of -log(biomarker)")
    axes[0].set_ylabel(f"summed SHAP, {ALGO_LABELS[algo]}")
    save(fig, figs / "shap_vs_truth.png")


def fig_pr(res, figs, info):
    pr = pd.read_csv(res / "pr_curve.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for tag, color in zip(dict.fromkeys(pr.tag), ["#a8a69f", "#2a78d6"]):
        g = pr[pr.tag == tag]
        ax.plot(g.recall, g.precision, color=color, drawstyle="steps-post", label=f"{tag} {LADDER_LABELS[tag]} (AUPRC {g.auprc.iloc[0]:.2f})")
    prev = pr.prevalence.iloc[0]
    ax.axhline(prev, color=INK, ls="--", lw=1)
    ax.text(0.02, prev - 0.03, f"prevalence {prev:.2f}", fontsize=8, color=INK2, va="top")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f"Precision-recall, {ALGO_LABELS[info['champion']['algo']]}")
    ax.legend(loc="lower left", fontsize=7.5)
    save(fig, figs / "pr_curve.png")


def fig_calibration(res, figs, info):
    cc = pd.read_csv(res / "calibration_curve.csv")
    cs = pd.read_csv(res / "calibration_stats.csv")
    ch = info["champion"]
    row = cs[(cs.tag == ch["step"]) & (cs.algo == ch["algo"])].iloc[0]
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=1)
    ax.plot(cc.mean_pred, cc.obs, color="#2a78d6", marker="o", ms=7)
    ax.set_xlabel("Mean predicted probability (quintiles)")
    ax.set_ylabel("Observed share at target")
    ax.set_title(f"Calibration, {ch['step']} {ALGO_LABELS[ch['algo']]}")
    ax.text(0.04, 0.9, f"slope {row.slope:.2f}\nintercept {row.intercept:.2f}", fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, figs / "calibration.png")


def fig_decision(res, figs, info):
    dc = pd.read_csv(res / "decision_curve.csv")
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    tags = list(dict.fromkeys(dc.tag))
    for tag, color in zip(tags, ["#a8a69f", "#2a78d6"]):
        g = dc[dc.tag == tag]
        ax.plot(g.threshold, g.model, color=color, label=f"model, {tag}")
    g = dc[dc.tag == tags[-1]]
    ax.plot(g.threshold, g.treat_all, color="#eb6834", label="intensify everyone")
    ax.plot(g.threshold, g.treat_none, color=INK, lw=1, label="intensify no one")
    ax.set_ylim(-0.05, max(0.05, g.treat_all.max()) + 0.08)
    ax.set_xlabel("Threshold probability of not reaching target")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision curve, {ALGO_LABELS[info['champion']['algo']]}")
    ax.legend(fontsize=8)
    save(fig, figs / "decision_curve.png")


def fig_selection(res, figs):
    f = pd.read_csv(res / "selection_frequency.csv")
    f = f[f.tag == "L5"].pivot(index="feature", columns="algo", values="selected")[ALGOS]
    f = f.loc[f.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(5.8, 8.5))
    im = ax.imshow(f.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(f)), [pretty(c) for c in f.index], fontsize=7.5)
    ax.set_xticks(range(len(ALGOS)), [ALGO_LABELS[a] for a in ALGOS], rotation=20)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="share of outer folds selected", shrink=0.5)
    ax.set_title("Selection frequency, everything (L5)")
    save(fig, figs / "selection_frequency.png")


def fig_map(res, figs):
    est = pd.read_csv(res / "map_estimates.csv")
    tt = pd.read_csv(res / "truth_components.csv").set_index("id")
    mv = pd.read_csv(res / "map_vs_truth.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
    for ax, k in zip(axes, (1, 2, 3)):
        e = est[est.block == f"pkpd_v{k}"].set_index("id")
        ax.scatter(tt.loc[e.index, "IC50"], e["IC50"], s=12, color="#2a78d6", alpha=0.7, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        lim = [0.2, 60]
        ax.plot(lim, lim, color=INK2, ls="--", lw=1)
        r = mv[(mv.block == f"pkpd_v{k}") & (mv.parameter == "IC50")].iloc[0]
        ax.set_title(f"Data to infusion {k + 1}\nr = {r.r_log:.2f}, shrinkage {r.shrinkage:.2f}", fontsize=9.5)
        ax.set_xlabel("true IC50 (mg/L)")
    axes[0].set_ylabel("MAP IC50 (mg/L)")
    save(fig, figs / "map_ic50.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--figures", default=str(ROOT / "figures"))
    args = ap.parse_args()
    res, figs = Path(args.results), Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)
    info = json.loads((res / "run_info.json").read_text())
    fig_cohort(res, figs)
    fig_ladder(res, figs, info)
    fig_deltas(res, figs)
    fig_shap_bars(res, figs)
    fig_shap_dots(res, figs)
    fig_consensus(res, figs)
    fig_shap_truth(res, figs, info["champion"]["algo"])
    fig_pr(res, figs, info)
    fig_calibration(res, figs, info)
    fig_decision(res, figs, info)
    fig_selection(res, figs)
    fig_map(res, figs)
    print(f"figures written to {figs}")


if __name__ == "__main__":
    main()
