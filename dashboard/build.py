"""Build the results dashboard from the committed tables in results/.

    python -m dashboard.build            # writes site/index.html
    python dashboard/build.py --out site

Only plotly, pandas and numpy are needed; nothing is retrained. Every
number on the page is read from a file in results/, and the charts are
plotly figures built from the same tables. The matplotlib PNGs in figures/
are copied next to the page as a static fallback and download.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # run as a script: python dashboard/build.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "dashboard"

from dashboard import figures as F  # noqa: E402
from dashboard.theme import ALGO_COLORS, ALGO_LABELS, ALGOS, DARK_MAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/Wrlog/remission-prediction-shap"
PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-cartesian-dist-min@4.1.1/plotly-cartesian.min.js"
PLOTLY_SRI = "sha384-zc4SwKObGL/W0M3RUI09UPWjlgIZHfbwMgcZO6713mRh78tfUmIQKhzCyE2LIiUg"

IMPUTER_LABELS = {"median": "Per-visit median", "mean": "Mean", "knn": "KNN", "iterative": "Iterative", "locf": "Carry forward + median"}
WORDS = ["none", "one", "two", "three", "four"]
BLOCK_LABELS = {"pkpd_v1": "data to day 21", "pkpd_v2": "data to day 42", "pkpd_v3": "data to day 98"}


# ------------------------------------------------------------------ data
def load(results: Path) -> dict:
    r = {}
    for f in results.glob("*.csv"):
        r[f.stem] = pd.read_csv(f)
    r["run_info"] = json.loads((results / "run_info.json").read_text())
    r["design"] = json.loads((results / "design.json").read_text())
    for algo in ALGOS:
        r[f"shap_values_{algo}"] = r[f"shap_values_{algo}"].set_index("id")
        r[f"shap_grouped_{algo}"] = r[f"shap_grouped_{algo}"].set_index("id")
    r["features"] = r["features"].set_index("id")
    r["truth_components"] = r["truth_components"].set_index("id")
    return r


def join_and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def fmt_p(p: float) -> str:
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def signed(x: float, nd: int = 3) -> str:
    return f"{x:+.{nd}f}".replace("-", "&minus;")


def rng(lo: float, hi: float, nd: int = 3, sign: bool = False) -> str:
    f = (lambda v: signed(v, nd)) if sign else (lambda v: f"{v:.{nd}f}")
    return f"{f(lo)} to {f(hi)}"


# ------------------------------------------------------------------ html helpers
def table(headers: list[str], rows: list[list], sort_keys: list[list] | None = None, first_cols_left: int = 1, cls: str = "") -> str:
    """A plain table. Cells may carry a sort key so formatted numbers sort right."""
    head = "".join(
        f'<th scope="col"{" class=left" if i < first_cols_left else ""}><button type="button">{html.escape(h)}</button></th>'
        for i, h in enumerate(headers)
    )
    body = []
    for ri, row in enumerate(rows):
        cells = []
        for ci, c in enumerate(row):
            key = sort_keys[ri][ci] if sort_keys else None
            attr = f' data-sort="{key}"' if key is not None and not (isinstance(key, float) and np.isnan(key)) else ""
            left = " class=left" if ci < first_cols_left else ""
            cells.append(f"<td{left}{attr}>{c}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table class="sortable{" " + cls if cls else ""}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def plot(fid: str, variants: dict, png: str | None = None, alt: str = "", tall: bool = False) -> str:
    FIGS[fid] = variants
    link = f'<a class="png" href="figures/{png}" download>Static PNG</a>' if png else ""
    cls = "plot plot-auto" if tall else "plot"
    return (f'<figure class="fig"><div id="{fid}" class="{cls}" data-fig="{fid}" data-png="{png or ""}" '
            f'role="img" aria-label="{html.escape(alt)}"></div>{link}</figure>')


def select(fid: str, label: str, options: list[tuple[str, str]], default: str | None = None) -> str:
    opts = "".join(f'<option value="{html.escape(k)}"{" selected" if k == default else ""}>{html.escape(v)}</option>' for k, v in options)
    return f'<label class="control">{html.escape(label)}<select data-for="{fid}">{opts}</select></label>'


def card(title: str, sub: str, body: str, controls: str = "", wide: bool = False) -> str:
    head = f'<div><h2>{title}</h2><p class="sub">{sub}</p></div>'
    if controls:
        head = f'<div class="card-head">{head}<div class="controls">{controls}</div></div>'
    return f'<article class="card{" wide" if wide else ""}">{head}{body}</article>'


def tile(label: str, value: str, detail: str) -> str:
    return f'<div class="tile"><div class="label">{label}</div><div class="value">{value}</div><div class="detail">{detail}</div></div>'


FIGS: dict[str, dict] = {}


# ------------------------------------------------------------------ page
def build(results: Path, figures_dir: Path, out: Path) -> Path:
    FIGS.clear()
    r = load(results)
    info, design = r["run_info"], r["design"]
    s = info["settings"]
    champ = info["champion"]
    summ = r["cv_summary"]
    main = summ[summ.imputer == "median"].set_index(["tag", "algo"])
    con = r["ladder_contrasts"]
    ladder = r["ladder"]
    lad = ladder.set_index("step")
    finfo = r["feature_info"]
    coh = r["cohort_summary"].set_index("study")
    n_folds = s["OUTER_FOLDS"] * s["N_REPEATS"]
    test_n = int(r["cv_folds"].query("kind == 'cv'").n_test.min())

    def c(frm, to):
        return con[(con["from"] == frm) & (con["to"] == to)].set_index("algo")

    l1 = main.loc["L1"]
    l34 = c("L3", "L4")
    l12 = c("L1", "L2")
    raw = c("L4r", "L4")
    best_step = l34.delta_mean.idxmax()
    ceiling = champ["ceiling_auc_true_probability"]

    # ---- tiles
    tiles = "".join([
        tile("Best out-of-fold AUROC", f"{champ['auc_mean']:.3f}",
             f"{ALGO_LABELS[champ['algo']]} on {champ['step']}, bootstrap 95% CI {champ['bootstrap_ci'][0]:.3f} to {champ['bootstrap_ci'][1]:.3f}"),
        tile("Baseline data only", f"{l1.auc_mean.max():.3f}", f"best of four algorithms on L1, known on day {lad.loc['L1', 'known_day']}"),
        tile("L3 to L4 gain", signed(l34.delta_mean.max()), f"{ALGO_LABELS[best_step].lower()}, corrected p {fmt_p(l34.loc[best_step, 'p_corrected'])}; PK/PD refitted with data to day {lad.loc['L4', 'known_day']}"),
        tile("Ceiling", f"{ceiling:.3f}", "AUROC of the true simulated probability of being at target"),
        tile("Simulated cohort", f"{info['n_patients']}", f"patients in {len(coh) - 1} studies, {info['prevalence']:.0%} at target"),
    ])

    # ---- ladder table
    lrows = [[f"{st}{' (check)' if r_.kind == 'check' else ''}", html.escape(r_.label), html.escape(r_.blocks), r_.n_features, f"day {r_.known_day}"] for st, r_ in lad.iterrows()]
    lkeys = [[i, None, None, r_.n_features, r_.known_day] for i, (st, r_) in enumerate(lad.iterrows())]
    ladder_table = table(["Step", "Label", "Feature blocks", "Features", "Known by"], lrows, lkeys, first_cols_left=3, cls="ladder-table")

    # ---- AUROC table
    steps = F.STEPS + ["L4r"]
    arows, akeys = [], []
    for i, st in enumerate(steps):
        row = [st]
        key = [i]
        for a in ALGOS:
            m = main.loc[(st, a)]
            row.append(f"{m.auc_mean:.3f} <span class=sd>({m.auc_run_sd:.3f})</span>")
            key.append(round(m.auc_mean, 4))
        arows.append(row)
        akeys.append(key)
    auroc_table = table(["Step"] + [ALGO_LABELS[a] for a in ALGOS], arows, akeys)
    fold_sd = main.loc[main.index.get_level_values(0).isin(steps), "fold_sd"]

    # ---- contrast table
    crow, ckey = [], []
    for i, ((frm, to), g) in enumerate(con.groupby(["from", "to"], sort=False)):
        g = g.set_index("algo")
        for a in ALGOS:
            x = g.loc[a]
            crow.append([f"{frm} to {to}", ALGO_LABELS[a], signed(x.delta_mean), fmt_p(x.p_corrected), fmt_p(x.p_naive)])
            ckey.append([i, ALGOS.index(a), round(x.delta_mean, 5), x.p_corrected, x.p_naive])
    contrast_table = table(["Contrast", "Algorithm", "Mean change", "Corrected p", "Naive p"], crow, ckey, first_cols_left=2)

    # ---- LOSO table
    lo = r["loso"]
    lrow, lkey = [], []
    for (st, ho), g in lo.groupby(["tag", "held_out"]):
        g = g.set_index("algo")
        lrow.append([ho, st] + [f"{g.loc[a, 'auc']:.3f}" for a in ALGOS])
        lkey.append([ho, st] + [round(g.loc[a, "auc"], 4) for a in ALGOS])
    loso_table = table(["Held out", "Step"] + [ALGO_LABELS[a] for a in ALGOS], lrow, lkey, first_cols_left=2)

    # ---- calibration table
    cal = r["calibration_stats"]
    krow, kkey = [], []
    for st in F.STEPS:
        for a in ALGOS:
            x = cal[(cal.tag == st) & (cal.algo == a)].iloc[0]
            krow.append([st, ALGO_LABELS[a], f"{x.slope:.2f}", signed(x.intercept, 2)])
            kkey.append([st, ALGOS.index(a), round(x.slope, 4), round(x.intercept, 4)])
    cal_table = table(["Step", "Algorithm", "Slope", "Intercept"], krow, kkey, first_cols_left=2)

    # ---- SHAP vs truth table
    svt = r["shap_vs_truth"]
    trow, tkey = [], []
    for a in ALGOS:
        g = svt[svt.algo == a].set_index("group")
        row, key = [ALGO_LABELS[a]], [ALGOS.index(a)]
        for grp in F.GROUPS:
            rho = g.loc[grp, "spearman"]
            row.append("none kept" if np.isnan(rho) else f"{rho:.2f}")
            key.append(-1 if np.isnan(rho) else round(rho, 4))
        trow.append(row)
        tkey.append(key)
    svt_table = table(["Algorithm"] + [f"{g[0].upper()}{g[1:]}" for g in F.GROUPS], trow, tkey)

    # ---- cohort table
    corow, cokey = [], []
    for st, x in coh.iterrows():
        corow.append([st, int(x.n), f"{x.at_target:.0%}", f"{x.ada_by_outcome:.0%}", f"{x.weight_median:.1f}", f"{x.baseline_bio_median:.0f}"])
        cokey.append([st, int(x.n), x.at_target, x.ada_by_outcome, x.weight_median, x.baseline_bio_median])
    cohort_table = table(["Study", "Patients", "At target", "ADA by outcome", "Median weight (kg)", "Median baseline biomarker"], corow, cokey)

    # ---- MAP table
    mvt = r["map_vs_truth"]
    mrow, mkey = [], []
    for b, g in mvt.groupby("block"):
        g = g.set_index("parameter")
        row, key = [BLOCK_LABELS[b]], [b]
        for p in ["CL", "V1", "BASE", "IC50"]:
            row.append(f"{g.loc[p, 'r_log']:.2f} / {g.loc[p, 'shrinkage']:.2f}")
            key.append(round(g.loc[p, "r_log"], 4))
        mrow.append(row)
        mkey.append(key)
    map_table = table(["MAP fit with", "CL", "V1", "BASE", "IC50"], mrow, mkey)
    ic50 = mvt[mvt.parameter == "IC50"].set_index("block")

    # ---- numbers used in captions
    sens = r["imputation_sensitivity"]
    sens_spread = sens.groupby("algo")["mean"].agg(lambda v: v.max() - v.min()).max()
    chk = summ[summ.tag.isin(["missing_only", "study_only"])]
    mo = chk[chk.tag == "missing_only"].auc_mean
    so = chk[chk.tag == "study_only"].auc_mean
    xgb_slopes = cal[cal.algo == "xgb"].slope
    rf_int = cal[(cal.algo == "rf") & (cal.intercept > 0.15)]
    champ_cal = cal[(cal.tag == champ["step"]) & (cal.algo == champ["algo"])].iloc[0]
    pr = r["pr_curve"].groupby("tag").auprc.first()
    dc = r["decision_curve"]
    dc4 = dc[dc.tag == "L4"]
    beats = dc4[(dc4.model > dc4.treat_all) & (dc4.model > dc4.treat_none)].threshold
    dc_at = dc4.iloc[(dc4.threshold - 0.3).abs().argmin()]
    imp = r["shap_importance"]
    top_share = imp[imp.feature == "v3_IC50"].share
    kept = imp[imp.mean_abs_shap > 0].groupby("algo").size()
    held = lo[lo.tag == "L4"].groupby("held_out").auc.mean()
    studies_only = coh.drop("All")
    hardest_note = ""
    if studies_only.weight_median.idxmin() == held.idxmin() == studies_only.ada_by_outcome.idxmax():
        hardest_note = ", with the lightest patients and the most ADA,"
    freq = r["selection_frequency"]
    ic50_kept = freq[(freq.tag == "L5") & (freq.feature == "v3_IC50")].selected.min()
    true_exp = svt[(svt.algo == "enet") & (svt.group == "exposure")].true_share.iloc[0]
    shap_exp = svt[svt.group == "exposure"].shap_share
    base_rho = svt[svt.group == "baseline level"].spearman
    sens_rho = svt[svt.group == "drug sensitivity"].spearman
    lo_step, hi_step = l34.delta_mean.min(), l34.delta_mean.max()
    n_sig = int((l34.p_corrected < 0.05).sum())

    findings = [
        f"Baseline data alone give an AUROC of {rng(l1.auc_mean.min(), l1.auc_mean.max())} across the four algorithms. The true simulated probabilities reach {ceiling:.2f}, so a lot is left to explain.",
        f"The first PK/PD block (data to day {lad.loc['L2', 'known_day']}) adds {rng(l12.delta_mean.min(), l12.delta_mean.max(), sign=True)}. A naive paired t-test calls all four significant (p {rng(l12.p_naive.min(), l12.p_naive.max())}); the corrected resampled t-test does not (p {rng(l12.p_corrected.min(), l12.p_corrected.max(), 2)}).",
        f"Refitting PK/PD with data to day {lad.loc['L4', 'known_day']} is the big step: {rng(lo_step, hi_step, sign=True)}, corrected p below 0.05 for {WORDS[n_sig]} of the four algorithms. The same day-{lad.loc['L4', 'known_day']} data fed in as raw values (L4r) do about as well, so here the model mostly repackages what the raw data carry.",
        f"SHAP from all four refitted models puts the day-98 IC50 estimate first ({top_share.min():.0%} to {top_share.max():.0%} of the total). Summed SHAP tracks the true mechanism well for baseline level and drug sensitivity and poorly for exposure.",
    ]
    findings_html = "".join(f"<li>{f}</li>" for f in findings)

    algo_key = "".join(f'<span class="key-item"><i style="background:{ALGO_COLORS[a]}"></i>{ALGO_LABELS[a]}</span>' for a in ALGOS)

    # ---- figures and cards
    overview = f"""
      <div class="tiles">{tiles}</div>
      <div class="grid-2">
        <article class="card wide">
          <h2>Findings</h2>
          <ol class="findings">{findings_html}</ol>
        </article>
        {card("The feature ladder", f"Each step adds a block of features. The dot marks the day every feature in the step is known; the outcome is read between day {design['outcome_window'][0]:.0f} and day {design['outcome_window'][1]:.0f}. L4r is a side check outside the ladder.",
              plot("ladder", F.ladder_timeline(ladder, design), alt="Timeline of when each feature group becomes available") + ladder_table, wide=True)}
      </div>"""

    disc = "".join([
        card("AUROC by feature group and algorithm",
             f"Mean out-of-fold AUROC over {s['N_REPEATS']} repeats of {s['OUTER_FOLDS']}-fold nested CV. The repeat SD only reflects seed sensitivity; the SD over outer folds ({fold_sd.min():.2f} to {fold_sd.max():.2f}, with {test_n} to {test_n + 1} patients per test fold) is the better guide. Click a legend entry to hide an algorithm.",
             plot("auroc", F.auroc_ladder(summ, ceiling), png="auroc_ladder.png", alt="AUROC by feature group for four algorithms with error bars") + auroc_table,
             select("auroc", "Error bars", [("repeats", "SD over repeats"), ("folds", "SD over outer folds")]), wide=True),
        card("Change at each step",
             f"Small points are the {n_folds} paired per-fold differences, large points their mean. p values in the table use the Nadeau and Bengio corrected resampled t-test; the naive paired t-test is shown next to it to show how much it overstates the evidence.",
             plot("deltas", F.ladder_deltas(r["ladder_deltas"], con), png="ladder_deltas.png", alt="Per-fold change in AUROC for each contrast") + contrast_table, wide=True),
        card("Raw values against the PK/PD model",
             f"L4 and L4r use data up to the same day. Tree models do slightly better with the raw values ({rng(raw.drop('enet').delta_mean.min(), raw.drop('enet').delta_mean.max(), sign=True)} for L4 against L4r) and the elastic net slightly worse ({signed(raw.loc['enet', 'delta_mean'])}). None of these is significant (corrected p {rng(raw.p_corrected.min(), raw.p_corrected.max(), 2)}).",
             plot("raw", F.raw_check(summ, con), alt="AUROC for L4 against L4r per algorithm"), wide=True),
    ])

    checks_html = "".join([
        card("Leave one study out",
             f"Train on two studies, test on the third. At L4 the mean AUROC in the held-out study is {held.min():.2f} to {held.max():.2f}; study {held.idxmin()}{hardest_note} is the hardest. Columns only one study measured are dropped when that study is held out.",
             plot("loso", F.loso(lo, r["cohort_summary"], summ), alt="AUROC in each held-out study") + f"<details><summary>All leave-one-study-out results</summary>{loso_table}</details>",
             select("loso", "Feature group", [(st, st) for st in F.STEPS], default="L4")),
        card("Missingness and study-only checks",
             f"Missingness depends on the study. A model on missing-value indicators alone reaches {mo.min():.2f} to {mo.max():.2f}, and one on study membership alone {so.min():.2f}. Error bars are the SD over outer folds.",
             plot("checkplot", F.checks(summ), alt="AUROC for the missingness and study-only checks")),
        card("Missing data by study",
             f"Share of patients with each feature missing, for the features that are ever missing. Columns more than {s['MISSING_DROP']:.0%} missing in a training fold are dropped.",
             plot("missing", F.missingness(r["missingness_by_study"], finfo), alt="Heatmap of missingness by feature and study", tall=True)),
        card("Imputation sensitivity",
             f"L5 with five imputers, first repeat only, mean and SD over {s['OUTER_FOLDS']} folds. The spread within an algorithm is at most {sens_spread:.3f}, well inside the fold-to-fold SD.",
             plot("impute", F.imputation(sens, IMPUTER_LABELS), alt="AUROC by imputation method and algorithm")),
    ])

    calib = "".join([
        card("Calibration of the best model",
             f"{ALGO_LABELS[champ['algo']]} on {champ['step']}, out-of-fold predictions averaged over repeats and split into fifths. Slope {champ_cal.slope:.2f}, intercept {champ_cal.intercept:.2f}.",
             plot("calcurve", F.calibration_curve(r["calibration_curve"], f"{champ['step']} {ALGO_LABELS[champ['algo']].lower()}"), png="calibration.png", alt="Calibration curve")),
        card("Calibration slope and intercept",
             f"From logistic recalibration of the out-of-fold predictions. XGBoost is overconfident (slope {xgb_slopes.min():.2f} to {xgb_slopes.max():.2f}) and random forest underestimates the chance of being at target at {join_and(rf_int.tag.tolist())} (intercept {rf_int.intercept.min():.2f} to {rf_int.intercept.max():.2f}).",
             plot("calstats", F.calibration_stats(cal), alt="Calibration slope or intercept by step and algorithm") + f"<details><summary>Table</summary>{cal_table}</details>",
             select("calstats", "Show", [("slope", "Slope"), ("intercept", "Intercept")])),
        card("Precision and recall",
             f"AUPRC is {pr['L4']:.2f} at L4 against {pr['L1']:.2f} at L1, with a prevalence of {info['prevalence']:.2f}.",
             plot("pr", F.pr_curve(r["pr_curve"], {}), png="pr_curve.png", alt="Precision-recall curves")),
        card("Decision curve",
             f"The action is dose intensification for a patient predicted to miss target. The L4 model beats intensifying everyone or no one at every threshold from {beats.min():.2f} to {beats.max():.2f}, for example {dc_at.model:.2f} at {dc_at.threshold:.1f} where treating everyone gives {signed(dc_at.treat_all, 2)}.",
             plot("dca", F.decision_curve(dc), png="decision_curve.png", alt="Decision curve")),
    ])

    algo_opts = [(a, ALGO_LABELS[a]) for a in ALGOS]
    shapsec = "".join([
        card("Global SHAP importance",
             f"Top features of the model refitted on the full cohort (L5). The refitted models keep {kept['enet']} (elastic net), {kept['rf']} (random forest), {kept['xgb']} (XGBoost) and {kept['cat']} (CatBoost) features. SHAP is on the log-odds scale except for random forest, so sizes are not comparable across algorithms.",
             plot("shapbar", F.shap_importance(imp, finfo), png="shap_bar.png", alt="Mean absolute SHAP per feature", tall=True),
             select("shapbar", "Algorithm", algo_opts)),
        card("SHAP per patient",
             "Each dot is a patient. Color is the feature value within the cohort, from low (blue) to high (orange). Positive SHAP pushes the prediction towards target; a high IC50 or BASE estimate pushes it away.",
             plot("shapdot", F.shap_dots({a: r[f"shap_values_{a}"] for a in ALGOS}, r["features"], finfo), png="shap_dot_enet.png", alt="SHAP dot plot", tall=True),
             select("shapdot", "Algorithm", algo_opts)),
        card("Consensus across algorithms",
             f"Share of total |SHAP| per feature for the features with the largest mean share. The day-98 IC50 estimate is first for all four; the trees lean more on the raw day-42 biomarker.",
             plot("consensus", F.shap_consensus(r["shap_consensus"], finfo), png="shap_consensus.png", alt="Share of SHAP per feature by algorithm", tall=True)),
        card("Feature selection frequency",
             f"Share of the {n_folds} outer folds in which each feature was kept (RFE for the trees, a non-zero coefficient for the elastic net). On L5 the day-98 IC50 estimate is kept in {ic50_kept:.0%} of folds by every algorithm.",
             plot("selfreq", F.selection(freq, finfo, ladder), png="selection_frequency.png", alt="Heatmap of feature selection frequency", tall=True),
             select("selfreq", "Feature group", [(st, st) for st in ladder.step], default="L5")),
        card("SHAP against the known mechanism",
             f"The true late biomarker is split into exact Shapley parts for exposure, baseline level and drug sensitivity, and SHAP is summed within the feature groups that mostly carry each part. The truth gives exposure {true_exp:.0%}; the models give it {shap_exp.min():.0%} to {shap_exp.max():.0%}. Spearman correlations per patient are {rng(base_rho.min(), base_rho.max(), 2)} for baseline level and {rng(sens_rho.min(), sens_rho.max(), 2)} for drug sensitivity.",
             plot("svtshare", F.shap_vs_truth_shares(svt), png="shap_vs_truth.png", alt="Share of SHAP by mechanism group against the true split") + svt_table, wide=True),
        card("Summed SHAP against the true part, per patient",
             "Pick an algorithm and a mechanism group. XGBoost and CatBoost dropped every exposure feature during feature elimination, so their exposure panels are flat.",
             plot("svtscatter", F.shap_vs_truth_scatter({a: r[f"shap_grouped_{a}"] for a in ALGOS}, r["truth_components"], svt), alt="Summed SHAP against the true Shapley part"),
             select("svtscatter", "Algorithm", algo_opts) + select("svtscatter", "Group", [(g, g) for g in F.GROUPS], default="drug sensitivity"), wide=True),
    ])

    bio = r["biomarker_records"]
    study_of = bio.groupby("id").study.first()
    est = r["map_estimates"]
    cohort = "".join([
        card("Simulated biomarker trajectories",
             f"Observed biomarker for every patient, colored by outcome. A patient is at target if the first value in the outcome window is below {design['target']:.0f} units. Double-click a legend entry to show one group only.",
             plot("traj", F.trajectories(bio, design), png="cohort.png", alt="Biomarker over time for each simulated patient") + cohort_table,
             select("traj", "Study", [("All", "All studies")] + [(x, f"Study {x}") for x in sorted(bio.study.unique())]), wide=True),
        card("MAP estimates against the truth",
             f"Individual estimates from the sequential PK-then-PD fit against the simulated true values. IC50 is barely identifiable early (r {ic50.loc['pkpd_v1', 'r_log']:.2f} with data to day 21) and much clearer by day 98 (r {ic50.loc['pkpd_v3', 'r_log']:.2f}). The table gives r on the log scale and shrinkage.",
             plot("map", F.map_vs_truth(est, r["truth_components"], mvt, study_of, BLOCK_LABELS), png="map_ic50.png", alt="MAP estimate against the true value") + map_table,
             select("map", "Fit with", list(BLOCK_LABELS.items())) + select("map", "Parameter", [(p, p) for p in ["CL", "V1", "BASE", "IC50"]], default="IC50"), wide=True),
    ])

    sections = [
        ("overview", "Overview", overview),
        ("discrimination", "Discrimination", f'<div class="grid-2">{disc}</div>'),
        ("checks", "Checks", f'<div class="grid-2">{checks_html}</div>'),
        ("calibration", "Calibration", f'<div class="grid-2">{calib}</div>'),
        ("shap", "SHAP", f'<div class="grid-2">{shapsec}</div>'),
        ("cohort", "Cohort and PK/PD", f'<div class="grid-2">{cohort}</div>'),
    ]
    nav = "".join(f'<a href="#{k}">{t}</a>' for k, t, _ in sections)
    body = "".join(f'<section id="{k}" class="section"><h2 class="section-title">{t}</h2>{b}</section>' if k != "overview" else f'<section id="{k}" class="section">{b}</section>' for k, t, b in sections)

    figs_json = json.dumps(FIGS, separators=(",", ":")).replace("</", "<\\/")
    dark_json = json.dumps(DARK_MAP)
    today = dt.date.today().isoformat()

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drug X Response Prediction</title>
  <meta name="description" content="Results dashboard: predicting long-term biomarker response to a hypothetical drug from early-treatment data and PK/PD estimates, on simulated studies.">
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="assets/style.css">
  <script src="{PLOTLY_JS}" integrity="{PLOTLY_SRI}" crossorigin="anonymous" defer></script>
  <script src="assets/app.js" defer></script>
</head>
<body>
  <header class="site-header">
    <div class="wrap">
      <p class="eyebrow">Simulated Drug X studies</p>
      <h1>Predicting long-term response to Drug X from early-treatment data</h1>
      <p class="lede">Can the first weeks of treatment tell whether a patient's biomarker will be at target about a year later, and do individual PK/PD estimates help beyond the raw clinical data? Four algorithms, a feature-group ladder, nested CV and SHAP checked against the known mechanism.</p>
      <p class="banner">All data on this page are simulated from a made-up PK/PD model. It's a research and teaching demo, not validated and not for patient care.</p>
      <nav class="tabs" aria-label="Sections">{nav}<button type="button" id="theme" class="theme-toggle" aria-pressed="false">Dark mode</button></nav>
    </div>
  </header>
  <main class="wrap">
    <noscript><p class="note">The charts need JavaScript. The tables below have every number, and the static figures are in the <a href="{REPO_URL}/tree/main/figures">repository</a>.</p></noscript>
    {body}
  </main>
  <footer class="wrap footer">
    <p>Code, results and figures: <a href="{REPO_URL}">github.com/Wrlog/remission-prediction-shap</a>. Charts drawn with plotly from the tables in <code>results/</code>; the static PNGs come from matplotlib.</p>
    <p>Simulated data only. Not for patient care. Built {today} from a pipeline run that took {info['runtime_seconds'] / 60:.0f} minutes (Python {info['python']}, scikit-learn {info['versions']['scikit-learn']}).</p>
  </footer>
  <script id="figures" type="application/json">{figs_json}</script>
  <script id="dark-map" type="application/json">{dark_json}</script>
</body>
</html>
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    assets = out / "assets"
    assets.mkdir(exist_ok=True)
    for f in (Path(__file__).parent / "static").iterdir():
        shutil.copy2(f, assets / f.name)
    figs_out = out / "figures"
    figs_out.mkdir(exist_ok=True)
    for f in figures_dir.glob("*.png"):
        shutil.copy2(f, figs_out / f.name)
    (out / ".nojekyll").write_text("")
    return out / "index.html"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--figures", default=str(ROOT / "figures"))
    ap.add_argument("--out", default=str(ROOT / "site"))
    a = ap.parse_args()
    path = build(Path(a.results), Path(a.figures), Path(a.out))
    size = path.stat().st_size / 1e6
    print(f"wrote {path} ({size:.1f} MB, {len(FIGS)} charts)")


if __name__ == "__main__":
    main()
