"""Convert the standalone analysis scripts into numbered notebooks, in the deck's house style.

`analysis/11`, `analysis/12` and `analysis/13` were written as scripts because they were added
late, during the manuscript round. Everything else in this project is a numbered notebook with
markdown module headers, and the reproducibility statement refers to the pipeline that way, so
these three belong in `notebooks/` too.

    analysis/11_confound_signal_quality_and_trend_tests.py
        -> notebooks/11_Confound_Control_and_Trend_Tests.ipynb
    analysis/12_xai_elderly_confidence_pair.py
        -> notebooks/12_XAI_Elderly_Confidence_Pairs.ipynb
    analysis/13_multiseed_replication.py
        -> notebooks/13_MultiSeed_Replication.ipynb
    analysis/14_full_ensemble_multiseed.py
        -> notebooks/14_Full_Ensemble_MultiSeed.ipynb

The code is copied verbatim, split at the section markers the scripts already carry, with one
markdown cell written in front of each. Three adaptations are applied, and nothing else:

  * `ROOT = Path(__file__)...` becomes a cwd-based resolution, since notebooks have no __file__;
  * `if __name__ == '__main__': main()` becomes a plain `main()` call;
  * a closing module displays the PNGs the script saved, so the figures appear in the notebook
    without touching the plotting code (which keeps the Agg backend and its `plt.close`).

Keeping the code verbatim is the point: the notebook and the script must not drift, or we
reintroduce exactly the provenance problem this review round was about.

Usage:
    .venv/bin/python scratch_ppt/build_analysis_notebooks.py
    .venv/bin/python scratch_ppt/build_analysis_notebooks.py --execute 12
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
NOTEBOOKS = ROOT / "notebooks"

SECTION_RE = re.compile(r"^# -{3,}\s*(.+?)\s*$", re.M)

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.3",
                      "mimetype": "text/x-python", "file_extension": ".py",
                      "codemirror_mode": {"name": "ipython", "version": 3},
                      "pygments_lexer": "ipython3", "nbconvert_exporter": "python"},
}


# --------------------------------------------------------------------------- notebook specs
NB11 = dict(
    script="11_confound_signal_quality_and_trend_tests.py",
    out="11_Confound_Control_and_Trend_Tests.ipynb",
    title="🫀 PTB-XL Notebook 11 — Confound Control and Formal Trend Tests",
    summary="""## Executive Summary & Purpose

Review Items 2 and 3 asked two questions the earlier notebooks could not answer. Item 2: is the
age effect really about age, or is it a proxy for older patients producing noisier recordings?
Item 3: the monotonicity claims were being made by eye — where is the trend statistic?

This notebook answers both on the held-out TEST split (N = 2,198).

**Item 2 — the confound.** Four acquisition-quality covariates are computed for every record by
differencing the raw calibrated signal against the 0.5–40 Hz filtered version: in-band SNR,
baseline-wander RMS, high-frequency-noise RMS, and dead-lead count. Each is tested against age,
and every age-stratified outcome is then re-fit adjusting for all four.

**Item 3 — the trend statistics.** Spearman ρ against age, Kruskal–Wallis across the four bands,
and an HC0 heteroskedasticity-robust OLS slope per decade, for conformal set size (both bases),
per-record calibration error, exact-match accuracy, and label count as a candidate mediator.

**Two further results.** The Mondrian set-size trend is refitted over 300 bootstrap resamples of
the calibration split, which is what explains the non-monotone dip at 65–80; and the
explanation-consistency comparison is given a paired test it is honestly underpowered for.

Before computing anything new, Module 4 re-derives the published pipeline from the released
checkpoints and asserts it reproduces the archived macro-AUC and all four Mondrian set sizes.
If that guard fails, nothing downstream should be trusted.

**Outputs** — `outputs/dataset_validation/11_confound_and_trend_tests.json` and four figures in
`outputs/figures/nb11_confound_trends/`.""",
    modules={
        "data": ("Module 1 — Splits, Labels, Demographics and Age Bands",
                 "Loads the calibration and TEST splits and the master metadata, rebuilds the "
                 "demographic vector the DualBranch model expects (age standardised against the "
                 "TRAIN mean and SD, plus sex), and asserts the label arrays match the metadata "
                 "before anything uses them. Age bands are the project's standard four: "
                 "<40, 40–65, 65–80, 80+."),
        "models (verbatim from NB05)": (
            "Module 2 — Architectures, Copied Verbatim from NB05",
            "The three ensemble members are redefined here exactly as NB05 defines them. They are "
            "duplicated rather than imported so this notebook can load the released checkpoints "
            "without depending on another notebook's execution state — if these definitions ever "
            "drift from NB05, the checkpoint load fails loudly rather than silently mis-loading."),
        "member inference (cached)": (
            "Module 3 — Member Inference, Ensemble Weighting and Temperature Scaling",
            "Each member scores the calibration and TEST splits; results are cached to disk so "
            "re-runs are cheap. The members are then combined by calibration-AUC weighting, and "
            "a per-class temperature is fitted on the calibration split by bounded scalar NLL "
            "minimisation — the same procedure as NB05, reproduced rather than re-derived."),
        "reproduction check": (
            "Module 4 — Reproduction Guard",
            "**This is the gate.** The rebuilt pipeline is compared against the archived "
            "close-out metrics: macro-AUC to four decimals and all four Mondrian set sizes to "
            "three. Everything after this module is new analysis layered on the published "
            "pipeline, so if this check fails the rest is meaningless and the notebook says so."),
        "(A) signal quality": (
            "Module 5 — Signal-Quality Covariates (Review Item 2)",
            "Acquisition quality per record, from the residual between the raw calibrated signal "
            "(`X_test_100.npy`) and the 0.5–40 Hz filtered signal (`X_test_clean_100.npy`): "
            "in-band SNR in dB, baseline-wander RMS, high-frequency-noise RMS, dead-lead count. "
            "Each is then tested against age by Spearman correlation and Kruskal–Wallis across "
            "bands. The headline result is that **SNR has no association with age at all**, which "
            "is what a noise-driven explanation of the age effect would need."),
        "per-record outcomes": (
            "Module 6 — Per-Record Outcomes",
            "The quantities the trend tests operate on, one value per TEST record: conformal set "
            "size under the global threshold and under Mondrian refitting, absolute calibration "
            "error, exact-match correctness, and the record's true label count."),
        "(B) trend tests": (
            "Module 7 — Formal Trend Tests (Review Item 3)",
            "For each outcome: Kruskal–Wallis across the four bands, Spearman ρ against raw age, "
            "and OLS slope per decade with HC0 robust standard errors — unadjusted, adjusted for "
            "the four signal-quality covariates, and additionally adjusted for label count. The "
            "attenuation after adjustment is the number that answers Item 2 quantitatively. "
            "Label count is treated as a **mediator**, not a confound: older patients genuinely "
            "carry more concurrent conditions, so adjusting it away would remove part of the "
            "effect we are trying to describe."),
        "(C) calibration-resample stability": (
            "Module 8 — Calibration-Resampling Stability",
            "The whole Mondrian procedure is refitted over 300 bootstrap resamples of the "
            "calibration split. This answers the supervisor's clarification about monotonicity: "
            "the 65–80 dip is a property of calibration-cell size, not of the age effect, and the "
            "slope stays positive in every resample."),
        "(D) explanation-consistency trend test": (
            "Module 9 — Explanation-Consistency Test",
            "A paired comparison of split-half Integrated-Gradients rank agreement between "
            "high- and low-confidence elderly cases. It is paired over five classes only, so it "
            "is underpowered by construction; the notebook records that in the output rather "
            "than presenting the p-value as if it were decisive."),
        "figures": ("Module 10 — Figures",
                    "Four figures: signal quality by age band, the age effect before and after "
                    "adjustment, reliability by band, and the calibration-resampling stability "
                    "panel. These are Figures 16–18 and the reliability panel in the manuscript."),
        "save": ("Module 11 — Save",
                 "Everything computed above is written to a single JSON so the deck and the "
                 "manuscript can read the numbers rather than have them retyped."),
    },
)

NB12 = dict(
    script="12_xai_elderly_confidence_pair.py",
    out="12_XAI_Elderly_Confidence_Pairs.ipynb",
    title="🫀 PTB-XL Notebook 12 — Attribution Overlays for Confident and Unconfident Elderly Cases",
    summary="""## Executive Summary & Purpose

Review Item 7 asked for literal ECG-with-heat-map examples alongside the quantitative
importance-score comparison the deck already had. A reader should be able to *see* what an
attribution map looks like when the model is sure about an 80+ patient and when it is not.

For CD and HYP — the two superclasses the model discriminates worst — this notebook picks the
most- and least-confident **correctly labelled** 80+ TEST record by predictive entropy and draws,
side by side, the Integrated-Gradients overlay on all 12 leads plus a Grad-CAM-1D temporal strip.
The attribution code is the same as NB06's.

The point of the figure is that the attribution concentrates on the same morphological region in
both columns. That is the visual counterpart of the ρ = 0.955 vs 0.980 rank agreement reported
in NB06: a weak confidence effect, not the collapse an earlier draft claimed.

**Outputs** — `outputs/figures/nb12_xai_pairs/12_xai_pair_{CD,HYP}.png` and
`outputs/dataset_validation/12_xai_pair_cases.json` recording exactly which records were chosen.""",
    anchors=[
        ("class InceptionBlock1D",
         "Module 1 — The Attribution Model",
         "InceptionTime1D, the signal-only anchor architecture, defined verbatim as in NB05/NB06. "
         "Attribution is computed on this single model rather than the ensemble: Integrated "
         "Gradients needs one differentiable path from input to logit, and the ensemble's "
         "AUC-weighted average of three architectures does not give a single coherent one."),
        ("X_test = np.transpose",
         "Module 2 — Data, Checkpoint and Predictive Entropy",
         "Loads the TEST split and the released checkpoint, scores every record, and computes "
         "mean binary entropy across the five classes as the confidence measure. Entropy rather "
         "than the predicted probability of the target class, because we want records where the "
         "model is globally decisive or globally hesitant, not just confident about one label."),
        ("def integrated_gradients",
         "Module 3 — Integrated Gradients and Grad-CAM-1D",
         "Integrated Gradients with 64 steps and a zero baseline gives per-sample, per-lead "
         "attribution. Grad-CAM-1D on the fourth Inception block gives a coarse temporal "
         "saliency strip. Showing both guards against reading too much into either: IG is "
         "fine-grained but noisy, Grad-CAM is smooth but low-resolution."),
        ("elderly = age_test >= 80",
         "Module 4 — Case Selection and the Paired Overlay Figures",
         "Within the 80+ cohort, for each of CD and HYP, the correctly-labelled true positives "
         "are ranked by entropy and the extremes are taken. Both panels are therefore cases the "
         "model got *right* — the comparison is about confidence, not about correctness, which "
         "is what makes the stability of the attribution meaningful."),
        ("with open(ROOT",
         "Module 5 — Record the Chosen Cases",
         "The selected record indices, their probabilities, entropies, ages and true labels are "
         "written to JSON. Without this the figures would not be traceable to specific records, "
         "which is the standard this project now holds itself to."),
    ],
)

NB13 = dict(
    script="13_multiseed_replication.py",
    out="13_MultiSeed_Replication.ipynb",
    title="🫀 PTB-XL Notebook 13 — Multi-Seed Replication of the Primary Pipeline",
    summary="""## Executive Summary & Purpose

Review Item 5: every result in this project came from a single training seed (42), and a reviewer
is right to discount single-seed deep-learning numbers.

This notebook retrains the pipeline's anchor architecture — InceptionTime1D, depth 4, the
NB03/NB05 model — at seeds 1, 7 and 13 under the hyperparameters recorded in the manuscript's
reproducibility section, then pushes **each seed through the entire downstream pipeline**:
ensemble re-weighting, per-class temperature scaling, global and Mondrian conformal thresholds,
prediction-set size by age band, and the trend statistic.

Running the full pipeline per seed, rather than just reporting four macro-AUCs, is the point. The
question the review is really asking is not "how much does accuracy wobble" but "does the age
gradient survive reseeding", and only the downstream numbers answer that.

**Scope, stated plainly.** Training here is CPU-only, so the two heavier ensemble members
(DualBranchECGNet and ResNet1D101) are held at their released checkpoints and only the anchor is
reseeded. This measures seed variance in the component that can be honestly retrained; it is not
full ensemble seed variance, and the manuscript says so rather than overclaiming.

⚠️ **This notebook trains three models and takes several hours on CPU.** Module 5 is the
expensive one. The equivalent script can be run detached instead:
`.venv/bin/python analysis/13_multiseed_replication.py`

**Outputs** — `outputs/dataset_validation/13_multiseed_replication.json` and one checkpoint per
seed in `outputs/`.""",
    modules={
        "config": ("Module 0 — Configuration",
                   "Seeds, band definitions, and the training hyperparameters copied from "
                   "manuscript section 10 verbatim: AdamW at lr 1e-3, weight decay 1e-4, batch "
                   "64, maximum 10 epochs, ReduceLROnPlateau (factor 0.5, patience 2) and early "
                   "stopping (patience 4) on validation macro-AUC. Changing any of these would "
                   "make the new seeds incomparable with seed 42."),
        "data": ("Module 1 — Splits and Age Bands",
                 "TRAIN (17,418), VALIDATION (1,080), CALIBRATION (1,103) and TEST (2,198), plus "
                 "the demographic vector and the four age bands on both the calibration and TEST "
                 "sides. The calibration split matters as much as TEST here, because the "
                 "conformal thresholds are refitted per seed."),
        "models (verbatim from NB05)": (
            "Module 2 — Architectures",
            "All three members, defined as in NB05. Only InceptionTime1D is retrained; the other "
            "two are needed to load their released checkpoints and reproduce the ensemble."),
        "shared pipeline pieces": (
            "Module 3 — The Downstream Pipeline as a Function",
            "`downstream()` takes one seed's anchor probabilities and runs everything that "
            "follows: AUC-weighted ensembling, per-class temperature, global and Mondrian "
            "conformal thresholds, set size and exact-match by age band, and Spearman ρ against "
            "age. Writing it once means each seed is processed identically by construction."),
        "training": ("Module 4 — Training the Anchor at a New Seed",
                     "Class-balanced BCE with positive weights from the training label "
                     "frequencies, best-on-validation checkpointing, and early stopping. The "
                     "seed is set for torch, numpy and the DataLoader's shuffle generator, so the "
                     "only thing that differs between runs is the seed."),
        "main": ("Module 5 — Run All Seeds and Summarise",
                 "⚠️ **The expensive cell.** Scores seed 42 from its released checkpoint first "
                 "(so it is measured identically to the new seeds), then trains and evaluates "
                 "seeds 1, 7 and 13. The summary reports mean ± SD and range for the anchor and "
                 "ensemble macro-AUC and for the per-band set sizes, and — the part the review "
                 "actually needs — whether the age trend's direction, monotonicity and "
                 "significance hold in **every** seed."),
    },
)

NB14 = dict(
    script="14_full_ensemble_multiseed.py",
    out="14_Full_Ensemble_MultiSeed.ipynb",
    title="\U0001FAC0 PTB-XL Notebook 14 \u2014 Full Three-Architecture Ensemble Seed Variance",
    summary="""## Executive Summary & Purpose

Review Item 5, residual. NB13 reseeded one ensemble member of three, because training here is
CPU-only, and reported the consequence rather than hiding it: two of the three members were
identical across every row of its Table 8, so the \u00b10.0005 in that table's ensemble column is a
**lower bound, not an estimate**. The supervisor accepted that scoping and asked for the other
two members once the compute was available. This notebook is that run.

It retrains **DualBranchECGNet** and **ResNet1D101** at seeds 1, 7 and 13, pairs each with the
anchor checkpoint NB13 already produced for the same seed, and pushes every seed through the
identical downstream pipeline. Each row then has all three members trained at that seed, which
is what turns the ensemble column into a real estimate.

**Recipe fidelity \u2014 the point that decides whether this run means anything.** Seed 42 for these
two members *is* the NB02 checkpoint, so the retrained seeds use **NB02's** recipe (AdamW,
CosineAnnealingLR with T_max 10, 10 fixed epochs, no early stopping, best-on-VAL checkpoint),
not NB13's ReduceLROnPlateau-plus-early-stopping schedule. Training them under NB13's schedule
would mix a schedule change into what is supposed to be pure seed variance. The anchor is not
retrained at all here \u2014 NB13's checkpoints are reused untouched \u2014 so the anchor column is
identical across the two runs by construction, which is exactly what makes the two ensemble
columns comparable.

**The guard runs before the cost.** Module 6 re-derives NB13's seed-42 row from the released
checkpoints and aborts unless it matches `13_multiseed_replication.json` to 1e-9. A drifted
dataset or checkpoint fails in about a minute instead of twelve hours later.

\u26a0\ufe0f **This notebook trains six models and takes roughly 12 hours on CPU.** Module 5 is the
expensive one, and ResNet1D101 dominates it. Every (model, seed) checkpoint is written and
reloaded separately, so an interrupted run resumes rather than restarting. For a run this long,
prefer the detached script:

```
nohup .venv/bin/python analysis/14_full_ensemble_multiseed.py > outputs/14_run.log 2>&1 &
```

**Outputs** \u2014 `outputs/dataset_validation/14_full_ensemble_multiseed.json` and six checkpoints
`outputs/14_seed{S}_{Model}_best.pth`.""",
    modules={
        "config": ("Module 0 \u2014 Configuration",
                   "Seeds, band definitions, and NB02's training recipe: AdamW at lr 1e-3, weight "
                   "decay 1e-4, batch 64, 10 fixed epochs and CosineAnnealingLR with T_max 10, "
                   "with no early stopping. These are not NB13's hyperparameters, and the "
                   "difference is deliberate \u2014 they are the ones the seed-42 checkpoints for "
                   "DualBranchECGNet and ResNet1D101 were trained under."),
        "data": ("Module 1 \u2014 Splits and Age Bands",
                 "TRAIN, VALIDATION, CALIBRATION and TEST, plus the four age bands on both the "
                 "calibration and TEST sides. Demographics are built for every split here, not "
                 "just calibration and TEST, because unlike NB13 this run trains the dual-branch "
                 "member and so needs them on TRAIN and VALIDATION too."),
        "models (verbatim from NB05/NB13)": (
            "Module 2 \u2014 Architectures",
            "All three members, defined exactly as in NB05 and NB13. `ckpt_path()` is the single "
            "place that decides where a given (member, seed) checkpoint lives: released for seed "
            "42, NB13's file for the anchor at a new seed, and this run's file for the two "
            "members being retrained."),
        "shared pipeline pieces": (
            "Module 3 \u2014 The Downstream Pipeline as a Function",
            "`downstream()` takes all three members' probabilities and runs AUC-weighted "
            "ensembling, per-class temperature, global and Mondrian conformal thresholds, set "
            "size and exact-match by age band, and Spearman \u03c1 against age. NB13's version "
            "closed over two fixed members; this one takes the whole set, because here every "
            "member varies by seed. The body is otherwise unchanged, which is what keeps the two "
            "runs' rows comparable."),
        "training (NB02's recipe)": (
            "Module 4 \u2014 Retraining a Member at a New Seed",
            "NB02's loop reproduced: class-balanced BCE with positive weights from the training "
            "label frequencies, a cosine schedule stepped once per epoch, ten fixed epochs and "
            "the best-on-validation state kept. The seed is set for torch, numpy and the "
            "DataLoader's shuffle generator. An existing checkpoint is reloaded rather than "
            "retrained, which is what makes a twelve-hour run survive an interruption."),
        "preflight": ("Module 5 \u2014 Preflight and Per-Seed Scoring",
                      "`preflight()` refuses to start unless all nine checkpoints and NB13's "
                      "result file are on disk \u2014 including the three anchor checkpoints NB13 "
                      "produces, without which this run has nothing to pair its members with. "
                      "`score_seed()` loads or trains each member at one seed and hands the set "
                      "to the downstream pipeline."),
        "main": ("Module 6 \u2014 Guard, Run All Seeds, Summarise",
                 "\u26a0\ufe0f **The expensive cell.** Seed 42 is scored first and checked against NB13 to "
                 "1e-9; a mismatch aborts before anything is trained. Then seeds 1, 7 and 13 are "
                 "retrained and evaluated. The summary reports mean \u00b1 SD and range for the "
                 "anchor, the ensemble and each individual member, the per-band set sizes, "
                 "whether the age trend's direction, monotonicity and significance hold in "
                 "**every** seed, and \u2014 the number this run exists to produce \u2014 how far NB13's "
                 "lower-bound ensemble SD sat below the real one."),
    },
)

SPECS = [NB11, NB12, NB13, NB14]

MODULE_NUM_RE = re.compile(r"^Module\s+\d+[a-z]?\s+[—-]\s+")

FIGURE_MODULE = """Figures Produced by This Notebook

The plotting code above keeps the script's non-interactive backend and writes each figure to
disk, so this cell displays the saved PNGs inline. The files on disk are the ones the deck and
the manuscript reference."""


def adapt(src: str) -> str:
    """The three changes a notebook needs; nothing else is touched."""
    src = src.replace(
        "ROOT = Path(__file__).resolve().parent.parent",
        "# notebooks have no __file__; resolve the project root from the working directory\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()")
    src = re.sub(r"\nif __name__ == ['\"]__main__['\"]:\s*\n\s*main\(\)\s*\n?$",
                 "\nmain()\n", src)
    return src


def split_on_sections(src: str):
    """Split a script at its own `# ----- name` markers, preamble first."""
    marks = list(SECTION_RE.finditer(src))
    if not marks:
        return [("", src)]
    out = [("", src[:marks[0].start()])]
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(src)
        out.append((m.group(1), src[m.end():end]))
    return out


def split_on_anchors(src: str, anchors):
    """Split a script at given substrings, preamble first."""
    idx = []
    for a, _t, _d in anchors:
        i = src.index(a)
        idx.append(i)
    out = [("", src[:idx[0]])]
    for k, (a, _t, _d) in enumerate(anchors):
        end = idx[k + 1] if k + 1 < len(idx) else len(src)
        out.append((a, src[idx[k]:end]))
    return out


def strip_docstring(preamble: str) -> str:
    """The module docstring becomes the notebook's summary cell, so drop it from the code."""
    s = preamble.lstrip()
    if s.startswith('"""'):
        end = s.index('"""', 3) + 3
        return s[end:].lstrip("\n")
    return preamble


def figure_paths(spec) -> list[str]:
    if spec is NB11:
        return ["outputs/figures/nb11_confound_trends"]
    if spec is NB12:
        return ["outputs/figures/nb12_xai_pairs"]
    return []


def build(spec) -> Path:
    src = adapt((ANALYSIS / spec["script"]).read_text())
    nb = nbf.v4.new_notebook(metadata=KERNEL)
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {spec['title']}\n\n{spec['summary']}"))

    if "anchors" in spec:
        chunks = split_on_anchors(src, spec["anchors"])
        titles = {a: (t, d) for a, t, d in spec["anchors"]}
    else:
        chunks = split_on_sections(src)
        titles = spec["modules"]

    # modules are numbered here, not in the specs, so a script gaining or losing a
    # section never leaves two cells claiming to be the same module
    n = 0
    for k, (key, code) in enumerate(chunks):
        if k == 0:
            code = strip_docstring(code)
            head = ("Imports, Seeds and Paths",
                    "Imports, the global seed, and the project paths. The seed is set here and "
                    "nowhere else, so every stochastic step below inherits it.")
        else:
            head = titles.get(key)
            if head is None:
                head = (key, "")
        code = code.strip("\n")
        if not code:
            continue
        title, desc = head
        title = MODULE_NUM_RE.sub("", title)
        nb.cells.append(nbf.v4.new_markdown_cell(
            f"## Module {n} — {title}" + (f"\n\n{desc}" if desc else "")))
        nb.cells.append(nbf.v4.new_code_cell(code))
        n += 1

    figs = figure_paths(spec)
    if figs:
        nb.cells.append(nbf.v4.new_markdown_cell(f"## Module {n} — " + FIGURE_MODULE))
        nb.cells.append(nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "from IPython.display import Image, display, Markdown\n\n"
            f"for folder in {figs!r}:\n"
            "    for png in sorted((ROOT / folder).glob('*.png')):\n"
            "        display(Markdown(f'**{png.name}**'))\n"
            "        display(Image(filename=str(png)))"))

    dest = NOTEBOOKS / spec["out"]
    nbf.write(nb, dest)
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", nargs="*", default=None,
                    help="notebook numbers to execute after writing, e.g. --execute 12")
    a = ap.parse_args()

    built = {}
    for spec in SPECS:
        dest = build(spec)
        nb = nbf.read(dest, as_version=4)
        n_code = sum(1 for c in nb.cells if c.cell_type == "code")
        print(f"[BUILT] {dest.relative_to(ROOT)}  ({len(nb.cells)} cells, {n_code} code)")
        built[spec["out"][:2]] = dest

    for num in (a.execute or []):
        dest = built.get(num)
        if dest is None:
            print(f"[SKIP] no notebook {num}")
            continue
        print(f"[EXEC] {dest.name} — this runs the real analysis")
        from nbclient import NotebookClient
        nb = nbf.read(dest, as_version=4)
        NotebookClient(nb, timeout=-1, kernel_name="python3",
                       resources={"metadata": {"path": str(NOTEBOOKS)}}).execute()
        nbf.write(nb, dest)
        print(f"[EXEC] {dest.name} done")


if __name__ == "__main__":
    main()
