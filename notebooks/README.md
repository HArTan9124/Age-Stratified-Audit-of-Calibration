# 📓 Notebooks — what each one does and how

## The big picture

**Goal of the project:** read a 12‑lead ECG and predict which of **5 broad diagnosis groups** it shows,
then go two steps further than a plain classifier — check that the model's probabilities are **honest**
(*calibration*), and show **which part of the ECG** drove each answer (*explainability*) — with special
attention to whether the model is **fair across patient age**.

**A few terms, in plain words:**

| Term | Meaning |
|---|---|
| **12‑lead ECG** | 12 simultaneous "views" of the heart's electrical activity (leads I, II, III, aVR, aVL, aVF, V1–V6). Here each recording is 10 seconds at 100 Hz → a `12 × 1000` grid of numbers. |
| **5 superclasses** | `NORM` = normal · `MI` = heart attack (myocardial infarction) · `STTC` = ST/T‑wave changes · `CD` = conduction problem · `HYP` = thickened heart muscle (hypertrophy). |
| **Multi‑label** | One ECG can be in **several** groups at once (e.g. `MI` *and* `STTC`). So the model outputs 5 independent yes/no probabilities, not one "pick the class" answer. |
| **Patient‑safe split** | Train / validation / calibration / test sets never share a patient, so the model can't cheat by recognising a person instead of a disease. |
| **macro‑AUC** | Average over the 5 groups of the "area under the ROC curve" — a 0.5→1.0 score for how well the model separates sick from not‑sick, independent of any cut‑off. **0.934 is the published state of the art** for this exact task — Strodthoff et al. 2020 and Gitau et al. 2025 both reach it with six-model ensembles; their best *single* models sit at 0.918–0.930. |
| **Calibration / ECE** | Does "70% chance of MI" actually come true ~70% of the time? **ECE** (Expected Calibration Error) is the average gap between the stated probability and reality; 0 = perfect. |

---

## Run order

```text
01_Data_Preparation.ipynb              ← run once, builds the dataset files everything else reads
        │
        ▼
02_3Model_Heterogeneous_Ensemble.ipynb ← trains 3 models, combines them (simple average)
        │
        ▼
03_Optimized_Best_Model.ipynb          ← the recommended model: smarter combining + fixes
        │
        ▼
05_Calibration_and_Uncertainty.ipynb   ← are the probabilities honest? how sure is the model?
        │
        ▼
06_Explainability.ipynb                ← which ECG leads / which moments drove each prediction?

04_ResNet18_vs_Transformer_Baseline.ipynb  ← independent side experiment; run any time after 01
```

Notebooks **02, 03, 05, 06** each just **reload** the files/checkpoints the earlier ones saved — none of
them retrains anything except its own new model. All of them only **read** from `data/processed/` and
**write** figures/checkpoints to `outputs/`; nothing is overwritten in place.

---

## 01 — `01_Data_Preparation.ipynb`  ·  build the dataset

**What it's for:** turn the raw PhysioNet PTB‑XL download into clean, aligned, ready‑to‑train files.

**How it does it, step by step:**

- **Find the data & check it's complete** — locates the PTB‑XL folders on disk and reports how many of
  the 21,837 recordings are present (100 Hz and 500 Hz). Has an optional in‑notebook downloader to fetch
  anything missing.
- **Read the labels & build patient features** — loads `ptbxl_database.csv`, maps every diagnostic code
  to the 5 superclasses, and derives `age`, `sex`, `BMI`, and age groups (`<40`, `40–65`, `65–80`, `80+`).
- **Look at the cohort (EDA)** — plots age/sex distributions, how common each diagnosis is, how often
  diagnoses co‑occur, and by age group.
- **Clean the signal** — applies a **zero‑phase 3rd‑order Butterworth band‑pass filter, 0.5–40 Hz**
  (`scipy.signal.filtfilt`). This removes slow baseline drift and high‑frequency noise **without shifting
  the waveform in time**. Shows a raw‑vs‑filtered comparison on a real 12‑lead strip.
- **Split the data by patient into 4 parts** — never sharing a patient (all 6 pairwise overlaps are empty):

| Split | Folds | Size | Used for |
| --- | --- | --- | --- |
| `TRAIN` | 1–8 | ~17,400 | training the models |
| `VALIDATION` | fold 9A | ~1,080 | picking hyper‑parameters / early stopping |
| `CONFORMAL_CALIBRATION` | fold 9B | ~1,100 | **reserved** for calibration (used in NB05) |
| `TEST` | fold 10 | ~2,198 | the final, untouched scoreboard |

- **Extract & cache the arrays** — writes the filtered signals as fast‑loading `.npy` tensors, plus one
  master table `MASTER_RESEARCH_METADATA.csv`. Normalisation (lead‑wise z‑score) is **fitted only on
  `TRAIN`** so no test information leaks in.
- **Build PyTorch `Dataset` / `DataLoader` objects** and run a **17‑point readiness audit** (shapes,
  alignment, leakage, class balance, NaNs, …) that must fully pass.

**Produces:** `data/processed/X_{train,val,cal,test}_clean_100.npy`, the matching label/age arrays,
`MASTER_RESEARCH_METADATA.csv`, `dataset_manifest.json`, and ~10 EDA figures in
`outputs/dataset_validation/`.

---

## 02 — `02_3Model_Heterogeneous_Ensemble.ipynb`  ·  three models, averaged

**What it's for:** a first strong model — train three **different** network types and average their
opinions, on the idea that they make different mistakes.

**How it does it:**

- **Loads** the `TRAIN` / `VALIDATION` / `TEST` arrays from NB01.
- **Trains 3 models** independently, each predicting the 5 yes/no labels with a class‑balanced
  binary‑cross‑entropy loss:

| Model | What's special about it | Size |
| --- | --- | --- |
| **DualBranchECGNet** | a CNN on the signal **plus a small branch that takes age + sex** and fuses them (ported from Atwa et al. 2025) | ~1.2 M |
| **InceptionTime1D** | looks at the signal with 3 kernel sizes at once (short / medium / long patterns) | ~0.4 M |
| **ResNet1D‑101** | a very deep residual CNN for fine waveform detail | ~3.1 M |

- **Combines them by "soft voting"** — averages the three probability vectors with **equal weight** (⅓
  each). No extra training.
- **Scores** every model and the ensemble on `TEST`, and runs an **age‑group fairness audit**.

**Measured on `TEST` (N = 2198):** ensemble **macro‑AUC 0.9243**, per‑label accuracy ~87%.

**Important honesty note (fixed in this notebook):** an earlier draft quoted "~84–85% accuracy /
~0.965 AUC" — those numbers were never produced by the code (they mixed up two different accuracy
definitions). Module 10 now shows the real, literature‑comparable numbers. See
`papers/Research_and_Analysis.md` §3.

**Also found here:** the equal‑weight average (0.9243) actually scores **slightly below its own best
member**, InceptionTime1D (~0.928), because the weakest model drags the average down — exactly the
effect Strodthoff et al. 2020 warn about. NB03 fixes this.

**Produces:** `outputs/ensemble_{DualBranchECGNet,InceptionTime1D,ResNet1D101}_best.pth` + figures.

---

## 03 — `03_Optimized_Best_Model.ipynb`  ·  the recommended model ⭐

**What it's for:** apply the concrete, paper‑backed fixes to NB02's weaknesses.

**How it does it (each change has a source):**
1. **Train the strongest single architecture properly** — retrains `InceptionTime1D` on its own, with a
   learning‑rate schedule that eases off when validation stops improving (`ReduceLROnPlateau`, from
   Atwa et al. 2025) and early stopping.
2. **Extra data for the rare class** — `HYP` is the rarest and weakest group, so every `HYP` training
   ECG is duplicated with a small amount of added Gaussian noise (augmentation recipe from Atwa et al.
   2025, Table 4). *(Whether this actually helps is flagged for an ablation in `papers/Whats_Needed_Next.md`.)*
3. **Weight the vote by skill** — instead of a flat ⅓/⅓/⅓ average, weight each model by its validation
   macro‑AUC, so the weakest model counts less. *(In practice the three are close, so the weights land
   near ⅓ anyway — a real "stacker" is listed as future work.)*
4. **Tune the decision threshold per class** — the raw model uses 0.5 as the yes/no cut‑off for every
   class; here each class's cut‑off is chosen on `VALIDATION` to maximise its F1 score. This is free
   (no retraining) and mostly helps the imbalanced classes.
5. **Head‑to‑head benchmark** of 4 configurations, an **age‑fairness audit**, and a **metric‑normalised
   literature comparison** (only comparing numbers that use the same task definition).

**Measured on `TEST` (N = 2198):** best config (AUC‑weighted ensemble + tuned thresholds) →
**macro‑AUC 0.9245**, per‑label accuracy **88.5%**, exact‑match accuracy 60.9%. This is **at parity
with the best published *single* models** (0.918–0.930) and about 0.007 **below** the six-model
ensembles (0.934) — not ahead of either, and that's the honest position for a three-model ensemble.

**Age audit:** accuracy still falls from ~78% (under 40) to ~43% (80+). Partly a scoring artefact
(older patients have more co‑occurring diagnoses, so "get all 5 labels right" is harder), partly real
difficulty. Followed up in NB05.

**Produces:** `outputs/05_optimized_InceptionTime1D_best.pth` + benchmark/ROC/confusion/age figures.

---

## 04 — `04_ResNet18_vs_Transformer_Baseline.ipynb`  ·  side experiment (not in the main chain)

**What it's for:** a clean, standalone comparison of two other architectures. Kept as a reference
baseline; the ensemble chain (02→03) does not depend on it.

**How it does it:**
1. Loads the `data/processed/` arrays (auto‑resolves whether run from the repo root or `notebooks/`).
2. Defines two models: **`ResNet1D‑18`** (a standard 1‑D residual CNN) and **`ResNet1D‑Transformer`**
   (a CNN stem followed by a multi‑head self‑attention Transformer encoder).
3. Trains both for 12 epochs with the same multi‑label loss and data.
4. Plots training curves, evaluates both on the held‑out `TEST` set with per‑class ROC curves, and runs
   the same **age‑group disparity audit** (`<40`, `40–65`, `65–80`, `80+`).

**Result:** both sit around macro‑AUC ~0.94 on the tasks they're scored on; the Transformer adds
attention but no decisive gain here.

---

## 05 — `05_Calibration_and_Uncertainty.ipynb`  ·  are the probabilities honest, and how sure is the model?

**What it's for:** a classifier can be accurate but **over‑confident**. This notebook measures that and
fixes what it can — the "uncertainty‑calibrated" part of the project title. It **reloads** NB02/NB03's
saved models and never retrains.

**Why the `CONFORMAL_CALIBRATION` split:** NB01 set aside ~1,100 patient‑separate records specifically
for this (it's written into `dataset_manifest.json`). Calibration must be fit on data the model has
**never seen for training *or* tuning** — `VALIDATION` was already used by NB03, so this reserved split
is the correct one, and no other notebook touches it.

**How it does it:**
1. **Rebuild the NB03 winning ensemble** and get its probabilities on the calibration and test sets
   (member AUCs match NB03 exactly — a sanity check that the reload is correct).
2. **Reliability diagrams + ECE / Brier** — bin the predictions by stated confidence and check whether,
   say, the "0.8 confidence" bin is right 80% of the time. **Result: macro‑ECE ≈ 0.082** — the raw model
   *is* meaningfully over/under‑confident.
3. **Temperature scaling** — fit **one number per class** on the calibration split that rescales the
   scores so probabilities match reality. It's monotonic, so **macro‑AUC is provably unchanged** (we
   assert this). **Result: ECE 0.082 → 0.075** — a modest improvement; the ensemble averaging had
   already done part of the job, and the rest is structural.
4. **Split‑conformal thresholds** — a distribution‑free method that turns scores into a per‑class
   "present / absent" decision with a **guaranteed ≥ 90% catch rate** for true positives.
   **Result:** realised catch rate 85–91% per class; the cost is a fairly high alert rate.
5. **Predictive uncertainty** — use the **disagreement between the 3 models** (and the prediction
   entropy) as an "how unsure am I" score. **Result: it flags the model's own errors with ROC‑AUC
   0.74–0.78** — uncertain cases really are more often wrong.
6. **Risk–coverage curve** — if the model **defers its least‑confident 30%** of cases to a cardiologist,
   accuracy on the rest rises from 55.6% → **64.4%** and macro‑AUC 0.9245 → **0.942**. This is the
   practical payoff of having an uncertainty score.
7. **Calibration by age** — **key finding:** the 80+ group is not just the least accurate, it's also the
   **worst‑calibrated** (ECE ~0.135, about 2.5× the under‑40 group), and a single global temperature
   **does not fix it** (the calibration data skews younger). The model *does* raise its uncertainty on
   older patients, so it "knows" it's struggling → next step is **age‑specific calibration**.

**Produces:** `outputs/dataset_validation/05cal_*.png` (5 figures).

---

## 06 — `06_Explainability.ipynb`  ·  what part of the ECG drove each prediction?

**What it's for:** the "explainable" part of the project title. Show **which lead** and **which moment
in the heartbeat** the model used — and, unlike most papers, **check that the explanation is real**,
not just plausible‑looking. It explains the **signal‑only `InceptionTime1D`** from NB03
(`outputs/05_optimized_InceptionTime1D_best.pth`) — chosen because it's the strongest single model and
takes only the raw signal, so a per‑lead story is clean.

**How it does it (each method has a source):**
1. **Attribution methods** (pure PyTorch, since `shap`/`captum` aren't installed):
   - **Integrated Gradients** — the principled stand‑in for SHAP: it adds up the model's gradient along
     a straight path from a "blank" ECG to the real one, and the pieces provably sum to the score
     difference (Sundararajan et al. 2017). This tells us **how much each point of each lead pushed the
     prediction**.
   - **Grad‑CAM‑1D** — highlights **which time‑windows** of the beat (P wave / QRS / ST / T) the model
     focused on (Selvaraju et al. 2017, adapted to 1‑D).
2. **Per‑lead importance** — average `|Integrated Gradients|` over time and over many test ECGs, per
   class, using **Atwa et al. 2025's exact formula** (`I_l = mean |attribution|`). Shown as a
   class × lead heatmap and compared to what a cardiologist would expect (e.g. MI → V2–V4 and II;
   HYP → V5–V6).
3. **Baseline check** — recompute the ranking with a different reference "blank" ECG and report the rank
   agreement; a ranking that survives this is more trustworthy.
4. **Attribution maps** — for one confidently‑correct ECG per class, paint the Integrated‑Gradients
   heat onto the actual 12‑lead waveform, with the Grad‑CAM time‑strip beneath (the style of
   Strodthoff et al. 2020, Fig. 8).
5. **Faithfulness test** (the check Atwa's paper skipped) — **delete the leads the method called
   important** (replace them with their average) and see how far the class AUC drops, versus deleting
   the same number of **random** leads. If the explanation is faithful, removing "important" leads must
   hurt more.
6. **Sanity check** (Adebayo et al. 2018) — recompute the attribution on a **randomly‑initialised**
   model. If the trained‑model and random‑model rankings match, the "explanation" is just reacting to
   signal shape, not to anything learned — so they should **not** match.

**Produces:** `outputs/dataset_validation/06xai_*.png` (per‑lead heatmap, 5 attribution maps,
faithfulness curves).

---

## 10 — `10_Final_Hypothesis_and_Rigor.ipynb`  ·  Mondrian conformal, age triangulation & statistical rigor ⭐

**What it's for:** closes out all 9 action items from professor feedback and delivers the central novel hypothesis test in Proposal Section J.

**How it does it:**
1. **Age-Conditional (Mondrian) Conformal Calibration:** Refits split-conformal non-conformity quantiles $q_k(g)$ separately per age band (`<40`, `40–65`, `65–80`, `80+`) on the reserved `CONFORMAL_CALIBRATION` split and evaluates mean set size $|S|$ and realized coverage on `TEST` with 1,000 bootstrap CIs.
2. **Age-Stratified Explanation Consistency:** Partitions elderly (`80+`) test cases by uncertainty and measures Integrated Gradients lead attribution rank stability via Spearman $\rho$.
3. **Random Forest Baseline Integration:** Trains MultiOutput Random Forest on 173 handcrafted HRV, QRS morphology, and Wavelet features.
4. **Bootstrap 95% CIs & Metrics:** Computes paired bootstrap CIs on Macro-AUC, Macro-AUPRC, Fmax, Macro-ECE, and exact match accuracy.
5. **Ablation Studies & Remediation Audit:** Evaluates HYP Gaussian noise augmentation ON vs OFF and benchmarks per-age-band temperature scaling for elderly patients.
6. **QA Documentation & Literature Gap Test:** Documents QA record exclusions and tests our 0.9245 against the published comparators. *Corrected 2026-10-02:* the 0.9290 figure used here previously is Strodthoff et al.'s **ALL**-task (71-label) result, not the five-class superdiagnostic task we run. The superdiagnostic comparators are 0.934 (six-model ensemble) and 0.918–0.930 (best single models). Our interval [0.9176, 0.9315] contains the single-model range but **not** 0.934.

---

## Scoreboard (all measured on the real `TEST` fold, N = 2198)

| Notebook | Model / step | Per‑label acc | macro‑AUC | macro‑ECE |
|---|---|---|---|---|
| 02 | Equal‑weight 3‑model ensemble | 87.3% | 0.9243 | — |
| **03** | **AUC‑weighted ensemble + tuned thresholds** ⭐ | **88.5%** | **0.9245** | — |
| 04 | ResNet1D‑18 / ResNet1D‑Transformer (baselines) | — | ~0.94 | — |
| 05 | NB03 ensemble **after temperature scaling** | 88.5% | 0.9245 *(unchanged)* | 0.082 → **0.075** |
| 09 | Multi-Output Random Forest Baseline (173 handcrafted features) | — | **0.8837** (Fmax 0.6733 · macro-AUPRC 0.7278) | — |
| **10** | **Mondrian Conformal + Bootstrap CIs** | **88.5%** | **0.9245 [0.916–0.933]** | **0.075 [0.068–0.082]** |

> **Corrected 2026-10-01.** The Random Forest row previously read 82.1% / 0.8250 [0.812–0.838] / 0.124.
> Those figures were never produced by an executed notebook. The values above are Notebook 09's stored
> output. See slide 56 of the deck ("Provenance and Corrections") and `papers/Draft_Paper.md` §4.4.

**Reference (superdiagnostic task, same multi‑label macro‑AUC framing):** Strodthoff et al. 2020
best‑single `resnet1d_wang` **0.930(06)** / 6‑model ensemble **0.934(05)**; Gitau et al. 2025
reproduce **0.918–0.929** across single models and **0.934** for the ensemble. The project is at
**parity with the best single models** and ~0.007 **behind** the ensembles; notebooks 05 and 06 add
the calibration and explainability that those papers leave open. Notebook 10 closes the Mondrian
conformal age-band hypothesis, and notebook 13 shows it survives reseeding.

> **Corrected 2026-10-02.** This block previously quoted 0.928/0.929 and claimed parity with SOTA.
> 0.929 is Strodthoff's **ALL**-task (71-label) number, not the superdiagnostic task this project
> runs. See `papers/Draft_Paper.md` §6 and slide 65 of the deck.

---

## Notes

- Runs on a plain CPU; the heavy notebooks (02, 03) take ~1–2 hours to train, the rest minutes. A free
  Colab / Kaggle T4 GPU is much faster.
- Every notebook auto‑resolves paths whether launched from the repo root or from inside `notebooks/`.
- Older, superseded notebooks (if any) live in `_deprecated/` and are not needed.

