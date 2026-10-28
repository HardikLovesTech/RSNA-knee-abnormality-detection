# RSNA-knee-abnormality-detection

# Knee MRI Abnormality Detection (RSNA) — Using Radiology Reports to Train an MRI-Only Model

> **Research / educational prototype. Not for diagnosis or treatment.**

This README explains the notebook `RSNA-1__EXPs___1_.ipynb` in **layers**. Read only as deep as you need:

| Layer | What you get | Time |
|---|---|---|
| **Layer 1** | The idea in plain words | 1 min |
| **Layer 2** | The pipeline in 6 steps | 3 min |
| **Layer 3** | Each notebook phase, one by one | 10 min |
| **Layer 4** | How to run it, outputs, results, limits | 5 min |

---

# Layer 1 — The Big Idea

**Goal:** Look at a knee MRI scan and predict three things:

| Finding | Meaning |
|---|---|
| **ACL** | Anterior cruciate ligament injury |
| **Medial Meniscus** | Medial meniscus tear |
| **Effusion** | Fluid in the knee joint |

**Problem:** Only **58 studies** in the dataset have all three *verified* labels. That is far too few to train a neural network well.

**Trick:** Thousands of studies come with a **written radiology report**. We read the reports with simple text rules, turn them into **"pseudo-labels"** (guessed labels), and use them as extra training data for the MRI model.

**Important:** Reports are used **only to create training labels**. The final model looks **only at the MRI** and never sees a report.

```
Radiology report ──(text rules)──► pseudo-labels ──┐
                                                    ├──► train MRI model ──► predict from MRI only
58 verified labels ─────────────────────────────────┘
```

---

# Layer 2 — The Pipeline in 6 Steps

1. **Load the data** — MRI scans (DICOM files), labels, and reports from the Kaggle RSNA dataset.
2. **Pick one MRI series per study** — prefer *sagittal*, *fluid-sensitive*, *fat-suppressed* images.
3. **Preprocess** — sort slices, normalise brightness, resize to 224×224, sample 8 (later 16) slices.
4. **Build the model** — a ResNet18 looks at each slice, then the slice features are combined (pooled) into one prediction per study.
5. **Train & compare experiments** — verified only vs. + pseudo-labels vs. + filtered pseudo-labels, etc.
6. **Evaluate & visualise** — ROC-AUC, cross-validation, MRI viewer, Grad-CAM-style explanations.

**Main metric: ROC-AUC** (0.5 = guessing, 1.0 = perfect). We report it per finding and as a **macro average** across the three.

---

# Layer 3 — Phase-by-Phase Walkthrough

The notebook is organised into phases. Each builds on the previous one.

## Phase 1–2: Baseline (cells 1–19)

**What it does**
- Sets seeds (`SEED=42`) so results are repeatable.
- Finds `train.csv` and `train_series.csv` under `/kaggle/input`.
- Plots how many positive/negative labels exist (EDA).
- Picks **one series per study** with a simple score:
  `sagittal = 4 points`, `fluid-sensitive = 2`, `fat-suppressed = 1`. Selection uses metadata only, never labels.
- Keeps only studies that have all 3 labels and a sagittal series.

**DICOM reading** (`sorted_dicoms`, `read_slice`)
- Sorts slices by their physical position in the scanner.
- Converts pixels to floats, clips the 1st–99th percentile, scales to 0–1.

**Model** (`KneeNet`)
- ResNet18 processes each slice on its own.
- Feature vectors from all slices are **averaged** (mean pooling).
- A final linear layer outputs 3 scores (one per finding).

**Training**
- Loss: `BCEWithLogitsLoss` (each finding is an independent yes/no).
- `pos_weight` compensates for rare positives.
- Study-level 80/20 split — all slices of a study stay together.

> First baseline result: macro AUC ≈ **0.59** (tiny dataset, random weights).

## Phase 3: Turn Reports into Pseudo-Labels (cell 20)

**What it does** — a **rule-based text reader** (not a clinical NLP system):

1. **Normalise text** — lowercase, strip accents.
2. **Split report into sentences** (`split_segments`).
3. **Look for keywords** per finding (`classify_segment`), in multiple languages (e.g. *derrame*, *épanchement* for effusion).
4. **Handle negation and uncertainty** — "no ACL tear" → negative; "possible tear" → unsure.
5. **Combine** sentence results into one label per report (`extract_label`).

**Golden rule:** anything the rules can't recognise stays **UNKNOWN**, *never* "negative".

**Checks and outputs**
- Compares extracted labels against the verified studies to measure accuracy.
- Saves `nlp_errors.csv` (mistakes) and `candidate_pseudo_labels.csv` (the new training labels).

## Phase 4: MRI-Only Weak Supervision (cells 21–22)

Three experiments, same model each time:

| ID | Training data |
|---|---|
| **E1** | Verified labels only |
| **E2** | Pseudo-labels only |
| **E3** | Verified + pseudo-labels |

New ideas here:
- **Masked loss** (`masked_loss`): if a label is UNKNOWN, it is simply ignored — no penalty, no guess.
- Verified validation studies are **held out** and never used for training.
- Model `KneeResNet` uses a **1-channel** first layer (MRI is grayscale).

## Phase 5: Transfer Learning (cells 24–27)

Instead of random starting weights, use **ImageNet-pretrained ResNet18**.

| ID | Setup |
|---|---|
| **E4** | Pretrained, backbone **frozen** (only the final layer learns), LR 1e-3 |
| **E5** | Pretrained, **fully fine-tuned**, LR 1e-4 |

## Phase 6: Are the Pseudo-Labels Any Good? (cells 28–32)

- **6A — Audit:** distribution of pseudo-labels, compared with verified ones.
- **6B — Error review:** read every wrong NLP prediction; flag vague words like *mild*, *trace*, *possible*.
- **6C — Conservative filtering:** turn ambiguous positives (e.g. "interstitial ACL tear", "mild effusion") into **UNKNOWN**.
- **6D — Experiments:**
  **E6** verified only · **E7** + original pseudo-labels · **E8** + filtered pseudo-labels.
- **6E — Speed-up:** decode each MRI **once**, keep it in RAM as float16, use mixed precision on GPU (`MRI_CACHE`, `CachedKneeDataset`, `FastKneeNet`).

## Phase 7: 3-Fold Cross-Validation (cell 33)

One small validation split is noisy. So the 58 verified studies are split into **3 folds**; each fold takes a turn as the test set. Results are reported as mean ± std.

## Phase 8: Inference & Interactive Viewer (cell 34)

- Loads the saved `E8_filtered_best.pt` model.
- Reproduces validation AUC.
- An `ipywidgets` viewer lets you **scroll through MRI slices** and see the three scores.
- Saves a demo image.

## Phase 9: Careful, Reproducible Cross-Validation (cells 35–36)

A stricter version of Phase 7:

- **3 repeats × 3 folds = 9 test runs** per setup.
- Fold assignments are saved and hashed so they can be reproduced.
- **Leakage checks** (asserts) make sure no test study appears in training.
- **Final-epoch only** — no picking the best epoch on test data.
- Adds a **step-matched control**: verified-only data trained for the *same number of optimizer steps* as the pseudo-label runs, so that more training steps don't unfairly explain a gain.
- Bootstrap confidence intervals and paired comparisons.

Run with: `P9 = p9_main()`

## Phase 10: The Improvement Ladder (cells 37–38)

Improvements are added **one at a time** so each one's effect is visible:

| Step | Change |
|---|---|
| `C8_mean` | Use the **central 60%** of slices (8 slices), mean pooling |
| `C8_attn` | Replace mean pooling with **attention pooling** |
| `C16_attn` | Use 16 slices |
| `C16_pre` | **ImageNet-pretrained** start |
| `C16_pre_aug` | Add **GPU data augmentation** |
| `ALL` | Use **all 1,609** filtered pseudo-labelled studies |
| `ALL_ens` | **3-seed ensemble** |
| `ALL_soft` | Soft targets, down-weighted pseudo-labels |
| `ALL_NS` | **Noisy student**: a teacher ensemble re-labels, keep labels where the teacher agrees |
| `ALL_V` | Control: same architecture, verified only, step-matched |

Run with: `P10 = p10_main()` (set `P10_SMOKE=True` first for a ~5-minute test run). It is resumable; a full run takes roughly 4–5 hours on a T4 GPU.

## Phase 11: Visualisation & Explainability (cells 39–40)

Uses saved results only — **no retraining**.

```python
p11_all()                              # dashboards + error analysis
p11_dashboard()                        # ROC, training curves, per-run plots
p11_show_study(uid, "ALL", r=0)        # slices + attention + scores vs truth
p11_show_study(uid, "ALL", r=0, label="ACL")   # adds Grad-CAM heatmap
p11_errors("ALL_ens")                  # best / worst studies, reliability plot
p11_browse("ALL")                      # interactive browser
p11_predict_dicom(study_uid)           # predict from raw DICOM
```

> A score is an honest prediction **only** for a study that was *held out* for that repeat/fold. The helpers print this status each time.

Cells 41–42 zip the figures for download.

---

# Layer 4 — Running It, Results, and Limits

## Setup

1. Open the notebook on **Kaggle**.
2. **Add Input** → attach the *RSNA Knee Abnormality Detection* competition data.
3. Turn on a **GPU** (T4 is enough).
4. Run cells **in order** — later phases reuse variables from earlier ones (`verified_val`, `MRI_CACHE`, `FastKneeNet`, `datasets`, …). If a variable is missing, the cell tells you which earlier one to run.

**Libraries:** `torch`, `torchvision`, `pydicom` (+ `pylibjpeg` for compressed DICOMs), `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `scipy`, `ipywidgets`.

## Key settings

| Setting | Value | Where |
|---|---|---|
| Slices per study | 8 (16 in Phase 10) | `SLICES`, `P10_CACHE_SLICES` |
| Image size | 224 × 224 | `IMAGE_SIZE` / `SIZE` |
| Epochs | 6–10 | per phase |
| Learning rate | 1e-3 (frozen) / 1e-4 (others) | per phase |
| Seed | 42 | `SEED` |

## Where outputs are saved (`/kaggle/working/`)

| Folder | Contents |
|---|---|
| `knee_project/` | Phase 1 plots, splits, `best.pt`, metrics |
| `knee_report_nlp/` | Pseudo-labels, NLP errors, label audit |
| `knee_phase4/` | Phases 4–8 models, filtered labels, CV results |
| `knee_phase9_cv/` | Reproducible CV predictions, fold files |
| `knee_phase10_*/` | Improvement-ladder predictions and checkpoints |
| `knee_phase11_viz/` | Final figures (PDF + PNG) |

## Results at a glance (Phase 9–10, mean macro AUC over 9 held-out runs)

| Setup | Macro AUC |
|---|---|
| Verified only | 0.567 |
| + original pseudo-labels | 0.615 |
| + filtered pseudo-labels | 0.652 |
| + central slices | 0.674 |
| + attention pooling | 0.691 |
| + ImageNet init | 0.740 |
| + augmentation | 0.760 |
| + all 1,609 pseudo-labels (`ALL`) | 0.806 |
| `ALL`, 3-seed ensemble | **0.834** |

**Takeaway:** each step helps, with the biggest gains from **pretrained weights** and **more pseudo-labelled data**. (Numbers come from the saved notebook output; re-running may differ slightly.)

## Honest limitations

- **Tiny verified set (58 studies)** → results have wide error bars (see the ± values).
- **Pseudo-labels are noisy** — built by simple rules, not a clinical NLP system.
- **Only sagittal series** are used; some findings may be missed.
- **Study-level labels** are weak supervision for individual slices.
- **No separate untouched test set** in the early phases; the same validation data was used for model selection and reporting. Phases 9–10 reduce this by using final-epoch evaluation.
- **Scores are not calibrated probabilities.**
- Patient/site IDs are unavailable, so patient- or site-independent evaluation cannot be guaranteed.

## Suggested next steps

- Hold out a truly untouched test set.
- Audit meniscus pseudo-labels (not filtered yet).
- Try fine-tuning stronger backbones and multi-plane input (axial, coronal).
- Validate on data from different hospitals.

---

*Reminder: for research and learning only — not a medical device.*
