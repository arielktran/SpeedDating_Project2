# Speed Dating with Autoencoder + Mixed-Effects Modeling Approaches

A two-stage modeling framework that uses an autoencoder to compress high-dimensional speed-dating survey data into latent features, then feeds those features into mixed-effects models to predict mutual matches while accounting for repeated observations per participant.

**Course:** CSS 206 / COGS 209 — Statistical Machine Learning for Social Scientists  
**Authors:** Dave Melkani, Nissa Carter, Oishani Bandopadhyay, Ariel Tran  
**Data:** [Fisman et al. (2006)](https://doi.org/10.1162/qjec.2006.121.2.673) — Columbia University speed dating experiment

---

## Overview

Speed dating events generate a naturally hierarchical dataset: each participant attends one wave, meets multiple partners in sequence, and records ratings before and after each interaction. This structure violates standard regression's independence assumption. The pipeline here handles that in two steps:

1. **Autoencoder** — compresses 63 preprocessed features into 8 latent dimensions (Z1–Z8), learning compact representations of preferences, partner evaluations, self-ratings, and demographic characteristics.
2. **Mixed-Effects Models** — uses Z1–Z8 as predictors with participant-level random intercepts to model match probability, including gender × latent feature interactions.

---

## Repository Contents

| File | Description |
|---|---|
| `main_modeling_code.ipynb` | Full pipeline: EDA → preprocessing → autoencoder → mixed-effects models → evaluation |
| `df_long.csv` | Long-format dataset (37,890 rows × 75 columns) after wide-to-long transform |
| `clean_df.csv` | Cleaned intermediate dataset |
| `full_data_speed_dating.csv` / `speed-dating.csv` | Raw source data |
| `best_model.pth` | Saved autoencoder weights (best validation checkpoint) |
| `loss_history.npy` | Training and validation MSE per epoch |
| `recon_errors_test.csv` | Per-observation reconstruction error on the test set |
| `per_missing.csv` | Column-level missingness summary used during EDA |
| `figure_autoencoder_loss.png` | Training vs. validation loss curve |
| `figure_latent_heatmap.png` | Correlation heatmap: latent dimensions vs. original features |
| `figure_roc_comparison.png` | Overlaid ROC curves for both models |
| `(KEY_and_SURVEY)_Speed_Dating_Data_Key.pdf` | Variable codebook |
| `(ORIGINAL_ACADEMIC_PAPER)_Fisman2006_...pdf` | Source paper |

---

## Data

- **37,890 observations**, 75 variables in long format
- **362 unique participants**, across **20 speed-dating waves**
- Each participant contributes 10–231 observations (one per partner interaction × time point)
- Participants are **nested within waves** (each attends exactly one), justifying random intercepts rather than crossed random effects
- Outcome: `match` — binary, ~17.2% positive (class imbalance noted)

**Key preprocessing decisions:**
- Dropped: structural IDs, decision columns (`dec`, `dec_o` — would perfectly predict `match`), high-cardinality strings (`field`, `career`, `from`, `zipcode`), and `shar`/`shar_o` (~37% missing)
- One-hot encoded: `pre_post`, `rating_ref_group`
- Median imputation (training set only) → StandardScaler (training set only) → applied to val/test

Train/val/test split was done **by participant** (not by row) to prevent leakage: 71.2% / 9.0% / 19.8%.

---

## Stage 1 — Autoencoder

```
Encoder: 63 → 48 → 24 → 8 (bottleneck)
Decoder: 8 → 24 → 48 → 63
```

- Activation: ReLU; Regularization: BatchNorm + Dropout (p=0.2)
- Optimizer: Adam (lr=0.001), batch size 256, up to 500 epochs
- Early stopping: patience=15 on validation MSE → stopped at epoch 67
- **Test MSE: 0.703** (val best: 0.725)

Reconstruction error was nearly identical for matched vs. unmatched observations (0.707 vs. 0.702), confirming the autoencoder learned general participant structure rather than overfitting to the outcome.

---

## Stage 2 — Mixed-Effects Models

Both models used the formula:

```
match ~ Z1 + Z2 + ... + Z8 + gender + gender:Z1 + ... + gender:Z8
```

with a random intercept grouped by participant (`iid`). Participant-level random intercept variance was estimated at **0.124**, confirming meaningful between-subject heterogeneity in baseline matchability.

**MELPM** (`statsmodels MixedLM`): linear probability model, predictions clipped to [0,1]. AIC/BIC returned NaN due to out-of-bounds fitted values — a known limitation of LPMs on binary outcomes.

**MELogR** (`BinomialBayesMixedGLM`, MAP estimation): logistic model, valid probabilities throughout, stable likelihood.

---

## Results

| Metric | MELPM | MELogR |
|---|---|---|
| AUC-ROC | 0.681 | **0.703** |
| Accuracy (threshold=0.5) | 79.5% | **80.8%** |
| Brier Score | 0.158 | 0.159 |
| Log Loss | 1.650 | **0.595** |

MELogR outperforms on all metrics, most notably log loss — reflecting better probability calibration from the logistic link function.

**Most predictive latent dimensions (MELogR):**

| Dimension | Top Correlates | Direction |
|---|---|---|
| Z2 | music, fun_o, like_o, prob | Strongly positive |
| Z4 | museums, theater, art vs. sports | Strongly negative |
| Z5 | date frequency, go_out, age | Negative |
| Z3 | fun_o, attr_o, like_o vs. gaming | Positive (logit) |

Gender was a significant predictor (β = −0.220, p < 0.001 in MELPM), and several gender × latent interactions reached significance, particularly for Z1, Z2, Z3, Z4, and Z5.

---

## Requirements

```
pandas, numpy, torch, scikit-learn, statsmodels, matplotlib, seaborn, tqdm, patsy
```

---

## Reference

Fisman, R., Iyengar, S. S., Kamenica, E., & Simonson, I. (2006). Gender differences in mate selection: Evidence from a speed dating experiment. *The Quarterly Journal of Economics, 121*(2), 673–697.
