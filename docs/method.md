# Method and design decisions

## 1. Why this problem

Raman spectroscopy measures the inelastic scattering of laser light by molecules.
Each molecule gives a spectrum of peaks (its vibrational fingerprint), but the
signal is weak, so **SERS** (surface-enhanced Raman scattering) places the
sample on a metal nanostructure that amplifies it by several orders of
magnitude. SERS is very sensitive but notoriously **hard to reproduce**: the
intensity depends on the substrate, the laser, the instrument and the operator.
Turning a spectrum into a concentration (quantification) is therefore a good
test of whether a model has learned chemistry or just the habits of one
laboratory. It also relates to the computational side of Raman imaging:
pre-processing, noise, and learning from spectra.

## 2. Data

Fornasaro et al., *Anal. Chem.* 2020 (Zenodo record 3572359, open access): an
interlaboratory study of adenine detected by SERS. The file used has 3,516
spectra from 15 laboratories, 4 substrates (silver and gold, colloidal and
solid), 2 lasers (532 and 785 nm), 15 samples with adenine concentrations from 0
to 9 (units as in the dataset), measured between 400 and 1999 cm⁻¹ with a step
of 3 cm⁻¹.

Two laboratories (P07, P15) measured a narrower range. Instead of inventing
missing values, the code keeps the **common range** where all 3,516 spectra are
complete (355 points).

## 3. Pre-processing

Per spectrum, in this order, each step optional:

1. **Baseline removal** with asymmetric least squares (AsLS; Eilers and Boelens,
   2005): fit a smooth curve that stays below the peaks (parameters `λ = 10⁵`,
   `p = 0.01`) and subtract it. Fluorescence usually produces such a slowly
   varying background.
2. **Normalisation**: SNV (zero mean, unit variance), area (unit L1 norm), or none.

The choice was made by **ablation** on the leave-labs-out protocol (see the
README), not by habit. Baseline removal turned out to hurt on this dataset.
Plausible reason: in SERS the broad background carries information about the
substrate and the amount of material, and the AsLS parameters are not tuned per
instrument. This is reported rather than hidden.

## 4. Models

- **PLS regression** (partial least squares): the standard chemometrics model. It
  projects the spectrum onto a few latent variables that explain the
  concentration. The number of components is chosen by 5-fold cross-validation
  inside the training set only.
- **Ridge regression** on standardised intensities: a linear baseline with L2
  regularisation, which handles the strong correlation between neighbouring
  wavenumbers.
- **1D CNN**: three blocks of convolution (kernels 9, 7, 5), batch normalisation,
  ReLU and max-pooling, then a small dense head. Augmentation: random gain
  (±5 %) and additive noise during training. AdamW, one-cycle learning rate,
  Smooth-L1 loss on the standardised target.

## 5. Evaluation protocols (the key decision)

- **A, official split**: the train and test labels shipped with the dataset
  (blank samples added to the training set). The same laboratories appear in both
  sets, with different samples.
- **B, leave labs out**: 5 folds grouped by laboratory. A model is trained on
  about 12 laboratories and tested on the 3 remaining ones, whose instruments
  and operators it has never seen. This answers the question that matters in
  practice: *does it work in a new lab?*

The gap between A and B measures the **domain shift** between laboratories.

Metrics: RMSE and MAE (in concentration units) and R². R² is negative when the
model is worse than predicting the mean concentration.

## 6. What is not done

- No calibration-transfer step (standardisation between instruments), which is
  the usual remedy for the gap between A and B.
- No hyperparameter search for the CNN beyond a single configuration, and a single
  seed: differences of a few percent between models are not significant.
- Concentration is predicted for all laboratories pooled; per-substrate models
  would likely do better.
