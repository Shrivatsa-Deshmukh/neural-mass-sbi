# neural-mass-sbi

A simulation-based inference (SBI) framework for Bayesian parameter estimation of neural mass models from EEG recordings, with the Jansen-Rit model as the reference implementation.

> **Code will be released upon publication.**

---

## Key finding

Feature gating is often treated as a cheap, free-by-product measure of interpretability; a mechanism trained jointly with a model should, in principle, tell you which inputs it actually relies on. We tested that assumption directly: does a learned feature gate recover the same feature importance as a gold-standard, retrain-based ablation?

**No.** Across 10 independent training seeds, gate-based and ablation-verified importance are **negatively correlated** (Spearman ρ ≈ −0.55 for sigmoid gating, −0.45 for softmax gating), despite the two gate variants agreeing closely with each other (ρ ≈ 0.83) and despite gating costing almost nothing in raw predictive accuracy. The single feature ablation ranks *most* important is consistently the one the gate down-weights *most*.

---

## Overview

Mechanistic neural mass models offer interpretable, physiologically grounded accounts of EEG dynamics, but fitting them to empirical data is non-trivial: the likelihood is intractable, forward simulations are expensive, and classical optimisation methods scale poorly to multi-parameter spaces. This project addresses all three by framing parameter estimation as amortised Bayesian inference.

Rather than optimising a likelihood for each new observation, a neural density estimator is trained once across the full parameter space using simulated data. At test time, the full posterior distribution over all model parameters is obtained instantaneously for any new EEG recording; without re-running the simulator.

The pipeline below shows the full data flow, from prior sampling through to posterior estimation. Each stage is modular: the forward model, noise model, and density estimator are independent of the choice of neural mass model, so the framework can be applied to any model that produces a time-series output.


---

## The Jansen-Rit Model

The Jansen-Rit model describes the mean membrane potential dynamics of a cortical column through three interacting neural populations; pyramidal cells, excitatory interneurons, and inhibitory interneurons - each modelled as a second-order linear system driven by a sigmoid nonlinearity. It is one of the canonical generative models for EEG oscillations and provides a well-understood testbed for parameter inference methods.

The model is governed by four parameters:

| Parameter | Symbol | Prior range | Physiological interpretation |
|-----------|--------|-------------|------------------------------|
| Connectivity | C | 135 – 270 | Scales the synaptic coupling strengths between all three populations (C₁ = C, C₂ = 0.8C, C₃ = C₄ = 0.25C) |
| Mean input | μ | 120 – 350 pps | Mean firing rate of afferent input to the pyramidal population |
| Time constant | κ | 0.75 – 1.25 | Multiplicative scale on the excitatory and inhibitory synaptic rate constants, holding their 2:1 ratio fixed |
| Inhibitory gain | g | 0.5 – 2.0 | Scales the inhibitory synaptic amplitude; primary determinant of band power |

Excitatory gain is held fixed rather than inferred, so that a decrease in excitatory gain and an increase in inhibitory gain; which shape the signal in near-mirror-image fashion cannot be confused for one another during inference.

All parameters are inferred in log-space. The model is integrated at 1 kHz (Euler method, Δt = 1 ms). The observed signal is the post-synaptic potential difference y₁ − y₂, which corresponds to the EEG-proximal output of the pyramidal population. A 2 s transient is discarded before retaining 3 s of output, which is then downsampled to 250 Hz using a zero-phase anti-aliasing filter.

---

## EEG Forward Model

To move from cortical source activity to scalp EEG, a lead field is computed using MNE-Python's fsaverage template head model. The source is placed at primary visual cortex, and the forward solution uses the pre-computed fsaverage three-layer boundary element model with an ico4 cortical source space and fixed surface-normal dipole orientation. Electrode positions follow the standard 10–20 montage.

Rather than collapsing to a single electrode, the 5 electrodes with the highest absolute sensitivity to the source are auto-selected, and the source signal is projected to each using its signed lead-field gain:

&nbsp;&nbsp;&nbsp;&nbsp;EEG_c(t) = L_c · source(t)&nbsp;&nbsp;&nbsp;&nbsp;for each selected electrode c

Using the signed gain (rather than its absolute value) preserves the real scalp topography, including sign inversion across electrode pairs, instead of collapsing every channel to the same polarity. The number of channels and the electrode-selection behaviour are both configurable; a single-electrode, absolute-gain mode is also available for simpler setups but multi-channel signed projection is the default. The forward solution is computed once per source location and cached locally.

---

## Observation Noise

Real EEG recordings contain structured background activity characterised by a 1/f power spectrum. Each simulated EEG epoch is therefore corrupted by pink noise (spectral exponent α = 1) scaled to a randomly drawn signal-to-noise ratio:

&nbsp;&nbsp;&nbsp;&nbsp;x(t) = EEG(t) + ε(t),&nbsp;&nbsp;&nbsp;&nbsp;ε ~ 1/f,&nbsp;&nbsp;&nbsp;&nbsp;SNR ~ Uniform(5, 25) dB

Noise is generated in the frequency domain: white noise is transformed via rfft, shaped by a 1/f amplitude filter, then transformed back via irfft and normalised to unit variance before scaling to the target SNR. Drawing a different SNR per simulation rather than using a fixed value trains the posterior to marginalise over noise conditions, producing an estimator that is robust to unknown noise levels at test time.

---

## Summary Statistics

Each 3 s epoch (750 samples at 250 Hz), per electrode, is compressed into seven interpretable summary statistics before being passed to the density estimator:

| Feature | Definition |
|---------|-----------|
| Skewness | Third standardised moment: asymmetry of the amplitude distribution |
| Kurtosis | Excess kurtosis (fourth moment − 3): tail heaviness relative to Gaussian |
| Spectral slope | OLS regression slope of log PSD on log frequency, 1–100 Hz |
| Total log power | Log of summed Welch PSD across all frequency bins (DC to Nyquist) |
| Dominant frequency | Frequency of peak Welch PSD in 1–100 Hz band, normalised by Nyquist (125 Hz) |
| Hjorth mobility | √(Var(x′) / Var(x)): proxy for mean signal frequency |
| Hjorth complexity | Mobility(x′) / Mobility(x): proxy for spectral bandwidth |

All features are z-scored using mean and standard deviation computed from the training set before being passed to the density estimator.

---

## Feature Gating & Interpretability

An optional learned gating mechanism can re-weight the 7 features during training, pooled across electrodes so that gating operates at the same granularity as the ablation procedure below. Two gate activations are supported:

- **Sigmoid** - each feature's weight is independent; features don't compete.
- **Softmax** - weights are forced to sum to 1 across the 7 features; features compete directly.

To test whether the learned gate is a trustworthy importance signal, this project also implements a gold-standard, **retrain-based ablation**: for each feature, the full pipeline is retrained from scratch with that feature excluded, and the resulting drop in posterior accuracy is measured directly. Across repeated training seeds, the two measures of importance are compared using per-seed Spearman rank correlation - see [Key finding](#key-finding) above for the result.

---

## Density Estimator

The posterior p(θ | x) is learned using Sequential Neural Posterior Estimation (SNPE-C), implemented via the [sbi](https://github.com/sbi-dev/sbi) library. The density estimator is a Neural Spline Flow (NSF) conditioned on the summary statistics:

| Component | Setting |
|-----------|---------|
| Architecture | Neural Spline Flow |
| Coupling transforms | 5 |
| Hidden units | 125 |
| Training simulations | 32,768 (2¹⁵) |
| Prior sampling | Sobol quasi-random sequence |
| Optimiser | Adam, lr = 5 × 10⁻⁴ |
| Batch size | 512 |
| Stopping criterion | 15 epochs without validation improvement (max 250) |
| Validation fraction | 15% |

Parameters are drawn from a Sobol quasi-random sequence rather than uniform random sampling, providing more uniform prior coverage for the same number of simulations; particularly important in the corners of the parameter space. The 32,768-simulation default was chosen from a compute-efficiency sweep across four budgets (8,192 / 16,384 / 32,768 / 65,536): accuracy keeps improving beyond this point, but training time grows faster than the accuracy gain.

---

## Results

The trained posterior is evaluated on 250 held-out test observations per training seed, averaged across 10 independently trained seeds. For each test point, 1,000 posterior samples are drawn and the posterior mean is taken as the point estimate. Metrics reported are: coefficient of determination (R²), Pearson correlation (r), normalised RMSE (NRMSE, normalised by prior range), and empirical coverage of 50%, 90%, and 95% credible intervals.

**Per-parameter recovery (ungated baseline):**

| Parameter | R² | Pearson r | NRMSE | Cov50 | Cov90 | Cov95 |
|-----------|------|-----------|-------|-------|-------|-------|
| C — connectivity | 0.780 | 0.884 | 0.135 | 0.457 | 0.883 | 0.940 |
| μ — mean input | 0.751 | 0.867 | 0.152 | 0.461 | 0.882 | 0.935 |
| κ — time constant | 0.720 | 0.850 | 0.157 | 0.432 | 0.826 | 0.904 |
| g — inhibitory gain | 0.921 | 0.962 | 0.086 | 0.499 | 0.891 | 0.950 |
| **Mean** | **0.793** | **0.891** | **0.132** | **0.462** | **0.871** | **0.932** |


**Gating (mean R² across 10 seeds):**

| Configuration | R² | NRMSE | Pearson r |
|---|---|---|---|
| No gating (default) | 0.793 (± 0.010) | 0.132 (± 0.003) | 0.891 (± 0.006) |
| Sigmoid gating | 0.788 (± 0.019) | 0.134 (± 0.007) | 0.887 (± 0.011) |
| Softmax gating | 0.778 (± 0.017) | 0.138 (± 0.006) | 0.882 (± 0.009) |

---

## Acknowledgements

- [MNE-Python](https://mne.tools) — EEG forward modelling and fsaverage template
- [sbi](https://github.com/sbi-dev/sbi) — SNPE implementation
