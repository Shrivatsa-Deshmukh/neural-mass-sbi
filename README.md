# Neural Mass SBI

A general simulation-based inference (SBI) framework for neural mass models, with the Jansen-Rit model as the reference implementation.

> **Note:** Full results and analysis code will be released upon publication.

---

## Overview

Fitting mechanistic brain models to EEG recordings is a long-standing challenge in computational neuroscience. This project frames it as an amortized Bayesian inference problem: rather than fitting a model to a single observation, a neural density estimator is trained across the full parameter space so that posterior inference at test time is near-instantaneous.

![Pipeline](assets/pipeline_overview.svg)

The pipeline has four stages:

1. **Simulate** — Generate signals from a neural mass model across a broad parameter space using Sobol-sampled priors.
2. **Project** — Apply a biophysically realistic EEG forward model (MNE fsaverage lead field) to map source activity to sensor space.
3. **Corrupt** — Add pink (1/f) observation noise at variable SNR to train a noise-robust posterior.
4. **Infer** — Extract 7 interpretable summary statistics and train a Neural Spline Flow via SNPE to map them to posterior distributions over model parameters.

The framework is designed to be model-agnostic. The Jansen-Rit model ships as the reference implementation — swapping in a different neural mass model requires only implementing a single function.

---

## Results (Jansen-Rit)

Evaluated on 150 held-out test observations with 1,000 posterior samples each. Training used 131,072 simulations with pink noise at variable SNR (5–25 dB).

| Parameter | R² | Pearson r | NRMSE | Cov50 | Cov90 | Cov95 |
|-----------|------|-----------|-------|-------|-------|-------|
| log C | 0.793 | 0.891 | 0.127 | 0.51 | 0.93 | 0.96 |
| log μ | 0.744 | 0.868 | 0.150 | 0.47 | 0.91 | 0.96 |
| log κ | 0.711 | 0.844 | 0.158 | 0.48 | 0.85 | 0.91 |
| log g | 0.925 | 0.962 | 0.081 | 0.55 | 0.91 | 0.94 |
| **Mean** | **0.793** | **0.891** | **0.129** | **0.50** | **0.90** | **0.94** |

90% credible intervals achieve near-nominal coverage across all parameters (0.85–0.93), indicating a well-calibrated posterior. The inhibitory gain (log g) is best-recovered; the time constant (log κ) is most challenging, consistent with its narrower prior range.

---

## Repository Structure

```
neural-mass-sbi/
├── pipeline.py              # General SBI framework (model-agnostic)
├── embedding.py             # Statistical feature embedding network
├── forward_model.py         # MNE-based EEG lead field forward model
├── run_jansen_rit.py        # CLI entry point for the JR reference model
├── simulators/
│   ├── __init__.py
│   └── jansen_rit.py        # Jansen-Rit ODEs + JansenRitConfig
├── assets/
│   └── pipeline_overview.svg
└── README.md
```

---

## Installation

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, MNE-Python ≥ 1.6, sbi ≥ 0.22, SciPy, NumPy, Matplotlib

```bash
git clone https://github.com/your-username/neural-mass-sbi.git
cd neural-mass-sbi

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install mne sbi scipy numpy matplotlib
```

The first run will automatically download the MNE `fsaverage` template (~200 MB) and cache the forward solution to `~/.mne/lead_field_cache/`.

---

## Usage

### Run the Jansen-Rit reference model

```bash
python run_jansen_rit.py                              # 4 params, fixed σ
python run_jansen_rit.py --infer_sigma                # 5 params, infer σ jointly
python run_jansen_rit.py --snr_min 10 --snr_max 30   # custom SNR range
python run_jansen_rit.py --eval_snr 15               # fixed SNR at evaluation
python run_jansen_rit.py --no_noise                  # disable observation noise
python run_jansen_rit.py --use_gating                # learned feature gating
python run_jansen_rit.py --retrain_ablation          # feature importance analysis
```

### Use your own neural mass model

Implement a simulator function and a config, then pass them to `train_and_evaluate`:

```python
from pipeline import BaseConfig, train_and_evaluate
import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class MyConfig(BaseConfig):
    PRIOR_MIN:   List[float] = field(default_factory=lambda: [0.0, 0.0])
    PRIOR_MAX:   List[float] = field(default_factory=lambda: [1.0, 1.0])
    PARAM_NAMES: List[str]   = field(default_factory=lambda: ["alpha", "beta"])

def my_simulator(theta_batch: torch.Tensor, config: MyConfig) -> torch.Tensor:
    """
    Args:
        theta_batch : [batch_size, n_params]
        config      : MyConfig
    Returns:
        source_signal : [batch_size, signal_length]  at config.FS_OUT Hz
    """
    # ... your model here ...
    return source_signal

results = train_and_evaluate(config=MyConfig(), simulator=my_simulator)
```

The pipeline automatically handles the forward model, noise, embedding, and SNPE training. See `simulators/jansen_rit.py` for a complete example.

---

## Methods

### Key design choices

**Pluggable simulator.** The pipeline is decoupled from any specific neural mass model. Any model that maps parameters to a time series at `config.FS_OUT` Hz can be dropped in without touching the inference machinery.

**Fixed σ.** The Jansen-Rit input noise std. is fixed at its geometric mean (4.5 pps) by default, because σ is non-identifiable under observation noise. Pass `--infer_sigma` to infer it jointly.

**Variable SNR training.** Each simulation is corrupted with pink noise at a randomly drawn SNR (5–25 dB), teaching the posterior to marginalise over noise conditions and generalise to real data.

**Sobol sampling.** Parameters are drawn from a Sobol low-discrepancy sequence for more uniform prior coverage than random uniform sampling.

**Interpretable embedding.** Seven handcrafted statistics replace a learned CNN encoder, making the embedding transparent and supporting principled ablation studies.

### Architecture summary

| Component | Details |
|-----------|---------|
| Simulator | Pluggable — JR reference: Euler @ 1 kHz, downsampled to 250 Hz |
| Forward model | MNE fsaverage BEM, surface-normal lead field, auto-selected electrode |
| Noise | Coloured 1/fᵅ, SNR ~ Uniform(5, 25) dB per sample |
| Embedding | 7 statistical features, z-scored, optional sigmoid gating |
| Density estimator | Neural Spline Flow, 5 transforms, 125 hidden units |
| Training | SNPE-C, Adam lr=5×10⁻⁴, early stopping, 131,072 simulations |

<!-- 
---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2025,
  title   = {Your paper title},
  author  = {Your Name and Co-authors},
  journal = {Journal Name},
  year    = {2025},
  note    = {Under review}
}
```
-->


---


## Acknowledgements

- [MNE-Python](https://mne.tools) for EEG forward modelling
- [sbi](https://github.com/sbi-dev/sbi) for the SNPE implementation
- Jansen & Rit (1995), *Biological Cybernetics* for the reference model
