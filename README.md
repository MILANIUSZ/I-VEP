# I‑VEP: Imperceptible Visual Evoked Potentials

**Authors:** Milán András Fodor, Ivan Volosyak  
**Paper:** “Towards Visual‑Fatigue‑Free BCI with Imperceptible Visual Evoked Potentials (I‑VEP)” (SMC 2025)

This repository contains all code, data‑processing scripts, and instructions to reproduce the two main pipelines in our I‑VEP study:

1. **Nested CV + Optuna + CNN**: `I-cVEP_ResNet-CNN_NCV.py`  
  Runs a 4‑fold outer nested cross‑validation over all CSVs, using Optuna to tune:
  - filter band centers & bandwidths
  - window length & label shift
  - CNN architecture (kernel size, filters, dropout, etc.)
  - learning rate, weight decay, batch size, early stopping  
  Outputs per‑fold `.h5` models, `.json` hyperparameters, and a summary `results.csv`.

2. **Leave‑One‑Trial‑Out SVC baseline**: `I-cVEP_SVC_decodability_check.py`
  Implements a subject‑agnostic SVM (offline) baseline with causal Hilbert‑based features:
  - Trial segmentation based on m-sequnce labels
  - Band‑power (env + harmonics), low‑freq, and flip‑potential features
  - Leave‑One‑Trial‑Out cross‑validation  
  Prints per‑file LOTO and overall bit‑accuracy, with an optional verbose mode.


If you use this code, please cite:
Fodor, M. A., & Volosyak, I. (2025).
Towards Visual‑Fatigue‑Free BCI with Imperceptible Visual Evoked Potentials (I‑VEP). In Proceedings of IEEE SMC 2025.
DOI: tba
