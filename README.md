# cmv_torch
Incomplete multiview clustering via Wyner Common Information

Accepted at 2025 IEEE Information Theory Workshop, Sydney, Australia

## Citation
```
@INPROCEEDINGS{11240493,
  author={Odeh, AbdAlRahman and Huang, Teng-Hui and El Gamal, Hesham},
  booktitle={2025 IEEE Information Theory Workshop (ITW)}, 
  title={Incomplete Multiview Learning via Wyner Common Information}, 
  year={2025},
  pages={1-6},
  doi={10.1109/ITW62417.2025.11240493}}
```

## Overview

`cmv_torch` is a PyTorch implementation of the incomplete multiview
clustering algorithm described in the above paper.  The repository
provides data loaders, model definitions, loss functions and
training/evaluation scripts for experimenting with various datasets of
multiview data when some views are missing.

The primary entry point is `main_incomplete.py`, which trains the
Wyner‑common‑information based model on a chosen dataset.  Several
auxiliary modules support preprocessing (`cvcl_dataprocessing.py`),
additional models (`networks.py`), evaluation (`evaluate.py`), and
utility routines (`utils.py`).

> ⚠️ This codebase is research‑oriented and may not include extensive
> production‑grade checks.  It is provided to reproduce the results in
the associated paper and to serve as a starting point for further
experiments.

## Requirements

- Python 3.8+ (tested with 3.10)
- [PyTorch](https://pytorch.org/) (compatible CUDA version if using GPU)
- `numpy`, `pandas`, `scikit-learn` (for evaluation metrics), `matplotlib`
  (optional for plotting)

You can install the Python dependencies with:

```bash
python -m pip install torch numpy pandas scikit-learn matplotlib
```

(or using a `requirements.txt` file if you prefer.)

## Usage

### Training an incomplete model

The `main_incomplete.py` script supports several built‑in benchmark
datasets.  At a minimum you must specify the dataset name and the
location of the data.

```bash
python main_incomplete.py --db MSRCv1 --datapath ./datasets \
    --batch_size 128 --missing_rate 0.1 --learning_rate 5e-4
```

Additional flags control the random seed (`--seed`), latent feature
size (`--feature_dim`), number of training epochs (`--mse_epochs` and
`--con_epochs`), whether to force CPU (`--cpu`), and the directory to
save the trained model (`--save_dir`).  Run `python main_incomplete.py
--help` for a full list of options.

### Evaluation

After training, the `evaluate.py` module can be used to compute
clustering accuracy, normalized mutual information (NMI) and adjusted
rand index (ARI) against ground truth labels.  See the top of
`evaluate.py` for usage examples or call it directly from Python.

### Data preprocessing

`cvcl_dataprocessing.py` contains the `IncompleteviewData` class used to
load and batch datasets with missing views.  It supports the datasets
listed in the paper and can be extended with new ones by modifying the
module.

### Additional models and utilities

- `networks.py` defines the neural architectures used in the paper
  (e.g. `GumbelWyner` and `GumbelWynerZ`) along with some helper
  encoders/decoders.
- `loss.py` implements the custom loss functions (`IMV_Loss` and
  `MFLVC_Loss`) required for training.
- `data_mflvc.py` includes routines for loading complete multiview data
  used in baseline experiments.
- `utils.py` contains miscellaneous functions such as device selection,
  seed setting, and logging helpers.

## Adding new datasets

To experiment with a new dataset, place the feature files in the
`datasets/` directory and update `cvcl_dataprocessing.py` to include a
loader for the new format.  Ensure you provide labels and a mask
indicating missing views.

---

Licensed under the MIT License.
