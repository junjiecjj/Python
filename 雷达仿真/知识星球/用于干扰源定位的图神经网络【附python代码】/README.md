# Graph Neural Networks for Jamming Source Localization

This repository accompanies our paper:

> **Graph Neural Networks for Jamming Source Localization**
> Dania Herzalla, Willian T Lunardi, Martin Andreoni
> *ECML PKDD 2025 - Applied Data Science Track*
> [Link to the paper (DOI)](doi.here)

## Overview

This project provides an implementation of our method for localizing jamming sources in wireless networks using graph neural networks (GNNs). Traditional geometric methods face limitations due to environmental uncertainties and dense interference scenarios. Our proposed framework leverages structured node representations and an attention-based GNN for robust and accurate localization under complex RF environments.

## Highlights

* **Graph-Based Learning:** Localization formulated as an inductive graph regression task.
* **Attention Mechanism:** Enhances neighborhood representation and robustness.
* **Confidence-Guided Estimation:** Combines GNN predictions with domain-informed priors dynamically.

## Methodology

### Global Context Encoding

A supernode aggregates global information derived from noise floor levels, providing structured global context to inform confidence-weighted localization decisions.

### Confidence-Guided Adaptive Position Estimation (CAGE)

Localization combines predictions from the GNN and Weighted Centroid Localization (WCL) based on dynamically computed confidence weights:

```math
\hat{x}_{\text{final}} = \alpha \odot \hat{x}_{\text{GNN}} + (1 - \alpha) \odot \hat{x}_{\text{WCL}}
```

### Training Strategy

Training minimizes a combined loss balancing independent GNN predictions and adaptive confidence weighting to ensure generalization and robustness:

```math
\mathcal{L}_{\text{CAGE}} = \frac{1}{2} (\mathcal{L}_{\text{GNN}} + \mathcal{L}_{\text{Adapt}}) + \lambda \sum_{m \in B} (1 - \alpha^{(m)})^2
```


## Data Availability

The dataset used in this project is publicly available at [Kaggle: Network Jamming Simulation Dataset](https://www.kaggle.com/datasets/daniaherzalla/network-jamming-dataset).

## Repository Structure

* **`main.py`**: Main entry point for training, validation, and inference.
* **`model.py`**: Defines the GNN architecture and attention mechanism.
* **`train.py`**: Training and evaluation logic.
* **`data_processing.py`**: Data loading and preprocessing utilities.
* **`utils.py`**: Helper functions for reproducibility and result handling.
* **`custom_logging.py`**: Logging utilities for debugging.
* **`data/`**: Directory containing datasets (`dynamic_data.pkl`).
* **`experiments/`**: Stores experimental logs, checkpoints, and results.

## Running the Code

To see available command-line options for preprocessing and model training:

```bash
python main.py --help
```

To train the model with default parameters:

```bash
python main.py
```

## Citation

If you find this project useful, please cite our paper:

```bibtex
@inproceedings{herzalla2025graph,
  title={Graph Neural Networks for Jamming Source Localization},
  author={Herzalla, Dania and Lunardi, Willian T and Andreoni, Martin},
  booktitle={ECML PKDD 2025 - Applied Data Science Track},
  year={2025},
  publisher={},
  doi={doi.here}
}
```
