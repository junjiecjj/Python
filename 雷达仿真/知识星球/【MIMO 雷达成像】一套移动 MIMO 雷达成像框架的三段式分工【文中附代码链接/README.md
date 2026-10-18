# Mobile-MIMO-Radar-Imaging
Mobile MIMO Radar Imaging via Synergistic Deep Learning and Wave Physics Modeling
# Mobile MIMO Radar Imaging via Synergistic Deep Learning and Wave Physics Modeling

This repository provides the implementation of the paper:

**Mobile MIMO Radar Imaging via Synergistic Deep Learning and Wave Physics Modeling**

The proposed framework decomposes mobile MIMO radar imaging into three main stages:

1. **RDSegNet-based valid target region (VTR) detection** on range-Doppler maps (RDMs);
2. **Modified hypothetical phase compensation (MHPC)** for Doppler velocity disambiguation and phase compensation;
3. **Wavenumber-domain approximation (WNDA)** for high-fidelity radar image reconstruction.

## Code Structure

```text
Mobile-MIMO-Radar-Imaging/
├── README.md
├── requirements.txt
│
├── 01_RDSegNet_VTR_Detection/
│   ├── README.md
│   ├── train_rdsegnet.py
│   ├── test_rdsegnet.py
│   ├── models/
│   │   ├── rdsegnet.py
│   │   ├── complex_conv.py
│   │   └── attention.py
│   └── losses/
│       ├── focal_loss.py
│       └── dice_loss.py
│
├── 02_MHPC_Velocity_Disambiguation/
│   ├── README.md
│   └── phase_compensation.m
│
└── 03_WNDA_Imaging_Reconstruction/
    ├── README.md
    ├── wnda_reconstruction.m
    └── kspace_mapping.m
```

## Module Description

### 1. RDSegNet VTR Detection

The folder `01_RDSegNet_VTR_Detection/` contains the implementation of RDSegNet for valid target region detection on RDMs. The network takes selected Tx-Rx RDM channels as input and outputs a binary segmentation mask indicating valid target regions.

Main files:

* `train_rdsegnet.py`: training script for RDSegNet.
* `test_rdsegnet.py`: testing script for RDSegNet.
* `models/rdsegnet.py`: main RDSegNet architecture.
* `models/complex_conv.py`: complex-valued convolution modules.
* `models/attention.py`: attention modules.
* `losses/focal_loss.py`: focal loss for class imbalance.
* `losses/dice_loss.py`: dice loss for segmentation.

### 2. MHPC Velocity Disambiguation

The folder `02_MHPC_Velocity_Disambiguation/` contains the implementation of the modified hypothetical phase compensation algorithm. This module estimates the Doppler ambiguity index and compensates the motion-induced phase residual for valid RDM cells.

Main file:

* `phase_compensation.m`: MHPC-based velocity disambiguation and phase compensation.

### 3. WNDA Imaging Reconstruction

The folder `03_WNDA_Imaging_Reconstruction/` contains the implementation of the wavenumber-domain approximation imaging method. This module maps the phase-compensated radar data into the wavenumber domain and reconstructs local DOI sub-images.

Main files:

* `wnda_reconstruction.m`: main WNDA reconstruction script.
* `kspace_mapping.m`: wavenumber-domain mapping function.


## Data Format

The RDSegNet module expects RDM data and corresponding binary masks. Each RDM is processed as a two-dimensional range-Doppler representation. The label mask indicates valid target regions on the RDM.

A typical data sample contains:

* `rdm`: range-Doppler map input;
* `mask`: binary valid target region label.

The full measured dataset is not included in this repository due to data-sharing restrictions. Users may prepare their own RDMs and labels following the same format.


## Notes

The released code is intended to provide the core implementation of the proposed framework. Some dataset-specific paths and radar configuration parameters may need to be modified according to the user's own experimental setup.
