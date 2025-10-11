# Learning to shape beams: using a neural network to control a beamforming antenna

This is a repository for the paper [[Learning to shape beams: using a neural network to control a beamforming antenna](https://doi.org/10.1016/j.comnet.2025.111544)]

# Abstract

The field of reconfigurable intelligent surfaces (RIS) has gained significant traction in recent years in the wireless communications domain, owing to the ability to dynamically reconfigure surfaces to change their electromagnetic radiance patterns in real-time. In this work, we propose utilising a novel deep learning model that innovatively employs only the parameters of each signal or beam as input, eliminating the need for the entire one-dimensional signal or its diffusion map (two-dimensional information). This approach enhances efficiency and reduces the overall complexity of the model, drastically reducing network size and enabling its implementation on low-cost devices. Furthermore, to enhance training effectiveness, the learning model attempts to estimate the discrete cosine transform applied to the output matrix rather than the raw matrix, significantly improving the achieved accuracy. This scheme is validated on a 1-bit programmable metasurface of size 10x10, achieving an accuracy close to 95\% using a K-fold methodology with K=10.

# Pretrained models

The github project associated to this repository has an associated release with a zip file with the pretrained models. To use this repository, clone it to your local computer and unzip the previously mentioned zip file into the repository (after unzipping it, the repository should contain a new subfolder: `trainings`). The zip for the pretrained models can be downloaded [[here](https://github.com/icai-uma/learning-to-shape-beams-using-a-neural-network-to-control-a-beamforming-antenna/releases/download/pretrained_models/pretrained_models.zip)].

# Datasets

Please reach to Marcos Baena Molina (co-author of the paper) for permission to use the datasets. Once permission is granted and the datasets are downloaded, they should be placed in a new folder named `datasets` in the repository. This folder should have two files: `datasets/10x10_1bit.mat` and `datasets/10x10_2bit.mat`.

# Code

This repository contains validation code, as well as some constant weights for the models with DCT output. Once you have both the pretrained models and the datasets, you can run the validation code.

After unzipping, each pretrained model is in a subfolder in `trainings`. Correspondence between folder names and model specifications in the paper (tables 1, 2, A.3 and A.4 in the paper) is as following:

| bits |  N  | output   | loss  |  n  |   FOLDER_NAME                                             |
|------|-----|----------|-------|-----|-----------------------------------------------------------|
|    1 |  25 | BIN      | BCE   |  1  |   fulldataset_25ch_1bit_10x10_straightBCE_test0           |
|    1 |  25 | BIN      | MSE   |  1  |   fulldataset_25ch_1bit_10x10_straightAngles_test0        |
|    1 |  25 | DCT      | MSE   |  1  |   fulldataset_25ch_1bit_10x10_straightDCT_test0           |
|    1 |  25 | BIN      | BCE   |  2  |   fulldataset_25ch_1bit_10x10_straightBCE2models_test0    |
|    1 |  25 | BIN      | MSE   |  2  |   fulldataset_25ch_1bit_10x10_straightAngles2models_test0 |
|    1 |  25 | DCT      | MSE   |  2  |   fulldataset_25ch_1bit_10x10_straightDCT2models_test0    |
|    1 |  50 | BIN      | BCE   |  1  |   fulldataset_50ch_1bit_10x10_straightBCE_test0           |
|    1 |  50 | BIN      | MSE   |  1  |   fulldataset_50ch_1bit_10x10_straightAngles_test0        |
|    1 |  50 | DCT      | MSE   |  1  |   fulldataset_50ch_1bit_10x10_straightDCT_test0           |
|    1 |  50 | BIN      | BCE   |  2  |   fulldataset_50ch_1bit_10x10_straightBCE2models_test0    |
|    1 |  50 | BIN      | MSE   |  2  |   fulldataset_50ch_1bit_10x10_straightAngles2models_test0 |
|    1 |  50 | DCT      | MSE   |  2  |   fulldataset_50ch_1bit_10x10_straightDCT2models_test0    |
|    2 |  25 | 4_VALUES | MSE   |  1  |   fulldataset_25ch_2bit_10x10_straightAngles_test0        |
|    2 |  25 | DCT      | MSE   |  1  |   fulldataset_25ch_2bit_10x10_straightDCT_test0           |
|    2 |  50 | 4_VALUES | MSE   |  1  |   fulldataset_50ch_2bit_10x10_straightAngles_test0        |
|    2 |  50 | DCT      | MSE   |  1  |   fulldataset_50ch_2bit_10x10_straightDCT_test0           |

Experiments in the paper involved, (a) doing a 10-fold cross-validation study for each parameter combination (results shown in Tables 1 and A.3 in the paper), as well as (b) training 10 models with the full dataset as training data for each parameter combination (results shown in Tables 2 and A.4 in the paper). Please note that this repository provides only one of each of the 10 models for each parameter combination, for case (b).

The validation code can be run with the following commands:

```

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_25ch_1bit_10x10_straightBCE_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_25ch_1bit_10x10_straightAngles_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_25ch_1bit_10x10_straightDCT_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_25ch_1bit_10x10_straightBCE2models_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_25ch_1bit_10x10_straightAngles2models_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_25ch_1bit_10x10_straightDCT2models_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_50ch_1bit_10x10_straightBCE_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_50ch_1bit_10x10_straightAngles_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_50ch_1bit_10x10_straightDCT_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_50ch_1bit_10x10_straightBCE2models_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_50ch_1bit_10x10_straightAngles2models_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_50ch_1bit_10x10_straightDCT2models_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_25ch_2bit_10x10_straightAngles_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_25ch_2bit_10x10_straightDCT_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_50ch_2bit_10x10_straightAngles_test0

python tablerito.py --device 0 --config_file configuration_arguments.txt --checkpoint_file model_best_errordB.pt --validate_log validation.txt --img_basename img_results_ --matfilename results.mat --traindir trainings/fulldataset_50ch_2bit_10x10_straightDCT_test0

```

# Requirements

Requeriments were installed with conda, but should work with any other installation method. Versions are orientative (no advanced features from any library were used, so the code should work with any reasonably recent version of any of these): Pytorch 1.10.2, Numpy 1.22.2, Scipy 1.8.0, Matplotlib 3.5.1, OpenCV's python bindings 4.5.3.

While the pytorch code can be converted to just use the CPU, that mode of execution would be very slow. An NVidia GPU is required to accelerate the execution of the model.

