# MD-AirComp
This repository contains code for [Massive Digital Over-the-Air Computation for Communication-Efficient Federated Edge Learning](https://arxiv.org/abs/2405.15969). The project focuses on uplink multi-user MIMO over-the-air computation systems and its application in federated edge learning.

## Installation

### Prerequisites

- Python >= 3.8
- Anaconda or Miniconda (for creating virtual environments)

### Setup

1. Create a Conda environment:

   ```bash
   conda create --name MDAirComp python=3.8
   conda activate MDAirComp

2. Install required packages:

   ```bash
   pip install numpy tqdm pandas torch tensorboardX scikit-learn faiss-cpu scipy torchvision

3. To run the MIMO channel simulation:

   ```bash
   python main_MIMO_channel.py

4. To run the error-free channel simulation:
   
   ```bash
   python main_error_free_channel.py

### Visit Our Lab

Check out our lab's research at [GaoZhen Lab](https://gaozhen16.github.io/) or [IPC Lab, Imperial College](https://www.imperial.ac.uk/information-processing-and-communications-lab/publications/). Discover our latest projects and meet the team!

### Citations

If you find this project useful, please cite the related original paper as:

```
@article{qiao2024massive,
  title={Massive Digital Over-the-Air Computation for Communication-Efficient Federated Edge Learning},
  author={Qiao, Li and Gao, Zhen and Mashhadi, Mahdi Boloursaz and G{\"u}nd{\"u}z, Deniz},
  journal={arXiv preprint arXiv:2405.15969},
  year={2024}
}
