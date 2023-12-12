# TurboSVM-FL
This is the repository for TurboSVM-FL, a novel federated learning algorithm that can significantly boost convergence even for lazy clients. TurboSVM-FL trades client models in a model-as-sample strategy and fits support vector machine (SVM) on these samples. With the help of support vectors and SVM separation hyperplane, TurboSVM-FL conducts selective aggregation and max-margin spread-out regularization.

## Teaser
<img src="https://github.com/wmd0701/TurboSVM-FL/assets/34072813/d40ea56b-faa0-4111-b5d7-eb0257da57c5" width="700">

## Usage
run `main_non_FL.py` for centralized learning.

run `main_FL.py` for federated learning.

run `experiments.sh` for reproducing experiments.

For detailed argument settings please check `utils.py`. 

## Environment
Important installed libraries and their versions by **2023 August 1st**:

| Library | Version |
| --- | ----------- |
| Python | 3.10.12 by Anaconda|
| PyTorch | 2.0.1 for CUDA 11.7 |
| TorchMetrics | 0.11.4 |
| Scikit-Learn | 1.2.2 |
| NumPy | 1.25.0 |

Others:
- There is no requirement on OS for the experiment itself. However, to do data preprocessing, Python environment on Linux is needed. If data preprocessing is done in Windows Subsystem Linux (WSL or WSL2), please make sure `unzip` is installed beforehand, i.e. `sudo apt install unzip` for WSL2 Ubuntu.

- We used **Weights & Bias** (https://wandb.ai/site) for figures instead of tensorboard. Please install and set up it properly beforehand.

- We used the Python function `match` in our implementation. This function only exists for Python version >= 3.10. Please replace it with `if-elif-else` statement if needed.

## Instructions on data preprocessing
We conducted experiments using three datasets: FEMNIST, CelebA, and Shakespeare (the Covid-19 dataset is not used anymore). The datasets can be obtained from https://leaf.cmu.edu/ together with bash code for reproducible data split.

Please dive into the `data` directory for further instructions.

## Citation
If you use this code, please cite our paper:
```
@inproceedings{wang2024turbosvm,
  title={TurboSVM-FL: Boosting Federated Learning through SVM Aggregation for Lazy Clients},
  author={Wang, Mengdi and Bodonhelyi, Anna and Bozkir, Efe and Kasneci, Enkelejda},
  booktitle={Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI-24)},
  year={2024}
}
```