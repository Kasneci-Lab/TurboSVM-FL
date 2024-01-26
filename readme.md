# TurboSVM-FL

> **[AAAI 2024] TurboSVM-FL: Boosting Federated Learning through SVM Aggregation for Lazy Clients**.
> TurboSVM-FL is a novel federated learning algorithm that can greatly reduce the number of aggregation rounds needed to approach convergence for federated classifications tasks without any additional computation burden on the client side.

## 🖼️ Teaser
<img src="https://github.com/Kasneci-Lab/TurboSVM-FL/assets/34072813/2a09a189-52d7-4dd1-be27-942ba4cbe975" width="700">

## 🗼 Pipeline
The pipeline of TurboSVM-FL starts by trading client models in a model-as-sample strategy and fitting support vector machine (SVM) on these samples. Then, it carries out selective aggregation using only the class embeddings that form support vectors. Further, it conducts max-margin spread-out regularization on aggregated global representations that are projected back onto the SVM separation hyperplane. 

## 💁 Usage
1. Download data and carry out data preprocessing following the instructions below.

2. Create conda environment with `conda env create -f environment.yml`.

3. Run `main_non_FL.py` for centralized learning; `main_FL.py` for federated learning; `experiments.sh` for reproducing experiments.

For detailed argument settings please check `utils.py`. 

## 🔧 Environment
Important libraries and their versions by **January 25th, 2024**:

| Library | Version |
| --- | ----------- |
| Python | 3.11.7 by Anaconda|
| PyTorch | 2.1.2 for CUDA 12.1 |
| TorchMetrics | 1.2.1 |
| Scikit-Learn | 1.4.0 |
| WandB | 0.16.2 |

Others:
- There is no requirement on OS for the experiment itself. However, for data preprocessing, Python environment on Linux is desired. If data preprocessing is done in Windows Subsystem Linux (WSL or WSL2), please make sure `unzip` is installed beforehand, i.e. `sudo apt install unzip` for WSL2 Ubuntu.

- We used **Weights & Bias** (https://wandb.ai/site) for figures instead of tensorboard. Please install and set up it properly beforehand.

- We used the Python function `match` in our implementation. This function only exists for Python version >= 3.10. Please replace it with `if-elif-else` statement if needed.

## 🗺 Instructions on data preprocessing
We conducted experiments using three datasets: FEMNIST, CelebA, and Shakespeare (the Covid-19 dataset is not used anymore). The datasets can be obtained from https://leaf.cmu.edu/ together with bash code for reproducible data split.

Please dive into the `data` directory for further instructions.

## 📋 Citation
If you use this code, please cite our paper:
```bibtex
@misc{wang2024turbosvmfl,
      title={TurboSVM-FL: Boosting Federated Learning through SVM Aggregation for Lazy Clients}, 
      author={Mengdi Wang and Anna Bodonhelyi and Efe Bozkir and Enkelejda Kasneci},
      year={2024},
      eprint={2401.12012},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
},

@inproceedings{wang2024turbosvm,
  title={TurboSVM-FL: Boosting Federated Learning through SVM Aggregation for Lazy Clients},
  author={Wang, Mengdi and Bodonhelyi, Anna and Bozkir, Efe and Kasneci, Enkelejda},
  booktitle={Proceedings of the 38th AAAI Conference on Artificial Intelligence (AAAI-24)},
  year={2024}
}
```
