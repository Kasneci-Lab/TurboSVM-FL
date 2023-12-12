# Data preprocessing for CelebA dataset

## Source
https://github.com/TalwalkarLab/leaf/tree/master/data/celeba.

## Steps

Please go through the following steps in Linux Python exvironment that has already installed `numpy` and `pillow` packages. If running in Windows Subsystem Linux (WSL or WSL2), please make sure `unzip` in installed beforehand, i.e. `sudo apt install unzip` for WSL2 Ubuntu.

1. from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, download or request the metadata files `identity_CelebA.txt` and `list_attr_celeba.txt`, and place them in `data/raw/` directory.

2. git clone from https://github.com/TalwalkarLab/leaf/tree/master.

3. dive into `leaf/data/celeba/`.

4. run `./preprocess.sh -s niid --sf 1.0 -k 5 -t user --tf 0.9 --smplseed 0 --spltseed 0`.

5. place the json files and the image folder according to the path variables `--celeba_train_path`, `--celeba_test_path` and `--celeba_image_path` in `utils.py`.