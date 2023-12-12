# Instructions on data preprocessing

We conducted experiments using three datasets: FEMNIST, CelebA, and Shakespeare (the COVID-19 dataset is not used anymore). The datasets can be obtained from https://leaf.cmu.edu/ together with bash code for reproducible data split.

We always did user-independent data split, which means we have a held-out set of clients for validation, rather that a proportion of held-out samples for each client. This corresponds to `-t user` argument in bash code.

The train-test split rate is 90%-10%. This corresponds to `--tf 0.9` argument in bash code.

Please dive into the respective subdirectory for further instructions on each dataset.

Once data processing is done, please place the data according to the path variables in `utils.py`.