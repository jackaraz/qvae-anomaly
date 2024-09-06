# Quantum Autoencoder for anomaly detection

[![arxiv](https://img.shields.io/static/v1?style=plastic&label=arXiv&message=2409.xxxxx&color=brightgreen)](https://arxiv.org/abs/xxxx.xxxx)

## Datasets

* **The credit card fraud dataset** has been taken from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and preprocessed using `sklearn.preprocessing.MinMaxScaler(feature_range=(-np.pi, np.pi))`.

* Pedregosa, F., Varoquaux, Gael, Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(Oct), 2825–2830.

## Usage

The code to reproduce the results is available in `qVAE.py`, and all necessary dependencies can be installed using the `requirements.txt` file. Please note that the package versions are fixed to ensure complete reproducibility. For a list of execution options, run `./qVAE.py -h`.
