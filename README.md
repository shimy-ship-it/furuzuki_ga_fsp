# Industrial CSV Prediction

All source and split CSV files are stored in `dataset/`. Data is randomly split into an 80% training set and a 20% test set.

## Structure

```text
train_ga_svm.py        Independent GA-SVM training entry point
train_neural.py        Independent PyTorch neural-network entry point
dataset_utils.py       Dataset loading and 80/20 splitting
models/ga_svm.py      GA-SVM implementation
models/neural_network.py  PyTorch implementation
```

The models use separate implementations and separate training scripts. The neural network is implemented with a simple PyTorch feed-forward network, Adam, mini-batches, and CPU tensor operations without a validation split or early stopping.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Train separately

```powershell
python train_ga_svm.py
python train_neural.py
```

### GA-SVM options

```text
--population       GA population size (default: 10)
--generations      GA generations (default: 5)
--cv-folds         Fitness cross-validation folds (default: 3)
--dataset-dir      Split CSV directory (default: dataset)
--output-dir       Output directory (default: output-ga-svm)
```

### PyTorch neural-network options

```text
--hidden-layers    Comma-separated hidden sizes (default: 64,32)
--activation       relu, tanh, or logistic (default: relu)
--max-iter         Training epoch limit (default: 500)
--batch-size       Mini-batch size (default: 64)
--learning-rate    Adam learning rate (default: 0.001)
--dataset-dir      Split CSV directory (default: dataset)
--output-dir       Output directory (default: output-neural)
```

Both scripts accept `--data`, `--target`, `--seed`, and `--test-size`. The split is saved as `dataset/train.csv` and `dataset/test.csv`.
