# CS5487 Project Demo — Handwritten Digit Classification

A compact, self-contained pipeline for the seven-model digit classification
experiment reported in `doc/Project-Report/`.

## Layout

```
.
├── main.py                       # entry point: --mode {train,base,eval,challenge}
├── Makefile                      # build / checkenv / train / base / eval / challenge / clean
├── requirements.txt              # Python dependencies
├── check_env.py                  # verify packages and data files
├── data/
│   ├── raw/{base,challenge}/     # course-provided MNIST subset
│   └── trained-model/            # saved models live here after `make train`
├── results/
│   ├── evaluation.json           # mean/std accuracy from --mode base / train
│   ├── evaluation-challenge.json # mean/std accuracy from --mode challenge
│   ├── images/                   # per-(model, trial) confusion matrices (.pdf)
│   └── eval_figures/             # hyperparameter analysis figures (.pdf)
├── evaluation/
│   ├── eval_svm_kernel.py        # SVM kernel comparison
│   ├── eval_svm_grid.py          # SVM (C, γ) grid search heatmap
│   ├── eval_pca_variance.py      # PCA variance ratio sweep
│   ├── eval_kernel_pca.py        # Linear PCA vs Kernel PCA
│   ├── eval_speed_comparison.py  # from-scratch vs sklearn speed benchmark
│   └── eval_error_analysis.py   # per-class recall heatmap & confusion pairs
├── utils/
│   ├── data_io.py                # load_trial, load_challenge
│   ├── model_io.py               # save_model, load_model
│   └── results_io.py             # write_summary, save_confusion_matrix
└── pipeline/
    ├── preprocess.py             # 3 modes: noprocess / noPCA / wholeprocess
    ├── kernels.py                # euclidean_distance_sq, rbf_kernel
    ├── OneNN.py                  # 1-NN baseline (from-scratch, NumPy)
    ├── LogisticRegression.py     # multinomial softmax + L2 (from-scratch)
    ├── KernelSVM.py              # simplified SMO + OvR (from-scratch)
    ├── RandomForest.py           # sklearn
    ├── XGBoost.py                # xgboost
    ├── MLP.py                    # sklearn
    └── CNN.py                    # PyTorch LeNet-style
```

## Quick Start

```bash
make all          # run the full pipeline end-to-end (recommended)
```

Or step by step:

```bash
make build        # 1. install dependencies
make checkenv     # 2. verify Python env and data files
make train        # 3. train all 7 models on both trials
make base         # 4. evaluate on the base test set
make challenge    # 5. evaluate on the challenge set
make eval         # 6. run hyperparameter analysis scripts
```

Per-model invocation (train / base / challenge only):

```bash
python3 main.py --mode train     --model SVM
python3 main.py --mode base      --model CNN
python3 main.py --mode challenge --model 1NN
```

## Make Targets

| Command           | Description                                                              |
|-------------------|--------------------------------------------------------------------------|
| `make all`        | Run the full pipeline: build → checkenv → train → base → challenge → eval.|
| `make build`      | Install all Python dependencies from `requirements.txt`.                 |
| `make checkenv`   | Print Python version, verify packages and data files.                    |
| `make train`      | Train all models on both trials, save to `data/trained-model/`.          |
| `make base`       | Evaluate on base test set (loads models; trains if missing).             |
| `make eval`       | Run the four hyperparameter analysis scripts in `evaluation/`.           |
| `make challenge`  | Evaluate on the challenge dataset.                                       |
| `make clean`      | Remove saved models and `__pycache__` directories.                       |

## Preprocessing Modes

| Mode           | Steps                                                                       | Used by                 |
|----------------|-----------------------------------------------------------------------------|-------------------------|
| `noprocess`    | raw / 255                                                                   | 1-NN                    |
| `noPCA`        | median 3×3 → Gaussian σ=1 → centroid centering → Min-Max normalize          | XGBoost, RF, CNN        |
| `wholeprocess` | `noPCA` + PCA retaining specified variance ratio                            | LR, Kernel SVM, MLP     |

## Model Configuration

| Model               | Preprocessing  | PCA  | Key Hyperparameters                    |
|----------------------|---------------|------|----------------------------------------|
| 1-NN                | `noprocess`    | —    | k = 1                                  |
| Logistic Regression | `wholeprocess` | 0.99 | C = 1.0; max_iter = 300                |
| Kernel SVM          | `wholeprocess` | 0.95 | C = 10; γ = scale                      |
| Random Forest       | `noPCA`        | —    | 500 trees; max_features = √D           |
| XGBoost             | `noPCA`        | —    | 150 rounds; η = 0.1; depth = 6         |
| MLP                 | `wholeprocess` | 0.95 | Arch: (512, 256, 128); lr = 0.005      |
| CNN                 | `noPCA`        | —    | 50 epochs; lr = 0.001; batch = 64      |

## Results — Base Test Set

| Model               | Mean Acc. | Std.   | F1-Score | Recall |
|----------------------|----------|--------|----------|--------|
| **1-NN (Baseline)** | **0.9160** | 0.0025 | 0.9155 | 0.9160 |
| Logistic Regression  | 0.8832   | 0.0058 | 0.8827   | 0.8832 |
| **Kernel SVM**       | **0.9480** | 0.0010 | 0.9479 | 0.9480 |
| XGBoost              | 0.9170   | 0.0030 | 0.9168   | 0.9170 |
| Random Forest        | 0.9187   | 0.0033 | 0.9186   | 0.9187 |
| MLP                  | 0.9253   | 0.0172 | 0.9251   | 0.9252 |
| **CNN**              | **0.9810** | 0.0020 | 0.9810 | 0.9810 |

## Results — Challenge Dataset

| Model               | Mean Acc. | Std.   | F1-Score | Recall |
|----------------------|----------|--------|----------|--------|
| **1-NN (Baseline)** | **0.6833** | 0.0233 | 0.6757 | 0.6833 |
| Logistic Regression  | 0.7700   | 0.0167 | 0.7644   | 0.7700 |
| **Kernel SVM**       | **0.8567** | 0.0100 | 0.8560 | 0.8567 |
| XGBoost              | 0.7967   | 0.0300 | 0.7960   | 0.7967 |
| Random Forest        | 0.7933   | 0.0200 | 0.7892   | 0.7933 |
| MLP                  | 0.7967   | 0.0100 | 0.7946   | 0.7967 |
| **CNN**              | **0.9233** | 0.0033 | 0.9195 | 0.9233 |

## From-Scratch vs Sklearn Speed Comparison

Benchmarked on Trial 1 with identical preprocessing (PCA 0.95 variance):

| Component           | From-scratch    | Sklearn         | Ratio              |
|---------------------|----------------:|----------------:|--------------------|
| PCA                 | 0.07 s          | 0.45 s          | **0.2× (faster)**  |
| Logistic Regression | 0.16 s          | 0.02 s          | ~10× slower        |
| Kernel SVM          | 15.5 s          | 0.10 s          | ~150× slower       |
| Random Forest       | 16.3 s          | 0.19 s          | ~86× slower        |

> Our from-scratch PCA is faster due to a direct `numpy.linalg.eigh` path.
> Classifier speed gaps arise from pure-Python loops vs optimized C/Cython backends.
> Despite being slower, the from-scratch SVM achieves higher accuracy (94.90% vs 94.15%).
