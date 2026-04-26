# Spatiotemporal RUL Prediction

This repository contains a simple PyTorch implementation for remaining useful life (RUL) prediction on the CMAPSS aircraft engine dataset.

The current code is mainly for running experiments and reproducing basic training / prediction results. More details will be added after the related paper is published.

## Requirements

Recommended environment:

```bash
Python >= 3.8
PyTorch
pandas
numpy
matplotlib
scikit-learn
```

Install the common dependencies:

```bash
pip install torch pandas numpy matplotlib scikit-learn
```

## Data

Place the CMAPSS data files under:

```text
自己设计的/raw_data/
```

The expected files include:

```text
train_FD001.txt, test_FD001.txt, RUL_FD001.txt
train_FD002.txt, test_FD002.txt, RUL_FD002.txt
train_FD003.txt, test_FD003.txt, RUL_FD003.txt
train_FD004.txt, test_FD004.txt, RUL_FD004.txt
```

Processed CSV files will be saved under:

```text
自己设计的/raw_data/processed/
```

## Run

Before running, open `main.py` and modify the data paths according to your local directory:

```python
data_path_0 = "your_project_path/自己设计的/raw_data/"
data_path = "your_project_path/自己设计的/raw_data/processed/"
```

Select the dataset subset in `main.py`:

```python
subset_name = "003"
```

Then run:

```bash
python main.py
```

If the processed files do not exist, uncomment the preprocessing lines in `main.py` before training:

```python
process_and_save_data(data_path_0, subset_name)
train_data, test_data = cluster(data_path, subset_name)
oc_history_cols(data_path, subset_name, train_data, test_data, save=True)
```

## Output

Training results, prediction files, figures, and model weights will be generated in the corresponding result / model folders.

## Note

This project is released as runnable research code. The implementation is kept simple, and detailed method descriptions will be provided after the paper is available.
