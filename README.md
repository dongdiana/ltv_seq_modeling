# LTV Sequence Modeling

Predicting **Lifetime Value (LTV)** of game users based on sequence logs and tabular features using deep learning (LSTM/Longformer-based architectures).

---

## Project Overview

This project builds a pipeline for **multi-stage LTV prediction** in a gaming environment.  
It combines event-sequence modeling with tabular data using a hybrid architecture and supports modular configuration for flexible experimentation.

### Key Objectives
- Model user-level LTV from in-game event sequences.
- Support **multi-stage prediction pipeline** (Stage 1: payer classification → Stage 2: whale prediction → Stage 3: regression for payers).
- Enable scalable sequence preprocessing and efficient GPU training with custom collate and dataset classes.

---

## Repository Structure

```
ltv_seq_modeling-main/
│
├── main.py                     # Main entry point for training/inference
├── config.py                   # Global hyperparameter settings (HP dict)
├── utils.py                    # Logging, device setup, metric evaluation, etc.
├── datasets.py                 # Custom PyTorch Dataset and DataLoader utilities
├── models.py                   # Sequence model (Longformer/LSTM Regressor)
├── requirements.txt            # Required Python packages
│
├── data/                       # Input parquet or CSV files (train, val, test)
│   ├── train_df_5days.parquet
│   ├── val_df_5days.parquet
│   └── test_df_5days.parquet
├── seq/                       # Input parquet files (train, val, test)
│   ├── train_df_5days_seq.parquet
│   ├── val_df_5days_seq.parquet
│   └── test_df_5days_seq.parquet
│
└── README.md                   # This file
```

---

## Environment Setup

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # (Linux/macOS)
venv\Scripts\activate        # (Windows)
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

If requirements.txt is not provided, install manually:
```bash
pip install torch pandas numpy tqdm scikit-learn transformers
```

---

## How to Run

```bash
python run.py
```


---

## Configuration (config.py)

Example snippet:

```python
HP = dict(
    max_len=2048,
    batch_size=64,
    epochs=10,
    lr=1e-4,
    model_type="LongformerRegressor",
    seed=42,
)
```

You can modify `HP` to experiment with model size, sequence length, or learning rate.

---

## Logging and Outputs

- **Logs**: stored in `outputs/logs/` with timestamped filenames.  
- **Checkpoints**: auto-saved to `outputs/ckpt/`.  
- **Predictions**: CSV results for test set in `outputs/predictions/`.

---

## Model Components

| Module | Description |
|--------|--------------|
| `LongformerRegressor` | Transformer-based sequence regression model for LTV |
| `SeqDataset` | Loads and tokenizes user event sequences |
| `collate_batch` | Custom collation with padding & truncation |
| `evaluate_reg` | Computes MAE, RMSE, R² metrics |
