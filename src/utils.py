import os
import time
import logging
import json
import random
import pathlib
import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from transformers import LongformerConfig, LongformerModel


# --- Device & seed ---
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"

DEVICE, DEVICE_TYPE = get_device()
PIN_MEMORY = torch.cuda.is_available()

def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 유틸리티 함수 ---
def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def save_ckpt(path, model, opt, sched, epoch, best_metric, stoi, hp):
    ensure_dir(str(pathlib.Path(path).parent))
    obj = {
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict() if opt is not None else None,
        "sched_state": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
        "best_metric": float(best_metric) if best_metric is not None else None,
        "stoi": stoi,
        "hp": hp,
        "device_type": DEVICE_TYPE,
        "ts": time.time(),
    }
    torch.save(obj, path)

def load_ckpt(path, model=None, opt=None, sched=None, map_location=DEVICE):
    ckpt = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(ckpt["model_state"])
    if opt is not None and ckpt.get("opt_state") is not None:
        opt.load_state_dict(ckpt["opt_state"])
    if sched is not None and ckpt.get("sched_state") is not None:
        sched.load_state_dict(ckpt["sched_state"])
    return ckpt

def build_event_vocab(df, col='ACTION_DELTA', min_freq=1, top_k=None):
    cnt = Counter()
    for seq in df[col]:
        for ev in seq:
            event = ev[0] if isinstance(ev, (list, tuple)) and len(ev) > 0 else ev
            cnt[event] += 1
    items = [(ev, c) for ev, c in cnt.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    if top_k is not None:
        items = items[:top_k]
    stoi = {'<PAD>': 0, '<UNK>': 1}
    for ev, _ in items:
        stoi[ev] = len(stoi)
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

def setup_logger(name, log_file, level=logging.INFO):
    """로거를 설정하고 반환합니다."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # log 파일 저장 경로를 확인하고 생성합니다.
    pathlib.Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(stream_handler)
    
    return logger

log_file_path = os.path.join(os.path.dirname(__file__), "logs/training.log")
logger = setup_logger("main_logger", log_file_path)


def encode_sequence(seq, stoi):
    ev_ids = []
    for ev in seq:
        event = ev[0] if isinstance(ev, (list, tuple)) and len(ev) > 0 else ev
        ev_ids.append(stoi.get(event, stoi.get("<UNK>", 1)))
    if not ev_ids:
        ev_ids = [stoi.get("<UNK>", 1)]
    return torch.tensor(ev_ids, dtype=torch.long)

# --- 손실/평가 함수 ---
def class_balanced_focal_with_logits(logits, targets, beta=0.999, gamma=1.5):
    B = targets.numel()
    n_pos = targets.sum().clamp(min=1.0)
    n_neg = (B - n_pos).clamp(min=1.0)
    w_pos = (1 - beta) / (1 - beta**n_pos)
    w_neg = (1 - beta) / (1 - beta**n_neg)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    focal = (1 - pt).pow(gamma) * bce
    w = targets * w_pos + (1 - targets) * w_neg
    return (w * focal).mean()

def find_best_threshold(y_true, p_hat):
    precision, recall, thr = precision_recall_curve(y_true, p_hat)
    thr = np.append(thr, 1.0)
    f1s = 2 * precision * recall / np.clip(precision + recall, 1e-8, None)
    idx = np.nanargmax(f1s)
    return float(thr[idx]), float(f1s[idx])

@torch.no_grad()
def evaluate_cls(y_true, p_hat):
    """
    주어진 실제 레이블과 예측 확률을 기반으로 다양한 분류 지표를 계산합니다.
    """
    y_true = np.asarray(y_true).astype(int)
    p_hat = np.asarray(p_hat).astype(float)
    
    # PR-AUC와 ROC-AUC 계산
    pr_auc = average_precision_score(y_true, p_hat) if (y_true.max() > 0 and y_true.min() == 0) else float('nan')
    try:
        roc_auc = roc_auc_score(y_true, p_hat)
    except ValueError: # handle case with only one class in y_true
        roc_auc = float('nan')
    
    # 최적의 임계값 및 F1 스코어 계산
    best_thr, best_f1 = find_best_threshold(y_true, p_hat)
    y_pred = (p_hat >= best_thr).astype(int)
    
    # 상세 지표 계산
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    return {
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "best_threshold": float(best_thr),
        "best_f1": float(best_f1),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec)
    }

def _safe_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        full = np.zeros((2, 2), dtype=int)
        full[:cm.shape[0], :cm.shape[1]] = cm
        cm = full
    return cm

def classification_report_at_threshold(y_true, p_hat, thr, title="[REPORT]"):
    y_true = np.asarray(y_true).astype(int)
    p_hat = np.asarray(p_hat).astype(float)
    y_pred = (p_hat >= float(thr)).astype(int)
    cm = _safe_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    logger.info(f"{title} @thr={thr:.4f}")
    logger.info(f"CM: tn={tn:,}  fp={fp:,}  fn={fn:,}  tp={tp:,}")
    logger.info(f"acc={acc:.4f} | precision={prec:.4f} | recall={rec:.4f} | f1={f1:.4f}")
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}    

def load_seq_parquet(path, seq_col='ACTION_DELTA'):
    df = pd.read_parquet(path)
    if df[seq_col].dtype == 'object' and isinstance(df[seq_col].iloc[0], str):
        df[seq_col] = df[seq_col].apply(json.loads)
    return df

def ensure_binary_labels(df, amt_col='PAY_AMT', y_col='PAY_AMT_bin'):
    if y_col not in df.columns:
        df[y_col] = (df[amt_col].astype(float) > 0).astype(int)
    return df