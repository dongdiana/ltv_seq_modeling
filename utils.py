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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import boxcox, yeojohnson

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

def save_ckpt(path, model, opt, sched, epoch, best_metric, stoi, hp, transform_params=None):
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
        "transform_params": transform_params # 변환 파라미터 저장
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

def encode_sequence(seq, stoi):
    ev_ids = []
    for ev in seq:
        event = ev[0] if isinstance(ev, (list, tuple)) and len(ev) > 0 else ev
        ev_ids.append(stoi.get(event, stoi.get("<UNK>", 1)))
    if not ev_ids:
        ev_ids = [stoi.get("<UNK>", 1)]
    return torch.tensor(ev_ids, dtype=torch.long)

def load_seq_parquet(path, seq_col='ACTION_DELTA'):
    df = pd.read_parquet(path)
    if df[seq_col].dtype == 'object' and isinstance(df[seq_col].iloc[0], str):
        df[seq_col] = df[seq_col].apply(json.loads)
    return df

def class_balanced_mae_loss(logits, targets, beta=0.999):
    """
    클래스 불균형을 고려한 평균 절대 오차(MAE) 손실 함수.
    이진 분류의 클래스 가중치를 회귀 문제에 적용합니다.
    """
    targets_binary = (targets > 0).float()
    n_pos = targets_binary.sum().clamp(min=1.0)
    n_neg = (len(targets) - n_pos).clamp(min=1.0)
    w_pos = (1 - beta) / (1 - beta**n_pos)
    w_neg = (1 - beta) / (1 - beta**n_neg)
    
    # MAE 계산
    mae = F.l1_loss(logits, targets, reduction='none')
    
    # 가중치 적용
    w = targets_binary * w_pos + (1 - targets_binary) * w_neg
    return (w * mae).mean()

def evaluate_reg(y_true, y_pred):
    """
    회귀 모델의 성능 지표를 계산합니다.
    (RMSE, MAE, R2 score 포함)
    """
    metrics = {}
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['R2'] = r2_score(y_true, y_pred)
    return metrics

def transform_target(y, mode, params=None):
    """타겟 변수를 지정된 모드로 변환하고, 변환 파라미터를 반환합니다."""
    if mode == 'log1p':
        return np.log1p(y), None
    elif mode == 'box_cox':
        # Box-Cox 변환은 양수 데이터에만 적용 가능 (y > 0)
        # 0이 포함된 경우 데이터에 1을 더해서 변환합니다.
        # 변환된 값과 람다(lambda) 값을 반환합니다.
        # np.log1p와 np.expm1를 사용하지 않는다면 0을 처리하는 방법을 정의해야 합니다.
        y_trans, lam = boxcox(y + 1e-6)
        return y_trans, {'lambda': lam}
    elif mode == 'yeo_johnson':
        # Yeo-Johnson 변환은 음수, 양수, 0을 모두 처리할 수 있습니다.
        y_trans, lam = yeojohnson(y)
        return y_trans, {'lambda': lam}
    else:
        return y, None

def inverse_transform_target(y_transformed, mode, params):
    """변환된 타겟 변수를 원래 스케일로 되돌립니다."""
    if mode == 'log1p':
        return np.expm1(y_transformed)
    elif mode == 'box_cox':
        lam = params.get('lambda')
        # 변환된 값과 람다(lambda) 값을 반환합니다.
        # 0이 포함된 경우 데이터에 1을 더해서 변환합니다.
        if lam == 0:
            return np.exp(y_transformed) - 1e-6
        else:
            return (y_transformed * lam + 1)**(1/lam) - 1e-6
    elif mode == 'yeo_johnson':
        lam = params.get('lambda')
        # 음수, 양수, 0을 모두 처리
        return np.where(y_transformed >= 0, (y_transformed * lam + 1)**(1/lam) - 1e-6 if lam != 0 else np.exp(y_transformed) - 1e-6, 
                        (2 - (2 - lam * y_transformed)**(1/(2-lam)))/lam if lam != 2 else np.log(y_transformed + 1) / (-1))
    else:
        return y_transformed