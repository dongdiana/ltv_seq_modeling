import os
import argparse
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
from functools import partial
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import contextlib
import random
import traceback
import torch.nn.functional as F
from datetime import datetime  # datetime 모듈 추가


from utils import (
    setup_logger,
    get_device,
    set_seed,
    load_seq_parquet,
    build_event_vocab,
    save_ckpt,
    load_ckpt,
    evaluate_reg,
    transform_target,
    inverse_transform_target,
    DEVICE,
    DEVICE_TYPE,
    PIN_MEMORY,
)
from datasets import (
    SeqDataset,
    SeqDatasetInfer,
    collate_batch,
    collate_infer,
    make_length_sorted_loader,
)
from models import LongformerRegressor

# 하이퍼파라미터 설정을 config.py에서 불러옴
from config import HP

# ========== 로거 및 경로 설정 ==========
# 현재 시간을 'YYYYMMDD_HHmmss' 형식으로 가져옵니다.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 실행 시간 기반의 로그 파일 경로 설정
log_file_path = os.path.join(os.path.dirname(__file__), f"logs/training_{timestamp}.log")
logger = setup_logger("main_logger", log_file_path)

# 실행 시간 기반의 출력 디렉토리 설정
output_dir = os.path.join("outputs", timestamp)
# =========================================================

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ========== 전역 예외 처리 후크 추가 ==========
# 처리되지 않은 모든 예외를 로깅합니다.
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # KeyboardInterrupt는 기본 핸들러로 전달하여 프로그램이 정상 종료되도록 합니다.
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception
# =========================================================

def train_one(train_df, valid_df, stoi, max_len=None, batch_size=512, epochs=12, patience=3,
              d_model=96, nhead=4, nlayers=2, lr=3e-4, wd=1e-4, y_col='PAY_AMT',
              base_rate=0.03675, verbose=True, save_dir="checkpoints/seq_cls", run_name="default",
              resume=False, global_tokens=['그로아', '캐시 상점', '스탯'], tabular_data_train=None, tabular_data_valid=None, **kwargs):
    
    # kwargs에서 필요한 값들을 추출
    transformation_mode = kwargs.get('transformation_mode', 'log1p')
    loss_mode = kwargs.get('loss_mode', 'mae')
    huber_delta = kwargs.get('huber_delta', 1.0)
    num_workers = kwargs.get('num_workers', 0)
    regression_model_type = kwargs.get('regression_model_type', 'mlp')

    vocab_size = len(stoi)
    if verbose: logger.info(f"[INFO] Vocab size: {vocab_size}")

    ckpt_last = f"{save_dir}/{run_name}_last.pt"
    ckpt_best = f"{save_dir}/{run_name}_best.pt"

    # 태블러 데이터 차원 계산
    tabular_input_dim = tabular_data_train.shape[1]-1 if tabular_data_train is not None else 0
    
    # tabular_data_train 상태 로깅
    if tabular_data_train is not None:
        logger.info(f"[IDX] first 5 index values: {tabular_data_train.index[:5].tolist()}")
        
    tr_ds = SeqDataset(train_df, stoi, y_col=y_col, max_len=max_len, global_tokens=global_tokens, transformation_mode=transformation_mode, tabular_data=tabular_data_train)
    va_ds = SeqDataset(valid_df, stoi, y_col=y_col, max_len=max_len, global_tokens=global_tokens, transformation_mode=transformation_mode, tabular_data=tabular_data_valid)
    
    # DataLoader에 num_workers 인자 전달
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                                        collate_fn=partial(collate_batch, max_len=max_len), num_workers=num_workers, pin_memory=PIN_MEMORY)
    va_ld = torch.utils.data.DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                                        collate_fn=partial(collate_batch, max_len=max_len), num_workers=num_workers, pin_memory=PIN_MEMORY)
    logger.info("Train/Validation loaders created. Starting training.")

    model = LongformerRegressor(
        vocab_size=vocab_size, d_model=d_model, nhead=nhead, nlayers=nlayers,
        p=0.1, base_rate=base_rate, max_len=max_len, tabular_input_dim=tabular_input_dim, regression_model_type=regression_model_type
    ).to(DEVICE)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    if DEVICE_TYPE == "cuda":
        scaler = None
        amp_ctx = contextlib.nullcontext()
        
    elif DEVICE_TYPE == "mps":
        scaler = None
        amp_ctx = contextlib.nullcontext()
    else:
        scaler = None
        amp_ctx = contextlib.nullcontext()

    start_epoch = 1
    best_metric = float('inf')
    transform_params = tr_ds.transform_params
    
    if resume and os.path.exists(ckpt_last):
        try:
            ckpt = load_ckpt(ckpt_last, model=model, opt=opt, sched=sched)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_metric = ckpt.get("best_metric", float('inf'))
            transform_params = ckpt.get("transform_params")
            if verbose:
                logger.info(f"[resume] Loaded {ckpt_last} @epoch {start_epoch-1} | best RMSE={best_metric:.4f}")
        except Exception:
            logger.exception(f"Failed to load checkpoint from {ckpt_last}. Starting from scratch.")
            
    wait = 0
    
    if loss_mode == 'mae':
        loss_fn = F.l1_loss
    elif loss_mode == 'huber':
        loss_fn = partial(F.huber_loss, delta=huber_delta)
    else:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")
    
    for ep in range(start_epoch, epochs + 1):
        model.train()
        total_loss, n_samples = 0.0, 0
        
        if verbose:
            t_bar = tqdm(tr_ld, desc=f"Epoch {ep}/{epochs}")
        else:
            t_bar = tr_ld

        for batch in t_bar:
            try:
                ev = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                global_mask = batch.get('global_attention_mask', None)
                if global_mask is not None:
                    global_mask = global_mask.to(DEVICE)
                y = batch['labels'].float().to(DEVICE)
                tabular_features = batch['tabular_features'].to(DEVICE)

                opt.zero_grad(set_to_none=True)
                with amp_ctx:
                    position_ids = torch.arange(ev.shape[1], dtype=torch.long, device=DEVICE)
                    position_ids = position_ids.unsqueeze(0).expand_as(ev)
                    logits = model(ev, attention_mask=mask, global_attention_mask=global_mask, tabular_features=tabular_features, position_ids=position_ids)
                    
                    loss = loss_fn(logits, y, reduction='mean')

                if scaler is not None:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                total_loss += loss.item() * ev.size(0)
                n_samples += ev.size(0)

                if verbose:
                    t_bar.set_postfix(loss=total_loss / max(n_samples, 1))

            except Exception:
                logger.exception(f"Error during training epoch {ep}, batch {n_samples}:")
                raise

        tr_loss = total_loss / max(n_samples, 1)
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for batch in va_ld:
                ev = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                global_mask = batch.get('global_attention_mask', None)
                if global_mask is not None:
                    global_mask = global_mask.to(DEVICE)
                y = batch['labels'].to(DEVICE)
                tabular_features = batch['tabular_features'].to(DEVICE)

                out = model(ev, attention_mask=mask, global_attention_mask=global_mask, tabular_features=tabular_features)
                all_p.append(out.detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())
        
        y_true_valid = np.concatenate(all_y)
        p_hat_valid = np.concatenate(all_p)
        
        metrics_valid = evaluate_reg(y_true_valid, p_hat_valid)
        score = metrics_valid['RMSE']

        if verbose:
            logger.info(f"[{ep:02d}] loss {tr_loss:.4f} | RMSE {metrics_valid['RMSE']:.4f} | MAE {metrics_valid['MAE']:.4f} | R2 {metrics_valid['R2']:.4f}")

        sched.step()
        
        save_ckpt(ckpt_last, model, opt, sched, ep, best_metric, stoi, HP, transform_params)
        if score < best_metric - 1e-4:
            best_metric = score
            save_ckpt(ckpt_best, model, opt, sched, ep, best_metric, stoi, HP, transform_params)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose: logger.info(f"Early stopping at epoch {ep} (best RMSE={best_metric:.4f})")
                break

    @torch.no_grad()
    def collect_probs(df, tabular_data=None):
        ds = SeqDataset(df, stoi, y_col=y_col, max_len=max_len, global_tokens=global_tokens, transformation_mode=transformation_mode, tabular_data=tabular_data)
        ld = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_batch, max_len=max_len), num_workers=num_workers, pin_memory=PIN_MEMORY)
        model.eval()
        all_pid, all_p, all_y = [], [], []
        for batch in ld:
            ev = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            global_mask = batch.get('global_attention_mask', None)
            if global_mask is not None:
                global_mask = global_mask.to(DEVICE)
            tabular_features = batch['tabular_features'].to(DEVICE)
            
            position_ids = torch.arange(ev.shape[1], dtype=torch.long, device=DEVICE)
            position_ids = position_ids.unsqueeze(0).expand_as(ev)
            
            p = model(ev, attention_mask=mask, global_attention_mask=global_mask, tabular_features=tabular_features, position_ids=position_ids).cpu().numpy()

            all_pid.extend(batch['ids'])
            all_p.append(p)
            all_y.append(batch['labels'].numpy())
        return np.array(all_pid), np.concatenate(all_p), np.concatenate(all_y)

    pid_v, p1_v, y_v = collect_probs(valid_df, tabular_data=tabular_data_valid)
    
    pred_valid = pd.DataFrame({
        "PLAYERID": pid_v,
        "payer_pred_valid": inverse_transform_target(p1_v, transformation_mode, transform_params),
        "true_value": inverse_transform_target(y_v, transformation_mode, transform_params),
    })

    return model, stoi, pred_valid, best_metric

def prepare_tabular_features(df: pd.DataFrame, y_col: str):
    out = df.copy()

    # 1) 범주형 컬럼 추출 (PLAYERID, y_col 제외)
    cat_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["PLAYERID", y_col]]

    # 2) 원-핫 인코딩 (전체 df에 한 번에 → 분할 후 컬럼 불일치 방지)
    if cat_cols:
        out = pd.get_dummies(out, columns=cat_cols, dummy_na=False)

    # 3) y_col 제거 (피처만 남기려면)
    if y_col in out.columns:
        out = out.drop(columns=[y_col])

    # 4) 수치 변환 & 결측 대체 & 타입 통일
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Enable TEST_MODE (use small sample)")
    args, _ = parser.parse_known_args()
    TEST_MODE = args.test or (os.getenv("TEST_MODE", "0") in ["1", "true", "True"])
    
    try:
        # 태블러 데이터 로드 및 통합
        tabular_train_df_raw = pd.read_parquet('./data/train_df_5days.parquet') #/root/sblm/3stage/data/train_df_5days.parquet
        tabular_val_df_raw = pd.read_parquet('./data/val_df_5days.parquet') #/root/sblm/3stage/data/val_df_5days.parquet
        tabular_test_df_raw = pd.read_parquet('./data/test_df_5days.parquet') # /root/sblm/3stage/data/test_df_5days.parquet
        
        all_tabular_df = pd.concat([tabular_train_df_raw, tabular_val_df_raw, tabular_test_df_raw], ignore_index=True)
        all_tabular_df = all_tabular_df.set_index('PLAYERID')

        # categorical feature 처리
        all_tabular_df = prepare_tabular_features(all_tabular_df, y_col=HP["y_col"])
        logger.info("All tabular data loaded and merged.")
        
        # 시퀀스 데이터 로드
        train_df = load_seq_parquet('./seq/train_df_5days_seq.parquet')
        val_df = load_seq_parquet('./seq/val_df_5days_seq.parquet')
        test_df = load_seq_parquet('./seq/test_df_5days_seq.parquet')
        
        # 태블러 데이터에서 불필요한 컬럼 제거
        tabular_cols = [col for col in all_tabular_df.columns if col not in [HP['y_col']]]

        # 시퀀스 데이터의 플레이어 ID를 기준으로 태블러 데이터 분할
        # .loc[] 인덱싱을 통해 PLAYERID를 기준으로 데이터를 분리
        # .reset_index()를 사용하여 PLAYERID를 컬럼으로 복원
        tabular_train_df_for_model = all_tabular_df.loc[train_df['PLAYERID']].reset_index()
        tabular_val_df_for_model = all_tabular_df.loc[val_df['PLAYERID']].reset_index()
        tabular_test_df_for_model = all_tabular_df.loc[test_df['PLAYERID']].reset_index()

        # 모델에 전달할 데이터프레임에서 'PAY_AMT'를 제외한 피처만 선택
        tabular_train_df_for_model = tabular_train_df_for_model[tabular_cols+['PLAYERID']]
        tabular_val_df_for_model = tabular_val_df_for_model[tabular_cols+['PLAYERID']]
        tabular_test_df_for_model = tabular_test_df_for_model[tabular_cols+['PLAYERID']]
        
        logger.info(f"Tabular data partitioned based on sequence data's PLAYERIDs.")
        logger.info(f"[DEBUG] tabular_train_df_for_model shape: {tabular_train_df_for_model.shape}")
        logger.info(f"[DEBUG] tabular_val_df_for_model shape: {tabular_val_df_for_model.shape}")
        logger.info(f"[DEBUG] tabular_test_df_for_model shape: {tabular_test_df_for_model.shape}")
        
        train_full = pd.concat([train_df, val_df], ignore_index=True)
        tabular_train_full = pd.concat([tabular_train_df_for_model, tabular_val_df_for_model], ignore_index=True)

        del train_df, val_df
        logger.info("Data loaded and preprocessed successfully.")
        
        if TEST_MODE:
            logger.info("\n===== TEST MODE ENABLED: Sampling 1000 items =====")
            if len(train_full) > 1000:
                train_full = train_full.sample(n=1000, random_state=2025).reset_index(drop=True)
            logger.info(f"Sampled train_full shape: {train_full.shape}")
    
    except FileNotFoundError:
        logger.error("Parquet 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        sys.exit(1)
    except Exception:
        logger.exception("An unexpected error occurred during data loading.")
        sys.exit(1)

    set_seed(2025)

    try:
        # build_event_vocab 함수에 min_freq와 top_k_vocab 인자 전달
        stoi, _ = build_event_vocab(train_full, min_freq=HP.get('min_freq', 3), top_k=HP.get('top_k_vocab', None))
        vocab_size = len(stoi)
        logger.info(f"[INFO] Global Vocab size: {vocab_size}")

        logger.info(f"[DEBUG] HP.max_len = {HP['max_len']}, HP.min_freq = {HP['min_freq']}, HP.top_k_vocab = {HP['top_k_vocab']}")
        logger.info(f"[DEBUG] stoi sample: {dict(list(stoi.items())[:10])}")

        logger.info("\n===== Final Model Training =====")
        final_model, _, _, final_metric = train_one(
            train_full, train_full, stoi, tabular_data_train=tabular_train_full, tabular_data_valid=tabular_train_full, **HP, verbose=True
        )
        
        logger.info(f"Final model trained. Best validation RMSE: {final_metric:.4f}")
        
        logger.info("\n===== Test Data Inference =====")
        
        test_ds = SeqDatasetInfer(test_df, stoi, max_len=HP['max_len'], tabular_data=tabular_test_df_for_model)
        # DataLoader에 num_workers 인자 전달
        test_ld = torch.utils.data.DataLoader(
            test_ds, batch_size=HP['batch_size'], shuffle=False,
            collate_fn=partial(collate_infer, max_len=HP['max_len']),
            num_workers=HP.get('num_workers', 0), pin_memory=PIN_MEMORY, persistent_workers=(HP.get('num_workers', 0) > 0)
        )
        
        final_model.eval()
        test_pids, test_preds = [], []
        
        ckpt_path = f"checkpoints/seq_cls/default_best.pt"
        ckpt = load_ckpt(ckpt_path, map_location=DEVICE)
        transform_params = ckpt.get("transform_params")
        
        with torch.no_grad():
            for batch in test_ld:
                ev = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                global_mask = batch.get('global_attention_mask', None)
                if global_mask is not None:
                    global_mask = global_mask.to(DEVICE)
                tabular_features = batch['tabular_features'].to(DEVICE)
                
                out = final_model(ev, attention_mask=mask, global_attention_mask=global_mask, tabular_features=tabular_features)
                test_pids.extend(batch['ids'])
                test_preds.append(out.detach().cpu().numpy())
        
        test_preds = np.concatenate(test_preds)
        
        predictions_df = pd.DataFrame({
            "PLAYERID": test_pids,
            "payer_prediction": inverse_transform_target(test_preds, HP['transformation_mode'], transform_params),
        })
        
        # 수정된 부분: 실행 시간 기반의 폴더에 CSV 파일 저장
        os.makedirs("outputs", exist_ok=True)
        predictions_df.to_csv("outputs/test_predictions_reg.csv", index=False)
        logger.info("[saved] outputs/test_predictions_reg.csv")

    except Exception:
        logger.exception("An error occurred during training or inference.")
        sys.exit(1)