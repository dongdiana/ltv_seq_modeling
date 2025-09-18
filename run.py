import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
from functools import partial
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import ast
from sklearn.model_selection import StratifiedKFold
import contextlib
import random
import traceback
from config import HP

# pip install -r requirements.txt
# python longformer/run.py

from src.utils import (
    setup_logger,
    logger,
    get_device,
    set_seed,
    load_seq_parquet,
    ensure_binary_labels,
    build_event_vocab,
    save_ckpt,
    load_ckpt,
    class_balanced_focal_with_logits,
    evaluate_cls,
    find_best_threshold,
    DEVICE,
    DEVICE_TYPE,
    PIN_MEMORY,
    classification_report_at_threshold,
)
from src.datasets import (
    SeqDataset,
    SeqDatasetInfer,
    collate_batch,
    collate_infer,
    make_length_sorted_loader,
)
from src.models import SeqClassifier

# 로거 설정
log_file_path = os.path.join(os.path.dirname(__file__), "logs/training.log")
logger = setup_logger("main_logger", log_file_path)

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

# ========== tqdm 출력을 logger로 리디렉션하는 클래스 ==========
class TqdmToLogger(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write(self, s, file=None, end="\n", nolock=False):
        if s.strip():
            logger.info(s.strip())

# =========================================================

def train_one(train_df, valid_df, stoi, max_len=None, batch_size=512, epochs=12, patience=3,
              d_model=96, nhead=4, nlayers=2, lr=3e-4, wd=1e-4, gamma=1.5, beta=0.999,
              min_freq=3, top_k_vocab=8000, num_workers=0, y_col='PAY_AMT_bin',
              base_rate=0.03675, verbose=True, save_dir="checkpoints/seq_cls", run_name="default",
              resume=False, global_tokens=['exchange_등록', '캐시 상점', 'exchange_구매']):
    
    vocab_size = len(stoi)
    if verbose: logger.info(f"[INFO] Vocab size: {vocab_size}")

    ckpt_last = f"{save_dir}/{run_name}_last.pt"
    ckpt_best = f"{save_dir}/{run_name}_best.pt"

    tr_ld = make_length_sorted_loader(train_df, stoi, batch_size=batch_size, y_col=y_col, num_workers=num_workers, max_len=max_len)
    logger.info("Train loader created.")
    
    va_ds = SeqDataset(valid_df, stoi, y_col=y_col, max_len=max_len, global_tokens=global_tokens)
    va_ld = torch.utils.data.DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                                        collate_fn=partial(collate_batch, max_len=max_len),
                                        num_workers=num_workers, pin_memory=PIN_MEMORY, persistent_workers=False)
    logger.info("Validation loader created. Starting training.")

    model = SeqClassifier(
        vocab_size=vocab_size, d_model=d_model, nhead=nhead, nlayers=nlayers,
        p=0.1, base_rate=base_rate, max_len=max_len
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
    best_metric = -1.0
    if resume and os.path.exists(ckpt_last):
        try:
            ckpt = load_ckpt(ckpt_last, model=model, opt=opt, sched=sched)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_metric = ckpt.get("best_metric", -1.0)
            if verbose:
                logger.info(f"[resume] Loaded {ckpt_last} @epoch {start_epoch-1} | best PR-AUC={best_metric:.4f}")
        except Exception:
            logger.exception(f"Failed to load checkpoint from {ckpt_last}. Starting from scratch.")
            
    wait = 0
    
    for ep in range(start_epoch, epochs + 1):
        model.train()
        total_loss, n_samples = 0.0, 0
        
        # 💡 수정된 부분: tqdm을 직접 사용하고 logger에 명시적으로 기록
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
                opt.zero_grad(set_to_none=True)
                with amp_ctx:
                    position_ids = torch.arange(ev.shape[1], dtype=torch.long, device=DEVICE)
                    position_ids = position_ids.unsqueeze(0).expand_as(ev)
                    logits = model(ev, attention_mask=mask, global_attention_mask=global_mask, position_ids=position_ids)
                    loss = class_balanced_focal_with_logits(logits, y, beta=beta, gamma=gamma)
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

                # tqdm 상태 업데이트를 수동으로 기록합니다.
                if verbose:
                    t_bar.set_postfix(loss=total_loss / max(n_samples, 1))

            except Exception:
                logger.exception(f"Error during training epoch {ep}, batch {n_samples}:")
                raise  # 오류가 발생하면 프로그램을 중단합니다.

        tr_loss = total_loss / max(n_samples, 1)
        # 💡 수정된 부분: 검증 데이터에 대한 예측 및 평가 지표 계산
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
                out = model(ev, attention_mask=mask, global_attention_mask=global_mask)
                p = torch.sigmoid(out).squeeze(-1)
                all_p.append(p.detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())
        
        y_true_valid = np.concatenate(all_y)
        p_hat_valid = np.concatenate(all_p)
        
        # 💡 수정된 부분: 새로운 evaluate_cls 함수 호출
        metrics_valid = evaluate_cls(y_true_valid, p_hat_valid)
        score = metrics_valid['PR-AUC']

        if verbose:
            logger.info(f"[{ep:02d}] loss {tr_loss:.4f} | PR-AUC {metrics_valid['PR-AUC']:.4f} | ROC-AUC {metrics_valid['ROC-AUC']:.4f}")
            logger.info(f"        Acc: {metrics_valid['accuracy']:.4f} | Prec: {metrics_valid['precision']:.4f} | Rec: {metrics_valid['recall']:.4f} | F1: {metrics_valid['best_f1']:.4f} (at thr={metrics_valid['best_threshold']:.4f})")

        sched.step()
        
        save_ckpt(ckpt_last, model, opt, sched, ep, best_metric, stoi, {'max_len': max_len, 'd_model': d_model, 'nhead': nhead, 'nlayers': nlayers})
        if score > best_metric + 1e-4:
            best_metric = score
            save_ckpt(ckpt_best, model, opt, sched, ep, best_metric, stoi, {'max_len': max_len, 'd_model': d_model, 'nhead': nhead, 'nlayers': nlayers})
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose: logger.info(f"Early stopping at epoch {ep} (best PR-AUC={best_metric:.4f})")
                break

    @torch.no_grad()
    def collect_probs(df):
        ds = SeqDataset(df, stoi, y_col=y_col, max_len=max_len, global_tokens=global_tokens)
        ld = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_batch, max_len=max_len),
                                         num_workers=num_workers, pin_memory=PIN_MEMORY, persistent_workers=False)
        model.eval()
        all_pid, all_p, all_y = [], [], []
        for batch in ld:
            ev = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            global_mask = batch.get('global_attention_mask', None)
            if global_mask is not None:
                global_mask = global_mask.to(DEVICE)
            
            # 💡 수정된 부분: position_ids 생성 및 전달
            position_ids = torch.arange(ev.shape[1], dtype=torch.long, device=DEVICE)
            position_ids = position_ids.unsqueeze(0).expand_as(ev)
            
            p = torch.sigmoid(model(ev, attention_mask=mask, global_attention_mask=global_mask, position_ids=position_ids)).cpu().numpy()

            all_pid.extend(batch['ids'])
            all_p.append(p)
            all_y.append(batch['labels'].numpy())
        return np.array(all_pid), np.concatenate(all_p), np.concatenate(all_y)

    pid_v, p1_v, y_v = collect_probs(valid_df)
    best_thr, best_f1 = find_best_threshold(y_v, p1_v)
    yhat_v = (p1_v >= best_thr).astype(int)
    p0_v = 1.0 - p1_v
    
    # 💡 수정된 부분: best_thr 컬럼 추가
    pred_valid = pd.DataFrame({
        "PLAYERID": pid_v,
        "payer_p1_valid": p1_v,
        "payer_p0_valid": p0_v,
        "payer_pred_valid": yhat_v,
        "best_threshold": best_thr # 각 샘플에 동일한 임계값 저장
    })
    return model, stoi, pred_valid, best_thr, best_metric


def make_oof_meta_features(train_df, stoi, max_len=128, batch_size=512, epochs=12, patience=3,
                           d_model=48, nhead=3, nlayers=1, lr=3e-4, wd=1e-4, gamma=2.0, beta=0.999,
                           min_freq=3, top_k_vocab=8000, num_workers=0, y_col='PAY_AMT_bin',
                           n_splits=3, seed=2025, base_rate=0.03675,
                           global_tokens=['exchange_등록', '캐시 상점', 'exchange_구매']):
    y = train_df[y_col].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    parts = []
    fold_metrics = []
    
    with TqdmToLogger(enumerate(skf.split(train_df, y), 1),
                      total=n_splits, desc="OOF Folds") as t_bar:
        for fold, (tr_idx, va_idx) in t_bar:
            tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
            va_df = train_df.iloc[va_idx].reset_index(drop=True)

            logger.info(f"\n===== OOF Fold {fold}/{n_splits} =====")
            # ✅ 추가: 데이터셋 shape, positive rate, 평균 길이 출력
            logger.info(f"[DEBUG][Fold {fold}] tr_df shape: {tr_df.shape}, va_df shape: {va_df.shape}")
            logger.info(f"[DEBUG][Fold {fold}] tr_df pos_rate: {tr_df[y_col].mean():.5f}, va_df pos_rate: {va_df[y_col].mean():.5f}")
            
            # ACTION_DELTA 평균 길이 (시퀀스 길이 체크)
            try:
                tr_seq_len = tr_df['ACTION_DELTA'].apply(len).mean()
                va_seq_len = va_df['ACTION_DELTA'].apply(len).mean()
                logger.info(f"[DEBUG][Fold {fold}] mean seq_len (train): {tr_seq_len:.1f}, (val): {va_seq_len:.1f}")
            except Exception as e:
                logger.warning(f"[WARN] Could not compute seq length in fold {fold}: {e}")

            try:
                model_f, stoi_f, pred_va_f, thr_f, best_metric_f = train_one(
                    tr_df, va_df, stoi, max_len=max_len, batch_size=batch_size, epochs=epochs, patience=patience,
                    d_model=d_model, nhead=nhead, nlayers=nlayers, lr=lr, wd=wd, gamma=gamma, beta=beta,
                    min_freq=min_freq, top_k_vocab=top_k_vocab, num_workers=num_workers,
                    y_col=y_col, base_rate=base_rate, global_tokens=global_tokens, verbose=True
                )
            except Exception:
                logger.exception(f"Error during OOF Fold {fold}. Skipping this fold.")
                continue  # 오류 발생 시 해당 폴드를 건너뛰고 다음 폴드로 진행
            
            actual_labels = train_df.iloc[va_idx][[y_col, 'PLAYERID']].copy()
            pred_va_f = pd.merge(pred_va_f, actual_labels, on='PLAYERID', how='left')

            pred_va_f = pred_va_f.rename(columns={
                "payer_p1_valid": "payer_p1_oof",
                "payer_p0_valid": "payer_p0_oof",
                "payer_pred_valid": "payer_pred_oof"
            })
            # 💡 수정된 부분: y_col과 best_threshold 컬럼을 함께 추가
            parts.append(pred_va_f[["PLAYERID", "payer_p1_oof", "payer_p0_oof", "payer_pred_oof", y_col, "best_threshold"]])
            fold_metrics.append(best_metric_f)

            os.makedirs("outputs", exist_ok=True)
            pred_va_f.to_csv(f"outputs/oof_fold{fold}.csv", index=False)
            logger.info(f"[saved] outputs/oof_fold{fold}.csv")

            del model_f
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

    if not parts:
        logger.error("No OOF folds were successfully completed. Meta features cannot be created.")
        return None, fold_metrics
        
    meta_train_oof = pd.concat(parts, axis=0, ignore_index=True)
    if "PLAYERID" in meta_train_oof.columns:
        # ✅ 수정: OOF 결과 DataFrame을 PLAYERID로 정렬
        meta_train_oof = meta_train_oof.sort_values("PLAYERID").reset_index(drop=True)

    meta_train_oof.to_csv("outputs/meta_train_oof.csv", index=False)
    logger.info("[saved] outputs/meta_train_oof.csv")
    logger.info(f"\nOOF done. mean PR-AUC across folds: {np.nanmean(fold_metrics):.4f}")
    return meta_train_oof, fold_metrics

def run_inference(model, test_df, stoi, hp, final_thr):
    """테스트 데이터 추론을 수행하고 결과를 반환하는 함수"""
    logger.info("\n===== Test Data Inference =====")
    try:
        # ✅ 수정: 추론을 위해 test_df를 PLAYERID로 정렬하여 순서 보장
        test_df = test_df.sort_values("PLAYERID").reset_index(drop=True)
        logger.info("Test data loaded and sorted by PLAYERID successfully.")

        test_ds = SeqDataset(test_df, stoi, y_col='PAY_AMT_bin', max_len=hp['max_len'])
        test_ld = torch.utils.data.DataLoader(
            test_ds, batch_size=hp['batch_size'], shuffle=False,
            collate_fn=partial(collate_batch, max_len=hp['max_len']),
            num_workers=hp['num_workers'], pin_memory=PIN_MEMORY, persistent_workers=False
        )
        
        # 예측 수행
        model.eval()
        test_pids, test_probs, test_y = [], [], []
        with torch.no_grad():
            for batch in test_ld:
                ev = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                global_mask = batch.get('global_attention_mask', None)
                if global_mask is not None:
                    global_mask = global_mask.to(DEVICE)
                
                position_ids = torch.arange(ev.shape[1], dtype=torch.long, device=DEVICE)
                position_ids = position_ids.unsqueeze(0).expand_as(ev)
                
                y = batch['labels'].to(DEVICE)
                out = model(ev, attention_mask=mask, global_attention_mask=global_mask, position_ids=position_ids)
                p = torch.sigmoid(out).squeeze(-1)
                
                test_pids.extend(batch['ids'])
                test_probs.append(p.detach().cpu().numpy())
                test_y.append(y.detach().cpu().numpy())

        test_probs = np.concatenate(test_probs)
        test_y = np.concatenate(test_y)
        
        return test_pids, test_probs, test_y
    
    except FileNotFoundError:
        logger.error(f"테스트 데이터 파일 '{TEST_DATA_PATH}'을(를) 찾을 수 없습니다. 테스트 추론을 건너뜁니다.")
        return None, None, None
    except Exception:
        logger.exception("테스트 추론 과정에서 예상치 못한 오류가 발생했습니다.")
        return None, None, None

if __name__ == '__main__':
    TEST_MODE = False
    TRAIN_DATA_PATH = './seq/train_full_5days_seq.parquet' # 전체 학습 데이터 (train + val)
    TEST_DATA_PATH = './seq/test_df_5days_seq.parquet'

    try:
        if TEST_MODE and os.path.exists(TRAIN_DATA_PATH):
            logger.info(f"[TEST MODE] Loading sampled data from {TRAIN_DATA_PATH}")
            train_full = load_seq_parquet(TRAIN_DATA_PATH)
            logger.info("Data loaded from sampled file successfully.")
        else:
            train_df = load_seq_parquet('./seq/train_df_5days_seq.parquet')
            val_df = load_seq_parquet('./seq/val_df_5days_seq.parquet')
            train_df = ensure_binary_labels(train_df, y_col='PAY_AMT_bin')
            val_df = ensure_binary_labels(val_df, y_col='PAY_AMT_bin')
            train_full = pd.concat([train_df, val_df], ignore_index=True)
            del train_df, val_df
            logger.info("Data loaded and preprocessed successfully.")
            
            logger.info(f"[DEBUG] train_full shape: {train_full.shape}")
            logger.info(f"[DEBUG] train_full columns: {list(train_full.columns)}")
            logger.info(f"[DEBUG] Positive rate: {train_full['PAY_AMT_bin'].mean():.5f}")

            if TEST_MODE:
                logger.info("\n===== TEST MODE ENABLED: Sampling 1000 items =====")
                if len(train_full) > 1000:
                    train_full = train_full.sample(n=1000, random_state=2025).reset_index(drop=True)
                logger.info(f"Sampled train_full shape: {train_full.shape}")

                df_to_save = train_full.copy()
                df_to_save['ACTION_DELTA'] = df_to_save['ACTION_DELTA'].apply(json.dumps)
                os.makedirs(os.path.dirname(TRAIN_DATA_PATH), exist_ok=True)
                df_to_save.to_parquet(TRAIN_DATA_PATH, index=False)
                logger.info(f"Sampled data saved to {TRAIN_DATA_PATH}")
    
    except FileNotFoundError:
        logger.error("Parquet 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        sys.exit(1)
    except Exception:
        logger.exception("An unexpected error occurred during data loading.")
        sys.exit(1)

    set_seed(2025)

    pos_rate = float(train_full['PAY_AMT_bin'].mean())
    neg = (1 - pos_rate) * len(train_full)
    pos = pos_rate * len(train_full)
    pos_weight_val = float(neg / max(1.0, pos))
    base_rate_val = pos_rate

    logger.info(f"[Auto] (train_full) pos_weight≈{pos_weight_val:.2f}, base_rate≈{base_rate_val:.5f}")

    # HP 정의 부분을 삭제하고, base_rate만 업데이트
    HP['base_rate'] = base_rate_val

    try:
        stoi, _ = build_event_vocab(train_full, min_freq=HP['min_freq'], top_k=HP['top_k_vocab'])
        vocab_size = len(stoi)
        logger.info(f"[INFO] Global Vocab size: {vocab_size}")

        logger.info(f"[DEBUG] HP.max_len = {HP['max_len']}, HP.min_freq = {HP['min_freq']}, HP.top_k_vocab = {HP['top_k_vocab']}")
        logger.info(f"[DEBUG] stoi sample: {dict(list(stoi.items())[:10])}")

        meta_train_oof, fold_metrics = make_oof_meta_features(train_full, stoi=stoi, **HP)
        
        logger.info(f"OOF PR-AUC(mean): {np.nanmean(fold_metrics):.4f}")

        if meta_train_oof is not None:
            # 최종 모델 학습 및 테스트 데이터 예측
            logger.info("\n===== Final Model Training =====")
            
            # 💡 수정: train_one에 필요 없는 인자 제거
            final_hp = HP.copy()
            final_hp.pop('n_splits', None)
            final_hp.pop('seed', None) # seed 인자 제거
            
            # ✅ 수정: final_thr 변수를 반환받아 명확하게 사용
            final_model, _, _, final_thr, _ = train_one(
                train_full, train_full, stoi, **final_hp, verbose=True
            )
            
            # 여기서 final_thr 변수에 최종 임계값이 저장됩니다.
            logger.info(f"Final model trained. Best threshold from full training data: {final_thr:.4f}")
            
            # 테스트 추론 단계
            test_df = load_seq_parquet(TEST_DATA_PATH)
            test_df = ensure_binary_labels(test_df, y_col='PAY_AMT_bin')

            test_pids, test_probs, test_y = run_inference(final_model, test_df, stoi, HP, final_thr)
            
            if test_pids is not None:
                # 💡 추가: 테스트 데이터에 대한 평가 지표 계산
                test_metrics = evaluate_cls(test_y, test_probs)
                logger.info(f"\n===== Test Set Performance =====")
                logger.info(f"PR-AUC {test_metrics['PR-AUC']:.4f} | ROC-AUC {test_metrics['ROC-AUC']:.4f}")
                logger.info(f"Acc: {test_metrics['accuracy']:.4f} | Prec: {test_metrics['precision']:.4f} | Rec: {test_metrics['recall']:.4f} | F1: {test_metrics['best_f1']:.4f} (at thr={test_metrics['best_threshold']:.4f})")
                
                # 결과 DataFrame 생성 및 저장
                predictions_df = pd.DataFrame({
                    "PLAYERID": test_pids,
                    "payer_p1_test": test_probs,
                    # ✅ 수정: final_thr 변수를 사용하여 예측
                    "payer_pred_test": (test_probs >= final_thr).astype(int),
                    "best_threshold_used": final_thr,
                    "PAY_AMT_bin": test_y # ✅ 추가: 평가를 위해 실제 레이블을 DataFrame에 포함
                })
                
                # 디렉토리 존재 여부 확인 후 저장
                output_dir = "outputs"
                os.makedirs(output_dir, exist_ok=True)
                
                # ✅ 수정: float_format을 지정하여 정밀도 손실 최소화
                predictions_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False, float_format='%.8f')
                logger.info("[saved] outputs/test_predictions.csv")

                # ✅ 추가: 저장된 CSV 파일을 다시 불러와서 평가 지표를 재확인
                logger.info("\n===== Verifying Metrics from CSV =====")
                loaded_df = pd.read_csv(os.path.join(output_dir, "test_predictions.csv"))
                
                # 정렬되지 않은 원본 test_df와 순서가 일치하는지 확인하기 위해 PLAYERID를 기준으로 merge
                merged_df = pd.merge(test_df[['PLAYERID', 'PAY_AMT_bin']], loaded_df, on='PLAYERID', how='left')
                
                csv_y_true = merged_df['PAY_AMT_bin_x'].values
                csv_p_hat = merged_df['payer_p1_test'].values
                csv_metrics = evaluate_cls(csv_y_true, csv_p_hat)
                
                logger.info(f"CSV PR-AUC: {csv_metrics['PR-AUC']:.4f} | ROC-AUC: {csv_metrics['ROC-AUC']:.4f}")
                logger.info(f"CSV F1: {csv_metrics['best_f1']:.4f} (at thr={csv_metrics['best_threshold']:.4f})")

                # ✅ 추가: 최종 모델 학습 시 얻은 임계값(final_thr)을 사용하여 리포트 생성
                classification_report_at_threshold(csv_y_true, csv_p_hat, final_thr, title="[Final Report from CSV]")
                
    except Exception:
        logger.exception("An unrecoverable error occurred during the main process.")
        sys.exit(1)