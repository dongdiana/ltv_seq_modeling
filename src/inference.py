import os
import sys
import numpy as np
import pandas as pd
import torch
from functools import partial

# 필요한 모듈을 현재 경로에서 찾도록 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    setup_logger,
    logger,
    load_seq_parquet,
    load_ckpt
)
from src.datasets import (
    SeqDatasetInfer,
    collate_infer,
)
from src.models import SeqClassifier

# CUDA 사용 가능 여부 확인 및 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = DEVICE == "cuda"
logger.info(f"Using device: {DEVICE}")

# 로거 설정
log_file_path = os.path.join(os.path.dirname(__file__), "logs/inference.log")
logger = setup_logger("inference_logger", log_file_path)

def run_inference():
    # 최종 모델 체크포인트 경로
    CKPT_PATH = "checkpoints/seq_cls/default_last.pt"
    
    # ========== 체크포인트 파일에서 단어 사전 및 하이퍼파라미터 로드 ==========
    try:
        # 단어 사전과 하이퍼파라미터만 먼저 로드하여 모델을 정의합니다.
        ckpt_info = load_ckpt(CKPT_PATH)
        
        # 체크포인트에서 필요한 정보 추출
        stoi = ckpt_info['stoi']
        hp = ckpt_info['hp']
        
        # 'batch_size' 키가 누락되었으므로 직접 추가합니다.
        hp['batch_size'] = 64
        
        logger.info(f"체크포인트 파일 '{CKPT_PATH}'에서 단어 사전과 하이퍼파라미터를 성공적으로 불러왔습니다.")
        logger.info(f"단어 사전 크기: {len(stoi)}")
        logger.info(f"사용된 하이퍼파라미터: {hp}")
        
    except FileNotFoundError:
        logger.error(f"오류: 체크포인트 파일을 찾을 수 없습니다. 경로를 확인하세요: {CKPT_PATH}")
        return
    except Exception:
        logger.exception(f"오류: 체크포인트 파일 로드 중 문제가 발생했습니다.")
        return

    # ========== 모델 정의 및 최종 모델 체크포인트 로드 ==========
    final_model = SeqClassifier(
        vocab_size=len(stoi), d_model=hp['d_model'], nhead=hp['nhead'], nlayers=hp['nlayers'],
        p=0.1, base_rate=0.03675, max_len=hp['max_len']
    ).to(DEVICE)
    
    # 모델 가중치를 로드합니다.
    load_ckpt(CKPT_PATH, model=final_model)
    logger.info("모델 가중치를 성공적으로 로드했습니다.")
    
    final_model.eval()
    
    # 로그에서 확인된 최적의 임계값
    final_thr = 0.4548
    
    # ========== 테스트 데이터 추론 및 저장 ==========
    TEST_DATA_PATH = './seq/test_df_5days_seq.parquet'

    try:
        logger.info(f"\n===== 테스트 데이터 추론 시작 =====")
        test_df = load_seq_parquet(TEST_DATA_PATH)
        
        # 테스트 데이터셋 및 데이터로더 준비
        test_ds = SeqDatasetInfer(test_df, stoi, max_len=hp['max_len'])
        test_ld = torch.utils.data.DataLoader(
            test_ds, batch_size=hp['batch_size'], shuffle=False,
            collate_fn=partial(collate_infer, max_len=hp['max_len']),
            num_workers=hp.get('num_workers', 0), pin_memory=PIN_MEMORY, persistent_workers=False
        )
        
        # 예측 수행
        test_pids, test_probs = [], []
        with torch.no_grad():
            for batch in test_ld:
                ev = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                global_mask = batch.get('global_attention_mask', None)
                if global_mask is not None:
                    global_mask = global_mask.to(DEVICE)
                
                out = final_model(ev, attention_mask=mask, global_attention_mask=global_mask)
                p = torch.sigmoid(out).squeeze(-1)
                test_pids.extend(batch['ids'])
                test_probs.append(p.detach().cpu().numpy())
        
        test_probs = np.concatenate(test_probs)
        
        # 결과 DataFrame 생성 및 저장
        predictions_df = pd.DataFrame({
            "PLAYERID": test_pids,
            "payer_p1_test": test_probs,
            "payer_pred_test": (test_probs >= final_thr).astype(int),
            "best_threshold_used": final_thr
        })
        
        os.makedirs("outputs", exist_ok=True)
        output_file_path = "outputs/test_predictions.csv"
        predictions_df.to_csv(output_file_path, index=False)
        logger.info(f"[saved] {output_file_path}")
        
        logger.info("테스트 데이터 추론 및 저장이 완료되었습니다. outputs/test_predictions.csv 파일을 확인해 주세요.")
        
    except FileNotFoundError:
        logger.error(f"오류: 테스트 데이터를 찾을 수 없습니다. 경로를 확인하세요: {TEST_DATA_PATH}")
    except Exception:
        logger.exception("오류: 추론 과정에서 문제가 발생했습니다.")

if __name__ == '__main__':
    run_inference()