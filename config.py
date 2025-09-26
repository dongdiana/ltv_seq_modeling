# config.py
# 이 파일에 모든 하이퍼파라미터 설정을 담습니다.

HP = dict(
    # 시퀀스 최대 길이를 줄여 각 배치의 연산량을 감소시킵니다.
    max_len=128, 
    # 배치 크기를 늘리면 GPU 활용률이 높아져 학습 속도가 빨라질 수 있습니다.
    batch_size=128, 
    # 에폭 수를 줄여 전체 학습 시간을 단축합니다.
    epochs=1,
    # 조기 종료(Early stopping) 대기 횟수를 줄여 검증 지표가 개선되지 않으면 빠르게 종료합니다.
    patience=1, 
    # 모델의 크기(차원, 헤드 수, 레이어 수)를 줄여 연산량을 감소시킵니다.
    d_model=16,
    nhead=1,
    nlayers=2,
    lr=2e-4,
    wd=2e-4,
    y_col='PAY_AMT',
    min_freq=3,
    top_k_vocab=15,
    # 데이터 로딩에 사용되는 워커 수를 줄여 오버헤드를 감소시킵니다.
    num_workers=4,
    base_rate=0.03675,
    
    transformation_mode='log1p', 
    loss_mode='mae',              
    huber_delta=1.0,               
    regression_model_type='mlp',   
)