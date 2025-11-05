# config.py
# 이 파일에 모든 하이퍼파라미터 설정을 담습니다.

HP = dict(
    max_len=4096, 
    batch_size=32, 
    epochs=3,
    patience=1, 
    d_model=16,
    nhead=2,
    nlayers=2,
    lr=2e-4,
    wd=2e-4,
    y_col='PAY_AMT',
    min_freq=3,
    top_k_vocab=25,
    num_workers=4,
    base_rate=0.03675,
    
    transformation_mode=None, 
    loss_mode='mae',              
    huber_delta=1.0,               
    regression_model_type='mlp',   
)