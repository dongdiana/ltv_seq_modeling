import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x는 (batch_size, sequence_length, feature_dim) 형태여야 함
        lstm_out, _ = self.lstm(x)
        # 마지막 시점의 히든 스테이트를 사용
        last_hidden_state = lstm_out[:, -1, :]
        return self.fc(last_hidden_state).squeeze(-1)

class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_size)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)

class LongformerRegressor(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nlayers, p=0.1, base_rate=0.0314, max_len=4096, tabular_input_dim=0, regression_model_type='mlp'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        config = LongformerConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=nlayers,
            num_attention_heads=nhead,
            intermediate_size=d_model * 4,
            attention_probs_dropout_prob=p,
            hidden_dropout_prob=p,
            max_position_embeddings=max_len + 2,
            padding_idx=0,
        )
        self.encoder = LongformerModel(config)
        
        self.regression_model_type = regression_model_type
        
        # 태블러 데이터가 있다면 입력 차원을 추가
        total_input_dim = d_model + tabular_input_dim

        if regression_model_type == 'mlp':
            self.regressor = MLPRegressor(total_input_dim, d_model * 2)
        elif regression_model_type == 'lstm':
            # Longformer 출력과 태블러 데이터를 결합하여 LSTM 입력으로 사용
            self.regressor = LSTMRegressor(total_input_dim, d_model * 2)
        else:
            raise ValueError(f"Unsupported regression_model_type: {regression_model_type}")

    def forward(self, input_ids, attention_mask=None, global_attention_mask=None, tabular_features=None, position_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            position_ids=position_ids
        )
        # [CLS] 토큰의 히든 스테이트를 가져옴
        pooled_output = out.last_hidden_state[:, 0]
        
        if tabular_features is not None and tabular_features.size(1) > 0:
            # 시퀀스 임베딩과 태블러 임베딩을 결합
            combined_features = torch.cat((pooled_output, tabular_features), dim=1)
        else:
            combined_features = pooled_output
        
        # 선택된 회귀 모델을 사용
        if self.regression_model_type == 'mlp':
            logits = self.regressor(combined_features)
        elif self.regression_model_type == 'lstm':
            # LSTM은 시퀀스 입력을 받으므로, pooled_output을 재구성
            # 여기서는 Longformer의 출력 시퀀스 전체를 사용하는 것이 일반적이지만,
            # 간단한 결합을 위해 Longformer의 모든 시퀀스 출력과 태블러 데이터를 각 시점마다 concat하여 LSTM에 전달하는 방식도 가능함
            # 이 코드에서는 pooled_output과 tabular_features를 단순 결합하여 MLP에 넣는 방식을 채택
            # LSTM을 사용하려면 `out.last_hidden_state`와 태블러 데이터를 결합해야 함
            logits = self.regressor(combined_features.unsqueeze(1))
            
        return logits.squeeze(-1)