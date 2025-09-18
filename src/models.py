import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel

class SeqClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nlayers, p=0.1, base_rate=0.0314, max_len=4096):
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
        self.fc = nn.Linear(d_model, 1)
        self.base_rate = base_rate

    def forward(self, input_ids, attention_mask=None, global_attention_mask=None, position_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            position_ids=position_ids
        )
        pooled = out.last_hidden_state.mean(dim=1)
        logits = self.fc(pooled)
        return logits.squeeze(-1)