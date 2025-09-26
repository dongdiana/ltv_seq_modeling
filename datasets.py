import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from functools import partial
from utils import encode_sequence, PIN_MEMORY, transform_target, inverse_transform_target, load_seq_parquet
import numpy as np

# datasets.py의 collate_batch 함수 (우측 패딩 적용)
def collate_batch(batch, max_len=None):
    ids = [b["ids"] for b in batch]
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.float)
    tabular_features = torch.tensor([b["tabular_features"] for b in batch], dtype=torch.float)
    
    trimmed_input_ids = []
    trimmed_global_masks = []
    for b in batch:
        current_input_ids = b["input_ids"]
        current_global_mask = b["global_attention_mask"]
        
        if max_len is not None and len(current_input_ids) > max_len:
            # max_len을 초과하는 시퀀스는 뒤에서부터 자름
            trimmed_input_ids.append(current_input_ids[-max_len:])
            trimmed_global_masks.append(current_global_mask[-max_len:])
        else:
            trimmed_input_ids.append(current_input_ids)
            trimmed_global_masks.append(current_global_mask)

    L = max(len(ids) for ids in trimmed_input_ids) if trimmed_input_ids else 0
    B = len(batch)
    
    input_ids = torch.zeros(B, L, dtype=torch.long)
    attn_mask = torch.zeros(B, L, dtype=torch.bool)
    global_mask = torch.zeros(B, L, dtype=torch.long)

    for i, b_ids in enumerate(trimmed_input_ids):
        l = len(b_ids)
        # 우측 패딩: 시퀀스를 왼쪽 정렬하고 나머지를 0으로 채움
        input_ids[i, :l] = b_ids
        attn_mask[i, :l] = 1
        global_mask[i, :l] = trimmed_global_masks[i]
        
    return {
        "ids": ids,
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "global_attention_mask": global_mask,
        "labels": labels,
        "tabular_features": tabular_features,
    }

# collate_infer 함수 수정: 마찬가지로 우측 패딩 적용
def collate_infer(batch, max_len=None):
    ids = [b["ids"] for b in batch]
    tabular_features = torch.tensor([b["tabular_features"] for b in batch], dtype=torch.float)

    trimmed_input_ids = []
    trimmed_global_masks = []
    for b in batch:
        current_input_ids = b["input_ids"]
        current_global_mask = b["global_attention_mask"]

        if max_len is not None and len(current_input_ids) > max_len:
            trimmed_input_ids.append(current_input_ids[-max_len:])
            trimmed_global_masks.append(current_global_mask[-max_len:])
        else:
            trimmed_input_ids.append(current_input_ids)
            trimmed_global_masks.append(current_global_mask)
    
    L = max(len(ids) for ids in trimmed_input_ids) if trimmed_input_ids else 0
    B = len(batch)
    
    input_ids = torch.zeros(B, L, dtype=torch.long)
    attn_mask = torch.zeros(B, L, dtype=torch.bool)
    global_mask = torch.zeros(B, L, dtype=torch.long)

    for i, b_ids in enumerate(trimmed_input_ids):
        l = len(b_ids)
        input_ids[i, :l] = b_ids
        attn_mask[i, :l] = 1
        global_mask[i, :l] = trimmed_global_masks[i]

    return {
        "ids": ids,
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "global_attention_mask": global_mask,
        "tabular_features": tabular_features,
    }


class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, stoi: dict, seq_col='ACTION_DELTA',
                 id_col='PLAYERID', y_col='PAY_AMT', max_len: int = None,
                 global_tokens: list = None, transformation_mode='log1p', tabular_data=None):
        self.id = df[id_col].tolist()
        
        # 타겟 변수 변환을 utils 함수로 분리
        self.y_raw = df[y_col].astype(float).values
        self.y, self.transform_params = transform_target(self.y_raw, transformation_mode)
        
        self.seqs = df[seq_col].tolist()
        self.stoi = stoi
        self.max_len = max_len
        self.global_tokens = global_tokens if global_tokens else []
        self.tabular_data = tabular_data.set_index(id_col) if tabular_data is not None else None


    def __len__(self):
        return len(self.id)

    def __getitem__(self, i):
        seq = self.seqs[i]
        seq = [ev[0] for ev in seq if isinstance(ev, (list, tuple)) and len(ev) > 0]
        ids = encode_sequence(seq, self.stoi)
        if self.max_len is not None and len(ids) > self.max_len:
            ids = ids[-self.max_len:]
        
        # ========== 수정된 부분: global_attention_mask 생성 로직 ==========
        global_id_set = {self.stoi[ev] for ev in self.global_tokens if ev in self.stoi}
        # ids와 동일한 크기의 0으로 채워진 텐서를 만듭니다.
        global_mask = torch.zeros(ids.size(), dtype=torch.long)
        # ids에 있는 각 토큰이 global_id_set에 포함되는지 확인하여 마스크를 설정합니다.
        global_mask[torch.isin(ids, torch.tensor(list(global_id_set)))] = 1
        # =========================================================

        # 태블러 데이터 추가
        player_id = self.id[i]
        if self.tabular_data is not None:
            tabular_features = self.tabular_data.loc[player_id].values.astype(np.float32)
        else:
            tabular_features = np.zeros(0, dtype=np.float32)

        return {
            "ids": self.id[i],
            "input_ids": ids,
            "global_attention_mask": global_mask,
            "labels": torch.tensor(self.y[i], dtype=torch.float),
            "tabular_features": tabular_features,
        }

class SeqDatasetInfer(Dataset):
    def __init__(self, df: pd.DataFrame, stoi: dict,
                 seq_col='ACTION_DELTA', id_col='PLAYERID',
                 max_len: int = None, global_tokens: list = None, tabular_data=None):
        self.id = df[id_col].tolist()
        self.seqs = df[seq_col].tolist()
        self.stoi = stoi
        self.max_len = max_len
        self.global_tokens = global_tokens if global_tokens else []
        self.tabular_data = tabular_data.set_index(id_col) if tabular_data is not None else None


    def __len__(self):
        return len(self.id)

    def __getitem__(self, i):
        seq = self.seqs[i]
        seq = [ev[0] for ev in seq if isinstance(ev, (list, tuple)) and len(ev) > 0]
        ids = encode_sequence(seq, self.stoi)
        if self.max_len is not None and len(ids) > self.max_len:
            ids = ids[-self.max_len:]
            
        global_mask = torch.zeros_like(ids)
        global_id_set = {self.stoi[ev] for ev in self.global_tokens if ev in self.stoi}
        for gi, token in enumerate(ids):
            if token.item() in global_id_set:
                global_mask[gi] = 1
        
        # 태블러 데이터 추가
        player_id = self.id[i]
        if self.tabular_data is not None:
            tabular_features = self.tabular_data.loc[player_id].values.astype(np.float32)
        else:
            tabular_features = np.zeros(0, dtype=np.float32)

        return {
            "ids": self.id[i],
            "input_ids": ids,
            "global_attention_mask": global_mask,
            "tabular_features": tabular_features,
        }
        
class BucketSampler(BatchSampler):
    def __init__(self, data_source, batch_size, bucket_size_multiplier=100):
        self.sampler = RandomSampler(data_source)
        self.batch_size = batch_size
        self.data_source = data_source
        self.bucket_size = batch_size * bucket_size_multiplier
        self.buckets = self.create_buckets()

    def create_buckets(self):
        lengths = [len(s) if isinstance(s, list) else 0 for s in self.data_source.seqs]
        df = pd.DataFrame({'idx': range(len(lengths)), 'len': lengths})
        df = df.sort_values('len').reset_index(drop=True)
        buckets = []
        for i in range(0, len(df), self.bucket_size):
            bucket_indices = df.iloc[i:i + self.bucket_size]['idx'].tolist()
            np.random.shuffle(bucket_indices)
            buckets.extend([bucket_indices[j:j + self.batch_size] for j in range(0, len(bucket_indices), self.batch_size)])
        np.random.shuffle(buckets)
        return buckets

    def __iter__(self):
        return iter(self.buckets)

    def __len__(self):
        return len(self.buckets)


def make_length_sorted_loader(df, stoi, batch_size=512, max_len=None, y_col='PAY_AMT', num_workers=4, transformation_mode='log1p', tabular_data=None):
    ds = SeqDataset(df, stoi, y_col=y_col, max_len=max_len, transformation_mode=transformation_mode, tabular_data=tabular_data)
    bucket_sampler = BucketSampler(ds, batch_size=batch_size)
    
    return DataLoader(
        ds,
        batch_sampler=bucket_sampler,
        collate_fn=partial(collate_batch, max_len=max_len),
        num_workers=num_workers,
        pin_memory=True
    )

def make_loader_from_dataset(ds, batch_size, num_workers=4):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_infer, max_len=ds.max_len),
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=(num_workers > 0)
    )