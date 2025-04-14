# RNN_Summarization_Project/src/preprocess.py

"""
Step 2:文本摘要任务预处理脚本：
1. 加载小样本 JSON 数据（cnn_dm_sample.json）
2. 构建词汇表（vocab）
3. 文本编码为索引序列
4. 构造 PyTorch Dataset 用于训练

使用方式：
    from preprocess import TextSummaryDataset, build_vocab, encode_text
"""

import json
import torch
from torch.utils.data import Dataset
from collections import Counter

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<SOS>": 2,
    "<EOS>": 3
}

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def build_vocab(data, max_size=5000):
    counter = Counter()
    for item in data:
        tokens = item["article"].split()
        counter.update(tokens)
    
    most_common = counter.most_common(max_size - 4)  # 预留4个特殊符号
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<SOS>": 2,
        "<EOS>": 3
    }
    for i, (word, _) in enumerate(most_common, start=4):
        vocab[word] = i
    return vocab

def encode_text(text, vocab, max_len=512):
    tokens = text.lower().split()
    indices = [vocab.get("<SOS>", 2)]
    for token in tokens:
        indices.append(vocab.get(token, vocab.get("<UNK>", 1)))
        if len(indices) >= max_len - 1:
            break
    indices.append(vocab.get("<EOS>", 3))
    return indices

def pad_sequence(seq, max_len, pad_idx=0):
    return seq + [pad_idx] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]

class TextSummaryDataset(Dataset):
    def __init__(self, samples, vocab, max_len=512):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        src = encode_text(sample["article"], self.vocab, self.max_len)
        trg = encode_text(sample["summary"], self.vocab, self.max_len)
        return torch.tensor(pad_sequence(src, self.max_len)), torch.tensor(pad_sequence(trg, self.max_len))
