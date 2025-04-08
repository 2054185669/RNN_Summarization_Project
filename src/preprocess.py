# RNN_Summarization_Project/src/preprocess.py

"""
Step 2: 数据预处理模块

功能：
1. 加载 `cnn_dm_sample.json` 样本
2. 分词（简单空格分词）
3. 构建词表（限制大小，如 5000）
4. 将文本转换为索引序列
5. 提供 Dataset 类供训练使用

运行前请确保 `extract_small_sample.py` 已成功运行。
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

class TextSummaryDataset(Dataset):
    def __init__(self, json_path, vocab_size=5000, max_len=100):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.max_len = max_len
        self.vocab = self.build_vocab(self.data, vocab_size)
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.vocab["<PAD>"]

    def build_vocab(self, data, vocab_size):
        counter = Counter()
        for item in data:
            counter.update(item['article'].split())
            counter.update(item['summary'].split())

        most_common = counter.most_common(vocab_size - len(SPECIAL_TOKENS))
        vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        for idx, (word, _) in enumerate(most_common, len(SPECIAL_TOKENS)):
            vocab[word] = idx
        return vocab

    def encode(self, text):
        tokens = text.split()
        indices = [self.vocab.get(w, self.vocab['<UNK>']) for w in tokens]
        indices = [self.vocab['<SOS>']] + indices + [self.vocab['<EOS>']]
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = self.encode(item['article'])
        tgt = self.encode(item['summary'])
        return src, tgt

    def __len__(self):
        return len(self.data)

# 示例用法
if __name__ == "__main__":
    dataset = TextSummaryDataset("../data/cnn_dm_sample.json")
    print(f"样本数: {len(dataset)}")
    print(f"词表大小: {dataset.vocab_size}")
    print("示例数据 (前5个词索引):")
    print("Article:", dataset[0][0][:5])
    print("Summary:", dataset[0][1][:5])
