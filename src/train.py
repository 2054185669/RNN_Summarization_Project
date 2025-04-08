# RNN_Summarization_Project/src/train.py

"""
Step 4: 模型训练脚本

- 加载数据集和词表
- 构建模型
- 执行训练循环

运行方式：
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import json
from preprocess import TextSummaryDataset, build_vocab, encode_text
from model import Encoder, Decoder, Seq2Seq

# 超参数
BATCH_SIZE = 32
EMB_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
EPOCHS = 10
MAX_LEN = 128

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载样本数据
with open("../data/cnn_dm_sample.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 构建词表和编码数据
vocab = build_vocab(raw_data, max_size=5000)
dataset = TextSummaryDataset(raw_data, vocab, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 构建模型
encoder = Encoder(len(vocab), EMB_DIM, HIDDEN_DIM, NUM_LAYERS)
decoder = Decoder(len(vocab), EMB_DIM, HIDDEN_DIM, NUM_LAYERS)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])

# 训练循环
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

# 保存模型
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/rnn_summarizer.pth")
print("[INFO] 模型保存到 ../models/rnn_summarizer.pth")
