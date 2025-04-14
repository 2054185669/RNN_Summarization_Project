"""
Step 5: 推理与摘要生成脚本

- 加载保存的模型
- 加载词表和测试文章
- 生成摘要并输出

运行方式：
    python inference.py
"""

import torch
import torch.nn.functional as F
import json
import os
import pickle  # [新增] 用于加载词表
from model import Encoder, Decoder, Seq2Seq
from preprocess import build_vocab, encode_text

# 设置参数
EMB_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
MAX_LEN = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [修改] 加载词表
try:
    with open("../vocab/vocab.pkl", "rb") as f:
        vocab_data = pickle.load(f)
    vocab = vocab_data['vocab']
    vocab_inv = vocab_data['vocab_inv']
    print(f"[INFO] 成功加载词表 | 词表大小: {len(vocab)} | 特殊标记: {vocab_data['special_tokens']}")
except FileNotFoundError:
    raise SystemExit("[ERROR] 未找到词表文件，请先运行train.py生成vocab.pkl")

# [新增] 词表验证
def validate_vocab(vocab):
    required_tokens = ['<UNK>', '<SOS>', '<EOS>']
    for token in required_tokens:
        if token not in vocab:
            raise ValueError(f"词表中缺少必要标记: {token}")

validate_vocab(vocab)

# 加载数据
with open("../data/cnn_dm_sample.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 加载模型
encoder = Encoder(len(vocab), EMB_DIM, HIDDEN_DIM, NUM_LAYERS)
decoder = Decoder(len(vocab), EMB_DIM, HIDDEN_DIM, NUM_LAYERS)
model = Seq2Seq(encoder, decoder, device).to(device)
model.load_state_dict(torch.load("../models/rnn_summarizer.pth", map_location=device))
model.eval()

def summarize(text):
    # [新增] 调试信息
    print("\n=== 输入验证 ===")
    test_words = ["the", "potter", "harry", "<UNK>"]
    print("测试词ID映射:", {w: vocab.get(w, vocab['<UNK>']) for w in test_words})
    
    tokens = text.lower().split()[:10]  # 只检查前10个token
    print("实际输入tokens示例:", tokens)
    print("对应IDs:", [vocab.get(w, vocab['<UNK>']) for w in tokens])
    
    # 原有处理逻辑
    tokens = text.lower().split()
    indices = [vocab.get(w, vocab['<UNK>']) for w in tokens][:MAX_LEN]
    src_tensor = torch.tensor(indices).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [vocab['<SOS>']]
    for _ in range(MAX_LEN):
        trg_tensor = torch.tensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        if pred_token == vocab['<EOS>']:
            break
        trg_indexes.append(pred_token)

    trg_tokens = [vocab_inv.get(idx, '<UNK>') for idx in trg_indexes[1:]]
    return ' '.join(trg_tokens)

# 测试一条文章
example = raw_data[0]["article"]
summary = summarize(example)
print("\n=== 结果输出 ===")
print("原文：", example[:500], "...\n")
print("生成的摘要：", summary)