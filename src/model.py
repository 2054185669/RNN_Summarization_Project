# RNN_Summarization_Project/src/model.py

"""
Step 3: 模型定义（Encoder + Decoder + Seq2Seq）

- Encoder：双向 LSTM 编码文章
- Decoder：LSTM 解码生成摘要
- Seq2Seq：整体组合结构
"""

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # 合并双向输出

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        # 拼接双向 hidden
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.fc(hidden).unsqueeze(0)
        cell = torch.cat((cell[-2], cell[-1]), dim=1)
        cell = self.fc(cell).unsqueeze(0)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[:, 0]  # <SOS>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            input = trg[:, t]  # Teacher forcing

        return outputs
