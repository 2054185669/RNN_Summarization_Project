"""
Step 1: 下载并准备 CNN/DailyMail 数据集（小样本）

本脚本完成以下工作：
1. 使用 Hugging Face 的 `datasets` 库下载 CNN/DailyMail 数据集。
2. 抽取训练集中前 1% 的样本（约 2800 条数据）。
3. 每条数据包含：
   - `article`: 新闻正文
   - `summary`: 人工生成的摘要
4. 保存为 JSON 文件：`data/cnn_dm_sample.json`

使用前请确保已安装 `datasets` 库：
    pip install datasets

运行本脚本：
    python extract_small_sample.py
"""
from datasets import load_dataset
import json
import os

# 创建 data 目录（如果不存在）
os.makedirs("../data", exist_ok=True)

print("[INFO] 正在加载 CNN/DailyMail 数据集（前 1%）...")
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")

# 抽取文章和摘要字段
sample = [{
    "article": item["article"],
    "summary": item["highlights"]
} for item in dataset]

# 保存为 JSON 文件
save_path = "../data/cnn_dm_sample.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)

print(f"[INFO] 共保存样本数量: {len(sample)} 条")
print(f"[INFO] 数据保存位置: {save_path}")