import os
import json
import random

# ===== 参数配置 =====
SAMPLE_SIZE = 500  # 采样数量，可在这里直接修改

# 路径配置
base_dir = "/mnt/petrelfs/panjiabao/huxiaobin/yunhang/Code/medlab/data/slake"
input_path = os.path.join(base_dir, "test.jsonl")
output_path = os.path.join(base_dir, "new_test.jsonl")

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                pass
    return data

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main(sample_size):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件未找到: {input_path}")

    # 读取数据
    all_records = read_jsonl(input_path)
    total = len(all_records)

    # 确定实际要抽取的数量
    k = min(sample_size, total)

    # 随机采样（不放回）
    sampled = random.sample(all_records, k) if k > 0 else []

    # 写入新文件
    write_jsonl(output_path, sampled)

    print(f"总样本数：{total}，已抽取：{k}，输出文件：{output_path}")

if __name__ == "__main__":
    main(SAMPLE_SIZE)