#!/usr/bin/env python3
"""
极简评估脚本 - 字符计数任务
用法: python eval_simple.py --model_path <模型路径>
"""
import re
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


def extract_answer(text):
    """从模型输出中提取 \\boxed{} 中的答案"""
    match = re.search(r'\\boxed\{(\d+)\}', text)
    if match:
        return int(match.group(1))
    return None


def evaluate(model_path, test_file='./data/char_count/sft/test.parquet', max_samples=None):
    # 加载模型和tokenizer
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.eval()

    # 加载测试数据
    print(f"Loading test data from {test_file}...")
    df = pd.read_parquet(test_file)
    if max_samples:
        df = df.head(max_samples)

    # 评估
    correct = 0
    total = 0

    print(f"Evaluating on {len(df)} samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        messages = row['messages']

        # 提取问题和正确答案
        question = messages[0]['content']
        gt_answer_text = messages[1]['content']
        gt_answer = extract_answer(gt_answer_text)

        if gt_answer is None:
            continue

        # 准备输入
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred_answer = extract_answer(response)

        # 计算准确率
        total += 1
        if pred_answer == gt_answer:
            correct += 1

        # 打印前几个样例
        if idx < 3:
            print(f"\n{'='*50}")
            print(f"Question: {question}")
            print(f"Ground Truth: {gt_answer}")
            print(f"Prediction: {pred_answer}")
            print(f"Model Output: {response[:200]}...")

    # 输出结果
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--test_file", type=str, default="./data/char_count/sft/test.parquet", help="测试集路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最大测试样本数（可选，用于快速测试）")
    args = parser.parse_args()

    evaluate(args.model_path, args.test_file, args.max_samples)
