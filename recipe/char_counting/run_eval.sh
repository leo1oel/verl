#!/usr/bin/env bash
set -xeuo pipefail

# 设置模型路径（你可以修改这里）
MODEL_PATH=${MODEL_PATH:-./experiments/char_count/models/sft/fsdp/global_step_140/huggingface}

# 测试集路径
TEST_FILE=${TEST_FILE:-./data/char_count/sft/test.parquet}

# 可选：设置最大测试样本数（留空则测试全部）
MAX_SAMPLES=${MAX_SAMPLES:-}

# 运行评估
if [ -z "$MAX_SAMPLES" ]; then
    python3 recipe/char_counting/eval_simple.py \
        --model_path "$MODEL_PATH" \
        --test_file "$TEST_FILE"
else
    python3 recipe/char_counting/eval_simple.py \
        --model_path "$MODEL_PATH" \
        --test_file "$TEST_FILE" \
        --max_samples "$MAX_SAMPLES"
fi
