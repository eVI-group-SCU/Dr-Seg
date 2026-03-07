#!/bin/bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com

: "${REASONING_MODEL_PATH:=your/path/to/checkpoint}"

echo "Using reasoning model from: $REASONING_MODEL_PATH"

# 自动构建 GPU 列表：若已设置 CUDA_VISIBLE_DEVICES 则按其分割，否则自动检测全部 GPU
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
    GPU_LIST=( $(seq 0 $((NUM_GPUS - 1))) )
fi
echo "Using GPUs: ${GPU_LIST[*]}"

: "${SEGMENTATION_MODEL_PATH:=facebook/sam2-hiera-large}"

# 提取模型名：先去除多余的 //，再截取 pretrained_models/ 到 /actor/ 之间的部分
MODEL_DIR=$(echo $REASONING_MODEL_PATH | tr -s '/' | sed -E 's/.*pretrained_models\/(.*)\/actor\/.*/\1/')
: "${TEST_DATA_PATH:=Ricky06662/ReasonSeg_val}"
 # you can also use the following datasets:
# : "${TEST_DATA_PATH:=Ricky06662/ReasonSeg_test}"
# : "${TEST_DATA_PATH:=Ricky06662/Ricky06662/refcoco_val}"
# : "${TEST_DATA_PATH:=Ricky06662/Ricky06662/refcoco_testA}"
# : "${TEST_DATA_PATH:=Ricky06662/Ricky06662/refcocoplus_val}"
# : "${TEST_DATA_PATH:=Ricky06662/Ricky06662/refcocoplus_testA}"
# : "${TEST_DATA_PATH:=Ricky06662/Ricky06662/refcocog_val}"
# : "${TEST_DATA_PATH:=Ricky06662/Ricky06662/refcocog_test}"


TEST_NAME=$(echo $TEST_DATA_PATH | sed -E 's/.*\/([^\/]+)$/\1/')
OUTPUT_PATH="./new_reasonseg_eval_results/${MODEL_DIR}/${TEST_NAME}"

NUM_PARTS=${#GPU_LIST[@]}
: "${BATCH_SIZE:=16}"
echo "NUM_PARTS (= number of GPUs): ${NUM_PARTS}"

# Create output directory
mkdir -p $OUTPUT_PATH

# 按 GPU 数量自动分片，每块 GPU 处理一个分片
for idx in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$idx]}"
    echo "Launching worker idx=${idx} on GPU ${gpu} ..."
    CUDA_VISIBLE_DEVICES=$gpu python evaluation_scripts/evaluation_drseg.py \
        --reasoning_model_path $REASONING_MODEL_PATH \
        --segmentation_model_path $SEGMENTATION_MODEL_PATH \
        --output_path $OUTPUT_PATH \
        --test_data_path $TEST_DATA_PATH \
        --idx $idx \
        --num_parts $NUM_PARTS \
        --batch_size $BATCH_SIZE &
done

wait


python evaluation_scripts/calculate_iou.py --output_dir $OUTPUT_PATH
