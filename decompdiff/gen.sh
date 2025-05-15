#!/bin/bash

total_tasks=100
gpu_count=4
current_task=56  # 起始任务编号

# 定义一个函数，用于在指定 GPU 上运行任务
run_task_on_gpu() {
  local gpu_id=$1
  local task_id=$2
  CUDA_VISIBLE_DEVICES=$gpu_id python scripts/sample_diffusion_decomp.py -i $task_id --guide origin --outdir outputs_dpo_5 --config configs/dpo_sampling.yml &
  echo "启动任务 $task_id 在 GPU$gpu_id 上"
}

# 初始化 GPU 状态数组，记录每张 GPU 当前运行的任务
declare -a gpu_tasks
for ((j=0; j<gpu_count; j++)); do
  gpu_tasks[$j]=-1
done

# 主循环分配任务
while [ $current_task -lt $total_tasks ] || [ $(jobs -r | wc -l) -gt 0 ]; do
  for ((j=0; j<gpu_count; j++)); do
    # 检查当前 GPU 是否空闲
    if ! ps -p ${gpu_tasks[$j]} > /dev/null 2>&1; then
      if [ $current_task -lt $total_tasks ]; then
        # 分配新任务到空闲 GPU
        run_task_on_gpu $j $current_task
        gpu_tasks[$j]=$!
        current_task=$((current_task + 1))
      fi
    fi
  done
  # 避免过度占用 CPU，稍作休眠
  sleep 10
done
