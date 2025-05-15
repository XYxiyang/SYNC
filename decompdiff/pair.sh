#!/bin/bash

total_tasks=100000
gpu_count=8
current_task=8277  # 起始任务编号

# 定义一个函数，用于在指定 GPU 上运行任务
run_task_on_gpu() {
  local gpu_id=$1
  local task_id=$2
  local -a pids=()

  for offset in 0 1 2; do
    local id=$((task_id + offset))
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/sample_pairwise.py -i $id &
    pids+=($!)           # 把每次启动后的 PID 都收集起来
  done

  gpu_tasks_list[$gpu_id]="${pids[*]}"
  echo "启动任务 ${task_id} - $((task_id+2)) 在 GPU${gpu_id} 上 (PIDs: ${pids[*]})"
}

# 初始化 GPU 状态数组，记录每张 GPU 当前运行的任务
declare -a gpu_tasks_list
for ((j=0; j<gpu_count; j++)); do
  gpu_tasks_list[$j]="" 
done

# 主循环分配任务
while [ $current_task -lt $total_tasks ] || [ $(jobs -r | wc -l) -gt 0 ]; do
  for ((j=0; j<gpu_count; j++)); do
    # 检查这一 GPU 上上一次分配的那批 PID，是否都已退出
    all_done=true
    for pid in ${gpu_tasks_list[$j]}; do
      if ps -p $pid > /dev/null 2>&1; then
        all_done=false
        break
      fi
    done

    # 如果三条都结束了，而且还有剩任务，就派新一批
    if $all_done && [ $current_task -lt $total_tasks ]; then
      run_task_on_gpu $j $current_task
      current_task=$((current_task + 3))
    fi
  done

  sleep 10
done