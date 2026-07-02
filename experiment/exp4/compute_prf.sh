#!/bin/bash

# bash compute_prf.sh
# ps -ef|grep python | grep compute_prf.py | awk '{print $2}' | xargs kill -9


# methods=(group1 group2 group3 group4)
methods=(Gemma3-4b Gemma3-12b Gemma3-27b Gemma3n-2b GLM4.5-air gpt-oss-20b gpt-oss-120b KimiK2 Llama3.1-405B Llama3.3-70B qwen3-4b)
datasets=(dataset1 dataset2 dataset3 dataset4 dataset5 dataset6)

for method in ${methods[@]}; do
    for dataset in ${datasets[@]}; do
        nohup python compute_prf.py --dataset $dataset --method $method >log/run_compute_prf_${method}_${dataset}.log &
    done
done


for method in ${methods[@]}; do
    for dataset in ${datasets[@]}; do
        nohup python compute_prf.py --dataset $dataset --method $method --target llm >log_llm/run_compute_prf_${method}_${dataset}.log &
    done
done