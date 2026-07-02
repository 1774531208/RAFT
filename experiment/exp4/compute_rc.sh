#!/bin/bash

# bash compute_rc.sh
# ps -ef|grep python | grep compute_rc.py | awk '{print $2}' | xargs kill -9

# methods=(group1 group2 group3 group4)
methods=(Gemma3-4b Gemma3-12b Gemma3-27b Gemma3n-2b GLM4.5-air gpt-oss-20b gpt-oss-120b KimiK2 Llama3.1-405B Llama3.3-70B qwen3-4b)
datasets=(dataset1 dataset2 dataset3 dataset4 dataset5 dataset6)

for method in ${methods[@]}; do
    for dataset in ${datasets[@]}; do
        nohup python compute_rc.py --dataset $dataset --method $method >log/run_compute_rc_${method}_${dataset}.log &
    done
done

