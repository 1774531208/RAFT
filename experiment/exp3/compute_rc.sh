#!/bin/bash

# bash compute_rc.sh
# ps -ef|grep python | grep compute_rc.py | awk '{print $2}' | xargs kill -9


nohup python compute_rc.py --dataset dataset1 --method ours_grok >log/run_compute_rc_ours_grok_dataset1.log &
nohup python compute_rc.py --dataset dataset2 --method ours_grok >log/run_compute_rc_ours_grok_dataset2.log &
nohup python compute_rc.py --dataset dataset3 --method ours_grok >log/run_compute_rc_ours_grok_dataset3.log &
nohup python compute_rc.py --dataset dataset4 --method ours_grok >log/run_compute_rc_ours_grok_dataset4.log &
nohup python compute_rc.py --dataset dataset5 --method ours_grok >log/run_compute_rc_ours_grok_dataset5.log &
nohup python compute_rc.py --dataset dataset6 --method ours_grok >log/run_compute_rc_ours_grok_dataset6.log &

nohup python compute_rc.py --dataset dataset1 --method ours_deepseek >log/run_compute_rc_ours_deepseek_dataset1.log &
nohup python compute_rc.py --dataset dataset2 --method ours_deepseek >log/run_compute_rc_ours_deepseek_dataset2.log &
nohup python compute_rc.py --dataset dataset3 --method ours_deepseek >log/run_compute_rc_ours_deepseek_dataset3.log &
nohup python compute_rc.py --dataset dataset4 --method ours_deepseek >log/run_compute_rc_ours_deepseek_dataset4.log &
nohup python compute_rc.py --dataset dataset5 --method ours_deepseek >log/run_compute_rc_ours_deepseek_dataset5.log &
nohup python compute_rc.py --dataset dataset6 --method ours_deepseek >log/run_compute_rc_ours_deepseek_dataset6.log &

nohup python compute_rc.py --dataset dataset1 --method ours_gpt >log/run_compute_rc_ours_gpt_dataset1.log &
nohup python compute_rc.py --dataset dataset2 --method ours_gpt >log/run_compute_rc_ours_gpt_dataset2.log &
nohup python compute_rc.py --dataset dataset3 --method ours_gpt >log/run_compute_rc_ours_gpt_dataset3.log &
nohup python compute_rc.py --dataset dataset4 --method ours_gpt >log/run_compute_rc_ours_gpt_dataset4.log &
nohup python compute_rc.py --dataset dataset5 --method ours_gpt >log/run_compute_rc_ours_gpt_dataset5.log &
nohup python compute_rc.py --dataset dataset6 --method ours_gpt >log/run_compute_rc_ours_gpt_dataset6.log &
