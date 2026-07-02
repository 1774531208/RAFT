#!/bin/bash

# bash compute_prf.sh
# ps -ef|grep python | grep compute_prf.py | awk '{print $2}' | xargs kill -9

nohup python compute_prf.py --dataset dataset1 --method ours_gpt >log/run_compute_prf_ours_without_testcase_gpt_dataset1.log &
nohup python compute_prf.py --dataset dataset1 --method ours_grok >log/run_compute_prf_ours_without_testcase_grok_dataset1.log &
nohup python compute_prf.py --dataset dataset1 --method ours_deepseek >log/run_compute_prf_ours_without_testcase_deepseek_dataset1.log &
nohup python compute_prf.py --dataset dataset2 --method ours_gpt >log/run_compute_prf_ours_without_testcase_gpt_dataset2.log &
nohup python compute_prf.py --dataset dataset2 --method ours_grok >log/run_compute_prf_ours_without_testcase_grok_dataset2.log &
nohup python compute_prf.py --dataset dataset2 --method ours_deepseek >log/run_compute_prf_ours_without_testcase_deepseek_dataset2.log &
nohup python compute_prf.py --dataset dataset3 --method ours_gpt >log/run_compute_prf_ours_without_testcase_gpt_dataset3.log &
nohup python compute_prf.py --dataset dataset3 --method ours_grok >log/run_compute_prf_ours_without_testcase_grok_dataset3.log &
nohup python compute_prf.py --dataset dataset3 --method ours_deepseek >log/run_compute_prf_ours_without_testcase_deepseek_dataset3.log &
nohup python compute_prf.py --dataset dataset4 --method ours_gpt >log/run_compute_prf_ours_without_testcase_gpt_dataset4.log &
nohup python compute_prf.py --dataset dataset4 --method ours_grok >log/run_compute_prf_ours_without_testcase_grok_dataset4.log &
nohup python compute_prf.py --dataset dataset4 --method ours_deepseek >log/run_compute_prf_ours_without_testcase_deepseek_dataset4.log &
nohup python compute_prf.py --dataset dataset5 --method ours_gpt >log/run_compute_prf_ours_without_testcase_gpt_dataset5.log &
nohup python compute_prf.py --dataset dataset5 --method ours_grok >log/run_compute_prf_ours_without_testcase_grok_dataset5.log &
nohup python compute_prf.py --dataset dataset5 --method ours_deepseek >log/run_compute_prf_ours_without_testcase_deepseek_dataset5.log &
nohup python compute_prf.py --dataset dataset6 --method ours_gpt >log/run_compute_prf_ours_without_testcase_gpt_dataset6.log &
nohup python compute_prf.py --dataset dataset6 --method ours_grok >log/run_compute_prf_ours_without_testcase_grok_dataset6.log &
nohup python compute_prf.py --dataset dataset6 --method ours_deepseek >log/run_compute_prf_ours_without_testcase_deepseek_dataset6.log &

