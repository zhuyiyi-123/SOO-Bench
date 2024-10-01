#!/bin/bash
num=0 # will be reset to dimension of the problem * 1000
low=(0 10 20 25 30 40 0 0 0 0 0)
high=(50 60 70 75 80 90 60 70 80 90 100)
seed=(0 1 2 3 4 5 6 7)

tasks=("gtopx_data" "hybrid_data" "cec_data")
benchmarks_gtopx_data=(1 5 7)
benchmarks_hybrid_data=(0)
benchmarks_cec_data=(1 2 3 4 5)

## gtopx_data tasks
task=gtopx_data
for b in "${benchmarks_gtopx_data[@]}"; do
  for s in "${seed[@]}"; do
    for ((i=0; i<${#low[@]}; i++)); do
      python ./constraint/CARCOO/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./constraint/COMS_P/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s 
      python ./constraint/DE/main_PF.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./constraint/DE/main_SPF.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
    done
  done
done

# hybrid_data tasks
task=hybrid_data
for b in "${benchmarks_hybrid_data[@]}"; do
  for s in "${seed[@]}"; do
    for ((i=0; i<${#low[@]}; i++)); do
      python ./constraint/CARCOO/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./constraint/COMS_P/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s 
      python ./constraint/DE/main_PF.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./constraint/DE/main_SPF.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
    done
  done
done

# cec_data tasks
task=cec_data
for b in "${benchmarks_cec_data[@]}"; do
  for s in "${seed[@]}"; do
    for ((i=0; i<${#low[@]}; i++)); do
      python ./constraint/CARCOO/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./constraint/COMS_P/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s 
      python ./constraint/DE/main_PF.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./constraint/DE/main_SPF.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
    done
  done
done