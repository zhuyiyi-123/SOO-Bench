#!/bin/bash

num=0
low=(0 0 0 0)
high=(60 70 80 90)
seed=(0 1 2 3 4 5 6 7)

tasks=("gtopx_data" "cec_data")
# benchmarks_gtopx_data=(7)
benchmarks_cec_data=(2 3 4 5)

# gtopx_data tasks
for b in "${benchmarks_gtopx_data[@]}"; do
  for s in "${seed[@]}"; do
    for ((i=0; i<${#low[@]}; i++)); do
      python DE/main_PF.py --task=gtopx_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python DE/main_SPF.py --task=gtopx_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python CARCOO/main.py --task=gtopx_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python COMS_P/main.py --task=gtopx_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
    done
  done
done

# cec_data tasks
for b in "${benchmarks_cec_data[@]}"; do
  for s in "${seed[@]}"; do
    for ((i=0; i<${#low[@]}; i++)); do
      python DE/main_PF.py --task=cec_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python DE/main_SPF.py --task=cec_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python CARCOO/main.py --task=cec_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python COMS_P/main.py --task=cec_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
    done
  done
done
