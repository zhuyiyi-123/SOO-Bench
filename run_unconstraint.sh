#!/bin/bash

num=0 # will be reset to dimension of the problem * 1000
low=(0 10 20 25 30 40 0 0 0 0 0)
high=(50 60 70 75 80 90 60 70 80 90 100)
seed=(0 1 2 3 4 5 6 7)

tasks=("gtopx_data" "hybrid_data" "mujoco_data")
benchmarks_gtopx_data=(2 3 4 6)
benchmarks_hybrid_data=(1)
benchmarks_mujoco_data=(1 2)
## gtopx_data tasks
task=gtopx_data
for b in "${benchmarks_gtopx_data[@]}"; do
  for s in "${seed[@]}"; do
    for ((i=0; i<${#low[@]}; i++)); do
      python ./unconstraint/ARCOO/arcoo/__init__.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./unconstraint/BO/bo_qei.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s 
      python ./unconstraint/CBAS/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./unconstraint/CMAES/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./unconstraint/DEEA/TTDDEAMain.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./unconstraint/Tri-mentoring/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
    done
  done
done
#
# hybrid_data tasks
task=hybrid_data
for b in "${benchmarks_hybrid_data[@]}"; do
  for s in "${seed[@]}"; do
    for ((i=0; i<${#low[@]}; i++)); do
      python ./unconstraint/ARCOO/arcoo/__init__.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./unconstraint/BO/bo_qei.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s 
      python ./unconstraint/CBAS/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./unconstraint/CMAES/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./unconstraint/DEEA/TTDDEAMain.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./unconstraint/Tri-mentoring/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
    done
  done
done

# mujoco_data tasks
task=mujoco_data
for b in "${benchmarks_mujoco_data[@]}"; do
  for s in "${seed[@]}"; do
    for ((i=0; i<${#low[@]}; i++)); do
      python ./unconstraint/ARCOO/arcoo/__init__.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./unconstraint/BO/bo_qei.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s 
      python ./unconstraint/CBAS/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./unconstraint/CMAES/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./unconstraint/DEEA/TTDDEAMain.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      python ./unconstraint/Tri-mentoring/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
    done
  done
done
