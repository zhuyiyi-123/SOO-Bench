#!/bin/bash

num=0
low=(0)
high=(50)
seed=(0 1 2 3 4)

tasks=("gtopx_data" "hybrid_data" "mujoco_data")
benchmarks_gtopx_data=(2 3 4 6)
# benchmarks_hybrid_data=(1)
# benchmarks_mujoco_data=(1 2)

## gtopx_data tasks
for b in "${benchmarks_gtopx_data[@]}"; do
 for s in "${seed[@]}"; do
   for ((i=0; i<${#low[@]}; i++)); do
     python DEEA/TTDDEAMain.py --task=gtopx_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
    #  python ARCOO/core/arcoo/__init__.py --task=gtopx_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      #  python BO/bo_qei.py --task=gtopx_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s 
   done
 done
done
#
## hybrid_data tasks
#for b in "${benchmarks_hybrid_data[@]}"; do
#  for s in "${seed[@]}"; do
#    for ((i=0; i<${#low[@]}; i++)); do
#      python Tri-mentoring/main.py --task=hybrid_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
#      python ARCOO/core/arcoo/__init__.py --task=hybrid_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
      #  python BO/bo_qei.py --task=gtopx_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s 
#    done
#  done
#done

# mujoco_data tasks
# for b in "${benchmarks_mujoco_data[@]}"; do
#   for s in "${seed[@]}"; do
#     for ((i=0; i<${#low[@]}; i++)); do
#       python Tri-mentoring/main.py --task=mujoco_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
#       python ARCOO/core/arcoo/__init__.py --task=mujoco_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s
#       # python BO/bo_qei.py --task=gtopx_data --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s 
#     done
#   done
# done
