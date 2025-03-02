#!/bin/bash
cd "$(dirname "$0")"

info='unconstraint'
nums=(1000) # will be reset to (dimension of the problem * 1000)
# low=(25)
# high=(75)
low=(0 10 20 25 30 40 0 0 0 0)
high=(50 60 70 75 80 90 60 70 80 90)
seed=(0 1 2 3 4 5 6 7)
algorithms=(\
      "./unconstraint/ARCOO/arcoo/__init__.py"\
      "./unconstraint/BO/bo_qei.py"\
      "./unconstraint/CBAS/main.py"\
      "./unconstraint/DEEA/TTDDEAMain.py"\
      "./unconstraint/Tri-mentoring/main.py"\
      "./unconstraint/CC-DDEA/example.py"\
      "./unconstraint/CMAES/main.py"\
      )
sample_method="sample_bound"  # sample_bound, sample_limit 
change_optimization_step=150
cuda=0,3

tasks=("gtopx_data" "mujoco_data")
benchmarks_gtopx_data=(2 3 4 6)
# benchmarks_gtopx_data=()


benchmarks_mujoco_data=(1 2)
# benchmarks_mujoco_data=()


## gtopx_data tasks
task=gtopx_data
for algorithm in "${algorithms[@]}"; do
  for num in "${nums[@]}"; do
    for b in "${benchmarks_gtopx_data[@]}"; do
      for s in "${seed[@]}"; do
        for ((i=0; i<${#low[@]}; i++)); do
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/ARCOO/arcoo/__init__.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/BO/bo_qei.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s  --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/CBAS/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/CMAES/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/DEEA/TTDDEAMain.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/Tri-mentoring/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/CC-DDEA/example.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          CUDA_VISIBLE_DEVICES=$cuda python $algorithm  \
                --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s \
                --sample_method=$sample_method --change_optimization_step=$change_optimization_step
        done
      done
    done
  done
done


# # mujoco_data tasks
task=mujoco_data
for algorithm in "${algorithms[@]}"; do
  for num in "${nums[@]}"; do
    for b in "${benchmarks_mujoco_data[@]}"; do
      for s in "${seed[@]}"; do
        for ((i=0; i<${#low[@]}; i++)); do
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/ARCOO/arcoo/__init__.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/BO/bo_qei.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s  --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/CBAS/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/CMAES/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/DEEA/TTDDEAMain.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/Tri-mentoring/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./unconstraint/CC-DDEA/example.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          CUDA_VISIBLE_DEVICES=$cuda python $algorithm  \
                --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s \
                --sample_method=$sample_method --change_optimization_step=$change_optimization_step
        done
      done
    done
  done
done

python ./scripts/result_gather.py
touch ./done_unconstraint.flag
python ~/notifier.py $info