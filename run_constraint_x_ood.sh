#!/bin/bash
cd "$(dirname "$0")"

info='"constraint_x_ood"'
nums=(1000) # will be reset to (dimension of the problem * num)

low=(0 10 20 25 30 40 0 0 0 0)
high=(50 60 70 75 80 90 60 70 80 90)
seed=(0 1 2 3 4 5 6 7)
algorithms=("./constraint/CARCOO/main.py" "./constraint/COMS_P/main.py" "./constraint/DE/main_PF.py" "./constraint/DE/main_SPF.py")
sample_method="sample_limit"  # sample_bound, sample_limit 
change_optimization_step=150
cuda=0,3

tasks=("gtopx_data" "cec_data")
benchmarks_gtopx_data=(1 5 7)
# benchmarks_gtopx_data=()

benchmarks_cec_data=(1 2 3 4 5)
# benchmarks_cec_data=()


## gtopx_data tasks
task=gtopx_data
for algorithm in "${algorithms[@]}"; do
  for num in "${nums[@]}"; do
    for b in "${benchmarks_gtopx_data[@]}"; do
      for s in "${seed[@]}"; do
        for ((i=0; i<${#low[@]}; i++)); do
          # CUDA_VISIBLE_DEVICES=$cuda python ./constraint/CARCOO/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./constraint/COMS_P/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s  --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./constraint/DE/main_PF.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./constraint/DE/main_SPF.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          CUDA_VISIBLE_DEVICES=$cuda python $algorithm  \
              --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s \
              --sample_method=$sample_method --change_optimization_step=$change_optimization_step
        done
      done
    done
  done
done

# # cec_data tasks
task=cec_data
for algorithm in "${algorithms[@]}"; do
  for num in "${nums[@]}"; do
    for b in "${benchmarks_cec_data[@]}"; do
      for s in "${seed[@]}"; do
        for ((i=0; i<${#low[@]}; i++)); do
          # CUDA_VISIBLE_DEVICES=$cuda python ./constraint/CARCOO/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./constraint/COMS_P/main.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s  --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./constraint/DE/main_PF.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          # CUDA_VISIBLE_DEVICES=$cuda python ./constraint/DE/main_SPF.py --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s --sample_method=$sample_method
          CUDA_VISIBLE_DEVICES=$cuda python $algorithm  \
              --task=$task --benchmark=$b --num=$num --low=${low[i]} --high=${high[i]} --seed=$s \
              --sample_method=$sample_method --change_optimization_step=$change_optimization_step
        done
      done
    done
  done
done

python ./scripts/result_gather.py
touch ./done_constraint.flag
python ~/notifier.py $info