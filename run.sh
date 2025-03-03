#!/bin/bash

# y-ood
bash ./run_unconstraint_y_ood.sh
bash ./run_constraint_y_ood.sh
mv ./results ./results_y_ood

# x-ood
bash ./run_unconstraint_x_ood.sh
bash ./run_constraint_x_ood.sh
mv ./results ./results_x_ood

# different num
bash ./run_unconstraint_different_n.sh
bash ./run_constraint_different_n.sh
mv ./results ./results_different_n