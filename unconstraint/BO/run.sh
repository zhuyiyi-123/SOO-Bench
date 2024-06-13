# for i in $(seq 0 4);do
#     python bo_qei.py --task="mujoco_data" --benchmark=1
# done
python bo_qei.py --task="gtopx_data" --benchmark=3 --low=10 --high=50