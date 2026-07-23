#!/bin/bash
unset PYTHONPATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/home/syx/ModelZoo-PyTorch/ACL_PyTorch/built-in/embodied_ai/GenPosePlus:/home/syx/ModelZoo-PyTorch/ACL_PyTorch/built-in/embodied_ai/GenPosePlus/GenPose2:$PYTHONPATH
export LD_LIBRARY_PATH=/home/syx/ModelZoo-PyTorch/ACL_PyTorch/built-in/embodied_ai/GenPosePlus/ascendc_kernels/out/lib:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export USE_TORCHAIR=1
export TORCHAIR_PT2=1
export TORCHAIR_CORES=8
export POINTNET2_BACKEND=ascendc
CUDA_VISIBLE_DEVICES=0 python runners/evaluation_tracking.py \
--pretrained_score_model_path results/ckpts/ScoreNet/scorenet.pth \
--pretrained_energy_model_path results/ckpts/EnergyNet/energynet.pth \
--pretrained_scale_model_path results/ckpts/ScaleNet/scalenet.pth \
--data_path omni6dpose-000000-sub/ROPE/ \
--sampler_mode ode \
--percentage_data_for_test 1.0 \
--batch_size 4 \
--seed 0 \
--result_dir tracking \
--eval_repeat_num 50 \
--clustering 1 \
--T0 0.25 \
--dino pointwise \
--num_worker 0
