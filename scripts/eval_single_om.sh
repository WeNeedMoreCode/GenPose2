#!/bin/bash
# OM Single Evaluation. Auto-detects device via lspci (mirrors env_utils.is_rc_device):
# - 300V PRO (Ascend 310P3): pointnet2 runs as monolithic OM (pointnet2_from_*.om)
# - 200I PRO RC (Ascend 310P1, CANN 8.3RC1): pointnet2 on CPU via graspnet (.pth), head nets OM
if lspci 2>/dev/null | grep -q accelerators; then
  PN2_SCORE=om_models/pointnet2_from_score.om
  PN2_ENERGY=om_models/pointnet2_from_energy.om
  RESULT_DIR=single_om
else
  PN2_SCORE=results/ckpts/ScoreNet/scorenet.pth
  PN2_ENERGY=results/ckpts/EnergyNet/energynet.pth
  RESULT_DIR=single_om_rc
fi

CUDA_VISIBLE_DEVICES=0 python runners/evaluation_single.py \
--pretrained_dino_model_path om_models/dinov2_vits14.om \
--pretrained_pointnet2_score_model_path $PN2_SCORE \
--pretrained_pointnet2_energy_model_path $PN2_ENERGY \
--pretrained_score_model_path om_models/scorenet.om \
--pretrained_energy_model_path om_models/energynet.om \
--pretrained_scale_model_path om_models/scalenet.om \
--data_path omni6dpose-000000/ROPE/ \
--sampler_mode ode \
--percentage_data_for_test 1.0 \
--batch_size 4 \
--seed 0 \
--result_dir $RESULT_DIR \
--eval_repeat_num 50 \
--clustering 1 \
--T0 0.55 \
--dino pointwise \
--num_worker 32 \
--real_drop 3 \
--device npu:0
