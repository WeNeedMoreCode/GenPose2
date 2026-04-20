#!/bin/bash
# OM Model Tracking Evaluation Script
CUDA_VISIBLE_DEVICES=0 python runners/evaluation_tracking.py \
--pretrained_dino_model_path om_models/dinov2_vits14.om \
--pretrained_pointnet2_score_model_path om_models/pointnet2_from_score.om \
--pretrained_pointnet2_energy_model_path om_models/pointnet2_from_energy.om \
--pretrained_score_model_path om_models/scorenet.om \
--pretrained_energy_model_path om_models/energynet.om \
--pretrained_scale_model_path om_models/scalenet.om \
--data_path Omni6DPose_ROPE_PATH \
--sampler_mode ode \
--percentage_data_for_test 1.0 \
--batch_size 128 \
--seed 0 \
--result_dir tracking_om \
--eval_repeat_num 50 \
--clustering 1 \
--T0 0.25 \
--dino pointwise \
--num_worker 32 \
--device npu:0
