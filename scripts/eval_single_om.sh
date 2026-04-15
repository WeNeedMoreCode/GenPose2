#!/bin/bash
# OM Model Evaluation Script
# Uses separate PointNet2 OM and ScoreNet OM models for better performance
CUDA_VISIBLE_DEVICES=0 python runners/evaluation_single.py \
--pretrained_dino_model_path om_models/dinov2_vits14.om \
--pretrained_pointnet2_score_model_path om_models/pointnet2_from_score.om \
--pretrained_pointnet2_energy_model_path om_models/pointnet2_from_energy.om \
--pretrained_score_model_path om_models/scorenet.om \
--pretrained_energy_model_path om_models/energynet.om \
--pretrained_scale_model_path om_models/scalenet.om \
--data_path omin6dpose-000a/ROPE/ \
--sampler_mode ode \
--percentage_data_for_test 1.0 \
--batch_size 16 \
--seed 0 \
--result_dir single_om \
--eval_repeat_num 50 \
--clustering 1 \
--T0 0.55 \
--dino pointwise \
--num_worker 32 \
--real_drop 3 \
--device npu:0
