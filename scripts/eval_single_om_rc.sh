#!/bin/bash
# RC (200I PRO / 310P RC, CANN 8.3RC1) OM Single Evaluation Script.
# pointnet2 runs on CPU via graspnet_cpu_patches (cfg.pretrained_pointnet2_*_model_path
# points to .pth: scorenet.pth/energynet.pth holding the PointNet2 weights);
# head networks (score/energy/scale/dino) run OM. Auto-detected via is_rc_device().
CUDA_VISIBLE_DEVICES=0 python runners/evaluation_single.py \
--pretrained_dino_model_path om_models/dinov2_vits14.om \
--pretrained_pointnet2_score_model_path results/ckpts/ScoreNet/scorenet.pth \
--pretrained_pointnet2_energy_model_path results/ckpts/EnergyNet/energynet.pth \
--pretrained_score_model_path om_models/scorenet.om \
--pretrained_energy_model_path om_models/energynet.om \
--pretrained_scale_model_path om_models/scalenet.om \
--data_path omni6dpose-000000/ROPE/ \
--sampler_mode ode \
--percentage_data_for_test 1.0 \
--batch_size 4 \
--seed 0 \
--result_dir single_om_rc \
--eval_repeat_num 50 \
--clustering 1 \
--T0 0.55 \
--dino pointwise \
--num_worker 32 \
--real_drop 3 \
--device npu:0
