import os
import sys
import time
import cv2
import glob
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import pickle
import torch
import torch_npu
import random
torch_npu.npu.set_compile_mode(jit_compile=False)
import torch.nn as nn
import torch.nn.functional as F
import copy
import shutil
import hashlib
import random
import gc
from sklearn.cluster import DBSCAN

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ipdb import set_trace

from networks.posenet_agent import PoseNet
from networks.reward import sort_poses_by_energy, ranking_loss
from networks.score_wrapper import create_score_network, create_ode_sampler
from networks.gf_algorithms.sde import init_sde
from datasets.datasets_omni6dpose import Omni6DPoseDataSet, array_to_SymLabel, array_to_CameraIntrinsicsBase, process_batch
from utils.metrics import get_rot_matrix
from utils.transforms import matrix_to_quaternion, quaternion_to_matrix
from utils.misc import average_quaternion_batch
from utils.genpose_utils import get_pose_dim
from utils.so3_visualize import visualize_so3
from utils.visualize import create_grid_image
from cutoop.eval_utils import DetectMatch, Metrics
from configs.config import get_config


''' load config '''
cfg = get_config()
cfg.load_per_object = True

torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)

# Performance statistics
perf_stats = {
    'score_time': [],
    'score_samples': 0,
    'energy_time': [],
    'energy_samples': 0,
    'aggregate_time': [],
    'aggregate_samples': 0,
    'scale_time': [],
    'scale_samples': 0,
    'bbox_time': [],
    'bbox_samples': 0,
}


def apply_spec_ops_patches():
    def patched_bilinear2d(input, output_size, align_corners,scale_factors):
        original_device = input.device
        input_cpu = input.cpu()

        result_tmp = _original_bilinear2d(
            input_cpu,
            output_size,
            align_corners, 
            scale_factors,
        )
        result = result_tmp.to(device=original_device)
        return result

    def patched_max_pool2d(input, kernel_size):
        original_device = input.device
        input_cpu = input.cpu()
        result_tmp = _original_max_pool2d(input_cpu, kernel_size)
        result = result_tmp.to(device=original_device)
        return result
    
    _original_bilinear2d = torch._C._nn.upsample_bilinear2d
    _original_max_pool2d = F.max_pool2d

    torch._C._nn.upsample_bilinear2d = patched_bilinear2d
    F.max_pool2d = patched_max_pool2d

def get_dataloader():
    dataset = Omni6DPoseDataSet(
        cfg=cfg,
        dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
        deform_2d_params=cfg.DEFORM_2D_PARAMS,
        source='Omni6DPose',
        mode='real',
        data_dir=cfg.data_path,
        n_pts=1024,
        img_size=cfg.img_size,
        per_obj=None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        persistent_workers=True,
        drop_last=False,
        pin_memory=True,
    )
    return dataloader




# ============================================================================
# DINOv2 Feature Extraction (Preprocessing)
# ============================================================================

_dino_model = None

def get_dino_model():
    """Lazy load DINOv2 model to save memory.

    NOTE: We intentionally do NOT call eval() to match the behavior in GFObjectPose.
    This ensures consistent feature extraction behavior.
    """
    global _dino_model
    if _dino_model is None and cfg.dino != 'none':
        print("Loading DINOv2 model for preprocessing...")
        import torch.hub
        _dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(cfg.device)
        # Intentionally NOT calling eval() to match GFObjectPose behavior (posenet.py:44-45)
        # _dino_model.eval()  # <-- DO NOT enable this
        _dino_model.requires_grad_(False)
        print("DINOv2 loaded successfully")
    return _dino_model


def extract_dino_features(batch_sample):
    """
    Extract DINOv2 features as a preprocessing step.

    Args:
        batch_sample: Dict with 'roi_rgb', 'roi_xs', 'roi_ys'

    Returns:
        rgb_feat: [B, 1024, 384] or None if dino='none'
    """
    if cfg.dino == 'none':
        return None

    dino = get_dino_model()
    if dino is None:
        return None


    roi_rgb = batch_sample['roi_rgb']  # [B, 3, H, W]
    roi_xs = batch_sample['roi_xs']    # [B, 1024]
    roi_ys = batch_sample['roi_ys']    # [B, 1024]

    # Extract DINOv2 features from penultimate layer
    # Note: No torch.no_grad() wrapper to match original behavior in posenet.py:113-121
    # Note: No n=1 parameter to match original behavior in posenet.py:115
    feat = dino.get_intermediate_layers(roi_rgb)[0]  # [B, 384, H/14, W/14]

    # Extract features for each point based on their (x, y) coordinates
    xs = roi_xs // 14
    ys = roi_ys // 14
    pos = xs * 16 + ys  # 224x224 input -> 16x16 feature map
    pos = torch.unsqueeze(pos, -1).expand(-1, -1, 384)

    rgb_feat = torch.gather(feat, 1, pos)  # [B, 1024, 384]
    # Match original behavior in posenet.py:121 - use requires_grad_(False) instead of detach()
    rgb_feat.requires_grad_(False)

    return rgb_feat




# ============================================================================
# Inference Functions
# ============================================================================

def inference_score(save_path):
    if os.path.exists(save_path):
        return

    cfg.agent_type = 'score'
    score_agent:PoseNet = PoseNet(cfg)
    score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
    score_agent.eval()

    all_pred_pose = []
    all_score_feature = []
    total_samples = 0

    for i, test_batch in enumerate(tqdm(dataloader, desc="score sampling")):
        torch.npu.synchronize()
        start_time = time.time()
        batch_sample = process_batch(
            batch_sample = test_batch,
            device=cfg.device,
            pose_mode=cfg.pose_mode,
        )

        # Extract DINOv2 features as preprocessing (outside the model)
        rgb_feat = extract_dino_features(batch_sample)
        if rgb_feat is not None:
            batch_sample['precomputed_rgb_feat'] = rgb_feat

        pred_results = score_agent.pred_func(
            data=batch_sample,
            repeat_num=cfg.eval_repeat_num,
            T0=cfg.T0,
            return_average_res=False,
            return_process=False
        )
        pred_pose, _ = pred_results
        all_pred_pose.append(pred_pose)
        all_score_feature.append({
            'pts_feat': batch_sample['pts_feat'].cpu(),
            'rgb_feat': (None if batch_sample['rgb_feat'] is None else batch_sample['rgb_feat'].cpu()),
        })
        torch.npu.synchronize()
        elapsed = time.time() - start_time
        perf_stats['score_time'].append(elapsed)
        total_samples += pred_pose.shape[0]
        perf_stats['score_samples'] = total_samples

        if i % 4 == 3:
            gc.collect()
    pickle.dump((all_pred_pose, all_score_feature), open(save_path, 'wb'))

def inference_score_decoupled(save_path):
    """
    Unified decoupled inference using ScoreNetworkWrapper and ODESamplerExternal.

    Supports both PyTorch and OM models:
    - PyTorch: Uses internal PyTorch PointNet2 encoder
    - OM: Uses separate PointNet2 OM + ScoreNet OM (if available)

    This unified interface enables direct comparison between PyTorch and OM outputs
    for debugging precision issues.

    Args:
        save_path: Path to save cached results
    """
    if os.path.exists(save_path):
        return

    # Initialize SDE components
    prior_fn, marginal_prob_fn, sde_fn, sampling_eps, T = init_sde('ve')

    pointnet2_om_path = getattr(cfg, 'pretrained_pointnet2_model_path',
                                      None) or './onnx_models/pointnet2.om'


    # Create Score Network wrapper (automatically uses OM or PyTorch)
    score_net = create_score_network(
        checkpoint_path=cfg.pretrained_score_model_path,
        device=cfg.device,
        pointnet2_om_path=pointnet2_om_path  # Pass None for PyTorch, path for OM
    )



    # Create ODE sampler with the Score Network
    sampler = create_ode_sampler(
        score_network=score_net,
        sde={'prior_fn': prior_fn, 'sde_fn': sde_fn},
        device=cfg.device
    )

    all_pred_pose = []
    all_score_feature = []
    total_samples = 0

    print(f"\nRunning unified decoupled inference ({'OM' if score_net.is_om else 'PyTorch'})...")
    for i, test_batch in enumerate(tqdm(dataloader, desc="score sampling")):
        torch.npu.synchronize()
        start_time = time.time()

        batch_sample = process_batch(
            batch_sample=test_batch,
            device=cfg.device,
            pose_mode=cfg.pose_mode,
        )

        # Extract DINOv2 features as preprocessing
        rgb_feat = extract_dino_features(batch_sample)
        if rgb_feat is not None:
            batch_sample['precomputed_rgb_feat'] = rgb_feat

        # Extract point cloud features using unified interface
        # Automatically uses OM PointNet2 (if available) or PyTorch PointNet2
        with torch.no_grad():
            pts_feat = score_net.extract_pts_feat(
                pts=batch_sample['pts'],
                rgb_feat=rgb_feat
            )

        # Get batch info
        bs = batch_sample['pts'].shape[0]
        pose_dim = get_pose_dim(cfg.pose_mode)
        pts_center = batch_sample.get('pts_center', None)

        # Generate random initial values for all repeats at once
        init_x_all = prior_fn((bs, cfg.eval_repeat_num, pose_dim), T=cfg.T0).to(cfg.device)

        # Repeat features and init_x to process all at once
        pts_feat_repeated = pts_feat.unsqueeze(1).repeat(1, cfg.eval_repeat_num, 1).view(bs * cfg.eval_repeat_num, -1)

        # In pointwise mode, rgb_feat is already fused into pts_feat, so pass None
        rgb_feat_repeated = None

        init_x_repeated = init_x_all.view(bs * cfg.eval_repeat_num, pose_dim)
        pts_center_repeated = None if pts_center is None else \
            pts_center.unsqueeze(1).repeat(1, cfg.eval_repeat_num, 1).view(bs * cfg.eval_repeat_num, -1)

        # Single call to sampler for all repeats
        with torch.no_grad():
            _, sampled_pose = sampler.sample(
                pts_feat=pts_feat_repeated,
                rgb_feat=rgb_feat_repeated,
                batch_size=bs * cfg.eval_repeat_num,
                pose_dim=pose_dim,
                T=cfg.T0,
                eps=sampling_eps,
                rtol=1e-5,
                atol=1e-5,
                denoise=True,
                init_x=init_x_repeated,
                pts_center=pts_center_repeated
            )

        # Reshape result from [bs*repeat_num, pose_dim] to [bs, repeat_num, pose_dim]
        pred_pose = sampled_pose.view(bs, cfg.eval_repeat_num, pose_dim)

        # Save pred_pose and features
        all_pred_pose.append(pred_pose)
        all_score_feature.append({
            'pts_feat': pts_feat.cpu(),
            'rgb_feat': (None if rgb_feat is None else rgb_feat.cpu()),
        })

        torch.npu.synchronize()
        elapsed = time.time() - start_time
        perf_stats['score_time'].append(elapsed)
        total_samples += pred_pose.shape[0]
        perf_stats['score_samples'] = total_samples

        if i % 4 == 3:
            gc.collect()

    pickle.dump((all_pred_pose, all_score_feature), open(save_path, 'wb'))
    print(f"Unified decoupled inference complete! ({'OM' if score_net.is_om else 'PyTorch'})")


def inference_energy(score_path, save_path):
    if os.path.exists(save_path):
        return
    assert os.path.exists(score_path)
    all_pred_pose, _ = pickle.load(open(score_path, 'rb'))

    cfg.agent_type = 'energy'
    energy_agent = PoseNet(cfg)
    energy_agent.load_ckpt(model_dir=cfg.pretrained_energy_model_path, model_path=True, load_model_only=True)
    energy_agent.eval()

    all_pred_energy = []
    total_samples = 0

    for i, test_batch in enumerate(tqdm(dataloader, desc="energy")):
        torch.npu.synchronize()
        start_time = time.time()

        batch_sample = process_batch(
            batch_sample = test_batch,
            device=cfg.device,
            pose_mode=cfg.pose_mode,
        )

        # Extract DINOv2 features as preprocessing (outside the model)
        rgb_feat = extract_dino_features(batch_sample)
        if rgb_feat is not None:
            batch_sample['precomputed_rgb_feat'] = rgb_feat

        pred_energy = energy_agent.get_energy(
            data=batch_sample,
            pose_samples=all_pred_pose[i],
            T=1e-5,
            mode='test',
            extract_feature=True
        )
        all_pred_energy.append(pred_energy.cpu())

        torch.npu.synchronize()
        elapsed = time.time() - start_time
        perf_stats['energy_time'].append(elapsed)
        total_samples += pred_energy.shape[0]
        perf_stats['energy_samples'] = total_samples

        if i % 4 == 3:
            gc.collect()

    pickle.dump(all_pred_energy, open(save_path, 'wb'))

def aggregate_pose(score_path, energy_path, save_path):
    if os.path.exists(save_path):
        return
    assert os.path.exists(score_path)
    all_pred_pose, _ = pickle.load(open(score_path, 'rb'))
    if energy_path is not None:
        assert os.path.exists(energy_path)
        all_pred_energy = pickle.load(open(energy_path, 'rb'))
    else:
        all_pred_energy = [torch.ones(*(all_pred_pose[i].shape[:2]), 2) 
                           for i in range(len(all_pred_pose))]

    all_aggregated_pose = []
    total_samples = 0

    for i, (pred_pose, pred_energy) in enumerate(tqdm(zip(all_pred_pose, all_pred_energy), desc="aggregate")):
        torch.npu.synchronize()
        start_time = time.time()
        sorted_pose, sorted_energy = sort_poses_by_energy(pred_pose, pred_energy)
        bs = pred_pose.shape[0]
        retain_num = int(cfg.eval_repeat_num * cfg.retain_ratio)
        good_pose = sorted_pose[:, :retain_num, :]
        rot_matrix = get_rot_matrix(good_pose[:, :, :-3].reshape(bs * retain_num, -1), cfg.pose_mode)
        quat_wxyz = matrix_to_quaternion(rot_matrix).reshape(bs, retain_num, -1)
        aggregated_quat_wxyz = average_quaternion_batch(quat_wxyz)
        if cfg.clustering:
            for j in range(bs):
                # https://math.stackexchange.com/a/90098
                # 1 - ⟨q1, q2⟩ ^ 2 = (1 - cos theta) / 2
                pairwise_distance = 1 - torch.sum(quat_wxyz[j].unsqueeze(0) * quat_wxyz[j].unsqueeze(1), dim=2) ** 2
                dbscan = DBSCAN(eps=cfg.clustering_eps, min_samples=int(cfg.clustering_minpts * retain_num)).fit(pairwise_distance.cpu().cpu().numpy())
                labels = dbscan.labels_
                if np.any(labels >= 0):
                    bins = np.bincount(labels[labels >= 0])
                    best_label = np.argmax(bins)
                    aggregated_quat_wxyz[j] = average_quaternion_batch(quat_wxyz[j, labels == best_label].unsqueeze(0))[0]
        aggregated_trans = torch.mean(good_pose[:, :, -3:], dim=1)
        aggregated_pose = torch.zeros(bs, 4, 4)
        aggregated_pose[:, 3, 3] = 1
        aggregated_pose[:, :3, :3] = quaternion_to_matrix(aggregated_quat_wxyz)
        aggregated_pose[:, :3, 3] = aggregated_trans
        all_aggregated_pose.append(aggregated_pose)

        torch.npu.synchronize()
        elapsed = time.time() - start_time
        perf_stats['aggregate_time'].append(elapsed)
        total_samples += bs
        perf_stats['aggregate_samples'] = total_samples

        if i % 10 == 9:
            gc.collect()
    
    pickle.dump(all_aggregated_pose, open(save_path, 'wb'))

def inference_scale(score_path, aggregate_path, save_path):
    if os.path.exists(save_path):
        return
    assert os.path.exists(score_path)
    _, all_score_feature = pickle.load(open(score_path, 'rb'))
    assert os.path.exists(aggregate_path)
    all_aggregated_pose = pickle.load(open(aggregate_path, 'rb'))

    if cfg.pretrained_scale_model_path is None:
        all_final_length = []
        total_samples = 0

        for i, test_batch in enumerate(tqdm(dataloader, desc="bbox")):
            torch.npu.synchronize()
            start_time = time.time()
            pcl: torch.Tensor = test_batch['pcl_in'] # [bs, 1024, 3]
            rotation: torch.Tensor = all_aggregated_pose[i][:, :3, :3] # [bs, 3, 3]
            rotation_t = torch.transpose(rotation, 1, 2) # [bs, 3, 3]
            translation: torch.Tensor = all_aggregated_pose[i][:, :3, 3] # [bs, 3]

            n_pts = pcl.shape[1]
            pcl = pcl - translation.unsqueeze(1) # [bs, 1024, 3]
            pcl = pcl.reshape(-1, 3, 1) # [bs * 1024, 3, 1]
            rotation_t = torch.repeat_interleave(rotation_t, n_pts, dim=0) # [bs * 1024, 3, 3]
            pcl = torch.bmm(rotation_t, pcl).reshape(-1, n_pts, 3) # [bs, 1024, 3]

            bbox_length, _ = torch.max(torch.abs(pcl), dim=1)
            bbox_length *= 2
            all_final_length.append(bbox_length.cpu())

            torch.npu.synchronize()
            elapsed = time.time() - start_time
            perf_stats['bbox_time'].append(elapsed)
            total_samples += pcl.shape[0]
            perf_stats['bbox_samples'] = total_samples

            if i % 10 == 9:
                gc.collect()

        pickle.dump((all_aggregated_pose, all_final_length), open(save_path, 'wb'))
        return
    
    cfg.agent_type = 'scale'
    scale_agent = PoseNet(cfg)
    scale_agent.load_ckpt(model_dir=cfg.pretrained_scale_model_path, model_path=True, load_model_only=True)
    scale_agent.eval()

    all_final_pose = []
    all_final_length = []
    total_samples = 0

    for i, test_batch in enumerate(tqdm(dataloader, desc="scale")):
        torch.npu.synchronize()
        start_time = time.time()

        batch_sample = process_batch(
            batch_sample = test_batch, 
            device=cfg.device, 
            pose_mode=cfg.pose_mode,
        )
        batch_sample.update({key: (None if value is None else value.to(cfg.device)) 
                             for key, value in all_score_feature[i].items()})
        batch_sample['axes'] = all_aggregated_pose[i][:, :3, :3].to(cfg.device)
        cal_mat, length = scale_agent.pred_scale_func(batch_sample)
        final_pose = all_aggregated_pose[i].clone()
        final_pose[:, :3, :3] = cal_mat.cpu()
        all_final_pose.append(final_pose.cpu())
        all_final_length.append(length.cpu())

        torch.npu.synchronize()
        elapsed = time.time() - start_time
        perf_stats['scale_time'].append(elapsed)
        total_samples += length.shape[0]
        perf_stats['scale_samples'] = total_samples

        if i % 4 == 3:
            gc.collect()
    
    pickle.dump((all_final_pose, all_final_length), open(save_path, 'wb'))

def get_detect_match(cls_path, save_path):
    if os.path.exists(save_path):
        return
    assert os.path.exists(cls_path)
    all_final_pose, all_final_length = pickle.load(open(cls_path, 'rb'))

    all_dm = []

    for i, test_batch in enumerate(tqdm(dataloader, desc="detect match")):
        pred_pose = all_final_pose[i].numpy()
        pred_length = torch.clamp(all_final_length[i], min=1e-3).numpy()
        gt_pose = test_batch['affine'].numpy()
        gt_length = test_batch['bbox_side_len'].numpy()
        dm = DetectMatch(
            gt_affine=gt_pose, gt_size=gt_length, 
            gt_sym_labels=array_to_SymLabel(test_batch['sym_info']), gt_class_labels=test_batch['class_label'],
            pred_affine=pred_pose, pred_size=pred_length,
            image_path=[path + 'color.png' for path in test_batch['path']],
            camera_intrinsics=array_to_CameraIntrinsicsBase(test_batch['intrinsics'])
        )
        all_dm.append(dm)
        if i % 10 == 9:
            gc.collect()

    all_dm = DetectMatch.concat(all_dm)
    all_dm = all_dm.calibrate_rotation()

    pickle.dump(all_dm, open(save_path, 'wb'))

def get_criterion(dm_path, save_path):
    if os.path.exists(save_path):
        return
    assert os.path.exists(dm_path)
    all_dm: DetectMatch = pickle.load(open(dm_path, 'rb'))

    criterion = all_dm.criterion()

    pickle.dump(criterion, open(save_path, 'wb'))

def print_metrics(dm_path, criterion_path, save_path):
    assert os.path.exists(dm_path)
    all_dm: DetectMatch = pickle.load(open(dm_path, 'rb'))
    assert os.path.exists(criterion_path)
    criterion: "tuple[np.ndarray, np.ndarray, np.ndarray]" = pickle.load(open(criterion_path, 'rb'))
    
    metrics: Metrics = all_dm.metrics(
        criterion=criterion,
        iou_auc_ranges=[
            (0.25, 1, 0.075),
            (0.5, 1, 0.005),
            (0.75, 1, 0.0025),
        ],
        pose_auc_ranges=[
            ((0, 5, 0.05), (0, 2, 0.02)),
            ((0, 5, 0.05), (0, 5, 0.05)),
            ((0, 10, 0.1), (0, 2, 0.02)),
            ((0, 10, 0.1), (0, 5, 0.05)),
        ],
    )
    print("iou_mean:", metrics.class_means.iou_mean)
    print("iou_acc (0.25, 0.50, 0.75):", metrics.class_means.iou_acc)
    print("deg_mean:", metrics.class_means.deg_mean)
    print("sht_mean:", metrics.class_means.sht_mean)
    print("pose_acc [(5, 2), (5, 5), (10, 2), (10, 5)]:", metrics.class_means.pose_acc)
    print("AUC @ IoU 25:", metrics.class_means.iou_auc[0].auc)
    print("AUC @ IoU 50:", metrics.class_means.iou_auc[1].auc)
    print("AUC @ IoU 75:", metrics.class_means.iou_auc[2].auc)
    print("VUS @ 5 deg 2 cm:", metrics.class_means.pose_auc[0].auc)
    print("VUS @ 5 deg 5 cm:", metrics.class_means.pose_auc[1].auc)
    print("VUS @ 10 deg 2 cm:", metrics.class_means.pose_auc[2].auc)
    print("VUS @ 10 deg 5 cm:", metrics.class_means.pose_auc[3].auc)
    
    metrics.dump_json(save_path)

def print_performance_stats():
    """Print performance statistics for each inference stage."""
    print("\n" + "="*60)
    print("Performance Statistics")
    print("="*60)

    stages = [
        ('Score Network', 'score_time', 'score_samples'),
        ('Energy Network', 'energy_time', 'energy_samples'),
        ('Pose Aggregation', 'aggregate_time', 'aggregate_samples'),
        ('Scale Network', 'scale_time', 'scale_samples'),
        ('Bbox Calculation', 'bbox_time', 'bbox_samples'),
    ]

    for stage_name, time_key, samples_key in stages:
        if len(perf_stats[time_key]) > 0:
            times = perf_stats[time_key]
            samples = perf_stats[samples_key]
            total_time = sum(times)
            avg_batch_time = total_time / len(times)
            fps = samples / total_time if total_time > 0 else 0

            print(f"\n{stage_name}:")
            print(f"  Total batches: {len(times)}")
            print(f"  Total samples: {samples}")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Avg batch time: {avg_batch_time:.3f}s")
            print(f"  FPS: {fps:.3f}")
            print(f"  Avg latency per sample: {1000/fps if fps > 0 else 0:.2f}ms")

    # Calculate overall statistics
    total_samples = perf_stats['score_samples']
    total_time = sum(perf_stats['score_time']) + sum(perf_stats.get('energy_time', [0])) + \
                 sum(perf_stats.get('aggregate_time', [0])) + \
                 sum(perf_stats.get('scale_time', []) if perf_stats['scale_time'] else perf_stats.get('bbox_time', []))

    if total_time > 0:
        overall_fps = total_samples / total_time
        print(f"\n{'='*60}")
        print(f"Overall Pipeline:")
        print(f"  Total samples: {total_samples}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Overall FPS: {overall_fps:.3f}")
        print(f"  Avg latency per sample: {1000/overall_fps:.2f}ms")
        print("="*60)

def visualize_pose_distribution(score_path, dm_path):
    all_pred_pose, _ = pickle.load(open(score_path, 'rb'))
    all_dm: DetectMatch = pickle.load(open(dm_path, 'rb'))

    for i, test_batch in enumerate(tqdm(dataloader)):
        pred_pose = all_pred_pose[i][:, :, :-3]
        pose_rot = get_rot_matrix(pred_pose.reshape(pred_pose.shape[0] * cfg.eval_repeat_num, -1), cfg.pose_mode) \
                    .reshape(pred_pose.shape[0], cfg.eval_repeat_num, 3, 3)
        avg_pose_rot = get_rot_matrix(average_quaternion_batch(matrix_to_quaternion(pose_rot)), 'quat_wxyz')
        gt_pose_rot = test_batch['rotation']
        for j in range(pred_pose.shape[0]):
            index = i * cfg.batch_size + j
            visualize_so3(
                save_path='./so3_distribution.png', 
                pred_rotations=pose_rot[j].cpu().numpy(),
                pred_rotation=all_dm.pred_affine[index, :3, :3],
                gt_rotation=gt_pose_rot[j].cpu().numpy(),
                # probabilities=confidence
            )
            all_dm.draw_image(index=index)
            set_trace()

if __name__ == '__main__':
    dataloader = get_dataloader()
    apply_spec_ops_patches()
    os.makedirs(f'results/evaluation_results/{cfg.result_dir}', exist_ok=True)

    score_model_name = '_'.join(cfg.pretrained_score_model_path.split('/')[-2:])
    score_save_path = f'results/evaluation_results/{cfg.result_dir}/score_prediction_{score_model_name}.pkl'

    # Auto-detect model type
    is_om_model = cfg.pretrained_score_model_path.endswith('.om')

    # For OM models, verify PointNet2 OM exists
    if is_om_model:
        pointnet2_om_path = getattr(cfg, 'pretrained_pointnet2_model_path',
                                      None) or './onnx_models/pointnet2.om'
        if not os.path.exists(pointnet2_om_path):
            raise FileNotFoundError(
                f"PointNet2 OM not found: {pointnet2_om_path}\n"
                f"Please export PointNet2 to OM first:\n"
                f"  python runners/export_onnx.py --agent_type pointnet2\n"
                f"  python runners/onnx2om.py --onnx_path ./onnx_models/pointnet2.onnx"
            )

    # Unified decoupled inference (works for both PyTorch and OM)
    print("="*60)
    if is_om_model:
        print("Using UNIFIED DECOUPLED OM inference")
        print(f"ScoreNet OM: {cfg.pretrained_score_model_path}")
        print(f"PointNet2 OM: {pointnet2_om_path}")
    else:
        print("Using UNIFIED DECOUPLED PyTorch inference")
        print(f"Model: {cfg.pretrained_score_model_path}")
    print("This enables direct PyTorch vs OM comparison for debugging")
    print("="*60)
    inference_score_decoupled(score_save_path)

    aggregate_save_path = f'results/evaluation_results/{cfg.result_dir}/aggregated.pkl'
    if cfg.pretrained_energy_model_path is not None:
        energy_model_name = '_'.join(cfg.pretrained_energy_model_path.split('/')[-2:])
        energy_save_path = f'results/evaluation_results/{cfg.result_dir}/energy_prediction_{energy_model_name}.pkl'
        inference_energy(score_save_path, energy_save_path)
        aggregate_pose(score_save_path, energy_save_path, aggregate_save_path)
    else:
        aggregate_pose(score_save_path, None, aggregate_save_path)

    if cfg.pretrained_scale_model_path is not None:
        scale_model_name = '_'.join(cfg.pretrained_scale_model_path.split('/')[-2:])
    else:
        scale_model_name = 'scale-none'
    cls_save_path = f'results/evaluation_results/{cfg.result_dir}/scale_prediction_{scale_model_name}.pkl'
    inference_scale(score_save_path, aggregate_save_path, cls_save_path)

    dm_save_path = f'results/evaluation_results/{cfg.result_dir}/detect_match.pkl'
    get_detect_match(cls_save_path, dm_save_path)

    criterion_save_path = f'results/evaluation_results/{cfg.result_dir}/criterion.pkl'
    get_criterion(dm_save_path, criterion_save_path)

    metrics_save_path = f'results/evaluation_results/{cfg.result_dir}/metrics.json'
    print_metrics(dm_save_path, criterion_save_path, metrics_save_path)

    # Print performance statistics
    print_performance_stats()