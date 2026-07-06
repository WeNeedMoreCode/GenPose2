import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import get_config


''' load config '''
cfg = get_config()
cfg.load_per_object = True


import sys
import time
import cv2
import glob
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import shutil
import hashlib
import random
import gc
from sklearn.cluster import DBSCAN

from ipdb import set_trace

from networks.posenet_agent import PoseNet
from networks.reward import sort_poses_by_energy, ranking_loss
from datasets.datasets_tracking import Omni6DPoseDataSet, array_to_SymLabel, array_to_CameraIntrinsicsBase, process_batch
from utils.metrics import get_rot_matrix
from utils.transforms import matrix_to_quaternion, quaternion_to_matrix
from utils.misc import average_quaternion_batch, get_pose_dim, get_pose_representation
from utils.so3_visualize import visualize_so3
from utils.visualize import create_grid_image
from utils.tracking_utils import add_noise_to_RT
from cutoop.eval_utils import DetectMatch, Metrics
from cutoop.data_loader import Dataset

torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)

def get_dataloader(data_dir: str):
    dataset = Omni6DPoseDataSet(
        cfg=cfg,
        dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
        deform_2d_params=cfg.DEFORM_2D_PARAMS,
        source='Omni6DPose',
        mode='real',
        data_dir=data_dir,
        n_pts=1024,
        img_size=cfg.img_size,
        per_obj=None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset.num_valid,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        pin_memory=True,
    )
    return iter(dataloader)

# PTH/OM model loading
is_om_model = cfg.pretrained_score_model_path.endswith('.om')

if not is_om_model:
    import torch_npu
    torch_npu.npu.set_compile_mode(jit_compile=False)
    from ascendc_kernels.fps_ascendc import patch_pointnet2_fps
    patch_pointnet2_fps(num_cores=8)
    from ascendc_kernels.group_points_ascendc import patch_group_points
    patch_group_points(num_cores=8)
    from ascendc_kernels.ball_query_ascendc import patch_ball_query
    patch_ball_query(num_cores=8)
    cfg.agent_type = 'score'
    score_agent = PoseNet(cfg)
    score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
    score_agent.eval()

    cfg.agent_type = 'energy'
    energy_agent = PoseNet(cfg)
    energy_agent.load_ckpt(model_dir=cfg.pretrained_energy_model_path, model_path=True, load_model_only=True)
    energy_agent.eval()

    if cfg.pretrained_scale_model_path:
        cfg.agent_type = 'scale'
        scale_agent = PoseNet(cfg)
        scale_agent.load_ckpt(model_dir=cfg.pretrained_scale_model_path, model_path=True, load_model_only=True)
        scale_agent.eval()
else:
    from om_wrappers import (create_score_network, create_ode_sampler,
                             DINOv2Wrapper, EnergyNetWrapper,
                             PointNet2EncoderWrapper, PointNet2SplitOM,
                             PointNet2AscendC, PointNet2SplitAscendC,
                             ScaleNetWrapper)
    from networks.gf_algorithms.sde import init_sde, ve_sde_numpy
    from datasets.datasets_omni6dpose import process_batch_numpy
    from utils.misc import get_pose_dim

    prior_fn, _, sde_fn, sampling_eps, _ = init_sde('ve')
    pointnet2_score_om_path = getattr(cfg, 'pretrained_pointnet2_score_model_path', None)

    # POINTNET2_BACKEND options:
    #   split_om:      CPU indexing (fpsample+OMP) + per-SA OM MLP
    #   split_ascendc: NPU indexing (AscendC kernels) + per-SA OM MLP
    #   ascendc:       full PTH model + AscendC kernel patches (no split)
    #   (default):     monolithic PointNet2 OM
    pn2_backend = os.environ.get('POINTNET2_BACKEND', '')
    split_om_dir = os.environ.get('POINTNET2_SPLIT_OM_DIR', './om_models')
    score_pth_path = './results/ckpts/ScoreNet/scorenet.pth'
    energy_pth_path = './results/ckpts/EnergyNet/energynet.pth'

    score_pointnet2_encoder = None
    energy_pointnet2_encoder = None

    if pn2_backend == 'split_om':
        score_pointnet2_encoder = PointNet2SplitOM(
            om_dir=split_om_dir, device=cfg.device, name_suffix='_from_score')
        energy_pointnet2_encoder = PointNet2SplitOM(
            om_dir=split_om_dir, device=cfg.device, name_suffix='_from_energy')
        print(f"Using split PointNet2 (CPU indexing + OM MLP) from {split_om_dir}")
    elif pn2_backend == 'split_ascendc':
        score_pointnet2_encoder = PointNet2SplitAscendC(
            om_dir=split_om_dir, device=cfg.device, name_suffix='_from_score')
        energy_pointnet2_encoder = PointNet2SplitAscendC(
            om_dir=split_om_dir, device=cfg.device, name_suffix='_from_energy')
        print(f"Using split PointNet2 (AscendC indexing + OM MLP) from {split_om_dir}")
    elif pn2_backend == 'ascendc':
        score_pointnet2_encoder = PointNet2AscendC(
            pth_checkpoint_path=score_pth_path, device=cfg.device, patch_mode='ascendc')
        energy_pointnet2_encoder = PointNet2AscendC(
            pth_checkpoint_path=energy_pth_path, device=cfg.device, patch_mode='ascendc')
        print(f"Using AscendC PointNet2 (full PTH + AscendC NPU kernels)")
    elif pn2_backend == 'cpu_ops':
        score_pointnet2_encoder = PointNet2AscendC(
            pth_checkpoint_path=score_pth_path, device=cfg.device, patch_mode='graspnet_cpu')
        energy_pointnet2_encoder = PointNet2AscendC(
            pth_checkpoint_path=energy_pth_path, device=cfg.device, patch_mode='graspnet_cpu')
        print(f"Using CPU-ops PointNet2 (full PTH + CPU fpsample/OMP)")

    # When using a custom PointNet2 encoder, don't pass pointnet2_om_path (avoid loading monolithic OM)
    score_om_path = None if score_pointnet2_encoder else pointnet2_score_om_path

    score_net = create_score_network(
        checkpoint_path=cfg.pretrained_score_model_path,
        device=cfg.device,
        pointnet2_om_path=score_om_path,
        pointnet2_encoder=score_pointnet2_encoder,
    )
    sde_dir = {'prior_fn': prior_fn, 'sde_fn': ve_sde_numpy}
    sampler = create_ode_sampler(score_network=score_net, sde=sde_dir, device=cfg.device)

    energy_net = EnergyNetWrapper(cfg.pretrained_energy_model_path, device=cfg.device)
    if energy_pointnet2_encoder is not None:
        pointnet2_energy_encoder = energy_pointnet2_encoder
    else:
        pointnet2_energy_encoder = PointNet2EncoderWrapper(
            cfg.pretrained_pointnet2_energy_model_path, device=cfg.device)
    print(f"Using EnergyNet OM: {cfg.pretrained_energy_model_path}")

    if cfg.pretrained_scale_model_path:
        scale_net = ScaleNetWrapper(cfg.pretrained_scale_model_path, device=cfg.device)
        print(f"Using ScaleNet OM: {cfg.pretrained_scale_model_path}")

    if cfg.dino != 'none':
        dino_om_path = getattr(cfg, 'pretrained_dino_model_path', None)
        if dino_om_path is not None and dino_om_path.endswith('.om'):
            dino_model = DINOv2Wrapper(dino_om_path, device=cfg.device)
            print(f"Using DINOv2 OM: {dino_om_path}")
        else:
            import torch.hub
            dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(cfg.device)
            dino_model.requires_grad_(False)
            print("Using DINOv2 PyTorch")

        def extract_dino_features(batch_sample):
            roi_rgb = batch_sample['roi_rgb']
            roi_xs = batch_sample['roi_xs']
            roi_ys = batch_sample['roi_ys']
            if is_om_model:
                return dino_model(roi_rgb, roi_xs, roi_ys)
            feat = dino_model.get_intermediate_layers(roi_rgb)[0]
            xs = roi_xs // 14
            ys = roi_ys // 14
            pos = xs * 16 + ys
            pos = torch.unsqueeze(pos, -1).expand(-1, -1, 384)
            rgb_feat = torch.gather(feat, 1, pos)
            rgb_feat.requires_grad_(False)
            return rgb_feat
    else:
        def extract_dino_features(batch_sample):
            return None

def work_batch(test_batch, prev_pose):
    if is_om_model:
        batch_sample = process_batch_numpy(test_batch, pose_mode=cfg.pose_mode)
    else:
        batch_sample = process_batch(
            batch_sample = test_batch,
            device=cfg.device,
            pose_mode=cfg.pose_mode,
        )

    bs = prev_pose.shape[0]
    pose_dim = get_pose_dim(cfg.pose_mode) if is_om_model else prev_pose.shape[1]
    repeat_num = cfg.eval_repeat_num

    if is_om_model:
        # OM score path
        t0 = time.time()
        # DINOv2 + PointNet2 feature extraction
        rgb_feat = extract_dino_features(batch_sample)
        t_dino = time.time()
        pts_feat = score_net.extract_pts_feat(batch_sample['pts'], rgb_feat)
        t_pt2 = time.time()
        if os.environ.get('POINTNET2_DEBUG'):
            print(f"  DINOv2: {(t_dino-t0)*1000:.1f}ms, PointNet2: {(t_pt2-t_dino)*1000:.1f}ms")

        # Construct init_x: repeat prev_pose and add noise 
        _prev_pose = prev_pose.cpu().numpy().copy()
        _prev_pose[:, -3:] -= batch_sample['pts_center']
        noise = prior_fn((bs * repeat_num, pose_dim), T=cfg.T0).numpy()
        prev_pose_repeated = np.repeat(_prev_pose, repeat_num, axis=0)
        init_x_repeated = prev_pose_repeated + noise

        pts_feat_repeated = np.repeat(pts_feat[np.newaxis, ...], repeat_num, axis=1).reshape(bs * repeat_num, -1)

        _, sampled_pose = sampler.sample(
            pts_feat=pts_feat_repeated,
            rgb_feat=None,
            batch_size=bs * repeat_num,
            pose_dim=pose_dim,
            T=cfg.T0,
            eps=sampling_eps,
            rtol=1e-5,
            atol=1e-5,
            denoise=True,
            init_x=init_x_repeated,
            pts_center=None if batch_sample.get('pts_center') is None else
                np.repeat(batch_sample['pts_center'][:, np.newaxis, :], repeat_num, axis=1).reshape(bs * repeat_num, -1),
        )
        score_pred_results = sampled_pose.reshape(bs, repeat_num, pose_dim)

        score_feature = {
            'pts_feat': pts_feat,
            'rgb_feat': rgb_feat,
        }

        # OM energy
        pts_with_rgb = np.concatenate([batch_sample['pts'], rgb_feat], axis=-1)  # [bs, 1024, 387]
        pts_feat_energy = pointnet2_energy_encoder(pts_with_rgb)

        pose_samples = score_pred_results.reshape(bs * repeat_num, -1).astype(np.float32)
        pose_samples[:, -3:] -= np.repeat(batch_sample['pts_center'], repeat_num, axis=0)
        t = np.full((bs * repeat_num, 1), 1e-5, dtype=np.float32)
        pts_feat_repeated_energy = np.repeat(pts_feat_energy[np.newaxis, ...], repeat_num, axis=1).reshape(bs * repeat_num, -1)

        with torch.no_grad():
            pred_energy = energy_net(pts_feat_repeated_energy, pose_samples, t)
        energy_pred_results = pred_energy.reshape(bs, repeat_num, -1)
        perf_stats['score_time'].append(time.time() - t0)
        perf_stats['score_samples'] += bs

    else:
        # PTH path (original) 
        t0 = time.time()
        _prev_pose = prev_pose.clone()
        _prev_pose[:, -3:] -= batch_sample['pts_center']
        cfg.agent_type = 'score'
        score_pred_results, _ = score_agent.pred_func(
            data=batch_sample,
            repeat_num=cfg.eval_repeat_num,
            T0=cfg.T0,
            init_x=_prev_pose,
            return_average_res=False,
            return_process=False,
        )
        score_feature = {
            'pts_feat': batch_sample['pts_feat'].clone(),
            'rgb_feat': (None if batch_sample['rgb_feat'] is None else batch_sample['rgb_feat'].clone()),
        }

        cfg.agent_type = 'energy'
        energy_pred_results = energy_agent.get_energy(
            data=batch_sample,
            pose_samples=score_pred_results,
            T=1e-5,
            mode='test',
            extract_feature=True
        )
        perf_stats['score_time'].append(time.time() - t0)
        perf_stats['score_samples'] += bs

    # Convert numpy to tensor for sort and aggregate operations
    if is_om_model:
        score_pred_results = torch.from_numpy(score_pred_results)
        energy_pred_results = torch.from_numpy(energy_pred_results)

    # aggregate + scale
    t0 = time.time()
    sorted_pose, sorted_energy = sort_poses_by_energy(
        score_pred_results, energy_pred_results)
    bs = score_pred_results.shape[0]
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

    pred_pose = aggregated_pose.numpy()
    gt_pose = test_batch['affine'].numpy()
    gt_length = test_batch['bbox_side_len'].numpy()

    if cfg.pretrained_scale_model_path:
        if is_om_model:
            pred_length = scale_net(pts_feat, aggregated_pose[:, :3, :3])
        else:
            cfg.agent_type = 'scale'
            batch_sample.update(score_feature)
            batch_sample['axes'] = aggregated_pose[:, :3, :3].to(cfg.device)
            with torch.no_grad():
                pred_length = scale_agent.net(batch_sample)
            pred_length = pred_length.cpu().numpy()
    else:
        pred_length = np.ones((pred_pose.shape[0], 3))

    detect_match = DetectMatch(
        gt_affine=gt_pose, gt_size=gt_length,
        gt_sym_labels=array_to_SymLabel(test_batch['sym_info']),
        gt_class_labels=test_batch['class_label'],
        pred_affine=pred_pose, pred_size=pred_length,
        # image_path=[path + 'color.png' for path in test_batch['path']],
        camera_intrinsics=array_to_CameraIntrinsicsBase(test_batch['intrinsics'])
    )
    perf_stats['aggregate_time'].append(time.time() - t0)
    perf_stats['aggregate_samples'] += bs

    prev_pose = torch.zeros_like(
        prev_pose, 
        device=cfg.device if not is_om_model else 'cpu'
    )
    prev_pose[:, :-3] = get_pose_representation(aggregated_pose[:, :3, :3], cfg.pose_mode)
    prev_pose[:, -3:] = aggregated_pose[:, :3, 3]

    return detect_match, prev_pose

img_list = Dataset.glob_prefix(root = cfg.data_path)
video_paths = sorted({os.path.dirname(path) for path in img_list})

dataloaders: "set[torch.utils.data.DataLoader]" = set()

idx = 0
def add_dataloader():
    global idx
    while idx < len(video_paths):
        path = video_paths[idx]
        idx += 1
        save_path = f"results/evaluation_results/{cfg.result_dir}/{path.replace('/', '-')}/all_detect_match.pkl"
        if os.path.exists(save_path):
            continue
        dataloader = get_dataloader(path)
        dataloader.save_path = save_path
        dataloader.all_detect_match = []
        dataloaders.add(dataloader)
        break

total_objects = 0
for path in tqdm(video_paths):
    save_path = f"results/evaluation_results/{cfg.result_dir}/{path.replace('/', '-')}/all_detect_match.pkl"
    if os.path.exists(save_path):
        continue
    dataset = Omni6DPoseDataSet(
        cfg=cfg,
        dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
        deform_2d_params=cfg.DEFORM_2D_PARAMS,
        source='Omni6DPose',
        mode='real',
        data_dir=path,
        n_pts=1024,
        img_size=cfg.img_size,
        per_obj=None,
    )
    total_objects += len(dataset)
pbar = tqdm(total=total_objects)

for i in range(30):
    add_dataloader()

perf_stats = {
    'score_time': [],
    'score_samples': 0,
    'aggregate_time': [],
    'aggregate_samples': 0,
}

while 1:
    test_batch = []
    prev_pose = []
    split_pos = [(0,  None)]
    dd = set()
    for i, dataloader in enumerate(dataloaders):
        try:
            batch = dataloader.__next__()
        except StopIteration:
            if len(dataloader.all_detect_match): # otherwise, an error occurred in the dataset
                all_detect_match = DetectMatch.concat(dataloader.all_detect_match)
                all_detect_match = all_detect_match.calibrate_rotation()
                os.makedirs(os.path.dirname(dataloader.save_path), exist_ok="True")
                pickle.dump(all_detect_match, open(dataloader.save_path, "wb"))
            dd.add(dataloader)
            continue
        except AssertionError:
            with open("tracking_fail.txt", "a") as f:
                f.write(dataloader.save_path + '\n')
            dd.add(dataloader)
            continue
        length = dataloader._dataset.num_valid
        if batch.get('_corrupted', torch.tensor(False)).any():
            print(f"[SKIP] Corrupted frame in {dataloader.save_path}")
            pbar.update(length)
            continue
        test_batch.append(batch)
        try:
            prev_pose.append(dataloader.prev_pose)
        except:
            pose = torch.zeros(
                length, get_pose_dim(cfg.pose_mode), 
                device=cfg.device if not is_om_model else 'cpu'
            )
            assert batch['affine'].shape[0] == length, set_trace()
            for j in range(length):
                noise_gt_pose = add_noise_to_RT(batch['affine'][j].to(
                    cfg.device if not is_om_model else 'cpu'
                ).unsqueeze(0))[0]
                pose[j, :-3] = get_pose_representation(
                    noise_gt_pose[:3, :3].unsqueeze(0), 
                    pose_mode=cfg.pose_mode
                )[0]
                pose[j, -3:] = noise_gt_pose[:3, 3]
            prev_pose.append(pose)
        split_pos.append((split_pos[-1][0] + length, dataloader))
        if split_pos[-1][0] > cfg.batch_size - 8:
            break
    if test_batch == []:
        for dl in dd:
            dataloaders.remove(dl)
        if len(dataloaders) == 0:
            break
        continue
    
    keys = {key for key, value in test_batch[0].items() if type(value) != list}
    test_batch = {
        key: torch.concat([batch[key] for batch in test_batch]) for key in keys
    }
    prev_pose = torch.concat(prev_pose)
    
    detect_match, prev_pose = work_batch(test_batch, prev_pose)
    for i in range(len(split_pos) - 1):
        l, r = split_pos[i][0], split_pos[i+1][0]
        dataloader = split_pos[i+1][1]
        dataloader.all_detect_match.append(detect_match[l:r])
        dataloader.prev_pose = prev_pose[l:r]
    
    pbar.update(split_pos[-1][0])
        
    for dl in dd:
        dataloaders.remove(dl)
        add_dataloader()
    
    gc.collect()

pbar.close()

# Print performance statistics
print("\n" + "="*60)
print("Performance Statistics")
print("="*60)

stages = [
    ('Score + Energy', 'score_time', 'score_samples'),
    ('Aggregate + Scale', 'aggregate_time', 'aggregate_samples'),
]

for stage_name, time_key, samples_key in stages:
    times = perf_stats[time_key]
    samples = perf_stats[samples_key]
    if len(times) > 0:
        t_total = sum(times)
        fps = samples / t_total if t_total > 0 else 0
        print(f"\n{stage_name}:")
        print(f"  Total batches: {len(times)}")
        print(f"  Total samples: {samples}")
        print(f"  Total time: {t_total:.3f}s")
        print(f"  Avg batch time: {t_total/len(times):.3f}s")
        print(f"  FPS: {fps:.3f}")
        print(f"  Avg latency per sample: {1000/fps:.2f}ms" if fps > 0 else "")

total_samples = perf_stats['score_samples']
total_time = sum(perf_stats['score_time']) + sum(perf_stats['aggregate_time'])

if total_time > 0:
    overall_fps = total_samples / total_time
    print(f"\n{'='*60}")
    print(f"Overall Pipeline:")
    print(f"  Total samples: {total_samples}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Overall FPS: {overall_fps:.3f}")
    print(f"  Avg latency per sample: {1000/overall_fps:.2f}ms")
    print("="*60)

all_dm = []
all_crit = []
for path in tqdm(video_paths):
    save_path = f"results/evaluation_results/{cfg.result_dir}/{path.replace('/', '-')}/all_detect_match.pkl"
    if not os.path.exists(save_path):
        continue
    prefix = os.path.dirname(save_path)
    dm: DetectMatch = pickle.load(open(save_path, "rb"))
    all_dm.append(dm)
    crit_path = os.path.join(prefix, "criterion.pkl")
    if not os.path.exists(crit_path):
        criterion = dm.criterion(computeIOU=True)
        pickle.dump(criterion, open(crit_path, "wb"))
    else:
        criterion = pickle.load(open(crit_path, "rb"))
    all_crit.append(criterion)

all_dm = DetectMatch.concat(all_dm)
all_crit = [np.concatenate([all_crit[j][i] for j in range(len(all_crit))]) for i in range(3)]

metrics: Metrics = all_dm.metrics(
    criterion=all_crit,
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
metrics.dump_json(os.path.join(f"results/evaluation_results/{cfg.result_dir}", "metrics.json"))