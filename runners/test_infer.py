import os
import sys

# 禁用OpenCV GUI，适用于无图形界面环境
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import numpy as np
import torch
import torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
import random
import cv2
import json
import glob

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import get_config
from datasets.datasets_infer import InferDataset


class GenPose2:
    def __init__(self, score_model_path:str, energy_model_path:str, scale_model_path:str):
        ''' load config '''
        self.cfg = self._get_config(score_model_path, energy_model_path, scale_model_path)

        ''' set random seed '''
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def _get_config(self, score_model_path:str, energy_model_path:str, scale_model_path:str):
        cfg = get_config()
        cfg.pretrained_score_model_path = score_model_path
        cfg.pretrained_energy_model_path = energy_model_path
        cfg.pretrained_scale_model_path = scale_model_path
        cfg.sampler_mode=['ode']
        cfg.T0 = 0.55
        cfg.seed = 0
        cfg.eval_repeat_num = 50
        cfg.dino = 'pointwise'
        return cfg


def create_genpose2(score_model_path:str, energy_model_path:str, scale_model_path:str):
    return GenPose2(score_model_path, energy_model_path, scale_model_path)


def main():
    ######################################## PARAMETERS ########################################
    # 方式1：遍历目录加载（推荐用于批量测试）
    DATA_PATH = 'data/Omni6DPose/ROPE/000007'  # 修改为你的数据目录

    # 方式2：直接指定单张图像（推荐用于快速测试）
    # 直接在下面指定具体的文件路径
    RGB_PATH = 'path/to/your/color.png'
    DEPTH_PATH = 'path/to/your/depth.exr'
    MASK_PATH = 'path/to/your/mask.exr'
    META_PATH = 'path/to/your/meta.json'

    SCORE_MODEL_PATH = 'results/ckpts/ScoreNet/scorenet.pth'
    ENERGY_MODEL_PATH = 'results/ckpts/EnergyNet/energynet.pth'
    SCALE_MODEL_PATH = 'results/ckpts/ScaleNet/scalenet.pth'
    ######################################## PARAMETERS ########################################

    print("=== 测试数据加载 ===")

    # 创建 GenPose2 实例
    genpose2 = create_genpose2(
        score_model_path=SCORE_MODEL_PATH,
        energy_model_path=ENERGY_MODEL_PATH,
        scale_model_path=SCALE_MODEL_PATH,
    )

    # ========== 方式1：遍历目录加载 ==========
    print(f"\n方式1：遍历目录 {DATA_PATH}")
    color_images = sorted(glob.glob(DATA_PATH + '/*_color.png'))
    print(f"找到 {len(color_images)} 张图像")

    for index, color_image in enumerate(color_images[:1]):  # 只测试第一张
        print(f"\n处理第 {index} 张图像: {color_image}")
        data_prefix = color_image.replace('color.png', '')
        print(f"  data_prefix: {data_prefix}")

        try:
            data = InferDataset.alternetive_init(
                data_prefix,
                img_size=genpose2.cfg.img_size,
                device=genpose2.cfg.device,
                n_pts=genpose2.cfg.num_points
            )
            print(f"  数据加载成功！")
            print(f"  图像形状: {data.color.shape}")
            print(f"  深度形状: {data.depth.shape}")
            print(f"  掩码形状: {data.mask.shape}")

            # 测试获取物体数据
            objects = data.get_objects()
            print(f"  物体数量: {objects['pts'].shape[0]}")
            print(f"  点云形状: {objects['pts'].shape}")
        except Exception as e:
            print(f"  数据加载失败: {e}")
            import traceback
            traceback.print_exc()

    # ========== 方式2：直接指定文件路径加载 ==========
    # 取消下面的注释来测试方式2
    """
    print(f"\n方式2：直接指定文件路径")
    print(f"  RGB: {RGB_PATH}")
    print(f"  Depth: {DEPTH_PATH}")
    print(f"  Mask: {MASK_PATH}")
    print(f"  Meta: {META_PATH}")

    try:
        # 加载数据文件
        rgb = cv2.imread(RGB_PATH, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = Dataset.load_depth(DEPTH_PATH)  # 或者用 np.load(DEPTH_PATH) 如果是 .npy 格式
        mask = Dataset.load_mask(MASK_PATH)
        meta = json.load(open(META_PATH, 'r'))

        print(f"  RGB形状: {rgb.shape if rgb is not None else 'None'}")
        print(f"  Depth形状: {depth.shape if depth is not None else 'None'}")
        print(f"  Mask形状: {mask.shape if mask is not None else 'None'}")

        data = InferDataset(
            data={'color': rgb, 'depth': depth, 'mask': mask, 'meta': meta},
            img_size=genpose2.cfg.img_size,
            device=genpose2.cfg.device,
            n_pts=genpose2.cfg.num_points,
        )
        print(f"  数据加载成功！")

        # 测试获取物体数据
        objects = data.get_objects()
        print(f"  物体数量: {objects['pts'].shape[0]}")
        print(f"  点云形状: {objects['pts'].shape}")
    except Exception as e:
        print(f"  数据加载失败: {e}")
        import traceback
        traceback.print_exc()
    """

    print("\n=== 测试完成 ===")


if __name__ == '__main__':
    main()
