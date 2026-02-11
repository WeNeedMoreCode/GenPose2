# import cv2
# import numpy as np
# from cutoop.data_loader import Dataset
# mask = cv2.imread('/home/huawei/genpose2_without_cuda/data/mask_genpose.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# mask = cv2.bitwise_not(mask)
# cv2.imwrite("/home/huawei/genpose2_without_cuda/data/mask_genpose_2.png", mask)

# print(mask.shape)
# print(mask)

# print(type(mask))
# print(sum(sum(mask)))
# mask = Dataset.load_mask('/home/huawei/genpose2_without_cuda/data/ROPE/000000/mask/000000_mask.exr')
# img = cv2.imread('/home/huawei/genpose2_without_cuda/data/ROPE/000000/mask/000000_mask.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# if len(img.shape) == 3:
#     img = img[:, :, 2]
# mask = np.array(img * 255, dtype=np.uint8)
# cv2.imwrite("/home/huawei/genpose2_without_cuda/data/mask_temp.png", mask)


# rgb = cv2.imread('/home/huawei/genpose2_without_cuda/data/rgb_genpose.png', cv2.IMREAD_UNCHANGED)
# print(rgb.shape)
# print(rgb)

import numpy as np

# 原始6D位姿矩阵（4x4）
T = np.array([
    [0.10677963, -0.01527576,  0.99416536,  0.28084162],
    [-0.492441,   -0.86944747,  0.03953188,  0.1410087 ],
    [0.8637708,   -0.49378902, -0.1003617,   0.76861596],
    [0.0,          0.0,          0.0,          1.0       ]
])

# 定义坐标系转换的置换矩阵
# 新x轴 = 原z轴，新y轴 = 原x轴，新z轴 = 原y轴
S = np.array([
    [0, 0, 1],  # 新x轴对应原z轴
    [1, 0, 0],  # 新y轴对应原x轴
    [0, 1, 0]   # 新z轴对应原y轴
])

# 提取旋转矩阵和平移向量
R = T[:3, :3]  # 3x3旋转矩阵
t = T[:3, 3]   # 3x1平移向量

# 计算变换后的旋转矩阵和平移向量
# 旋转矩阵变换：R_new = S @ R
# 平移向量也需要相应变换：t_new = S @ t
R_new = S @ R
t_new = S @ t

# 组合新的4x4位姿矩阵
T_new = np.eye(4)
T_new[:3, :3] = R_new
T_new[:3, 3] = t_new

# 输出结果
print("原始位姿矩阵：")
print(T)
print("\n转换后的位姿矩阵（z→x, x→y, y→z）：")
print(T_new)

# 验证旋转矩阵的正交性（应接近单位矩阵）
print("\n验证新旋转矩阵的正交性（应接近单位矩阵）：")
print(R_new.T @ R_new)

# 验证旋转矩阵的行列式（应接近1）
print("\n新旋转矩阵的行列式（应接近1）：")
print(np.linalg.det(R_new))
    