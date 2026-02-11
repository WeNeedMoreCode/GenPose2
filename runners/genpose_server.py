import json
import os
import sys
import socket
import pickle
import numpy as np
import torch
import cv2
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.posenet_agent import PoseNet
from utils.metrics import get_rot_matrix
from utils.misc import average_quaternion_batch, get_pose_dim, get_pose_representation
from utils.transforms import matrix_to_quaternion, quaternion_to_matrix
from datasetsgp.datasets_infer import InferDataset
from configs.config import get_config
from infer import visualize_pose, GenPose2, create_genpose2

class GenPoseTCPServer:
    def __init__(self, host='0.0.0.0', port=5000, buffer_size=4096):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.server_socket = None
        self.genpose = None
        self.is_running = False
        self.tracking = True
        self.tracking_t0 = 0.15
        self.init_genpose()

    def init_genpose(self):
        '''初始化GenPose模型'''
        self.score_model_path = '../results/ckpts/ScoreNet/scorenet.pth'
        self.energy_model_path = '../results/ckpts/EnergyNet/energynet.pth'
        self.scale_model_path = '../results/ckpts/ScaleNet/scalenet.pth'
        # self.genpose = self._init_genpose_model()
        self.genpose = create_genpose2(
            score_model_path=self.score_model_path, 
            energy_model_path=self.energy_model_path,
            scale_model_path=self.scale_model_path,
        )
        self.prev_pose = None

    # def _init_genpose_model(self):
    #     '''初始化GenPose2模型'''
    #     cfg = get_config()
    #     cfg.pretrained_score_model_path = self.score_model_path
    #     cfg.pretrained_energy_model_path = self.energy_model_path
    #     cfg.pretrained_scale_model_path = self.scale_model_path
    #     cfg.sampler_mode=['ode']
    #     cfg.T0 = 0.55
    #     cfg.seed = 0
    #     cfg.eval_repeat_num = 50
    #     cfg.dino = 'pointwise'
    #     cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #     # 加载score模型
    #     cfg.agent_type = 'score'
    #     score_agent = PoseNet(cfg)
    #     score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
    #     score_agent.eval()

    #     # 加载energy模型
    #     cfg.agent_type = 'energy'
    #     energy_agent = PoseNet(cfg)
    #     energy_agent.load_ckpt(model_dir=cfg.pretrained_energy_model_path, model_path=True, load_model_only=True)
    #     energy_agent.eval()

    #     # 加载scale模型
    #     cfg.agent_type = 'scale'
    #     scale_agent = PoseNet(cfg)
    #     scale_agent.load_ckpt(model_dir=cfg.pretrained_scale_model_path, model_path=True, load_model_only=True)
    #     scale_agent.eval()

    #     return {
    #         'cfg': cfg,
    #         'score_agent': score_agent,
    #         'energy_agent': energy_agent,
    #         'scale_agent': scale_agent
    #     }

    def start(self):
        '''启动TCP服务器'''
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.is_running = True
        print(f"GenPose TCP Server listening on {self.host}:{self.port}")

        while self.is_running:
            client_socket, client_address = self.server_socket.accept()
            print(f"Accepted connection from {client_address}")
            client_handler = threading.Thread(target=self.handle_client, args=(client_socket,))
            client_handler.start()

    def handle_client(self, client_socket):
        '''处理客户端连接'''
        try:
            # 接收数据长度
            data_len_bytes = client_socket.recv(4)
            if not data_len_bytes:
                return
            data_len = int.from_bytes(data_len_bytes, byteorder='big')
            r_time = time.perf_counter()
            # 接收数据
            data_bytes = b''
            while len(data_bytes) < data_len:
                chunk = client_socket.recv(min(self.buffer_size, data_len - len(data_bytes)))
                if not chunk:
                    break
                data_bytes += chunk

            # 反序列化数据
            data_dict = pickle.loads(data_bytes)
            t0 = time.perf_counter()
            print(f'receive time: {(t0 - r_time)*1000:.2f}ms')
            print(f'data len: {data_len}')

            # 执行推理
            t1 = time.perf_counter()
            result = self.inference(data_dict)
            t2 = time.perf_counter()
            print(f'inference time: {(t2 - t1)*1000:.2f}ms')

            # 序列化结果
            result_bytes = pickle.dumps(result)
            result_len = len(result_bytes).to_bytes(4, byteorder='big')
            t3 = time.perf_counter()
            print(f'serialize time: {(t3 - t2)*1000:.2f}ms')
            # 发送结果
            client_socket.sendall(result_len + result_bytes)
            t4 = time.perf_counter()
            print(f'send time: {(t4 - t3)*1000:.2f}ms')
            print(f'total time: {(t4 - r_time)*1000:.2f}ms')

        except Exception as e:
            print(f"Error handling client: {str(e)}")
            error_result = {'status': 'error', 'message': str(e)}
            error_bytes = pickle.dumps(error_result)
            error_len = len(error_bytes).to_bytes(4, byteorder='big')
            client_socket.sendall(error_len + error_bytes)
        finally:
            client_socket.close()

    def inference(self, data_dict):
        '''执行推理'''
        # 从字典中提取数据
        # rgb = data_dict['rgb']
        rgb = cv2.imread("/home/huawei/yolov10_om_infer/data/rgb_yolo2.png")
               
        # 转换为NumPy数组
        depth = data_dict['depth']
        mask = data_dict['mask']
        # meta = data_dict['meta']
        with open('/home/huawei/genpose2_without_cuda/data/meta_genpose.json', 'r') as f:
            meta = json.load(f)
        # mask = np.array(mask * 255, dtype=np.uint8)
        mask = cv2.bitwise_not(mask)

        print("------------------depth--------------------------")
        print(depth.dtype)
        print("depth:", depth.shape)
        # print(depth)
        print("------------------mask--------------------------")
        print(mask.dtype)
        print("mask:", mask.shape)
        # print(mask)
        np.save("/home/huawei/genpose2_without_cuda/data/depth_genpose.npy", depth)
        cv2.imwrite("/home/huawei/genpose2_without_cuda/data/rgb_genpose.png", rgb)
        cv2.imwrite("/home/huawei/genpose2_without_cuda/data/mask_genpose.png", mask)
        # json.dump(meta, open("/home/huawei/genpose2_without_cuda/data/meta_genpose.json", 'w'))
        
        # 创建InferDataset实例
        dataset = InferDataset(
            data={'color': rgb, 'depth': depth, 'mask': mask, 'meta': meta},
            img_size=self.genpose.cfg.img_size,
            device=self.genpose.cfg.device,
            n_pts=self.genpose.cfg.num_points,
        )

        # dataset = InferDataset(
        #     data={'color': rgb, 'depth': depth, 'mask': mask, 'meta': meta},
        #     img_size=self.genpose['cfg'].img_size,
        #     device=self.genpose['cfg'].device,
        #     n_pts=self.genpose['cfg'].num_points
        # )

        # 执行推理
        # data_objects = dataset.get_objects()
        # all_pred_pose, all_score_feature = self._inference_score(dataset)
        # all_pred_energy = self._inference_energy(dataset, all_pred_pose)
        # all_aggregated_pose = self._aggregate_pose(all_pred_pose, all_pred_energy)
        # all_final_pose, all_final_length = self._inference_scale(dataset, all_score_feature, all_aggregated_pose)

        all_final_pose, all_final_length = self.genpose.inference(dataset, self.prev_pose, self.tracking, self.tracking_t0)

        path1 = '/home/huawei/genpose2_without_cuda/data/rgb_genpose_outputs.png'
        color_image_w_pose = visualize_pose(dataset, all_final_pose, all_final_length, visualize_image=False, path=path1)


        # ==== 添加位姿转换代码 ====
        # 提取旋转矩阵R和平移向量T (假设pose[0]是目标物体位姿)
        R = all_final_pose[0][0][:3, :3]  # 3x3旋转矩阵
        T = all_final_pose[0][0][:3, 3]   # 平移向量 [x, y, z]

        # 获取物体高度 (假设length的第二个元素是Z轴高度)
        obj_height = all_final_length[0][0][1]  # 0.2617901

        # 物体局部坐标系中从中心到底部的平移 (沿Z轴负方向移动半个高度)
        translation_local = np.array([0, -obj_height / 2, 0])  # [0, 0, -0.130895]

        # 将局部平移转换到相机坐标系 (旋转矩阵×局部平移)
        translation_camera = R @ translation_local

        # 更新平移向量 (原平移 + 转换后的局部平移)
        new_T = T + translation_camera
        all_final_pose[0][0][:3, 3] = new_T
        # ===========================

        # 可视化结果
        path2 = '/home/huawei/genpose2_without_cuda/data/rgb_genpose_outputs_2.png'
        color_image_w_pose = visualize_pose(dataset, all_final_pose, all_final_length, visualize_image=False, path=path2)
        # cv2.imshow('rgb', color_image_w_pose)
        # cv2.waitKey(1)

        poses = []
        pose = all_final_pose[0].numpy()

        # for pose, length in zip(all_final_pose[0], all_final_length[0]):
        #     init_pose = np.eye(4)
        #     init_pose[:3, :3] = pose[:3, :3]
        #     init_pose[:3, 3] = pose[:3, 3]
        #     # poses.append({
        #     #     'rotation': pose[:3, :3].tolist(),
        #     #     'translation': pose[:3, 3].tolist(),
        #     #     'scale': length.tolist()
        #     # })
        #     poses.append(init_pose)

        return {'status': 'success', 'poses': pose[0]}


    def stop(self):
        '''停止服务器'''
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()
        print("GenPose TCP Server stopped")

if __name__ == '__main__':
    server = GenPoseTCPServer(host='192.168.2.11', port=9000)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()