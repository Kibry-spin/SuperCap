import numpy as np
import json
import os
import time
import cv2
import trimesh
import torch
import open3d as o3d
from open3d import utility as o3du
from open3d import geometry as o3dg
from scipy.spatial.transform import Rotation as sciR
from manotorch.manolayer import ManoLayer
# from raisimPy.examples.mano_utils import quaternion_to_angle_axis

def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def hand_mesh_instance(verts, faces, trans=0., uni_color=[0.9, 0.5, 0.4]):
    """创建手部网格实例"""
    hand_mesh = o3dg.TriangleMesh()
    if torch.is_tensor(verts):
        verts = verts.detach().cpu().squeeze().numpy()

    if torch.is_tensor(faces):
        faces = faces.cpu().squeeze().numpy()

    verts[:, 0] += trans

    hand_mesh.vertices = o3du.Vector3dVector(verts)
    hand_mesh.triangles = o3du.Vector3iVector(faces)
    hand_mesh.paint_uniform_color(uni_color)
    hand_mesh.compute_vertex_normals()
    return hand_mesh

def get_finger_data(frame, hand='right', finger_order=['thumb', 'index', 'middle', 'ring', 'pinky']):
    """获取指定手指的欧拉角数据"""
    finger_indices = {
        'right': {
            'thumb': slice(7, 19),
            'index': slice(19, 31),
            'middle': slice(31, 43),
            'ring': slice(43, 55),
            'pinky': slice(55, 67)
        },
        'left': {
            'thumb': slice(74, 86),
            'index': slice(86, 98),
            'middle': slice(98, 110),
            'ring': slice(110, 122),
            'pinky': slice(122, 134)
        }
    }

    hand_data = []
    frame = np.array(frame)  # 转换为numpy数组
    for finger_name in finger_order:
        indices = finger_indices[hand][finger_name]
        finger_data = frame[indices].reshape(3, 4)  # 每个关节3个欧拉角
        hand_data.append(finger_data)
    hand_data = np.stack(hand_data, axis=0)  # (5,3,4)
    return hand_data


class IMU2MANO:
    def __init__(self):
        self.mano_layer = ManoLayer(mano_assets_root='E:\Project\ForheartUnity\MANO',
                                    use_pca=False, ncomps=45, flat_hand_mean=True)
        self.hand_faces_np = self.mano_layer.th_faces.cpu().squeeze().numpy()

        self.mano_shape = torch.zeros(1, 10)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=False, width=840, height=560)


        FOR1 = o3dg.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.vis.add_geometry(FOR1)

    @staticmethod
    def quat2axis(quat_params):
        """
        :param quat_params: (N,4)
        :return: roevec_params (N, 3)
        """
        roevec_params = quaternion_to_angle_axis(torch.tensor(quat_params))
        # transform_R = sciR.from_quat(quat_params)
        # rotmat_params = transform_R.as_matrix()
        # roevec_params = transform_R.as_rotvec()
        return roevec_params.cpu().numpy()
    @staticmethod
    def add_text_to_image(image, text):
        # 在图像上添加文字 (例：在左上角添加注释)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        position = (50, 50)  # 文字的起始位置 (x, y)
        font_scale = 3  # 字体大小
        color = (0, 254, 0)  # BGR 颜色（绿色）
        thickness = 3  # 线条粗细
        cv2.putText(image, text, position, font, font_scale, color, thickness)
        return image

    def data_load(self):
        """从raw_data.txt加载数据"""
        data_dir = 'E:\Project\ForheartUnity\glove_data/20250120_192115'
        raw_data_path = os.path.join(data_dir, 'raw_data.txt')
        print(f"从文件加载数据: {raw_data_path}")
        
        hand_seq_data = []
        with open(raw_data_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # 查找数据起始位置
                    start_idx = line.find("0.15,0,0.2,")
                    if start_idx == -1:
                        continue
                    
                    # 提取数值部分
                    values_str = line[start_idx:]
                    values = [float(x) for x in values_str.strip().split(',') if x]  # 过滤空字符串
                    
                    if len(values) >= 134:
                        print(f"处理第 {line_num} 行，数据长度: {len(values)}")
                        frame_data = get_finger_data(values[:134], 'left')
                        hand_seq_data.append(frame_data)
                    else:
                        print(f"第 {line_num} 行数据不完整，长度为: {len(values)}")
                    
                except Exception as e:
                    print(f"第 {line_num} 行解析错误: {str(e)}")
                    continue
        
        if not hand_seq_data:
            raise ValueError("没有找到有效数据")
            
        hand_seq_data = np.stack(hand_seq_data, axis=0)  # (Nframe,5,3,4)
        print(f"数据加载完成，形状: {hand_seq_data.shape}")
        return hand_seq_data

    def euler_to_mano_params(self, seq_data):
        """将欧拉角数据转换为MANO参数"""
        N_frame, N_finger, N_link = seq_data.shape[:3]
        # 直接使用欧拉角数据
        mano_params = seq_data.reshape(N_frame, N_finger*N_link*3) #(n_frame,45)

        # 添加手腕参数
        init_wrist_params = np.zeros((N_frame,6))
        mano_params = np.concatenate([init_wrist_params,mano_params],axis=1) #(n_frame, 51)
        return mano_params

    def mano_forward(self, mano_params):
        """MANO前向传播"""
        if torch.is_tensor(mano_params)==False:
            mano_params = torch.tensor(mano_params,dtype=torch.float32)

        bs = mano_params.shape[0]
        mano_output = self.mano_layer(mano_params[:, 3:], self.mano_shape.repeat(bs, 1))
        hand_joints = mano_output.joints
        hand_verts = mano_output.verts
        hand_trans = mano_params[:, :3].view(-1, 1, 3)
        hand_joints += hand_trans
        hand_verts += hand_trans

        return hand_verts, hand_joints


    def get_o3d_images(self, hand_mesh):
        """获取Open3D渲染图像"""
        self.vis.add_geometry(hand_mesh)

        view_control = self.vis.get_view_control()
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 0, 1])
        view_control.set_zoom(1.0)
        view_control.set_front([0., -1., -1.])

        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.02)

        image = self.vis.capture_screen_float_buffer(do_render=True)
        image = np.asarray(image)  # [...,[2,1,0]]
        image = (254 * image).astype(np.uint8)

        self.vis.remove_geometry(hand_mesh, reset_bounding_box=False)
        return image

    def get_video_from_images(self, img_list, video_path, id, text=None, width=840, height=560):
        """将图像序列保存为视频"""
        video_path = os.path.join(video_path, f"{id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
        
        for frame_i in img_list:
            frame_i = cv2.cvtColor(frame_i, cv2.COLOR_BGR2RGB)
            if text is not None:
                frame_i = self.add_text_to_image(frame_i, text)
            video_writer.write(frame_i)
        video_writer.release()

    def test(self):
        """测试函数"""
        print("开始加载数据...")
        hand_seq_data = self.data_load()
        print(f"数据加载完成，共 {len(hand_seq_data)} 帧")
        
        print("转换为MANO参数...")
        mano_params = self.euler_to_mano_params(hand_seq_data)
        print("MANO前向传播...")
        hand_seq_verts, _ = self.mano_forward(mano_params)
        hand_seq_verts = hand_seq_verts.cpu().numpy()
        
        print("开始生成视频帧...")
        img_list = []
        total_frames = len(hand_seq_verts[::20])
        for i, v_i in enumerate(hand_seq_verts[::20]):
            print(f"处理第 {i+1}/{total_frames} 帧...")
            hand_mesh = hand_mesh_instance(v_i, self.hand_faces_np)
            img = self.get_o3d_images(hand_mesh)
            img_list.append(img)
        
        print("保存视频...")
        save_path = '../Video'
        os.makedirs(save_path, exist_ok=True)
        self.get_video_from_images(img_list, video_path=save_path, id=0)
        print(f"视频已保存到: {os.path.join(save_path, '0.mp4')}")


if __name__ == '__main__':
    print("初始化MANO模型...")
    tester = IMU2MANO()
    tester.test()
    print("处理完成！")
