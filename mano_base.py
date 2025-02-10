import torch
import numpy as np
import open3d as o3d
from manotorch.manolayer import ManoLayer, MANOOutput
import scipy.spatial.transform as sciR
import cv2
import time
import torch.nn.functional as F
from torch import einsum
import transforms3d.euler as txe
import open3d.visualization as o3dv

# 大拇指有坐标轴是反的

class MANOVisualizer:
    def __init__(self):
        # 指定MANO资源文件夹路径
        self.mano_assets_root = 'E:/Project/ForheartUnity/MANO'
        
        # 初始化MANO层
        self.mano_layer = ManoLayer(
            mano_assets_root=self.mano_assets_root,
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45
        )

        
        # 左手模型
        self.mano_layer_left = ManoLayer(
            mano_assets_root=self.mano_assets_root,
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45,
            side='left'
        )
        
        # 获取手部面片数据
        self.faces_rh = self.mano_layer.th_faces.cpu().numpy()
        self.faces_lh = self.mano_layer_left.th_faces.cpu().numpy()

        
        # 初始化可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="MANO手型可视化", width=1280, height=720)
        
        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.vis.add_geometry(coordinate_frame)
        
        # 初始化手部网格
        self.right_hand = None
        self.left_hand = None
        
        # 加载运动数据
        self.motion_data = np.load("glove_data/20250206_195901/rotations.npy")
        print(f"加载数据，形状: {self.motion_data.shape}")
        self.current_frame = 0
        
        # 添加状态控制
        self.is_playing = False
        self.initialized = False

    def create_hand_mesh(self, vertices, color=[0.9, 0.7, 0.7],side='left'):
        """创建手部网格"""
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        if side=='left':
            mesh.triangles = o3d.utility.Vector3iVector(self.faces_lh)
        else:
            mesh.triangles = o3d.utility.Vector3iVector(self.faces_rh)
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        return mesh

    def euler_to_axis_angle(self, euler_angles):
        """将欧拉角转换为轴角表示"""
        # 确保输入形状正确
        euler_angles = euler_angles.reshape(-1, 3)  # 重塑为(N, 3)形状
        
        # 转换为弧度
        euler_angles = torch.tensor(euler_angles, dtype=torch.float32)
        euler_angles = euler_angles * np.pi / 180.0
        
        # 使用scipy的Rotation进行转换
        euler_angles[-2:]=0
        rot = sciR.Rotation.from_euler('ZYX', euler_angles.numpy())

        rotvec = rot.as_rotvec()
        
        return torch.tensor(rotvec, dtype=torch.float32)

    def update_visualization(self, frame_data=None):
        """更新可视化"""
        if frame_data is None:
            frame_data = self.motion_data[self.current_frame]
        
        # MANO关节顺序重映射（从数据格式到MANO格式）
        # 数据格式: [wrist, thumb1,2,3, index1,2,3, middle1,2,3, ring1,2,3, pinky1,2,3]
        # MANO格式: [wrist(0), 
        #           index(1-3), middle(4-6), pinky(7-9), ring(10-12),
        #           thumb(13-15)]
        mano_order = [0,                           # wrist
                      4, 5, 6,                     # index
                      7, 8, 9,                     # middle
                      13, 14, 15,                  # pinky
                      10, 11, 12,                  # ring
                      1, 2, 3]                     # thumb
        
        # 重新排序右手数据
        right_hand_data = frame_data[:16]
        right_hand_reordered = np.zeros_like(right_hand_data)
        for i, idx in enumerate(mano_order):
            euler = right_hand_data[idx].copy()  # [Z, Y, X]顺序
            if i == 0:  # 手腕关节
                # 手腕需要特殊处理，因为它是全局旋转
                euler[2] *= 1   # X轴
                euler[1] *= -1  # Y轴
                euler[0] *= -1  # Z轴
            else:  # 手指关节
                # 手指关节是相对于父关节的局部旋转
                a=1
                # euler[2] *= -1  # X轴（弯曲）
                # euler[1] *= 1   # Y轴（侧向）
                # euler[0] *= -1  # Z轴（扭转）
            right_hand_reordered[i] = euler
        
        # 重新排序左手数据
        left_hand_data = frame_data[16:]
        left_hand_reordered = np.zeros_like(left_hand_data)
        for i, idx in enumerate(mano_order):
            euler = left_hand_data[idx].copy()  # [Z, Y, X]顺序
            if i == 0:  # 手腕关节
                # 左手手腕需要镜像处理
                euler[2] *= 1   # X轴
                euler[1] *= 1   # Y轴
                euler[0] *= 1   # Z轴
            else:  # 手指关节
                # 左手手指需要镜像处理
                a=1
                euler[2] *= -1  # X轴（弯曲）
                euler[1] *= -1  # Y轴（侧向）
                euler[0] *= -1  # Z轴（扭转）
            left_hand_reordered[i] = euler
        
        # 转换为MANO参数格式
        right_pose_params = torch.zeros((1, 48))  # [global_rot(3) + 15_joints(45)]
        left_pose_params = torch.zeros((1, 48))
        
        # 转换为弧度并设置参数
        right_hand_rad = right_hand_reordered * np.pi / 180.0
        left_hand_rad = left_hand_reordered * np.pi / 180.0
        
        # 使用scipy的Rotation进行转换，注意使用'ZYX'顺序
        #euler_to_axis_angle
        # left_hand_rad[:,-1]=0
        # left_hand_rad[:,0]=0
        left_hand_rad = left_hand_rad[:,[1,2,0]]



        right_rot = sciR.Rotation.from_euler('ZYX', right_hand_rad)
        left_rot = sciR.Rotation.from_euler('ZYX', left_hand_rad)
        
        right_pose_params[0] = torch.from_numpy(right_rot.as_rotvec().reshape(-1))
        left_pose_params[0] = torch.from_numpy(left_rot.as_rotvec().reshape(-1))
        
        # 生成手部网格
        with torch.no_grad():
            # 右手
            right_verts = self.mano_layer(
                pose_coeffs=right_pose_params,
                betas=torch.zeros(1, 10)
            ).verts[0].cpu().numpy()
            
            # 左手
            left_verts = self.mano_layer_left(
                pose_coeffs=left_pose_params,
                betas=torch.zeros(1, 10)
            ).verts[0].cpu().numpy()
        
        # 更新可视化
        if not self.initialized:
            self.right_hand = self.create_hand_mesh(right_verts, [0.9, 0.7, 0.7],side='right')
            self.left_hand = self.create_hand_mesh(left_verts, [0.7, 0.7, 0.7],side='left')
            self.vis.add_geometry(self.right_hand)
            self.vis.add_geometry(self.left_hand)
            self.initialized = True
        else:
            self.right_hand.vertices = o3d.utility.Vector3dVector(right_verts)
            self.left_hand.vertices = o3d.utility.Vector3dVector(left_verts)
            self.right_hand.compute_vertex_normals()
            self.left_hand.compute_vertex_normals()
            self.vis.update_geometry(self.right_hand)
            self.vis.update_geometry(self.left_hand)
        
        self.vis.poll_events()
        self.vis.update_renderer()
        
        if self.is_playing:
            self.current_frame = (self.current_frame + 1) % len(self.motion_data)

    def run(self):
        """运行可视化"""
        # 设置默认视角
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.7)
        ctr.set_lookat([0.1, 0, 0])
        
        print("显示初始状态...")
        print("按空格键开始播放")
        print("按 ESC 退出")
        
        try:
            import msvcrt  # Windows系统
            while True:
                if not self.initialized:
                    self.show_initial_pose()
                    self.initialized = True
                
                if self.is_playing:
                    self.update_visualization()
                
                if not self.vis.poll_events():
                    break
                
                # 检查键盘输入
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b' ':  # 空格键
                        self.is_playing = not self.is_playing
                        if self.is_playing:
                            print("\n开始播放动作序列...")
                        else:
                            print("\n暂停播放")
                        time.sleep(0.2)
                    elif key == b'\x1b':  # ESC键
                        break
                
                self.vis.update_renderer()
                
                if self.is_playing:
                    time.sleep(0.033)
                
        except KeyboardInterrupt:
            print("\n用户中断播放")
        finally:
            self.vis.destroy_window()

    def show_initial_pose(self):
        """显示初始T-pose"""
        # 手型参数（使用默认值）
        shape = torch.zeros(1, 10)
        
        # 生成手部网格
        with torch.no_grad():
            # 右手
            right_pose_coeffs = torch.zeros(1, 48)  # 全零表示T-pose
            right_output = self.mano_layer(
                pose_coeffs=right_pose_coeffs,
                betas=shape
            )
            right_verts = right_output.verts[0].cpu().numpy()
            
            # 左手
            left_pose_coeffs = torch.zeros(1, 48)  # 全零表示T-pose
            left_output = self.mano_layer_left(
                pose_coeffs=left_pose_coeffs,
                betas=shape
            )
            left_verts = left_output.verts[0].cpu().numpy()
            
            # 更新右手网格
            if self.right_hand is None:
                self.right_hand = self.create_hand_mesh(right_verts, [0.9, 0.7, 0.7],side='right')
                self.vis.add_geometry(self.right_hand)
            else:
                self.right_hand.vertices = o3d.utility.Vector3dVector(right_verts)
                self.vis.update_geometry(self.right_hand)
            
            # 更新左手网格
            if self.left_hand is None:
                self.left_hand = self.create_hand_mesh(left_verts, [0.7, 0.7, 0.9])
                # o3dv.draw_geometries([self.left_hand])
                a=1
                self.vis.add_geometry(self.left_hand)
            else:
                self.left_hand.vertices = o3d.utility.Vector3dVector(left_verts)
                self.vis.update_geometry(self.left_hand)



def main():
    visualizer = MANOVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()