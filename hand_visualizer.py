import numpy as np
import open3d as o3d
import time
import cv2
import transforms3d.euler as txe
import msvcrt

class HandVisualizer:
    def __init__(self):
        # 初始化可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="手型可视化", width=1280, height=720)
        
        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(coordinate_frame)
        
        # 初始化手部关节和骨骼
        self.joints = []
        self.bones = []
        self.joint_positions = {}  # 存储关节位置
        self.create_hand_skeleton()
        
        # 加载运动数据
        self.motion_data = np.load("glove_data/20250206_195901/rotations.npy")
        print(f"加载数据，形状: {self.motion_data.shape}")
        self.current_frame = 0
        
        # 状态控制
        self.is_playing = False
        
    def create_hand_skeleton(self):
        """创建手部骨骼结构"""
        # 定义关节位置（调整后的版本）
        self.joint_positions = {
            'wrist': np.array([0, 0, 0]),
            # 拇指 - 调整位置更符合实际手型
            'thumb1': np.array([0.02, 0, 0.02]),
            'thumb2': np.array([0.04, 0, 0.03]),
            'thumb3': np.array([0.06, 0, 0.035]),
            # 食指
            'index1': np.array([0.07, 0, 0.01]),
            'index2': np.array([0.09, 0, 0.01]),
            'index3': np.array([0.11, 0, 0.01]),
            # 中指
            'middle1': np.array([0.07, 0, 0]),
            'middle2': np.array([0.09, 0, 0]),
            'middle3': np.array([0.11, 0, 0]),
            # 无名指
            'ring1': np.array([0.07, 0, -0.01]),
            'ring2': np.array([0.09, 0, -0.01]),
            'ring3': np.array([0.11, 0, -0.01]),
            # 小指
            'pinky1': np.array([0.06, 0, -0.02]),
            'pinky2': np.array([0.08, 0, -0.02]),
            'pinky3': np.array([0.10, 0, -0.02])
        }
        
        # 创建关节球体
        for name, pos in self.joint_positions.items():
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
            sphere.translate(pos)
            sphere.paint_uniform_color([1, 0.7, 0.7])
            self.joints.append(sphere)
            self.vis.add_geometry(sphere)
        
        # 定义骨骼连接
        connections = [
            # 拇指
            ('wrist', 'thumb1'), ('thumb1', 'thumb2'), ('thumb2', 'thumb3'),
            # 食指
            ('wrist', 'index1'), ('index1', 'index2'), ('index2', 'index3'),
            # 中指
            ('wrist', 'middle1'), ('middle1', 'middle2'), ('middle2', 'middle3'),
            # 无名指
            ('wrist', 'ring1'), ('ring1', 'ring2'), ('ring2', 'ring3'),
            # 小指
            ('wrist', 'pinky1'), ('pinky1', 'pinky2'), ('pinky2', 'pinky3')
        ]
        
        # 创建骨骼圆柱体
        for start, end in connections:
            start_pos = self.joint_positions[start]
            end_pos = self.joint_positions[end]
            
            # 计算圆柱体参数
            direction = end_pos - start_pos
            length = np.linalg.norm(direction)
            direction = direction / length
            
            # 创建圆柱体
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=length)
            
            # 计算旋转矩阵
            z_axis = np.array([0, 1, 0])
            rotation_axis = np.cross(z_axis, direction)
            rotation_angle = np.arccos(np.dot(z_axis, direction))
            
            if np.any(rotation_axis):
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
                cylinder.rotate(R, center=[0, 0, 0])
            
            # 移动到正确位置
            cylinder.translate(start_pos)
            cylinder.paint_uniform_color([0.8, 0.8, 0.8])
            
            self.bones.append(cylinder)
            self.vis.add_geometry(cylinder)
    
    def update_joint_rotations(self, frame_data):
        """更新关节旋转"""
        # 获取当前帧的欧拉角数据
        euler_angles = frame_data  # ZYX顺序
        
        # 更新每个关节的旋转
        for i, joint in enumerate(self.joints[1:], 1):  # 跳过手腕
            # 转换欧拉角为旋转矩阵（ZYX顺序）
            R = txe.euler2mat(
                euler_angles[i][0] * np.pi / 180.0,  # Z
                euler_angles[i][1] * np.pi / 180.0,  # Y
                euler_angles[i][2] * np.pi / 180.0,  # X
                'rzyx'
            )
            
            # 应用旋转
            joint.rotate(R, center=self.joint_positions['wrist'])
            self.vis.update_geometry(joint)
    
    def run(self):
        """运行可视化"""
        # 设置默认视角
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.7)
        ctr.set_lookat([0.05, 0, 0])
        
        print("显示初始状态...")
        print("按空格键开始播放")
        print("按 ESC 退出")
        
        try:
            while True:
                if self.is_playing:
                    self.update_joint_rotations(self.motion_data[self.current_frame])
                    self.current_frame = (self.current_frame + 1) % len(self.motion_data)
                
                if not self.vis.poll_events():
                    break
                
                # 检查键盘输入
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b' ':  # 空格键
                        self.is_playing = not self.is_playing
                        print("\n开始播放..." if self.is_playing else "\n暂停播放")
                    elif key == b'\x1b':  # ESC键
                        break
                
                self.vis.update_renderer()
                
                if self.is_playing:
                    time.sleep(0.033)  # 约30fps
                    
        except KeyboardInterrupt:
            print("\n用户中断播放")
        finally:
            self.vis.destroy_window()

if __name__ == "__main__":
    visualizer = HandVisualizer()
    visualizer.run() 