import numpy as np
import open3d as o3d
import time
import keyboard
import os
import sys
from enum import IntEnum
# 获取项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到Python路径
sys.path.append(project_root)
from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.euler import euler2mat, mat2euler
from Utils.SkeletonIndex import JOINT_EULER_FATHER

class HandSkeletonIndex(IntEnum):
    # 右手
    RightHand = 0
    RightHandThumb1 = 1
    RightHandThumb2 = 2
    RightHandThumb3 = 3
    RightHandIndex1 = 4
    RightHandIndex2 = 5
    RightHandIndex3 = 6
    RightHandMiddle1 = 7
    RightHandMiddle2 = 8
    RightHandMiddle3 = 9
    RightHandRing1 = 10
    RightHandRing2 = 11
    RightHandRing3 = 12
    RightHandPinky1 = 13
    RightHandPinky2 = 14
    RightHandPinky3 = 15
    
    # 左手
    LeftHand = 16
    LeftHandThumb1 = 17
    LeftHandThumb2 = 18
    LeftHandThumb3 = 19
    LeftHandIndex1 = 20
    LeftHandIndex2 = 21
    LeftHandIndex3 = 22
    LeftHandMiddle1 = 23
    LeftHandMiddle2 = 24
    LeftHandMiddle3 = 25
    LeftHandRing1 = 26
    LeftHandRing2 = 27
    LeftHandRing3 = 28
    LeftHandPinky1 = 29
    LeftHandPinky2 = 30
    LeftHandPinky3 = 31

class GloveVisualizer:
    def __init__(self):
        # 定义骨骼连接关系（32个关节的版本）
        self.lines = np.array([
            # 右手
            # 拇指
            [HandSkeletonIndex.RightHand, HandSkeletonIndex.RightHandThumb1],
            [HandSkeletonIndex.RightHandThumb1, HandSkeletonIndex.RightHandThumb2],
            [HandSkeletonIndex.RightHandThumb2, HandSkeletonIndex.RightHandThumb3],
            # 食指
            [HandSkeletonIndex.RightHand, HandSkeletonIndex.RightHandIndex1],
            [HandSkeletonIndex.RightHandIndex1, HandSkeletonIndex.RightHandIndex2],
            [HandSkeletonIndex.RightHandIndex2, HandSkeletonIndex.RightHandIndex3],
            # 中指
            [HandSkeletonIndex.RightHand, HandSkeletonIndex.RightHandMiddle1],
            [HandSkeletonIndex.RightHandMiddle1, HandSkeletonIndex.RightHandMiddle2],
            [HandSkeletonIndex.RightHandMiddle2, HandSkeletonIndex.RightHandMiddle3],
            # 无名指
            [HandSkeletonIndex.RightHand, HandSkeletonIndex.RightHandRing1],
            [HandSkeletonIndex.RightHandRing1, HandSkeletonIndex.RightHandRing2],
            [HandSkeletonIndex.RightHandRing2, HandSkeletonIndex.RightHandRing3],
            # 小指
            [HandSkeletonIndex.RightHand, HandSkeletonIndex.RightHandPinky1],
            [HandSkeletonIndex.RightHandPinky1, HandSkeletonIndex.RightHandPinky2],
            [HandSkeletonIndex.RightHandPinky2, HandSkeletonIndex.RightHandPinky3],
            # 掌骨连接
            [HandSkeletonIndex.RightHandThumb1, HandSkeletonIndex.RightHandIndex1],
            [HandSkeletonIndex.RightHandIndex1, HandSkeletonIndex.RightHandMiddle1],
            [HandSkeletonIndex.RightHandMiddle1, HandSkeletonIndex.RightHandRing1],
            [HandSkeletonIndex.RightHandRing1, HandSkeletonIndex.RightHandPinky1],
            
            # 左手（镜像右手的连接关系）
            # 拇指
            [HandSkeletonIndex.LeftHand, HandSkeletonIndex.LeftHandThumb1],
            [HandSkeletonIndex.LeftHandThumb1, HandSkeletonIndex.LeftHandThumb2],
            [HandSkeletonIndex.LeftHandThumb2, HandSkeletonIndex.LeftHandThumb3],
            # 食指
            [HandSkeletonIndex.LeftHand, HandSkeletonIndex.LeftHandIndex1],
            [HandSkeletonIndex.LeftHandIndex1, HandSkeletonIndex.LeftHandIndex2],
            [HandSkeletonIndex.LeftHandIndex2, HandSkeletonIndex.LeftHandIndex3],
            # 中指
            [HandSkeletonIndex.LeftHand, HandSkeletonIndex.LeftHandMiddle1],
            [HandSkeletonIndex.LeftHandMiddle1, HandSkeletonIndex.LeftHandMiddle2],
            [HandSkeletonIndex.LeftHandMiddle2, HandSkeletonIndex.LeftHandMiddle3],
            # 无名指
            [HandSkeletonIndex.LeftHand, HandSkeletonIndex.LeftHandRing1],
            [HandSkeletonIndex.LeftHandRing1, HandSkeletonIndex.LeftHandRing2],
            [HandSkeletonIndex.LeftHandRing2, HandSkeletonIndex.LeftHandRing3],
            # 小指
            [HandSkeletonIndex.LeftHand, HandSkeletonIndex.LeftHandPinky1],
            [HandSkeletonIndex.LeftHandPinky1, HandSkeletonIndex.LeftHandPinky2],
            [HandSkeletonIndex.LeftHandPinky2, HandSkeletonIndex.LeftHandPinky3],
            # 掌骨连接
            [HandSkeletonIndex.LeftHandThumb1, HandSkeletonIndex.LeftHandIndex1],
            [HandSkeletonIndex.LeftHandIndex1, HandSkeletonIndex.LeftHandMiddle1],
            [HandSkeletonIndex.LeftHandMiddle1, HandSkeletonIndex.LeftHandRing1],
            [HandSkeletonIndex.LeftHandRing1, HandSkeletonIndex.LeftHandPinky1]
        ])
        self.xAxisReverse_LeftHand = False  # 左手X轴是否反转
        self.isOriginSkeletonAngleCalculated = False
        self.gloveRelativeTrackerCtrl = {
            'xAddDis_LeftHand': 0,
            'yAddDis_LeftHand': 0,
            'zAddDis_LeftHand': 0,
            'xAddEuler_LeftHand': 0,
            'yAddEuler_LeftHand': 0,
            'zAddEuler_LeftHand': 0,
            'xAddDis_RightHand': 0,
            'yAddDis_RightHand': 0,
            'zAddDis_RightHand': 0,
            'xAddEuler_RightHand': 0,
            'yAddEuler_RightHand': 0,
            'zAddEuler_RightHand': 0
        }
        # 初始化可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        # 创建坐标系
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.coord_frame)
        
        # 创建手腕坐标系
        self.right_wrist_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        self.left_wrist_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        self.vis.add_geometry(self.right_wrist_frame)
        self.vis.add_geometry(self.left_wrist_frame)

        # 添加原始旋转记录
        self.original_rots = []  # 对应 orignalRots
        self.original_parent_rots = []  # 对应 orignalParentRots
        self.is_original_skeleton_angle_calculated = False
        
        # 初始化手部几何体
        self.joints = []
        self.bones = []
        self._init_hand_geometries()
        
        # 计算原始旋转
        self.calculate_original_rot()

    def _init_hand_geometries(self):
        """初始化手部几何体"""
        # 创建关节球体
        for i in range(32):
            # 手腕和指尖用大球体，其他关节用小球体
            radius = 0.011 if i in [HandSkeletonIndex.RightHand, HandSkeletonIndex.LeftHand] else 0.007
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            
            # 设置颜色
            if i in [HandSkeletonIndex.RightHand, HandSkeletonIndex.LeftHand]:  # 手腕为绿色
                sphere.paint_uniform_color([0, 1, 0])
            elif i in [HandSkeletonIndex.RightHandThumb3, HandSkeletonIndex.RightHandIndex3,
                      HandSkeletonIndex.RightHandMiddle3, HandSkeletonIndex.RightHandRing3,
                      HandSkeletonIndex.RightHandPinky3,
                      HandSkeletonIndex.LeftHandThumb3, HandSkeletonIndex.LeftHandIndex3,
                      HandSkeletonIndex.LeftHandMiddle3, HandSkeletonIndex.LeftHandRing3,
                      HandSkeletonIndex.LeftHandPinky3]:  # 指尖为红色
                sphere.paint_uniform_color([1, 0, 0])
            else:  # 其他关节为蓝色
                sphere.paint_uniform_color([0, 0, 1])
            
            self.joints.append(sphere)
            self.vis.add_geometry(sphere)

        # 创建骨骼圆柱体
        for _ in self.lines:
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.003, height=1.0)
            cylinder.paint_uniform_color([0.7, 0.7, 0.7])  # 骨骼为灰色
            self.bones.append(cylinder)
            self.vis.add_geometry(cylinder)

    def calculate_original_rot(self):
        """计算原始旋转（对应CaluateOrignalRot）"""
        if self.is_original_skeleton_angle_calculated:
            return
            
        # 初始化原始旋转列表
        self.original_rots = [np.eye(3) for _ in range(32)]  # 3x3旋转矩阵
        self.original_parent_rots = [np.eye(3) for _ in range(32)]
        
        # 记录初始旋转状态
        for i in range(32):
            if i == HandSkeletonIndex.LeftHand or i == HandSkeletonIndex.RightHand:
                continue
                
            parent_idx = JOINT_EULER_FATHER[i]
            if parent_idx != -1:
                self.original_rots[i] = np.eye(3)  # 初始姿态
                self.original_parent_rots[i] = np.eye(3)  # 父骨骼初始姿态
        
        self.is_original_skeleton_angle_calculated = True

    def _create_default_hand_positions(self):
        """创建默认手型的关节位置"""
        positions = np.zeros((32, 3))
        
        # 右手位置
        positions[HandSkeletonIndex.RightHand] = [0.15, 0, 0.2]  # 右手腕位置
        
        # 右手拇指（调整角度）
        positions[HandSkeletonIndex.RightHandThumb1] = positions[HandSkeletonIndex.RightHand] + [0.02, 0.02, 0.02]
        positions[HandSkeletonIndex.RightHandThumb2] = positions[HandSkeletonIndex.RightHandThumb1] + [0.01, 0.02, 0.01]
        positions[HandSkeletonIndex.RightHandThumb3] = positions[HandSkeletonIndex.RightHandThumb2] + [0.01, 0.02, 0.01]
        
        # 右手其他手指（调整为自然张开状态）
        base_angles = np.array([10, 5, 0, -5, -10]) * np.pi / 180  # 各手指基础角度
        for i, finger_base in enumerate([
            HandSkeletonIndex.RightHandIndex1,
            HandSkeletonIndex.RightHandMiddle1,
            HandSkeletonIndex.RightHandRing1,
            HandSkeletonIndex.RightHandPinky1
        ]):
            angle = base_angles[i]
            finger_dir = np.array([np.sin(angle), np.cos(angle), 0])  # 手指方向
            
            # 设置三个关节的位置
            positions[finger_base] = positions[HandSkeletonIndex.RightHand] + finger_dir * 0.04
            positions[finger_base + 1] = positions[finger_base] + finger_dir * 0.03
            positions[finger_base + 2] = positions[finger_base + 1] + finger_dir * 0.025
        
        # 左手（镜像右手的位置）
        positions[HandSkeletonIndex.LeftHand] = [-0.15, 0, 0.2]
        
        # 左手拇指
        positions[HandSkeletonIndex.LeftHandThumb1] = positions[HandSkeletonIndex.LeftHand] + [-0.02, 0.02, 0.02]
        positions[HandSkeletonIndex.LeftHandThumb2] = positions[HandSkeletonIndex.LeftHandThumb1] + [-0.01, 0.02, 0.01]
        positions[HandSkeletonIndex.LeftHandThumb3] = positions[HandSkeletonIndex.LeftHandThumb2] + [-0.01, 0.02, 0.01]
        
        # 左手其他手指
        for i, finger_base in enumerate([
            HandSkeletonIndex.LeftHandIndex1,
            HandSkeletonIndex.LeftHandMiddle1,
            HandSkeletonIndex.LeftHandRing1,
            HandSkeletonIndex.LeftHandPinky1
        ]):
            angle = -base_angles[i]  # 注意这里取负值，因为是镜像
            finger_dir = np.array([np.sin(angle), np.cos(angle), 0])
            
            positions[finger_base] = positions[HandSkeletonIndex.LeftHand] + finger_dir * 0.04
            positions[finger_base + 1] = positions[finger_base] + finger_dir * 0.03
            positions[finger_base + 2] = positions[finger_base + 1] + finger_dir * 0.025
        
        return positions

    def _update_bone(self, start_point, end_point, cylinder):
        """更新骨骼圆柱体"""
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return
            
        direction = direction / length
        
        # 创建新圆柱体
        new_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.003, height=length)
        new_cylinder.paint_uniform_color([0.7, 0.7, 0.7])
        
        # 将圆柱体移动到起点
        new_cylinder.translate([0, 0, length/2])
        
        # 计算旋转
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.arccos(np.dot(z_axis, direction))
        
        # 应用旋转
        if np.linalg.norm(rotation_axis) > 0:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
            new_cylinder.rotate(R, center=[0, 0, 0])
        
        # 移动到正确位置
        new_cylinder.translate(start_point)
        
        # 更新现有圆柱体
        cylinder.vertices = new_cylinder.vertices
        cylinder.triangles = new_cylinder.triangles
        cylinder.vertex_colors = new_cylinder.vertex_colors
        cylinder.vertex_normals = new_cylinder.vertex_normals

    def update_hands(self, joint_positions):
        """更新手部位置"""
        if joint_positions.shape != (32, 3):
            print(f"关节位置数据格式错误: {joint_positions.shape}")
            return
        # 更新关节球体位置
        for i in range(32):
            sphere = self.joints[i]
            current_center = np.asarray(sphere.get_center())
            transformation = np.eye(4)
            transformation[:3, 3] = joint_positions[i] - current_center
            sphere.transform(transformation)
            self.vis.update_geometry(sphere)
            
        # 更新骨骼
        for i, (start_idx, end_idx) in enumerate(self.lines):
            start_point = np.asarray(self.joints[start_idx].get_center())
            end_point = np.asarray(self.joints[end_idx].get_center())
            self._update_bone(start_point, end_point, self.bones[i])
            self.vis.update_geometry(self.bones[i])

    def run(self):
        """运行可视化"""
        print("控制说明:")
        print("Q: 退出")
        
        # 使用默认位置
        positions = self._create_default_hand_positions()
        self.update_hands(positions)
        
        while True:
            self.vis.poll_events()
            self.vis.update_renderer()
            
            if keyboard.is_pressed('q'):
                print("退出程序")
                break
                
        self.vis.destroy_window()

    def convert_loc_from_motion_venus_to_unity3d(self, x_axis_reverse, x, y, z):
        if x_axis_reverse:
            return np.array([-x, y, z])
        return np.array([x, y, z])

    def apply_bone_trans_rot(self, data):
        """应用骨骼变换和旋转"""
        try:
            # 更新左手位置和旋转
            left_hand_pos = self.convert_loc_from_motion_venus_to_unity3d(
                self.xAxisReverse_LeftHand,
                data['left_hand_pos'][0],
                data['left_hand_pos'][1],
                data['left_hand_pos'][2]
            )
            
            # 添加手套与追踪器之间的偏移量
            left_hand_pos += np.array([
                self.gloveRelativeTrackerCtrl['xAddDis_LeftHand'] / 1000,
                self.gloveRelativeTrackerCtrl['yAddDis_LeftHand'] / 1000,
                self.gloveRelativeTrackerCtrl['zAddDis_LeftHand'] / 1000
            ])

            # 更新右手位置和旋转
            right_hand_pos = self.convert_loc_from_motion_venus_to_unity3d(
                self.xAxisReverse_LeftHand,
                data['right_hand_pos'][0],
                data['right_hand_pos'][1],
                data['right_hand_pos'][2]
            )
            
            # 添加手套与追踪器之间的偏移量
            right_hand_pos += np.array([
                self.gloveRelativeTrackerCtrl['xAddDis_RightHand'] / 1000,
                self.gloveRelativeTrackerCtrl['yAddDis_RightHand'] / 1000,
                self.gloveRelativeTrackerCtrl['zAddDis_RightHand'] / 1000
            ])

            # 处理手指关节的旋转
            positions = self._create_default_hand_positions()
            positions[HandSkeletonIndex.LeftHand] = left_hand_pos
            positions[HandSkeletonIndex.RightHand] = right_hand_pos

            # 合并左右手的旋转数据
            all_rotations = []
            all_rotations.append(data['right_hand_rot'])
            all_rotations.extend(data['right_joint_rotations'])
            all_rotations.append(data['left_hand_rot'])
            all_rotations.extend(data['left_joint_rotations'])

            # 计算并应用手指关节的旋转
            for i in range(len(all_rotations)):
                if i >= 32:
                    break
                    
                quat = all_rotations[i]
                rot_mat = quat2mat(quat)
                
                # 手腕特殊处理
                if i == HandSkeletonIndex.LeftHand:
                    euler_offset = np.array([
                        self.gloveRelativeTrackerCtrl['xAddEuler_LeftHand'],
                        self.gloveRelativeTrackerCtrl['yAddEuler_LeftHand'],
                        self.gloveRelativeTrackerCtrl['zAddEuler_LeftHand']
                    ])
                    offset_mat = euler2mat(*np.radians(euler_offset))
                    rot_mat = rot_mat @ offset_mat
                elif i == HandSkeletonIndex.RightHand:
                    euler_offset = np.array([
                        self.gloveRelativeTrackerCtrl['xAddEuler_RightHand'],
                        self.gloveRelativeTrackerCtrl['yAddEuler_RightHand'],
                        self.gloveRelativeTrackerCtrl['zAddEuler_RightHand']
                    ])
                    offset_mat = euler2mat(*np.radians(euler_offset))
                    rot_mat = rot_mat @ offset_mat
                else:
                    # 应用原始旋转修正
                    original_rot = self.original_rots[i]
                    original_parent_rot = self.original_parent_rots[i]
                    used_rot = np.linalg.inv(original_parent_rot) @ rot_mat @ original_parent_rot
                    final_rot = used_rot @ original_rot

                    # 应用父子关系的旋转
                    parent_idx = JOINT_EULER_FATHER[i]
                    if parent_idx != -1:
                        parent_pos = positions[parent_idx]
                        current_pos = positions[i]
                        relative_pos = current_pos - parent_pos
                        rotated_pos = final_rot @ relative_pos
                        positions[i] = parent_pos + rotated_pos

            # 更新可视化
            self.update_hands(positions)
            
        except Exception as e:
            print(f"处理数据时出错: {str(e)}")

    def parse_udp_data(self, udp_data):
        """解析UDP数据包"""
        try:
            # 提取实际的数据部分（跳过头部信息）
            data_start = udp_data.find("subpackage 1/1,") + len("subpackage 1/1,")
            data_string = udp_data[data_start:]
            
            # 分割数据
            parts = data_string.split(',')
            if len(parts) < 134:  # 确保数据完整性
                print(f"数据不完整: 期望134个值，实际获得{len(parts)}个值")
                return None
            
            # 提取右手数据
            right_hand_pos = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
            right_hand_rot = np.array([float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])])
            
            # 提取右手15个骨骼的旋转数据
            right_joint_rotations = []
            for i in range(7, 67, 4):  # 7到66是右手骨骼旋转数据
                quat = np.array([
                    float(parts[i]),
                    float(parts[i+1]),
                    float(parts[i+2]),
                    float(parts[i+3])
                ])
                right_joint_rotations.append(quat)
            
            # 提取左手数据
            left_hand_pos = np.array([float(parts[67]), float(parts[68]), float(parts[69])])
            left_hand_rot = np.array([float(parts[70]), float(parts[71]), float(parts[72]), float(parts[73])])
            
            # 提取左手15个骨骼的旋转数据
            left_joint_rotations = []
            for i in range(74, 134, 4):  # 74到133是左手骨骼旋转数据
                quat = np.array([
                    float(parts[i]),
                    float(parts[i+1]),
                    float(parts[i+2]),
                    float(parts[i+3])
                ])
                left_joint_rotations.append(quat)
            
            data = {
                'right_hand_pos': right_hand_pos,
                'right_hand_rot': right_hand_rot,
                'right_joint_rotations': right_joint_rotations,
                'left_hand_pos': left_hand_pos,
                'left_hand_rot': left_hand_rot,
                'left_joint_rotations': left_joint_rotations
            }
            
            return data
            
        except Exception as e:
            print(f"解析UDP数据时出错: {str(e)}")
            return None

    def run_with_udp(self, udp_port=4396):
        """使用UDP接收数据并运行可视化"""
        import socket
        
        # 创建UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', udp_port))
        sock.setblocking(False)
        
        print(f"正在监听UDP端口 {udp_port}")
        print("控制说明:")
        print("Q: 退出")
        
        # 使用默认位置初始化
        positions = self._create_default_hand_positions()
        self.update_hands(positions)
        
        while True:
            try:
                # 尝试接收UDP数据
                data, addr = sock.recvfrom(4096)
                udp_data = data.decode('utf-8')
                
                # 解析数据
                parsed_data = self.parse_udp_data(udp_data)
                if parsed_data:
                    # 应用数据到可视化
                    self.apply_bone_trans_rot(parsed_data)
                
            except BlockingIOError:
                pass
            
            self.vis.poll_events()
            self.vis.update_renderer()
            
            if keyboard.is_pressed('q'):
                print("退出程序")
                break
        
        sock.close()
        self.vis.destroy_window()

def main():
    visualizer = GloveVisualizer()
    visualizer.run_with_udp()

if __name__ == "__main__":
    main()