import open3d as o3d
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
import time

class HandController:
    def __init__(self):
        # 定义手指关节枚举（类似KHHS32）
        self.HAND_JOINTS = {
            # 手腕
            'Wrist': 0,
            # 拇指
            'Thumb1': 1,
            'Thumb2': 2,
            'Thumb3': 3,
            # 食指
            'Index1': 4,
            'Index2': 5,
            'Index3': 6,
            # 中指
            'Middle1': 7,
            'Middle2': 8,
            'Middle3': 9,
            # 无名指
            'Ring1': 10,
            'Ring2': 11,
            'Ring3': 12,
            # 小指
            'Pinky1': 13,
            'Pinky2': 14,
            'Pinky3': 15
        }
        
        # 初始化可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        # 创建坐标系
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.coord_frame)
        
        # 初始化手部几何体
        self.joints = []
        self.bones = []
        self.original_rotations = []  # 存储初始旋转
        self.parent_rotations = []    # 存储父节点旋转
        
        # 初始化骨骼连接关系
        self.bone_connections = [
            # 拇指链
            (0, 1), (1, 2), (2, 3),
            # 食指链
            (0, 4), (4, 5), (5, 6),
            # 中指链
            (0, 7), (7, 8), (8, 9),
            # 无名指链
            (0, 10), (10, 11), (11, 12),
            # 小指链
            (0, 13), (13, 14), (14, 15)
        ]
        
        self._init_hand_geometries()
        self._calculate_original_rotations()

    def _init_hand_geometries(self):
        """初始化手部几何体"""
        # 创建关节球体
        for i in range(16):  # 16个关节点
            radius = 0.01 if i == 0 else 0.007  # 手腕关节大一点
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            
            # 设置颜色
            if i == 0:  # 手腕为绿色
                sphere.paint_uniform_color([0, 1, 0])
            elif i in [3, 6, 9, 12, 15]:  # 指尖为红色
                sphere.paint_uniform_color([1, 0, 0])
            else:  # 其他关节为蓝色
                sphere.paint_uniform_color([0, 0, 1])
                
            self.joints.append(sphere)
            self.vis.add_geometry(sphere)
        
        # 创建骨骼圆柱体
        for _ in self.bone_connections:
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=1.0)
            cylinder.paint_uniform_color([0.7, 0.7, 0.7])
            self.bones.append(cylinder)
            self.vis.add_geometry(cylinder)

    def _calculate_original_rotations(self):
        """计算并存储初始旋转状态"""
        # 初始化为单位四元数
        self.original_rotations = [np.array([0, 0, 0, 1]) for _ in range(16)]
        self.parent_rotations = [np.array([0, 0, 0, 1]) for _ in range(16)]
        
        # 设置初始手型 (T-pose)
        initial_rotations = {
            # 手腕保持不变
            'Wrist': [0, 0, 0, 1],
            # 拇指初始角度
            'Thumb1': [0, 0, 0.383, 0.924],  # 约45度
            'Thumb2': [0, 0, 0.259, 0.966],  # 约30度
            'Thumb3': [0, 0, 0.174, 0.985],  # 约20度
            # 其他手指保持伸直
            'Index1': [0, 0, 0, 1],
            'Index2': [0, 0, 0, 1],
            'Index3': [0, 0, 0, 1],
            'Middle1': [0, 0, 0, 1],
            'Middle2': [0, 0, 0, 1],
            'Middle3': [0, 0, 0, 1],
            'Ring1': [0, 0, 0, 1],
            'Ring2': [0, 0, 0, 1],
            'Ring3': [0, 0, 0, 1],
            'Pinky1': [0, 0, 0, 1],
            'Pinky2': [0, 0, 0, 1],
            'Pinky3': [0, 0, 0, 1],
        }
        
        # 应用初始旋转
        for joint_name, rotation in initial_rotations.items():
            joint_id = self.HAND_JOINTS[joint_name]
            self.original_rotations[joint_id] = np.array(rotation)

    def _update_bone(self, start_point, end_point, cylinder):
        """更新骨骼圆柱体"""
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        
        if length < 1e-6:
            return
            
        direction = direction / length
        
        # 创建新圆柱体
        new_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=length)
        new_cylinder.paint_uniform_color([0.7, 0.7, 0.7])
        
        # 将圆柱体移动到起点
        new_cylinder.translate([0, 0, length/2])
        
        # 计算旋转
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.arccos(np.dot(z_axis, direction))
        
        # 应用旋转
        if np.linalg.norm(rotation_axis) > 1e-6:
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

    def _get_joint_positions(self):
        """获取标准T-pose下的关节位置"""
        positions = np.zeros((16, 3))
        
        # 手腕位置作为原点
        positions[0] = [0, 0, 0]
        
        # 设置各关节相对位置
        # 拇指链
        positions[1] = [-0.03, 0, 0]      # Thumb1
        positions[2] = [-0.06, 0, 0.02]   # Thumb2
        positions[3] = [-0.08, 0, 0.04]   # Thumb3
        
        # 食指链
        positions[4] = [0, 0.07, 0]       # Index1
        positions[5] = [0, 0.11, 0]       # Index2
        positions[6] = [0, 0.14, 0]       # Index3
        
        # 中指链
        positions[7] = [0.02, 0.07, 0]    # Middle1
        positions[8] = [0.02, 0.11, 0]    # Middle2
        positions[9] = [0.02, 0.14, 0]    # Middle3
        
        # 无名指链
        positions[10] = [0.04, 0.06, 0]   # Ring1
        positions[11] = [0.04, 0.10, 0]   # Ring2
        positions[12] = [0.04, 0.13, 0]   # Ring3
        
        # 小指链
        positions[13] = [0.06, 0.05, 0]   # Pinky1
        positions[14] = [0.06, 0.08, 0]   # Pinky2
        positions[15] = [0.06, 0.11, 0]   # Pinky3
        
        return positions

    def update_joint_rotations(self, joint_rotations):
        """更新关节旋转"""
        positions = self._get_joint_positions()
        bone_lengths = {}  # 存储骨骼原始长度
        
        # 预先计算所有骨骼的长度
        for start_idx, end_idx in self.bone_connections:
            vec = positions[end_idx] - positions[start_idx]
            bone_lengths[(start_idx, end_idx)] = np.linalg.norm(vec)
        
        # 应用旋转
        for i in range(16):
            if i == 0:  # 手腕特殊处理
                continue
                
            # 找到当前关节的父关节
            for start_idx, end_idx in self.bone_connections:
                if end_idx == i:
                    parent_id = start_idx
                    break
            
            # 计算最终旋转
            parent_rot = self.parent_rotations[parent_id]
            original_rot = self.original_rotations[i]
            current_rot = joint_rotations[i]
            
            # 计算最终旋转
            inv_parent = np.array([parent_rot[0], parent_rot[1], parent_rot[2], -parent_rot[3]])
            used_rot = self._quaternion_multiply(
                self._quaternion_multiply(inv_parent, current_rot),
                parent_rot
            )
            final_rot = self._quaternion_multiply(used_rot, original_rot)
            
            # 更新关节位置，保持骨骼长度
            rot_matrix = quat2mat(final_rot)
            direction = positions[i] - positions[parent_id]
            direction = direction / np.linalg.norm(direction)  # 单位化方向向量
            
            # 使用原始骨骼长度
            bone_length = bone_lengths[(parent_id, i)]
            positions[i] = positions[parent_id] + direction * bone_length
            
            # 应用旋转
            relative_pos = positions[i] - positions[parent_id]
            rotated_pos = rot_matrix @ (relative_pos / np.linalg.norm(relative_pos)) * bone_length
            positions[i] = positions[parent_id] + rotated_pos
        
        # 更新可视化
        self._update_visualization(positions)

    def _quaternion_multiply(self, q1, q2):
        """四元数乘法"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def _update_visualization(self, positions):
        """更新可视化"""
        # 更新关节位置
        for i, sphere in enumerate(self.joints):
            transformation = np.eye(4)
            transformation[:3, 3] = positions[i] - sphere.get_center()
            sphere.transform(transformation)
            self.vis.update_geometry(sphere)
        
        # 更新骨骼
        for i, (start_idx, end_idx) in enumerate(self.bone_connections):
            self._update_bone(positions[start_idx], positions[end_idx], self.bones[i])
            self.vis.update_geometry(self.bones[i])

    def run_demo(self):
        """运行抓握动作演示"""
        print("运行抓握动作演示...")
        
        # 演示抓握动作
        t = 0
        while True:
            # 创建所有关节的旋转四元数
            joint_rotations = [np.array([0, 0, 0, 1]) for _ in range(16)]
            
            # 使用 0-1 的插值来控制抓握程度
            grip_factor = (np.sin(t) + 1) / 2  # 0 到 1 循环变化
            
            # 拇指的抓握动作
            thumb_angle = grip_factor * np.pi/3  # 最大60度
            for i in [1, 2, 3]:  # Thumb1, Thumb2, Thumb3
                joint_rotations[i] = np.array([
                    0,                    # x
                    np.sin(thumb_angle/2),# y (绕y轴转动使拇指内收)
                    0,                    # z
                    np.cos(thumb_angle/2) # w
                ])
            
            # 其他手指的抓握动作
            # 食指
            for i in [4, 5, 6]:  # Index1, Index2, Index3
                finger_angle = grip_factor * np.pi/2  # 最大90度
                if i == 4:  # 第一关节角度小一点
                    finger_angle *= 0.5
                joint_rotations[i] = np.array([
                    np.sin(finger_angle/2),  # x (绕x轴转动使手指弯曲)
                    0,                       # y
                    0,                       # z
                    np.cos(finger_angle/2)   # w
                ])
            
            # 中指（角度稍大）
            for i in [7, 8, 9]:  # Middle1, Middle2, Middle3
                finger_angle = grip_factor * np.pi/2 * 1.1
                if i == 7:  # 第一关节角度小一点
                    finger_angle *= 0.5
                joint_rotations[i] = np.array([
                    np.sin(finger_angle/2),
                    0,
                    0,
                    np.cos(finger_angle/2)
                ])
            
            # 无名指（角度再小一点）
            for i in [10, 11, 12]:  # Ring1, Ring2, Ring3
                finger_angle = grip_factor * np.pi/2 * 0.9
                if i == 10:  # 第一关节角度小一点
                    finger_angle *= 0.5
                joint_rotations[i] = np.array([
                    np.sin(finger_angle/2),
                    0,
                    0,
                    np.cos(finger_angle/2)
                ])
            
            # 小指（角度最小）
            for i in [13, 14, 15]:  # Pinky1, Pinky2, Pinky3
                finger_angle = grip_factor * np.pi/2 * 0.8
                if i == 13:  # 第一关节角度小一点
                    finger_angle *= 0.5
                joint_rotations[i] = np.array([
                    np.sin(finger_angle/2),
                    0,
                    0,
                    np.cos(finger_angle/2)
                ])
            
            # 更新关节旋转
            self.update_joint_rotations(joint_rotations)
            
            # 更新显示
            self.vis.poll_events()
            self.vis.update_renderer()
            
            t += 0.03  # 减慢动画速度
            time.sleep(0.03)
            
            # 检查窗口是否关闭
            if not self.vis.poll_events():
                break
        
        self.vis.destroy_window()

def main():
    controller = HandController()
    controller.run_demo()

if __name__ == "__main__":
    main()