import open3d as o3d
import numpy as np
import keyboard
from transforms3d.euler import euler2mat
import time

class HandVisualizer:
    def __init__(self):
        # 定义手指骨骼连接
        self.lines = np.array([
            # 拇指
            [1, 2], [2, 3], [3, 4],
            # 食指
            [5, 6], [6, 7], [7, 8],
            # 中指
            [9, 10], [10, 11], [11, 12],
            # 无名指
            [13, 14], [14, 15], [15, 16],
            # 小指
            [17, 18], [18, 19], [19, 20],
            # 掌骨连接
            [1, 5], [5, 9], [9, 13], [13, 17],
            # 连接手腕
            [0, 1], [17, 0]
        ])

        # 初始化可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        # 创建坐标系
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.coord_frame)

        # 创建世界坐标系（原点）
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.world_frame)
        
        # 创建手腕坐标系
        self.wrist_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        self.vis.add_geometry(self.wrist_frame)

        # 初始化手部几何体
        self.joints = []
        self.bones = []
        self._init_hand_geometries()

    def _init_hand_geometries(self):
        """初始化手部几何体"""
        # 创建关节球体
        for i in range(21):
            # 指尖和手腕用大球体，其他关节用小球体
            radius = 0.011 if i in [0, 4, 8, 12, 16, 20] else 0.007
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            
            # 设置颜色
            if i in [0, 1, 5, 9, 13, 17]:  # 手腕和掌指关节为绿色
                sphere.paint_uniform_color([0, 1, 0])
            elif i in [4, 8, 12, 16, 20]:  # 指尖为红色
                sphere.paint_uniform_color([1, 0, 0])
            else:  # 其他关节为蓝色
                sphere.paint_uniform_color([0, 0, 1])
            if i in [1, 2, 3, 4]:  # 拇指特殊标记为粉色
                sphere.paint_uniform_color([1, 0, 1])
            
            self.joints.append(sphere)
            self.vis.add_geometry(sphere)

        # 创建骨骼圆柱体
        for _ in self.lines:
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.003, height=1.0)
            cylinder.paint_uniform_color([1, 0, 0])
            self.bones.append(cylinder)
            self.vis.add_geometry(cylinder)

    def _create_static_hand_positions(self, wrist_position=None):
        """创建一个标准手型的关节位置"""
        if wrist_position is None:
            wrist_position = np.array([0, 0, 0])
            
        positions = np.zeros((21, 3))
        
        # 所有位置都是相对于手腕的偏移
        offsets = np.zeros((21, 3))
        
        # 手腕位置作为原点
        offsets[0] = [0, 0, 0]
        
        # 拇指 (弯曲姿态)
        offsets[1] = [-0.03, 0.02, 0]      # 掌指关节
        offsets[2] = [-0.05, 0.04, 0.02]   # 近节指关节
        offsets[3] = [-0.06, 0.06, 0.03]   # 远节指关节
        offsets[4] = [-0.07, 0.08, 0.04]   # 指尖i
        
        # 食指
        offsets[5] = [0, 0.07, 0]          # 掌指关节
        offsets[6] = [0, 0.11, 0]          # 近节指关节
        offsets[7] = [0, 0.14, 0]          # 远节指关节
        offsets[8] = [0, 0.16, 0]          # 指尖
        
        # 中指
        offsets[9] = [0.02, 0.07, 0]       # 掌指关节
        offsets[10] = [0.02, 0.12, 0]      # 近节指关节
        offsets[11] = [0.02, 0.15, 0]      # 远节指关节
        offsets[12] = [0.02, 0.17, 0]      # 指尖
        
        # 无名指
        offsets[13] = [0.04, 0.06, 0]      # 掌指关节
        offsets[14] = [0.04, 0.11, 0]      # 近节指关节
        offsets[15] = [0.04, 0.14, 0]      # 远节指关节
        offsets[16] = [0.04, 0.16, 0]      # 指尖
        
        # 小指
        offsets[17] = [0.06, 0.05, 0]      # 掌指关节
        offsets[18] = [0.06, 0.09, 0]      # 近节指关节
        offsets[19] = [0.06, 0.12, 0]      # 远节指关节
        offsets[20] = [0.06, 0.14, 0]      # 指尖
        
        # 将所有偏移加到手腕位置上
        for i in range(21):
            positions[i] = wrist_position + offsets[i]
        
        return positions

    def _update_bone(self, start_point, end_point, cylinder):
        """更新骨骼圆柱体"""
        # 计算圆柱体长度和方向
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        direction = direction / length

        # 创建新圆柱体
        new_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.003, height=length)
        new_cylinder.paint_uniform_color([1, 0, 0])
        
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

    def update_hand(self, joint_positions):
        """更新手部位置"""
        # 确保输入的关节位置数量正确
        if len(joint_positions) != 21:
            raise ValueError(f"Expected 21 joint positions, got {len(joint_positions)}")
        
        # 更新关节球体位置
        for i in range(21):  # 明确限制范围为0-20
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

    def update_hand_position(self, wrist_position, rotation=None):
        """更新整个手的位置和旋转"""
        # 获取新的关节位置
        positions = self._create_static_hand_positions(wrist_position)
        
        # 更新手腕坐标系位置
        transformation = np.eye(4)
        current_wrist_center = np.asarray(self.wrist_frame.get_center())
        transformation[:3, 3] = wrist_position - current_wrist_center
        if rotation is not None:
            transformation[:3, :3] = rotation
        self.wrist_frame.transform(transformation)
        self.vis.update_geometry(self.wrist_frame)
        
        # 如果提供了旋转矩阵，应用旋转
        if rotation is not None:
            # 将所有点相对于手腕旋转
            for i in range(1, 21):  # 跳过手腕本身
                relative_pos = positions[i] - positions[0]  # 相对于手腕的位置
                rotated_pos = rotation @ relative_pos      # 应用旋转
                positions[i] = positions[0] + rotated_pos  # 转回绝对位置
        
        # 更新手的位置
        self.update_hand(positions)
        
        # 更新显示
        for joint in self.joints:
            self.vis.update_geometry(joint)
        for bone in self.bones:
            self.vis.update_geometry(bone)

    def run(self):
        """运行可视化"""
        print("控制说明:")
        print("移动: W/S - 前后, A/D - 左右, R/F - 上下")
        print("旋转: I/K - X轴, J/L - Y轴, U/O - Z轴")
        print("Q: 退出")
        
        # 初始位置和旋转
        current_position = np.zeros(3)  # 使用 zeros 而不是 array([0, 0, 0])
        current_rotation = np.eye(3)
        pos_step = 0.01  # 减小位置移动步长，使移动更平滑
        rot_step = 0.1   # 旋转步长（弧度）
        
        while True:
            time.sleep(0.01)
            
            # 创建位置变化向量
            position_change = np.zeros(3)
            
            # 位置控制
            if keyboard.is_pressed('w'):
                position_change[2] -= pos_step  # 前移
            if keyboard.is_pressed('s'):
                position_change[2] += pos_step  # 后移
            if keyboard.is_pressed('a'):
                position_change[0] -= pos_step  # 左移
            if keyboard.is_pressed('d'):
                position_change[0] += pos_step  # 右移
            if keyboard.is_pressed('r'):
                position_change[1] += pos_step  # 上移
            if keyboard.is_pressed('f'):
                position_change[1] -= pos_step  # 下移
            
            # 更新位置
            if np.any(position_change != 0):
                current_position += position_change
                print(f"当前位置: [{current_position[0]:.3f}, {current_position[1]:.3f}, {current_position[2]:.3f}]")
            
            # 旋转控制
            rotation_changed = False
            if keyboard.is_pressed('i'):
                current_rotation = current_rotation @ euler2mat(rot_step, 0, 0)
                rotation_changed = True
                print("绕X轴正向旋转")
            if keyboard.is_pressed('k'):
                current_rotation = current_rotation @ euler2mat(-rot_step, 0, 0)
                rotation_changed = True
                print("绕X轴负向旋转")
            if keyboard.is_pressed('j'):
                current_rotation = current_rotation @ euler2mat(0, rot_step, 0)
                rotation_changed = True
                print("绕Y轴正向旋转")
            if keyboard.is_pressed('l'):
                current_rotation = current_rotation @ euler2mat(0, -rot_step, 0)
                rotation_changed = True
                print("绕Y轴负向旋转")
            if keyboard.is_pressed('u'):
                current_rotation = current_rotation @ euler2mat(0, 0, rot_step)
                rotation_changed = True
                print("绕Z轴正向旋转")
            if keyboard.is_pressed('o'):
                current_rotation = current_rotation @ euler2mat(0, 0, -rot_step)
                rotation_changed = True
                print("绕Z轴负向旋转")
            
            # 更新手的位置和旋转
            if np.any(position_change != 0) or rotation_changed:
                self.update_hand_position(current_position, current_rotation)
            
            self.vis.poll_events()
            self.vis.update_renderer()
            
            if keyboard.is_pressed('q'):
                print("退出程序")
                break
                
        self.vis.destroy_window()

def main():
    visualizer = HandVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main() 