import open3d as o3d
import numpy as np

class HandModel:
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
        
        # 初始化手的位置
        self._create_static_hand()

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
        offsets = np.zeros((21, 3))
        
        # 手腕位置作为原点
        offsets[0] = [0, 0, 0]
        
        # 拇指 (调整位置避免重叠)
        offsets[1] = [-0.02, 0.04, 0]      # 掌指关节
        offsets[2] = [-0.02, 0.08, 0]      # 近节指关节
        offsets[3] = [-0.02, 0.11, 0]      # 远节指关节
        offsets[4] = [-0.02, 0.13, 0]      # 指尖
        
        # 食指
        offsets[5] = [0, 0.09, 0]          # 掌指关节
        offsets[6] = [0, 0.13, 0]          # 近节指关节
        offsets[7] = [0, 0.16, 0]          # 远节指关节
        offsets[8] = [0, 0.18, 0]          # 指尖
        
        # 中指
        offsets[9] = [0.02, 0.10, 0]       # 掌指关节
        offsets[10] = [0.02, 0.14, 0]      # 近节指关节
        offsets[11] = [0.02, 0.17, 0]      # 远节指关节
        offsets[12] = [0.02, 0.19, 0]      # 指尖
        
        # 无名指
        offsets[13] = [0.04, 0.09, 0]      # 掌指关节
        offsets[14] = [0.04, 0.13, 0]      # 近节指关节
        offsets[15] = [0.04, 0.16, 0]      # 远节指关节
        offsets[16] = [0.04, 0.18, 0]      # 指尖
        
        # 小指
        offsets[17] = [0.06, 0.08, 0]      # 掌指关节
        offsets[18] = [0.06, 0.12, 0]      # 近节指关节
        offsets[19] = [0.06, 0.15, 0]      # 远节指关节
        offsets[20] = [0.06, 0.17, 0]      # 指尖
        
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

    def _create_static_hand(self):
        """创建静态手型"""
        # 获取关节位置
        positions = self._create_static_hand_positions()
        
        # 更新关节球体位置
        for i in range(21):
            sphere = self.joints[i]
            current_center = np.asarray(sphere.get_center())
            transformation = np.eye(4)
            transformation[:3, 3] = positions[i] - current_center
            sphere.transform(transformation)
            self.vis.update_geometry(sphere)
            
        # 更新骨骼
        for i, (start_idx, end_idx) in enumerate(self.lines):
            start_point = positions[start_idx]
            end_point = positions[end_idx]
            self._update_bone(start_point, end_point, self.bones[i])
            self.vis.update_geometry(self.bones[i])

    def run(self):
        """运行可视化"""
        # 设置默认视角
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.7)
        ctr.set_lookat([0, 0.1, 0])  # 将视点中心移到手的中心位置
        
        # 运行可视化
        self.vis.run()
        self.vis.destroy_window()

def main():
    hand_model = HandModel()
    hand_model.run()

if __name__ == "__main__":
    main() 