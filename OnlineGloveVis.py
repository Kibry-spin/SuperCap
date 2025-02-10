import numpy as np
import open3d as o3d
import socket
import time

class RealTimeHandVisualizer:
    def __init__(self, port=2211):
        # 初始化可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="双手形状可视化", width=1280, height=720)
        
        # 设置渲染选项
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
        render_option.point_size = 5.0
        
        # 添加坐标系
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        
        # 初始化几何体列表
        self.right_geometries = []
        self.left_geometries = []
        self.initialized = False
        
        # 初始化UDP接收器
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('localhost', port))
        self.sock.settimeout(0.001)

    def create_joint_sphere(self, position, color=[1, 0, 0]):
        """创建关节球体"""
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(position)
        sphere.paint_uniform_color(color)
        return sphere
    
    def create_bone_line(self, start_pos, end_pos, color=[0, 1, 0]):
        """创建骨骼连接线"""
        line_set = o3d.geometry.LineSet()
        points = np.array([start_pos, end_pos])
        lines = np.array([[0, 1]])
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color])
        return line_set

    def process_hand_data(self, values, start_idx, is_right_hand=True):
        """处理单手数据"""
        geometries = []
        finger_colors = [
            [1, 0, 0],  # 拇指 - 红色
            [0, 1, 0],  # 食指 - 绿色
            [0, 0, 1],  # 中指 - 蓝色
            [1, 1, 0],  # 无名指 - 黄色
            [1, 0, 1]   # 小指 - 紫色
        ]
        
        # 手腕位置
        wrist_pos = np.array(values[start_idx:start_idx+3])
        wrist_sphere = self.create_joint_sphere(wrist_pos, [1, 1, 1])  # 白色手腕
        geometries.append(wrist_sphere)
        
        # 处理每个手指
        for finger_idx in range(5):  # 5个手指
            prev_pos = wrist_pos
            finger_start = start_idx + 6 + finger_idx * 18  # 跳过手腕数据(6)，每个手指3个关节*6个值
            
            # 处理每个关节
            for joint_idx in range(3):  # 每个手指3个关节
                joint_start = finger_start + joint_idx * 6
                pos = np.array(values[joint_start:joint_start+3])
                
                # 创建关节球体
                sphere = self.create_joint_sphere(pos, finger_colors[finger_idx])
                geometries.append(sphere)
                
                # 创建骨骼连接线
                line = self.create_bone_line(prev_pos, pos, finger_colors[finger_idx])
                geometries.append(line)
                
                prev_pos = pos
        
        return geometries

    def update_visualization(self, values):
        """更新可视化"""
        # 清除之前的几何体
        if self.initialized:
            for geo in self.right_geometries + self.left_geometries:
                self.vis.remove_geometry(geo, False)
        
        # 处理右手数据
        self.right_geometries = self.process_hand_data(values, 0, True)
        # 处理左手数据
        self.left_geometries = self.process_hand_data(values, 67, False)
        
        # 添加所有几何体
        for geo in self.right_geometries + self.left_geometries:
            self.vis.add_geometry(geo, False)
        
        self.initialized = True
        self.vis.poll_events()
        self.vis.update_renderer()

    def parse_data(self, data_str):
        """解析UDP数据"""
        try:
            start_idx = data_str.find("0.15,0,0.2,")
            if start_idx == -1:
                return None
            values_str = data_str[start_idx:]
            values = [float(x) for x in values_str.strip().split(',')]
            return values
        except Exception as e:
            print(f"数据解析错误: {e}")
            return None

    def run(self):
        """运行可视化"""
        print(f"开始监听UDP端口 2211...")
        
        try:
            while True:
                try:
                    data, _ = self.sock.recvfrom(4096)
                    data_str = data.decode()
                    if data_str.startswith("Glove1"):
                        values = self.parse_data(data_str)
                        if values is not None:
                            self.update_visualization(values)
                except socket.timeout:
                    pass
                
                if not self.vis.poll_events():
                    break
                    
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.sock.close()
            self.vis.destroy_window()

def main():
    visualizer = RealTimeHandVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()
