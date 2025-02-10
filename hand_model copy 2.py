import open3d as o3d
import numpy as np
import socket
import threading
import time
import os
import sys

def force_cleanup_open3d():
    """强制清理所有Open3D资源"""
    try:
        temp_vis = o3d.visualization.Visualizer()
        temp_vis.destroy_window()
        del temp_vis
    except:
        pass
    time.sleep(1)  # 等待资源释放

class HandModel:
    def __init__(self):
        # 强制清理之前的Open3D资源
        force_cleanup_open3d()
        
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

        # UDP相关
        self.sock = None
        self.is_running = False
        self.frame_count = 0
        self.last_fps_time = time.perf_counter()
        self.fps = 0
        self.data_pending = False
        
        # 初始化可视化器相关变量
        self.vis = None
        self.coord_frame = None
        self.world_frame = None
        
        # 左右手的几何体
        self.left_joints = []
        self.left_bones = []
        self.right_joints = []
        self.right_bones = []
        
        # 左右手的坐标系
        self.left_wrist_frame = None
        self.right_wrist_frame = None

    def init_visualizer(self):
        """初始化可视化器"""
        try:
            # 确保之前的窗口已关闭
            if self.vis is not None:
                try:
                    self.vis.destroy_window()
                except:
                    pass
                self.vis = None
                time.sleep(0.5)
            
            # 创建新的可视化器
            self.vis = o3d.visualization.Visualizer()
            
            # 使用非阻塞方式创建窗口
            try:
                self.vis.create_window(window_name="双手模型可视化", 
                                     width=1280, 
                                     height=720,
                                     visible=True,
                                     left=50,
                                     top=50)
            except Exception as e:
                print(f"创建窗口失败: {e}")
                return False
                
            # 设置渲染选项
            opt = self.vis.get_render_option()
            opt.background_color = np.asarray([0.2, 0.2, 0.2])
            opt.point_size = 5.0
            opt.show_coordinate_frame = True
            opt.light_on = True
            
            # 创建世界坐标系
            self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.vis.add_geometry(self.world_frame)
            
            # 创建左右手腕坐标系
            self.left_wrist_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            self.right_wrist_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            self.vis.add_geometry(self.left_wrist_frame)
            self.vis.add_geometry(self.right_wrist_frame)

            # 初始化左右手的几何体
            self._init_hand_geometries()
            
            # 初始化左右手的位置
            self._create_static_hands()
            
            # 设置默认视角
            ctr = self.vis.get_view_control()
            ctr.set_zoom(0.7)
            ctr.set_lookat([0, 0.1, 0])
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, 1, 0])
            
            return True
            
        except Exception as e:
            print(f"可视化器初始化失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        self.is_running = False
        
        # 清理UDP资源
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
            
        # 清理可视化器资源
        if self.vis:
            try:
                self.vis.destroy_window()
            except:
                pass
            self.vis = None
            
        # 清空几何体列表
        self.left_joints.clear()
        self.left_bones.clear()
        self.right_joints.clear()
        self.right_bones.clear()
        self.coord_frame = None
        self.world_frame = None
        self.left_wrist_frame = None
        self.right_wrist_frame = None
        
        # 强制执行垃圾回收
        import gc
        gc.collect()
        
        time.sleep(0.5)  # 等待资源释放

    def _init_hand_geometries(self):
        """初始化左右手的几何体"""
        # 创建左手关节球体
        for i in range(21):
            # 指尖和手腕用大球体，其他关节用小球体
            radius = 0.008 if i in [0, 4, 8, 12, 16, 20] else 0.005
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            
            # 设置颜色
            if i in [0, 1, 5, 9, 13, 17]:  # 手腕和掌指关节为绿色
                sphere.paint_uniform_color([0, 1, 0])
            elif i in [4, 8, 12, 16, 20]:  # 指尖为红色
                sphere.paint_uniform_color([1, 0, 0])
            else:  # 其他关节为蓝色
                sphere.paint_uniform_color([0, 0, 1])
            
            self.left_joints.append(sphere)
            self.vis.add_geometry(sphere)

        # 创建右手关节球体
        for i in range(21):
            radius = 0.008 if i in [0, 4, 8, 12, 16, 20] else 0.005
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            
            if i in [0, 1, 5, 9, 13, 17]:
                sphere.paint_uniform_color([0, 1, 0])
            elif i in [4, 8, 12, 16, 20]:
                sphere.paint_uniform_color([1, 0, 0])
            else:
                sphere.paint_uniform_color([0, 0, 1])
            
            self.right_joints.append(sphere)
            self.vis.add_geometry(sphere)

        # 创建左手骨骼圆柱体
        for _ in self.lines:
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=1.0)
            cylinder.paint_uniform_color([0.7, 0.7, 0.7])
            self.left_bones.append(cylinder)
            self.vis.add_geometry(cylinder)

        # 创建右手骨骼圆柱体
        for _ in self.lines:
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=1.0)
            cylinder.paint_uniform_color([0.7, 0.7, 0.7])
            self.right_bones.append(cylinder)
            self.vis.add_geometry(cylinder)

    def _create_static_hand_positions(self, wrist_position=None, is_left=False):
        """创建一个标准手型的关节位置"""
        if wrist_position is None:
            wrist_position = np.array([-0.2 if is_left else 0.2, 0, 0])
            
        positions = np.zeros((21, 3))
        offsets = np.zeros((21, 3))
        
        # 手腕位置作为原点
        offsets[0] = [0, 0, 0]
        
        # 拇指
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
        
        # 如果是左手，对x轴取反
        if is_left:
            offsets[:, 0] = -offsets[:, 0]
        
        # 将所有偏移加到手腕位置上
        for i in range(21):
            positions[i] = wrist_position + offsets[i]
        
        return positions

    def _create_static_hands(self):
        """创建静态的左右手"""
        # 创建左手
        left_positions = self._create_static_hand_positions(is_left=True)
        for i in range(21):
            sphere = self.left_joints[i]
            current_center = np.asarray(sphere.get_center())
            transformation = np.eye(4)
            transformation[:3, 3] = left_positions[i] - current_center
            sphere.transform(transformation)
            self.vis.update_geometry(sphere)
            
        for i, (start_idx, end_idx) in enumerate(self.lines):
            start_point = left_positions[start_idx]
            end_point = left_positions[end_idx]
            self._update_bone(start_point, end_point, self.left_bones[i])
            self.vis.update_geometry(self.left_bones[i])
            
        # 创建右手
        right_positions = self._create_static_hand_positions(is_left=False)
        for i in range(21):
            sphere = self.right_joints[i]
            current_center = np.asarray(sphere.get_center())
            transformation = np.eye(4)
            transformation[:3, 3] = right_positions[i] - current_center
            sphere.transform(transformation)
            self.vis.update_geometry(sphere)
            
        for i, (start_idx, end_idx) in enumerate(self.lines):
            start_point = right_positions[start_idx]
            end_point = right_positions[end_idx]
            self._update_bone(start_point, end_point, self.right_bones[i])
            self.vis.update_geometry(self.right_bones[i])

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

    def init_udp(self):
        """初始化UDP接收"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('0.0.0.0', 2211))
            self.sock.settimeout(0.001)  # 设置超时
            print("UDP服务器已启动，正在监听端口2211...")
            return True
        except Exception as e:
            print(f"UDP初始化失败: {e}")
            return False

    def parse_udp_data(self, data_str):
        """解析UDP数据"""
        try:
            # 查找数据部分的起始位置
            start_idx = data_str.find("0.15,0,0.2,")
            if start_idx == -1:
                return None
            
            # 提取数据部分并转换为浮点数
            data_values = data_str[start_idx:].strip().split(',')
            values = [float(x) for x in data_values]
            
            # 确保数据长度正确
            if len(values) >= 134:
                hand_data = {
                    'right_wrist_pos': values[0:3],
                    'right_wrist_rot': values[3:7],
                    'right_joints_rot': [
                        values[7:11],   # 拇指第一关节
                        values[11:15],  # 拇指第二关节
                        values[15:19],  # 拇指第三关节
                        values[19:23],  # 食指第一关节
                        values[23:27],  # 食指第二关节
                        values[27:31],  # 食指第三关节
                        values[31:35],  # 中指第一关节
                        values[35:39],  # 中指第二关节
                        values[39:43],  # 中指第三关节
                        values[43:47],  # 无名指第一关节
                        values[47:51],  # 无名指第二关节
                        values[51:55],  # 无名指第三关节
                        values[55:59],  # 小指第一关节
                        values[59:63],  # 小指第二关节
                        values[63:67]   # 小指第三关节
                    ],
                    'left_wrist_pos': values[67:70],
                    'left_wrist_rot': values[70:74],
                    'left_joints_rot': [
                        values[74:78],   # 拇指第一关节
                        values[78:82],   # 拇指第二关节
                        values[82:86],   # 拇指第三关节
                        values[86:90],   # 食指第一关节
                        values[90:94],   # 食指第二关节
                        values[94:98],   # 食指第三关节
                        values[98:102],  # 中指第一关节
                        values[102:106], # 中指第二关节
                        values[106:110], # 中指第三关节
                        values[110:114], # 无名指第一关节
                        values[114:118], # 无名指第二关节
                        values[118:122], # 无名指第三关节
                        values[122:126], # 小指第一关节
                        values[126:130], # 小指第二关节
                        values[130:134]  # 小指第三关节
                    ]
                }
                return hand_data
            return None
        except Exception as e:
            print(f"数据解析错误: {e}")
            return None

    def update_hand_model(self, hand_data):
        """更新手模型"""
        if not hand_data:
            return

        # 更新左手
        left_wrist_pos = np.array(hand_data['left_wrist_pos'])
        left_positions = self._create_static_hand_positions(left_wrist_pos, is_left=True)
        
        for i in range(21):
            sphere = self.left_joints[i]
            current_center = np.asarray(sphere.get_center())
            transformation = np.eye(4)
            transformation[:3, 3] = left_positions[i] - current_center
            sphere.transform(transformation)
            self.vis.update_geometry(sphere)
        
        for i, (start_idx, end_idx) in enumerate(self.lines):
            start_point = left_positions[start_idx]
            end_point = left_positions[end_idx]
            self._update_bone(start_point, end_point, self.left_bones[i])
            self.vis.update_geometry(self.left_bones[i])

        # 更新右手
        right_wrist_pos = np.array(hand_data['right_wrist_pos'])
        right_positions = self._create_static_hand_positions(right_wrist_pos, is_left=False)
        
        for i in range(21):
            sphere = self.right_joints[i]
            current_center = np.asarray(sphere.get_center())
            transformation = np.eye(4)
            transformation[:3, 3] = right_positions[i] - current_center
            sphere.transform(transformation)
            self.vis.update_geometry(sphere)
        
        for i, (start_idx, end_idx) in enumerate(self.lines):
            start_point = right_positions[start_idx]
            end_point = right_positions[end_idx]
            self._update_bone(start_point, end_point, self.right_bones[i])
            self.vis.update_geometry(self.right_bones[i])

    def udp_thread(self):
        """UDP接收线程"""
        while self.is_running:
            try:
                data, _ = self.sock.recvfrom(4096)
                data_str = data.decode()
                
                if data_str.startswith("Glove1"):
                    self.frame_count += 1
                    current_time = time.perf_counter()
                    elapsed_time = current_time - self.last_fps_time
                    
                    if elapsed_time >= 1.0:
                        self.fps = self.frame_count / elapsed_time
                        print(f"\r实时帧率: {self.fps:.2f} FPS", end='', flush=True)
                        self.frame_count = 0
                        self.last_fps_time = current_time
                    
                    # 解析数据并更新模型
                    hand_data = self.parse_udp_data(data_str)
                    if hand_data:
                        self.update_hand_model(hand_data)
                        
            except socket.timeout:
                continue
            except Exception as e:
                print(f"\nUDP接收错误: {e}")

    def run_with_udp(self):
        """运行UDP可视化"""
        try:
            print("正在初始化...")
            
            # 确保之前的资源被清理
            self.cleanup()
            time.sleep(0.5)
            
            # 初始化可视化器
            if not self.init_visualizer():
                print("可视化器初始化失败")
                return
                
            print("可视化器初始化成功")
            
            # 初始化UDP
            if not self.init_udp():
                print("UDP初始化失败")
                self.cleanup()
                return
                
            print("UDP初始化成功")
            
            self.is_running = True
            udp_thread = threading.Thread(target=self.udp_thread)
            udp_thread.daemon = True
            udp_thread.start()
            
            print("开始运行...")
            
            # 运行可视化
            try:
                while self.is_running:
                    if not self.vis.poll_events():
                        break
                    self.vis.update_renderer()
                    time.sleep(0.001)
            except Exception as e:
                print(f"可视化运行错误: {e}")
            finally:
                self.cleanup()
                
        except Exception as e:
            print(f"程序运行错误: {e}")
            self.cleanup()

def main():
    # 确保程序开始时没有遗留的可视化器窗口
    force_cleanup_open3d()
    
    try:
        hand_model = HandModel()
        hand_model.run_with_udp()
    except Exception as e:
        print(f"主程序错误: {e}")
    finally:
        if 'hand_model' in locals():
            hand_model.cleanup()
        # 最后再次清理资源
        force_cleanup_open3d()

if __name__ == "__main__":
    main() 