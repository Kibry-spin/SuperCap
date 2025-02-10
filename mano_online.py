import torch
import numpy as np
import open3d as o3d
from manotorch.manolayer import ManoLayer
import scipy.spatial.transform as sciR
import socket
import time
import threading

class MANOOnlineVisualizer:
    def __init__(self, port=4396):
        """初始化在线可视化器
        Args:
            port: UDP接收端口号,默认4396
        """
        # 初始化MANO模型
        self.mano_assets_root = 'E:/Project/ForheartUnity/MANO'
        self.mano_layer = ManoLayer(
            mano_assets_root=self.mano_assets_root,
            use_pca=False,
            flat_hand_mean=True,
            ncomps=45
        )
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
        self.vis.create_window(window_name="MANO实时手型可视化", width=1280, height=720)
        
        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(coordinate_frame)
        
        # 初始化手部网格
        self.right_hand = None
        self.left_hand = None
        self.initialized = False
        
        # UDP相关
        self.port = port
        self.sock = None
        self.running = True
        self.frame_count = 0
        self.last_fps_time = time.perf_counter()
        self.fps = 0.0
        
        # 数据处理相关
        self.latest_data = None
        self.data_lock = threading.Lock()
        
        # MANO关节顺序映射
        self.mano_order = [0,                    # wrist
                          4, 5, 6,               # index
                          7, 8, 9,               # middle
                          13, 14, 15,            # pinky
                          10, 11, 12,            # ring
                          1, 2, 3]               # thumb
        
        # 初始化UDP接收器
        self._init_udp()
        
    def _init_udp(self):
        """初始化UDP接收器"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('localhost', self.port))
            self.sock.settimeout(0.001)  # 非阻塞模式
            print(f"UDP接收器初始化成功,监听端口 {self.port}")
        except Exception as ex:
            print(f"UDP接收器初始化失败: {ex}")
            raise ex
            
    def create_hand_mesh(self, vertices, color=[0.9, 0.7, 0.7], side='left'):
        """创建手部网格"""
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        if side == 'left':
            mesh.triangles = o3d.utility.Vector3iVector(self.faces_lh)
        else:
            mesh.triangles = o3d.utility.Vector3iVector(self.faces_rh)
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        return mesh
        
    def parse_udp_data(self, data_str):
        """解析UDP数据"""
        try:
            # 查找数据起始位置
            start_idx = data_str.find("subpackage 1/1,") + len("subpackage 1/1,")
            if start_idx == -1:
                return None
                
            # 提取数值部分
            values_str = data_str[start_idx:]
            values = [float(x) for x in values_str.strip().split(',')]
            
            # 确保数据完整性
            if len(values) < 192:  # 16*6*2 = 192 (两只手)
                return None
                
            return np.array(values[:192])  # 只取前192个值
                
        except Exception as e:
            print(f"数据解析错误: {e}")
            return None
            
    def update_visualization(self, frame_data):
        """更新可视化"""
        # 重新排序右手数据
        right_hand_data = frame_data[:96].reshape(-1, 6)  # 重塑为(16, 6)的数组
        right_hand_reordered = np.zeros((16, 3))  # 创建(16, 3)的数组存储欧拉角
        for i, idx in enumerate(self.mano_order):
            euler = right_hand_data[idx, :3].copy()  # 只取旋转数据(前3个值)
            
            # 全局旋转处理（手腕）
            if i == 0:
                # 右手手腕调整
                euler[0] *= 1   # Z轴
                euler[1] *= 1   # Y轴
                euler[2] *= -1  # X轴
            else:
                # 手指关节调整
                euler[0] *= 1   # Z轴（扭转）
                euler[1] *= -1  # Y轴（侧向）
                euler[2] *= 1   # X轴（弯曲）
                
            right_hand_reordered[i] = euler
        
        # 重新排序左手数据
        left_hand_data = frame_data[96:].reshape(-1, 6)  # 重塑为(16, 6)的数组
        left_hand_reordered = np.zeros((16, 3))  # 创建(16, 3)的数组存储欧拉角
        for i, idx in enumerate(self.mano_order):
            euler = left_hand_data[idx, :3].copy()  # 只取旋转数据(前3个值)
            
            # 全局旋转处理（手腕）
            if i == 0:
                # 左手手腕镜像处理
                euler[0] *= 1   # Z轴
                euler[1] *= 1   # Y轴
                euler[2] *= 1   # X轴
            else:
                # 手指关节镜像处理
                euler[0] *= -1  # Z轴（扭转）
                euler[1] *= -1  # Y轴（侧向）
                euler[2] *= -1  # X轴（弯曲）
                
            left_hand_reordered[i] = euler
            
        # 转换为MANO参数格式
        right_pose_params = torch.zeros((1, 48))
        left_pose_params = torch.zeros((1, 48))
        
        # 转换为弧度
        right_hand_rad = right_hand_reordered * np.pi / 180.0
        left_hand_rad = left_hand_reordered * np.pi / 180.0
        
        # 使用不同的旋转顺序
        right_rot = sciR.Rotation.from_euler('YZX', right_hand_rad)
        left_rot = sciR.Rotation.from_euler('YZX', left_hand_rad)
        
        # 获取旋转向量
        right_rotvec = right_rot.as_rotvec()
        left_rotvec = left_rot.as_rotvec()
        
        right_pose_params[0] = torch.from_numpy(right_rotvec.reshape(-1))
        left_pose_params[0] = torch.from_numpy(left_rotvec.reshape(-1))
        
        # 生成手部网格
        with torch.no_grad():
            right_verts = self.mano_layer(
                pose_coeffs=right_pose_params,
                betas=torch.zeros(1, 10)
            ).verts[0].cpu().numpy()
            
            left_verts = self.mano_layer_left(
                pose_coeffs=left_pose_params,
                betas=torch.zeros(1, 10)
            ).verts[0].cpu().numpy()
            
        # 更新可视化
        if not self.initialized:
            self.right_hand = self.create_hand_mesh(right_verts, [0.9, 0.7, 0.7], side='right')
            self.left_hand = self.create_hand_mesh(left_verts, [0.7, 0.7, 0.9], side='left')
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
        
    def udp_thread(self):
        """UDP数据接收线程"""
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                data_str = data.decode()
                
                if data_str.startswith("Glove1"):
                    parsed_data = self.parse_udp_data(data_str)
                    if parsed_data is not None:
                        with self.data_lock:
                            self.latest_data = parsed_data
                            
                        # 更新帧率
                        self.frame_count += 1
                        current_time = time.perf_counter()
                        elapsed_time = current_time - self.last_fps_time
                        
                        if elapsed_time >= 1.0:
                            self.fps = self.frame_count / elapsed_time
                            print(f"\r实时显示中... FPS: {self.fps:.1f}", end='', flush=True)
                            self.frame_count = 0
                            self.last_fps_time = current_time
                            
            except socket.timeout:
                continue
            except Exception as e:
                print(f"\nUDP接收错误: {e}")
                continue
                
    def run(self):
        """运行可视化"""
        # 设置默认视角
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.7)
        ctr.set_lookat([0.1, 0, 0])
        
        # 启动UDP接收线程
        udp_thread = threading.Thread(target=self.udp_thread)
        udp_thread.daemon = True
        udp_thread.start()
        
        print("开始实时显示...")
        print("按 ESC 退出")
        
        try:
            while self.running:
                # 获取最新数据
                with self.data_lock:
                    current_data = self.latest_data
                    
                if current_data is not None:
                    self.update_visualization(current_data)
                    
                if not self.vis.poll_events():
                    break
                    
                self.vis.update_renderer()
                time.sleep(0.001)  # 避免CPU占用过高
                
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.running = False
            self.sock.close()
            self.vis.destroy_window()

def main():
    visualizer = MANOOnlineVisualizer(port=4396)
    visualizer.run()

if __name__ == "__main__":
    main()
