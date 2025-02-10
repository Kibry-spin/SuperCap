import os
import open3d as o3d
import numpy as np
import time

class PointCloudVisualizer:
    def __init__(self, cloud_dir):
        """初始化点云可视化器
        Args:
            cloud_dir: 点云文件目录
        """
        self.cloud_dir = cloud_dir
        
        # 初始化可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("点云序列播放", width=1280, height=720)
        
        # 设置渲染选项
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
        render_option.point_size = 3.0  # 增大点的大小
        
        # 添加世界坐标系（原点）
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0,  # 增大坐标轴长度为1米
            origin=[0, 0, 0]
        )
        self.vis.add_geometry(self.coordinate_frame)
        
        # 设置默认视角
        self.setup_camera()
        
        # 加载点云文件列表
        self.cloud_files = sorted([f for f in os.listdir(cloud_dir) if f.endswith('.ply')])
        print(f"找到 {len(self.cloud_files)} 个点云文件")
        
        # 当前显示的点云
        self.current_cloud = None
        
    def setup_camera(self):
        """设置相机视角"""
        ctr = self.vis.get_view_control()
        
        # 设置相机参数
        cam_params = {
            'zoom': 0.45,          # 缩放因子
            'front': [0.5, -0.5, -0.5],  # 相机朝向
            'lookat': [0, 0, 0],   # 看向原点
            'up': [0, 0, 1]        # 向上方向
        }
        
        # 应用相机参数
        ctr.set_zoom(cam_params['zoom'])
        ctr.set_front(cam_params['front'])
        ctr.set_lookat(cam_params['lookat'])
        ctr.set_up(cam_params['up'])
    
    def update_point_cloud(self, cloud_file):
        """更新显示的点云"""
        # 移除当前点云
        if self.current_cloud is not None:
            self.vis.remove_geometry(self.current_cloud, False)
        
        # 加载新点云
        cloud_path = os.path.join(self.cloud_dir, cloud_file)
        pcd = o3d.io.read_point_cloud(cloud_path)
        
        # 可选：对点云进行下采样以提高显示性能
        pcd = pcd.voxel_down_sample(voxel_size=0.01)  # 1cm体素下采样
        
        # 添加新点云
        self.current_cloud = pcd
        self.vis.add_geometry(self.current_cloud, False)
        
        # 更新渲染
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def run(self, frame_delay=0.1):
        """运行可视化
        Args:
            frame_delay: 帧间延迟（秒）
        """
        print("\n开始播放点云序列...")
        print("按 'Q' 退出播放")
        
        try:
            while True:
                for i, cloud_file in enumerate(self.cloud_files):
                    # 更新点云
                    self.update_point_cloud(cloud_file)
                    
                    # 显示进度
                    print(f"\r播放进度: {i+1}/{len(self.cloud_files)}", end='')
                    
                    # 延迟
                    time.sleep(frame_delay)
                    
                    # 检查是否退出
                    if not self.vis.poll_events():
                        break
                
                # 检查是否退出
                if not self.vis.poll_events():
                    break
                    
        except KeyboardInterrupt:
            print("\n用户中断播放")
        finally:
            self.vis.destroy_window()
            print("\n播放结束")

def main():
    # 指定点云数据目录
    cloud_dir = 'MultimodalData/20250208_190448/world_clouds'
    
    try:
        visualizer = PointCloudVisualizer(cloud_dir)
        visualizer.run(frame_delay=0.1)  # 0.1秒的帧间延迟（约10FPS）
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()