import os
import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
import json
import argparse
from datetime import datetime

class PointCloudGenerator:
    def __init__(self, data_dir, calibration_file):
        """初始化点云生成器
        Args:
            data_dir: 录制数据的目录路径
            calibration_file: 手眼标定结果文件路径
        """
        self.data_dir = data_dir
        
        # 加载手眼标定结果
        self.T_cam2ee = np.load(calibration_file)['arr_0']
        print(f"已加载手眼标定结果: {calibration_file}")
        
        # 加载相机内参
        self._load_camera_intrinsics()
        
        # 加载并对齐时间戳和位姿数据
        self._load_timestamps_and_poses()
        
        # 创建输出目录
        self.output_dir = os.path.join(self.data_dir, 'world_clouds')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _load_camera_intrinsics(self):
        """加载相机内参"""
        intrinsics_file = os.path.join(self.data_dir, 'camera_intrinsics.txt')
        intrinsics_data = {}
        with open(intrinsics_file, 'r') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    key, value = line.strip().split(': ')  # 使用': '分割
                    intrinsics_data[key] = float(value)
            
        self.intrinsics = rs.intrinsics()
        self.intrinsics.width = int(intrinsics_data['width'])
        self.intrinsics.height = int(intrinsics_data['height'])
        self.intrinsics.fx = intrinsics_data['fx']
        self.intrinsics.fy = intrinsics_data['fy']
        self.intrinsics.ppx = intrinsics_data['ppx']
        self.intrinsics.ppy = intrinsics_data['ppy']
        self.intrinsics.model = rs.distortion.none
        self.intrinsics.coeffs = [0, 0, 0, 0, 0]
        
        print("已加载相机内参")
        
    def _load_timestamps_and_poses(self):
        """加载时间戳和位姿数据"""
        # 加载相机时间戳
        camera_timestamps = np.load(os.path.join(self.data_dir, 'timestamp.npy'))
        
        # 加载Tracker数据
        tracker_dir = os.path.join(self.data_dir, 'tracker_data')
        tracker_timestamps = np.load(os.path.join(tracker_dir, 'timestamp.npy'))
        raw_poses = np.load(os.path.join(tracker_dir, 'realsense_tracker_poses.npy'))
        
        print(f"相机帧数: {len(camera_timestamps)}")
        print(f"Tracker帧数: {len(tracker_timestamps)}")
        
        # 对齐数据
        aligned_indices = []
        aligned_poses = []
        
        # 对每个相机帧，找到最近的Tracker帧
        for cam_time in camera_timestamps:
            # 计算时间差
            time_diffs = np.abs(tracker_timestamps - cam_time)
            closest_idx = np.argmin(time_diffs)
            
            # 如果时间差太大（超过100ms），则跳过该帧
            if time_diffs[closest_idx] > 0.1:
                continue
            
            # 将欧拉角位姿转换为4x4变换矩阵
            pose = raw_poses[closest_idx]
            x, y, z = pose[0:3]
            roll, pitch, yaw = np.radians(pose[3:6])  # 转换为弧度
            
            # 创建旋转矩阵
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(roll), -np.sin(roll)],
                          [0, np.sin(roll), np.cos(roll)]])
            
            Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                          [0, 1, 0],
                          [-np.sin(pitch), 0, np.cos(pitch)]])
            
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])
            
            # 组合旋转矩阵
            R = Rz @ Ry @ Rx
            
            # 创建4x4变换矩阵
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            
            aligned_indices.append(closest_idx)
            aligned_poses.append(T)
        
        self.camera_indices = list(range(len(aligned_indices)))  # 保留的相机帧索引
        self.tracker_poses = np.array(aligned_poses)
        
        print(f"时间对齐后的帧数: {len(self.tracker_poses)}")
        
    def create_point_cloud(self, color_image, depth_image):
        """从RGB-D图像创建点云"""
        # 创建Open3D的相机内参
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.intrinsics.width, self.intrinsics.height,
            self.intrinsics.fx, self.intrinsics.fy,
            self.intrinsics.ppx, self.intrinsics.ppy
        )
        
        # 转换为Open3D格式
        depth_o3d = o3d.geometry.Image(depth_image)
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        
        # 创建RGBD图像
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False
        )
        
        # 创建点云
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, camera_intrinsic
        )
        
        return pcd
        
    def process_frames(self):
        """处理所有帧"""
        rgb_dir = os.path.join(self.data_dir, 'rgb')
        depth_dir = os.path.join(self.data_dir, 'depth')
        
        # 创建可视化器用于调试（可选）
        debug_vis = o3d.visualization.Visualizer()
        debug_vis.create_window("点云生成调试", width=1280, height=720)
        
        # 设置渲染选项
        render_option = debug_vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        render_option.point_size = 2.0
        
        # 添加世界坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        debug_vis.add_geometry(coordinate_frame)
        
        # 设置默认视角（与实时可视化保持一致）
        ctr = debug_vis.get_view_control()
        ctr.set_zoom(0.3)
        ctr.set_front([0, 0, -1])   # 设置前方向
        ctr.set_lookat([0, 0, 0])   # 设置视点中心为原点
        ctr.set_up([0, -1, 0])      # 设置上方向
        
        # 获取所有图像文件
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
        
        total_frames = len(self.camera_indices)
        print(f"\n开始处理 {total_frames} 帧数据...")
        
        current_pcd = None  # 用于调试显示
        
        for i, cam_idx in enumerate(self.camera_indices):
            # 读取图像
            color_image = cv2.imread(os.path.join(rgb_dir, f'color_{cam_idx:06d}.jpg'))
            depth_image = cv2.imread(os.path.join(depth_dir, f'depth_{cam_idx:06d}.png'), -1)
            
            if color_image is None or depth_image is None:
                print(f"\n警告: 无法读取图像 {cam_idx}")
                continue
            
            # 创建点云
            pcd = self.create_point_cloud(color_image, depth_image)
            
            # 获取当前帧的Tracker位姿
            T_tracker = self.tracker_poses[i]
            
            # 计算相机在世界坐标系下的位姿
            T_cam = T_tracker @ np.linalg.inv(self.T_cam2ee)
            
            # 将点云转换到世界坐标系
            pcd.transform(T_cam)
            
            # 下采样点云以减小文件大小
            pcd = pcd.voxel_down_sample(voxel_size=0.01)  # 1cm的体素下采样
            
            # 实时显示点云（用于调试）
            if current_pcd is None:
                current_pcd = pcd
                debug_vis.add_geometry(current_pcd)
            else:
                current_pcd.points = pcd.points
                current_pcd.colors = pcd.colors
                debug_vis.update_geometry(current_pcd)
            
            debug_vis.poll_events()
            debug_vis.update_renderer()
            
            # 保存点云
            output_file = os.path.join(self.output_dir, f'cloud_{cam_idx:06d}.ply')
            o3d.io.write_point_cloud(output_file, pcd)
            
            # 显示进度
            if (i + 1) % 10 == 0:
                print(f"\r处理进度: {i+1}/{total_frames}", end='')
                
        print("\n点云生成完成!")
        print(f"输出目录: {self.output_dir}")
        
        # 关闭调试窗口
        debug_vis.destroy_window()

def main():
    # 直接指定数据目录和标定文件路径
    data_dir = 'MultimodalData/20250208_190448'  # 数据目录
    calibration_file = 'data3/20250121_100022\FinalTransforms\T_cam2EE_Method_0.npz'  # 标定文件
    
    try:
        generator = PointCloudGenerator(data_dir, calibration_file)
        generator.process_frames()
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()