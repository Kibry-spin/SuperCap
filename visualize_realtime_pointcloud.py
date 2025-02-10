import os
import numpy as np
import pyrealsense2 as rs
import cv2
import time
from datetime import datetime
import keyboard
import triad_openvr
import open3d as o3d
import json
import threading
import queue

class RealSenseTrackerRecorder:
    def __init__(self, calibration_file):
        """初始化记录器
        Args:
            calibration_file: 手眼标定结果文件路径
        """
        self.recording = False
        self.running = True
        self.frame_count = 0
        self.last_fps_time = time.perf_counter()
        self.fps = 0.0
        
        # 加载手眼标定结果
        self.T_cam2ee = np.load(calibration_file)['arr_0']
        
        # 添加线程和队列相关的属性
        self.tracker_queue = queue.Queue(maxsize=1)  # 只保留最新的Tracker数据
        self.tracker_thread = None
        self.tracker_running = False
        
        try:
            # 初始化RealSense
            self._init_realsense()
            print("RealSense相机初始化成功")
            
            # 初始化Tracker
            self._init_tracker()
            print("Tracker初始化成功")
            
            # 初始化可视化器
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window("实时点云可视化")
            
            # 创建并添加世界坐标系
            self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            self.vis.add_geometry(self.world_frame)
            
            # 初始化点云对象（用于更新显示）
            self.current_pcd = None
            
            # 设置可视化参数
            render_option = self.vis.get_render_option()
            render_option.point_size = 2.0
            render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
            
            # 设置默认视角
            ctr = self.vis.get_view_control()
            ctr.set_zoom(0.3)
            ctr.set_front([0, 0, -1])   # 设置前方向
            ctr.set_lookat([0, 0, 0])   # 设置视点中心为原点
            ctr.set_up([0, -1, 0])      # 设置上方向
            
        except Exception as ex:
            print(f"初始化失败: {ex}")
            raise ex
            
    def _init_realsense(self):
        """初始化RealSense相机"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # 配置深度和彩色流
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 开始流传输
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        # 获取相机内参
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
    def _init_tracker(self):
        """初始化Tracker"""
        try:
            self.vr = triad_openvr.triad_openvr()
            print("Tracker初始化成功")
            self.vr.print_discovered_objects()
            self.tracker_running = True
            # 启动Tracker数据采集线程
            self.tracker_thread = threading.Thread(target=self._tracker_loop)
            self.tracker_thread.daemon = True
            self.tracker_thread.start()
        except Exception as ex:
            print(f"Tracker初始化失败: {ex}")
            raise ex
        
    def _tracker_loop(self):
        """Tracker数据采集循环"""
        while self.tracker_running:
            try:
                # 获取最新的Tracker数据
                T = np.eye(4)
                for deviceName in self.vr.devices:
                    if deviceName == 'tracker_1':
                        [x, y, z, roll, pitch, yaw] = self.vr.devices[deviceName].get_pose_euler()
                        
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
                        
                        R = Rz @ Ry @ Rx
                        T[:3, :3] = R
                        T[:3, 3] = [x, y, z]
                        break
                
                # 更新队列中的数据（如果队列满了会自动丢弃旧数据）
                try:
                    self.tracker_queue.put_nowait(T)
                except queue.Full:
                    try:
                        self.tracker_queue.get_nowait()  # 移除旧数据
                        self.tracker_queue.put_nowait(T)  # 放入新数据
                    except queue.Empty:
                        pass
                
            except Exception as e:
                print(f"\nTracker数据采集错误: {e}")
            
            # 短暂休眠以避免过度占用CPU
            time.sleep(0.001)
            
    def get_tracker_pose(self):
        """获取最新的Tracker位姿"""
        try:
            return self.tracker_queue.get_nowait()
        except queue.Empty:
            return np.eye(4)
        
    def start_recording(self):
        """开始记录"""
        if not self.recording:
            # 创建保存目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = os.path.join('realsense_tracker_data', f'session_{timestamp}')
            self.depth_dir = os.path.join(self.save_dir, "depth")
            self.rgb_dir = os.path.join(self.save_dir, "rgb")
            self.pose_dir = os.path.join(self.save_dir, "poses")
            self.cloud_dir = os.path.join(self.save_dir, "clouds")
            
            os.makedirs(self.depth_dir, exist_ok=True)
            os.makedirs(self.rgb_dir, exist_ok=True)
            os.makedirs(self.pose_dir, exist_ok=True)
            os.makedirs(self.cloud_dir, exist_ok=True)
            
            # 保存相机内参
            self._save_camera_intrinsics()
            
            self.recording = True
            self.frame_count = 0
            self.last_fps_time = time.perf_counter()
            print("\n开始记录")
            
    def _save_camera_intrinsics(self):
        """保存相机内参"""
        intrinsics_file = os.path.join(self.save_dir, "camera_intrinsics.json")
        intrinsics_data = {
            'fx': self.intrinsics.fx,
            'fy': self.intrinsics.fy,
            'ppx': self.intrinsics.ppx,
            'ppy': self.intrinsics.ppy,
            'width': self.intrinsics.width,
            'height': self.intrinsics.height
        }
        with open(intrinsics_file, 'w') as f:
            json.dump(intrinsics_data, f, indent=4)
            
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
        
        # 创建点云并进行下采样以提高性能
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, camera_intrinsic
        )
        pcd = pcd.voxel_down_sample(voxel_size=0.01)  # 1cm的体素下采样
        
        return pcd
        
    def record_frame(self):
        """实时可视化一帧数据"""
        try:
            # 获取RealSense数据
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return
                
            # 获取Tracker位姿
            T_tracker = self.get_tracker_pose()
            
            # 计算相机在世界坐标系下的位姿
            T_cam = T_tracker @ np.linalg.inv(self.T_cam2ee)
            
            # 打印位姿信息
            print("\n当前位姿:")
            print("Tracker位置: [{:.3f}, {:.3f}, {:.3f}]".format(
                T_tracker[0,3], T_tracker[1,3], T_tracker[2,3]))
            print("相机位置: [{:.3f}, {:.3f}, {:.3f}]".format(
                T_cam[0,3], T_cam[1,3], T_cam[2,3]))
            
            # 转换图像数据
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # 创建点云并转换到世界坐标系
            pcd = self.create_point_cloud(color_image, depth_image)
            pcd.transform(T_cam)  # 将点云转换到世界坐标系
            
            # 更新可视化
            if self.current_pcd is None:
                self.current_pcd = pcd
                self.vis.add_geometry(self.current_pcd)
            else:
                self.current_pcd.points = pcd.points
                self.current_pcd.colors = pcd.colors
                self.vis.update_geometry(self.current_pcd)
            
            # 更新显示
            self.vis.poll_events()
            self.vis.update_renderer()
            
            # 更新计数器和FPS
            self.frame_count += 1
            current_time = time.perf_counter()
            elapsed_time = current_time - self.last_fps_time
            
            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                print(f"FPS: {self.fps:.2f}")
                self.frame_count = 0
                self.last_fps_time = current_time
                
        except Exception as e:
            print(f"\n处理帧时出错: {e}")
            
    def stop_recording(self):
        """停止记录"""
        if self.recording:
            self.recording = False
            print(f"\n停止记录")
            print(f"数据保存在: {self.save_dir}")
            
    def cleanup(self):
        """清理资源"""
        # 停止Tracker线程
        self.tracker_running = False
        if self.tracker_thread:
            self.tracker_thread.join(timeout=1.0)
            
        # 停止RealSense
        if self.pipeline:
            self.pipeline.stop()
            
        # 关闭可视化窗口
        if self.vis:
            self.vis.destroy_window()
            
    def run(self):
        """运行可视化"""
        print("按Q退出")
        
        try:
            while self.running:
                if keyboard.is_pressed('q'):
                    self.running = False
                    break
                    
                self.record_frame()
                time.sleep(0.001)  # 小延迟避免CPU占用过高
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            self.cleanup()
            
def main():
    # 手眼标定结果文件路径
    # calibration_file = "Calibration_data/20250210_152255\FinalTransforms\T_cam2EE_Method_0.npz"
    calibration_file = "Calibration_data/20250210_163043\FinalTransforms\T_cam2EE_Method_0.npz"
    try:
        recorder = RealSenseTrackerRecorder(calibration_file)
        recorder.run()
    except Exception as e:
        print(f"程序运行出错: {e}")
        
if __name__ == "__main__":
    main() 