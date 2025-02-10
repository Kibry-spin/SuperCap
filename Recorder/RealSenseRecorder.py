import os
import numpy as np
import pyrealsense2 as rs
import cv2
import keyboard
import time
from datetime import datetime
from Recorder.BaseRecorder import BaseRecorder

class RealSenseRecorder(BaseRecorder):
    def __init__(self):
        """初始化RealSense相机记录器"""
        super().__init__()
        self.pipeline = None
        self.config = None
        self.align = None
        self.timestamps = []  # 存储时间戳
        
        try:
            self._init_device()
            print("RealSense相机初始化成功")
            self.device_ready = True
        except Exception as ex:
            print(f"RealSense相机初始化失败: {ex}")
            raise ex

    def _init_device(self):
        """初始化RealSense相机"""
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 配置深度和彩色流
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # 开始流传输
        self.pipeline_profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        # 获取并保存相机内参
        self.color_intrinsics = self.pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def _check_device(self):
        """检查设备状态"""
        return self.pipeline is not None and self.device_ready

    def start_recording(self, save_dir=None):
        """开始记录"""
        if super().start_recording(save_dir):
            # 创建保存目录
            self.depth_dir = os.path.join(self.save_dir, "depth")
            self.rgb_dir = os.path.join(self.save_dir, "rgb")
            os.makedirs(self.depth_dir, exist_ok=True)
            os.makedirs(self.rgb_dir, exist_ok=True)
            
            # 清空时间戳数组
            self.timestamps = []
            
            # 保存相机内参
            self._save_camera_intrinsics()
            return True
        return False

    def record_frame(self):
        """记录一帧数据"""
        if not self.is_recording():
            return False

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return False

            with self._data_lock:
                # 获取当前时间戳
                current_time = time.perf_counter()
                self.timestamps.append(current_time)
                
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # 使用总帧数作为文件名
                frame_idx = self.total_frames if hasattr(self, 'total_frames') else self.frame_count
                cv2.imwrite(os.path.join(self.rgb_dir, f"color_{frame_idx:06d}.jpg"), color_image)
                cv2.imwrite(os.path.join(self.depth_dir, f"depth_{frame_idx:06d}.png"), depth_image)
                
                # 更新帧率
                self.update_fps()
                return True

        except Exception as e:
            print(f"记录RealSense帧时出错: {e}")
            return False

    def save_data(self):
        """保存记录的数据"""
        # 相机数据是实时保存的，这里保存时间戳和元数据
        if self.save_dir:
            # 保存时间戳数据
            timestamps_array = np.array(self.timestamps)
            np.save(os.path.join(self.save_dir, 'timestamp.npy'), timestamps_array)
            
            info = {
                'resolution': {'width': 640, 'height': 480},
                'fps': self.fps,
                'total_frames': self.total_frames if hasattr(self, 'total_frames') else self.frame_count,
                'format': {
                    'color': 'jpg',
                    'depth': 'png'
                },
                'timestamps': {
                    'format': 'seconds since start (time.perf_counter)',
                    'total': len(self.timestamps)
                }
            }
            
            info_path = os.path.join(self.save_dir, 'camera_info.txt')
            with open(info_path, 'w') as f:
                for key, value in info.items():
                    f.write(f"{key}: {value}\n")

    def cleanup(self):
        """清理资源"""
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        self.device_ready = False

    def _save_camera_intrinsics(self):
        """保存相机内参"""
        intrinsics_path = os.path.join(self.save_dir, "camera_intrinsics.txt")
        with open(intrinsics_path, 'w') as f:
            f.write(f"fx: {self.color_intrinsics.fx}\n")
            f.write(f"fy: {self.color_intrinsics.fy}\n")
            f.write(f"ppx: {self.color_intrinsics.ppx}\n")
            f.write(f"ppy: {self.color_intrinsics.ppy}\n")
            f.write(f"width: {self.color_intrinsics.width}\n")
            f.write(f"height: {self.color_intrinsics.height}\n")

    def run(self):
        print("Press SPACE to start/stop recording, 'Q' to quit")
        
        try:
            while self.running:
                if keyboard.is_pressed('space'):
                    if not self.is_recording():
                        self.start_recording()
                    else:
                        self.stop_recording()
                    time.sleep(0.2)  # 防止重复触发
                
                if keyboard.is_pressed('q'):
                    self.running = False
                    break

                if self.is_recording():
                    self.record_frame()
                else:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
        finally:
            if self.is_recording():
                self.stop_recording()
            self.cleanup()

def main():
    recorder = RealSenseRecorder()
    recorder.run()

if __name__ == "__main__":
    main() 