import os
import numpy as np
import pyrealsense2 as rs
import cv2
import keyboard
import time
from datetime import datetime

class RealSenseRecorder:
    def __init__(self):
        """初始化RealSense相机记录器"""
        self.pipeline = None
        self.pipeline_profile = None
        self.align = None
        self.recording = False
        self.running = True
        self.frame_count = 0  # FPS计算用的计数器
        self.save_count = 0   # 保存图片用的计数器
        self.last_fps_time = time.perf_counter()
        self.fps = 0.0
        
        try:
            self.configure_realsense()
            print("Connected to RealSense camera")
        except Exception as ex:
            print(f"Failed to initialize RealSense camera: {ex}")
            raise ex

    def configure_realsense(self):
        self.pipeline = rs.pipeline()
        config = rs.config()

        # 配置深度和彩色流  
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # 开始流传输
        self.pipeline_profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # 获取并保存相机内参
        intrinsics = self.pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.save_camera_intrinsics(intrinsics)

    def save_camera_intrinsics(self, intrinsics):
        os.makedirs('realsense_data', exist_ok=True)
        with open(os.path.join('realsense_data', "camera_intrinsics.txt"), 'w') as f:
            f.write(f"fx {intrinsics.fx}\n")
            f.write(f"fy {intrinsics.fy}\n")
            f.write(f"ppx {intrinsics.ppx}\n")
            f.write(f"ppy {intrinsics.ppy}\n")
            f.write(f"width {intrinsics.width}\n")
            f.write(f"height {intrinsics.height}\n")

    def start_recording(self):
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_session = os.path.join('realsense_data', f'session_{timestamp}')
            self.depth_dir = os.path.join(self.current_session, "depth")
            self.rgb_dir = os.path.join(self.current_session, "rgb")
            os.makedirs(self.depth_dir, exist_ok=True)
            os.makedirs(self.rgb_dir, exist_ok=True)
            
            self.recording = True
            self.frame_count = 0
            self.save_count = 0  # 重置保存计数器
            self.last_fps_time = time.perf_counter()
            print("\nStarted recording")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            print(f"\nStopped recording")
            print(f"Final FPS: {self.fps:.2f}")
            print(f"Total frames saved: {self.save_count}")

    def record_frame(self):
        if not self.recording:
            return

        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 使用save_count保存图片
            cv2.imwrite(os.path.join(self.rgb_dir, f"color_{self.save_count:04d}.png"), color_image)
            cv2.imwrite(os.path.join(self.depth_dir, f"depth_{self.save_count:04d}.png"), depth_image)
            self.save_count += 1  # 增加保存计数器

            # FPS计算
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.last_fps_time

            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                print(f"\rFPS: {self.fps:.2f} | Frames saved: {self.save_count}", end='', flush=True)
                self.frame_count = 0  # 只重置FPS计数器
                self.last_fps_time = current_time

        except Exception as e:
            print(f"Error recording frame: {e}")

    def run(self):
        print("Press SPACE to start/stop recording, 'Q' to quit")
        
        try:
            while self.running:
                if keyboard.is_pressed('space'):
                    if not self.recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
                    time.sleep(0.2)  # 防止重复触发
                
                if keyboard.is_pressed('q'):
                    self.running = False
                    break

                if self.recording:
                    self.record_frame()
                else:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
        finally:
            if self.recording:
                self.stop_recording()
            self.pipeline.stop()

def main():
    recorder = RealSenseRecorder()
    recorder.run()

if __name__ == "__main__":
    main() 