import os
import numpy as np
import pyrealsense2 as rs
import cv2
import keyboard
import time
from datetime import datetime
import triad_openvr
import json

class RealSenseTracker:
    def __init__(self):
        """初始化RealSense相机和Vive Tracker"""
        self.init_realsense()
        self.init_vive_tracker()
        self.recording = False
        self.running = True
        self.frame_count = 0
        self.save_count = 0
        self.last_fps_time = time.perf_counter()
        self.fps = 0.0

    def init_realsense(self):
        """初始化RealSense相机"""
        try:
            self.rs_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline_profile = self.rs_pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            print("Connected to RealSense camera")
        except Exception as ex:
            print(f"Failed to initialize RealSense camera: {ex}")
            raise ex

    def init_vive_tracker(self):
        """初始化Vive Tracker"""
        try:
            self.vr = triad_openvr.triad_openvr()
            print("Connected to Vive Tracker")
            self.vr.print_discovered_objects()
        except Exception as ex:
            print(f"Failed to initialize Vive Tracker: {ex}")
            raise ex

    def get_tracker_pose(self):
        """获取Tracker的6D位姿"""
        for deviceName in self.vr.devices:
            if deviceName == 'tracker_1':  # 假设使用tracker_1
                return self.vr.devices[deviceName].get_pose_euler()
        return None

    def start_recording(self):
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_session = os.path.join('camera_tracker_data', f'session_{timestamp}')
            self.depth_dir = os.path.join(self.current_session, "depth")
            self.rgb_dir = os.path.join(self.current_session, "rgb")
            self.pose_dir = os.path.join(self.current_session, "pose")
            
            os.makedirs(self.depth_dir, exist_ok=True)
            os.makedirs(self.rgb_dir, exist_ok=True)
            os.makedirs(self.pose_dir, exist_ok=True)
            
            self.recording = True
            self.save_count = 0
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
            # 获取RealSense数据
            frames = self.rs_pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return

            # 获取Tracker位姿
            tracker_pose = self.get_tracker_pose()
            if tracker_pose is None:
                print("Warning: Cannot get tracker pose")
                return

            # 保存数据
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 保存图像
            cv2.imwrite(os.path.join(self.rgb_dir, f"color_{self.save_count:04d}.png"), color_image)
            cv2.imwrite(os.path.join(self.depth_dir, f"depth_{self.save_count:04d}.png"), depth_image)

            # 保存位姿数据
            pose_data = {
                'timestamp': time.time(),
                'frame_id': self.save_count,
                'tracker_pose': {
                    'position': {'x': tracker_pose[0], 'y': tracker_pose[1], 'z': tracker_pose[2]},
                    'rotation': {'roll': tracker_pose[3], 'pitch': tracker_pose[4], 'yaw': tracker_pose[5]}
                }
            }
            
            with open(os.path.join(self.pose_dir, f"pose_{self.save_count:04d}.json"), 'w') as f:
                json.dump(pose_data, f, indent=4)

            self.save_count += 1
            
            # 计算FPS
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.last_fps_time
            
            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                print(f"\rFPS: {self.fps:.2f} | Frames saved: {self.save_count}", end='', flush=True)
                self.frame_count = 0
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
                    time.sleep(0.2)
                
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
            self.rs_pipeline.stop()

def main():
    recorder = RealSenseTracker()
    recorder.run()

if __name__ == "__main__":
    main() 