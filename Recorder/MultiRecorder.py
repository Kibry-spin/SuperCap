import sys
import os
import time
import numpy as np
import keyboard
import pyrealsense2 as rs
import cv2
from datetime import datetime
import socket
import threading

# 添加Utils目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))
import triad_openvr

class MultiRecorder:
    def __init__(self, tracker_serials, glove_port=2211):
        """初始化多数据记录器
        Args:
            tracker_serials: 字典，包含三个Tracker的序列号
                {
                    'left_tracker': 'LHR-XXXXXX',
                    'right_tracker': 'LHR-XXXXXX',
                    'realsense_tracker': 'LHR-XXXXXX'
                }
            glove_port: 手套数据UDP端口号
        """
        self.recording = False
        self.running = True
        self.frame_count = 0
        self.last_fps_time = time.perf_counter()
        self.fps = 0.0
        self.save_dir = None
        
        # 初始化各个设备
        try:
            self._init_realsense()
            self._init_tracker(tracker_serials)
            self._init_glove(glove_port)
            print("所有设备初始化成功")
        except Exception as e:
            print(f"设备初始化失败: {e}")
            self.cleanup()
            raise e

    def _init_realsense(self):
        """初始化RealSense相机"""
        self.rs_pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.rs_profile = self.rs_pipeline.start(config)
        self.rs_align = rs.align(rs.stream.color)
        
        # 保存相机内参
        intrinsics = self.rs_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_intrinsics = {
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'ppx': intrinsics.ppx,
            'ppy': intrinsics.ppy,
            'width': intrinsics.width,
            'height': intrinsics.height
        }

    def _init_tracker(self, tracker_serials):
        """初始化Tracker"""
        self.tracker_serials = tracker_serials
        self.tracker_data = {
            'left_tracker': [],
            'right_tracker': [],
            'realsense_tracker': []
        }
        self.vr = triad_openvr.triad_openvr()
        self._verify_trackers()

    def _init_glove(self, port):
        """初始化手套数据接收"""
        self.glove_port = port
        self.glove_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.glove_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16384)
        server_address = ('localhost', port)
        self.glove_sock.bind(server_address)
        self.glove_sock.settimeout(0.001)
        self.glove_data = []

    def _verify_trackers(self):
        """验证所有指定的Tracker是否存在"""
        found_serials = []
        for device in self.vr.devices.values():
            serial = device.get_serial().decode('utf-8')
            found_serials.append(serial)
        
        for name, serial in self.tracker_serials.items():
            if serial not in found_serials:
                print(f"警告: 未找到{name} (序列号: {serial})")
        
        print("\n已找到的Tracker:")
        for name, serial in self.tracker_serials.items():
            if serial in found_serials:
                print(f"- {name}: {serial}")

    def start_recording(self):
        """开始记录数据"""
        if not self.recording:
            # 创建保存目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = os.path.join('recorded_data', timestamp)
            os.makedirs(os.path.join(self.save_dir, "depth"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "rgb"), exist_ok=True)
            
            # 保存相机内参
            with open(os.path.join(self.save_dir, "camera_intrinsics.txt"), 'w') as f:
                for key, value in self.camera_intrinsics.items():
                    f.write(f"{key} {value}\n")
            
            self.recording = True
            self.frame_count = 0
            self.last_fps_time = time.perf_counter()
            self.tracker_data = {name: [] for name in self.tracker_serials}
            self.glove_data = []
            print("\n开始记录数据...")

    def stop_recording(self):
        """停止记录数据"""
        if self.recording:
            self.recording = False
            print(f"\n停止记录")
            print(f"当前帧率: {self.fps:.1f} FPS")
            
            # 保存Tracker数据
            for name, data in self.tracker_data.items():
                if data:
                    filepath = os.path.join(self.save_dir, f"{name}.txt")
                    np.savetxt(filepath, data,
                              header='timestamp x y z roll pitch yaw',
                              fmt='%.6f')
                    print(f"已保存{name}数据，共 {len(data)} 帧")
            
            # 保存手套数据
            if self.glove_data:
                glove_path = os.path.join(self.save_dir, 'glove_data.txt')
                with open(glove_path, 'w', encoding='utf-8') as f:
                    for data_str in self.glove_data:
                        f.write(data_str + '\n')
                print(f"已保存手套数据，共 {len(self.glove_data)} 帧")

    def record_frame(self):
        """记录一帧数据"""
        if not self.recording:
            return

        try:
            current_time = time.perf_counter()
            frame_recorded = False
            
            # 记录RealSense数据
            frames = self.rs_pipeline.wait_for_frames()
            aligned_frames = self.rs_align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if depth_frame and color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                cv2.imwrite(os.path.join(self.save_dir, "rgb", f"color_{self.frame_count:04d}.png"), color_image)
                cv2.imwrite(os.path.join(self.save_dir, "depth", f"depth_{self.frame_count:04d}.png"), depth_image)
                frame_recorded = True
            
            # 记录Tracker数据
            for name, serial in self.tracker_serials.items():
                for device in self.vr.devices.values():
                    if device.get_serial().decode('utf-8') == serial:
                        [x, y, z, roll, pitch, yaw] = device.get_pose_euler()
                        frame_data = np.array([current_time, x, y, z, roll, pitch, yaw])
                        self.tracker_data[name].append(frame_data)
                        frame_recorded = True
                        break
            
            # 记录手套数据
            try:
                data, _ = self.glove_sock.recvfrom(4096)
                data_str = data.decode()
                if data_str.startswith("Glove1"):
                    self.glove_data.append(data_str)
                    frame_recorded = True
            except socket.timeout:
                pass
            
            # 更新帧率
            if frame_recorded:
                self.frame_count += 1
                elapsed_time = current_time - self.last_fps_time
                
                if elapsed_time >= 1.0:
                    self.fps = self.frame_count / elapsed_time
                    status = f"帧率: {self.fps:.1f} FPS | "
                    status += f"图像: {self.frame_count} | "
                    status += " | ".join([f"{name}: {len(data)}" 
                                        for name, data in self.tracker_data.items()])
                    status += f" | 手套: {len(self.glove_data)}"
                    print(f"\r{status}", end='', flush=True)
                    
                    self.frame_count = 0
                    self.last_fps_time = current_time

        except Exception as e:
            print(f"\n记录帧时出错: {e}")

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'rs_pipeline'):
            self.rs_pipeline.stop()
        if hasattr(self, 'glove_sock'):
            self.glove_sock.close()

    def run(self):
        """运行记录器"""
        print("按空格键开始/停止记录，按'Q'退出")
        
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
                
                time.sleep(0.001)  # 添加小延迟，避免CPU占用过高

        except KeyboardInterrupt:
            print("\n用户中断记录")
        finally:
            if self.recording:
                self.stop_recording()
            self.cleanup()

def main():
    # 指定三个Tracker的序列号
    tracker_serials = {
        'left_tracker': 'LHR-1CB8A619',      # 左手Tracker序列号
        'right_tracker': 'LHR-530B9203',      # 右手Tracker序列号
        'realsense_tracker': 'LHR-1CB8A619'   # RealSense Tracker序列号
    }
    
    recorder = MultiRecorder(tracker_serials,glove_port=2211)
    recorder.run()

if __name__ == '__main__':
    main() 