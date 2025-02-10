import sys
import os

# 添加Utils目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))
import triad_openvr
import time
import numpy as np
import math
from datetime import datetime
import keyboard

class TrackerRecorder:
    def __init__(self, tracker_serials):
        """初始化追踪器记录器
        Args:
            tracker_serials: 字典，包含三个Tracker的序列号
                {
                    'left_tracker': 'LHR-XXXXXX',
                    'right_tracker': 'LHR-XXXXXX',
                    'realsense_tracker': 'LHR-XXXXXX'
                }
        """
        self.tracker_serials = tracker_serials
        self.tracker_data = {
            'left_tracker': [],
            'right_tracker': [],
            'realsense_tracker': []
        }
        self.v = None
        self.recording = False
        self.frame_count = 0
        self.last_fps_time = time.perf_counter()
        self.last_frame_time = time.perf_counter()
        self.fps = 0.0
        self.running = True
        self.min_frame_interval = 0.01  # 限制最小帧间隔为10ms (约100fps)
        
        # 初始化VR系统
        try:
            self.v = triad_openvr.triad_openvr()
            print("已连接到VR系统")
            self.v.print_discovered_objects()
            
            # 验证所有指定的Tracker是否都存在
            self._verify_trackers()
            
        except Exception as ex:
            if (type(ex).__name__ == 'OpenVRError' and 
                ex.args[0] == 'VRInitError_Init_HmdNotFoundPresenceFailed (error number 126)'):
                print('无法找到Tracker')
                print('请检查:')
                print('1. SteamVR是否运行?')
                print('2. Vive Tracker是否开启并与SteamVR配对?')
                print('3. Lighthouse基站是否开启且Tracker在视野范围内?')
            raise ex

    def _verify_trackers(self):
        """验证所有指定的Tracker是否存在"""
        found_serials = []
        for device in self.v.devices.values():
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
            self.recording = True
            for name in self.tracker_data:
                self.tracker_data[name] = []
            self.frame_count = 0
            self.last_fps_time = time.perf_counter()
            self.fps = 0.0
            print("\n开始记录数据...")

    def stop_recording(self):
        """停止记录数据"""
        if self.recording:
            self.recording = False
            # 计算总体平均帧率
            if len(self.tracker_data['left_tracker']) > 0:
                total_time = self.tracker_data['left_tracker'][-1][0] - self.tracker_data['left_tracker'][0][0]
                avg_fps = len(self.tracker_data['left_tracker']) / total_time if total_time > 0 else 0
                print(f"\n停止记录")
                print(f"平均帧率: {avg_fps:.1f} FPS")
                for name in self.tracker_data:
                    print(f"{name} 记录帧数: {len(self.tracker_data[name])}")

    def save_data(self, filename=None):
        """保存记录的数据"""
        if not any(self.tracker_data.values()):
            print("没有数据可保存")
            return

        # 创建保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join('tracker_data', timestamp)
        os.makedirs(save_dir, exist_ok=True)

        # 保存每个Tracker的数据
        for name, data in self.tracker_data.items():
            if data:
                filepath = os.path.join(save_dir, f"{name}.txt")
                np.savetxt(filepath, data,
                          header='timestamp x y z roll pitch yaw',
                          fmt='%.6f')
                print(f"已保存{name}数据到 {filepath}")

        # 保存配置信息
        config = {
            'timestamp': timestamp,
            'tracker_serials': self.tracker_serials,
            'total_frames': {name: len(data) for name, data in self.tracker_data.items()},
            'average_fps': self.fps
        }
        
        config_path = os.path.join(save_dir, 'config.txt')
        with open(config_path, 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

    def record_frame(self):
        """记录一帧数据"""
        if not self.recording:
            return

        try:
            current_time = time.perf_counter()
            
            # 限制记录频率
            if current_time - self.last_frame_time < self.min_frame_interval:
                return
                
            frame_recorded = False
            
            # 记录每个Tracker的数据
            for name, serial in self.tracker_serials.items():
                for device in self.v.devices.values():
                    if device.get_serial().decode('utf-8') == serial:
                        [x, y, z, roll, pitch, yaw] = device.get_pose_euler()
                        frame_data = np.array([current_time, x, y, z, roll, pitch, yaw])
                        self.tracker_data[name].append(frame_data)
                        frame_recorded = True
                        break
            
            # 只有在成功记录数据时才更新帧率
            if frame_recorded:
                self.frame_count += 1
                self.last_frame_time = current_time
                
                # 更新帧率显示
                elapsed_time = current_time - self.last_fps_time
                if elapsed_time >= 1.0:  # 每秒更新一次帧率
                    self.fps = self.frame_count / elapsed_time
                    frames_info = " | ".join([f"{name}: {len(data)}" 
                                            for name, data in self.tracker_data.items()])
                    print(f"\r帧率: {self.fps:.1f} FPS | {frames_info}", end='', flush=True)
                    self.last_fps_time = current_time
                    self.frame_count = 0

        except Exception as e:
            print(f"\n记录帧时出错: {e}")

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
                        self.save_data()
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
                self.save_data()

def main():
    # 指定三个Tracker的序列号
    tracker_serials = {
        'left_tracker': 'LHR-2132E0A8',
        'right_tracker': 'LHR-530B9203',
        'realsense_tracker': 'LHR-1CB8A619'   # RealSense Tracker序列号
    }
    
    recorder = TrackerRecorder(tracker_serials)
    recorder.run()

if __name__ == '__main__':
    main()
