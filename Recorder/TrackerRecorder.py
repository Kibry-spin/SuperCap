import sys
import os
import numpy as np
import time
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))
import triad_openvr
from Recorder.BaseRecorder import BaseRecorder
# from BaseRecorder import BaseRecorder
class TrackerLostError(Exception):
    """当Tracker丢失追踪时抛出的异常"""
    pass

class TrackerRecorder(BaseRecorder):

    def __init__(self, tracker_name, tracker_serial):
        """初始化追踪器记录器
        Args:
            tracker_name: Tracker的名称（例如：'left_tracker'）
            tracker_serial: Tracker的序列号（例如：'LHR-XXXXXX'）
        """
        super().__init__()
        self.tracker_name = tracker_name
        self.tracker_serial = tracker_serial
        self.tracker_data = {
            'timestamps': [],  # 时间戳列表
            'poses': [],      # 位姿列表
            'lost_frames': 0  # 丢失追踪的帧数
        }
        self.vr = None
        self.min_frame_interval = 0.008  # 限制最小帧间隔为8ms (约120fps)
        self.last_frame_time = time.perf_counter()
        self.device_ready = False
        self.total_frames = 0  # 总帧数计数器
        
        # 初始化设备
        self._init_device()

    def _check_device(self):
        """检查设备状态"""
        return self.vr is not None and self.device_ready

    def _verify_tracker(self):
        """验证指定的Tracker是否存在"""
        found = False
        for device in self.vr.devices.values():
            serial = device.get_serial().decode('utf-8')
            if serial == self.tracker_serial:
                found = True
                print(f"找到 {self.tracker_name}: {self.tracker_serial}")
                break
        
        if not found:
            print(f"警告: 未找到 {self.tracker_name} (序列号: {self.tracker_serial})")
            
        return found

    def start_recording(self, save_dir=None):
        """开始记录"""
        if super().start_recording(save_dir):
            with self._data_lock:
                self.tracker_data = {
                    'timestamps': [],
                    'poses': [],
                    'lost_frames': 0
                }
                self.total_frames = 0  # 重置总帧数
            return True
        return False

    def _check_pose_valid(self, pose_data):
        """检查位姿数据是否有效
        Args:
            pose_data: [x, y, z, roll, pitch, yaw]数组
        Returns:
            bool: 位姿是否有效
        """
        # 检查位置数据是否全为0或接近0
        position = pose_data[:3]
        if np.allclose(position, 0, atol=1e-6):
            return False
            
        # 检查旋转数据是否合理
        rotation = pose_data[3:]
        if np.any(np.abs(rotation) > 360):  # 角度不应超过360度
            return False
            
        return True

    def stop_recording(self, force_stop=False):
        """停止记录
        Args:
            force_stop: 是否强制停止（不保存数据）
        """
        return super().stop_recording(force_stop)

    def record_frame(self):
        """记录一帧数据"""
        if not self.recording:
            return False

        try:
            current_time = time.perf_counter()
            
            # 限制记录频率
            if current_time - self.last_frame_time < self.min_frame_interval:
                time.sleep(0.001)  # 减小延迟到1ms
                return False
            
            frame_recorded = False
            
            # 记录Tracker数据
            for device in self.vr.devices.values():
                if device.get_serial().decode('utf-8') == self.tracker_serial:
                    # 获取位姿数据
                    pose_data = device.get_pose_euler()
                    if pose_data is None:
                        continue
                        
                    [x, y, z, roll, pitch, yaw] = pose_data
                    frame_data = np.array([x, y, z, roll, pitch, yaw])
                    
                    # 检查数据有效性
                    if not np.any(np.isnan(frame_data)) and not np.allclose(frame_data[:3], 0, atol=1e-5):
                        with self._data_lock:
                            self.tracker_data['timestamps'].append(current_time)
                            self.tracker_data['poses'].append(frame_data)
                            frame_recorded = True
                    else:
                        self.tracker_data['lost_frames'] += 1
                        print(f"\r{self.tracker_name} 追踪丢失...", end='', flush=True)
                    break
            
            # 更新时间戳和帧率
            if frame_recorded:
                self.last_frame_time = current_time
                self.update_fps()
                return True

        except Exception as e:
            print(f"\n记录Tracker帧时出错: {e}")
        
        return False

    def save_data(self):
        """保存记录的数据"""
        if not self.tracker_data['timestamps']:
            print(f"{self.tracker_name}: 没有数据可保存")
            return
            
        try:
            # 创建保存目录
            save_dir = os.path.join(self.save_dir, 'tracker_data')
            os.makedirs(save_dir, exist_ok=True)
            
            # 转换为numpy数组
            timestamps = np.array(self.tracker_data['timestamps'])
            poses = np.array(self.tracker_data['poses'])
            
            # 保存数据
            np.save(os.path.join(save_dir, f'{self.tracker_name}_poses.npy'), poses)
            np.save(os.path.join(save_dir, f'{self.tracker_name}_timestamp.npy'), timestamps)
            
            # 保存信息
            info = {
                'tracker_name': self.tracker_name,
                'tracker_serial': self.tracker_serial,
                'total_frames': len(timestamps),
                'average_fps': self.fps
            }
            
            info_path = os.path.join(save_dir, f'{self.tracker_name}_info.txt')
            with open(info_path, 'w') as f:
                for key, value in info.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"\n已保存{self.tracker_name}数据到: {save_dir}")
            print(f"总帧数: {len(timestamps)}")
            
        except Exception as e:
            print(f"保存{self.tracker_name}数据时出错: {e}")

    def cleanup(self):
        """清理资源"""
        try:
            if self.recording:
                self.stop_recording(force_stop=True)
            if self.vr is not None:
                # 确保所有设备资源被释放
                for device in self.vr.devices.values():
                    try:
                        device.close()
                    except:
                        pass
                self.vr = None
            self.device_ready = False
            time.sleep(0.1)  # 给系统一些时间来清理资源
        except Exception as e:
            print(f"清理 {self.tracker_name} 资源时出错: {e}")

    def _init_device(self):
        """初始化VR系统"""
        try:
            self.vr = triad_openvr.triad_openvr()
            print("已连接到VR系统")
            # self.vr.print_discovered_objects()
            
            # 验证Tracker是否存在
            if self._verify_tracker():
                self.device_ready = True
            
        except Exception as ex:
            if (type(ex).__name__ == 'OpenVRError' and 
                ex.args[0] == 'VRInitError_Init_HmdNotFoundPresenceFailed (error number 126)'):
                print('无法找到Tracker')
                print('请检查:')
                print('1. SteamVR是否运行?')
                print('2. Vive Tracker是否开启并与SteamVR配对?')
                print('3. Lighthouse基站是否开启且Tracker在视野范围内?')
            print(f"VR系统初始化失败: {ex}")
            raise ex

def main():
    # 配置参数
    a=1

if __name__ == '__main__':
    main()
