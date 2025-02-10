import socket
import time
import numpy as np
import os
from datetime import datetime
import json
from Recorder.BaseRecorder import BaseRecorder

class GloveRecorder(BaseRecorder):
    def __init__(self, port=2211):
        """初始化手套数据记录器"""
        super().__init__()
        self.port = port
        self.sock = None
        self.raw_data = []  # 存储原始数据字符串
        self.parsed_data = {
            'system_time': [],  # 系统时间戳
            'device_time': [],  # 设备时间戳
            'values': []
        }
        
        try:
            self._init_device()
            print(f"已连接到UDP服务器，端口 {self.port}")
            print("等待数据流...")
            self.device_ready = True
        except Exception as ex:
            print(f"UDP服务器初始化失败: {ex}")
            raise ex

    def _init_device(self):
        """初始化UDP接收器"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16384)
        server_address = ('localhost', self.port)
        self.sock.bind(server_address)
        self.sock.settimeout(0.001)  # 非阻塞模式

    def _check_device(self):
        """检查设备状态"""
        return self.sock is not None and self.device_ready

    def start_recording(self, save_dir=None):
        """开始记录数据"""
        if super().start_recording(save_dir):
            with self._data_lock:
                self.raw_data = []
                self.parsed_data = {
                    'system_time': [],  # 系统时间戳
                    'device_time': [],  # 设备时间戳
                    'values': []
                }
            return True
        return False

    def record_frame(self):
        """记录一帧数据"""
        if not self.is_recording():
            return False

        try:
            data, _ = self.sock.recvfrom(4096)
            data_str = data.decode()
            
            if data_str.startswith("Glove1"):
                # 获取系统时间戳
                current_time = time.perf_counter()
                
                parsed = self._parse_data(data_str)
                if parsed and len(parsed['values']) == 192:  # 修改为192个值
                    with self._data_lock:
                        self.raw_data.append(data_str)
                        self.parsed_data['system_time'].append(current_time)
                        self.parsed_data['device_time'].append(parsed['timestamp'])
                        self.parsed_data['values'].append(parsed['values'])
                    
                    # 更新帧率
                    self.update_fps()
                    return True
                    
        except socket.timeout:
            pass  # 忽略超时错误
        except UnicodeDecodeError as e:
            print(f"数据解码错误: {e}")
        except Exception as e:
            print(f"记录帧时出错: {e}")
        return False

    def save_data(self):
        """保存记录的数据"""
        with self._data_lock:
            if not self.parsed_data['system_time']:
                print("没有数据可保存")
                return

            try:
                # 创建保存目录
                save_dir = os.path.join(self.save_dir, 'glove_data')
                os.makedirs(save_dir, exist_ok=True)

                # 保存原始数据
                raw_path = os.path.join(save_dir, 'raw_data.txt')
                with open(raw_path, 'w', encoding='utf-8') as f:
                    for data_str in self.raw_data:
                        f.write(data_str + '\n')

                # 转换为numpy数组
                data = np.array(self.parsed_data['values'], dtype=np.float32)
                system_timestamps = np.array(self.parsed_data['system_time'])
                device_timestamps = np.array(self.parsed_data['device_time'])

                # 提取位置和旋转数据
                positions = []
                rotations = []
                
                for frame_data in data:
                    frame_positions = []
                    frame_rotations = []
                    
                    # 处理右手和左手
                    for hand_start in [0, 96]:  # 0是右手起始，96是左手起始
                        # 添加手腕数据
                        frame_positions.append(frame_data[hand_start:hand_start+3])  # 位置xyz
                        frame_rotations.append(frame_data[hand_start+3:hand_start+6])  # 旋转xyz
                        
                        # 处理15个手指关节
                        for i in range(15):  # 每只手15个手指关节
                            joint_start = hand_start + 6 + i * 6  # 跳过手腕数据(6)
                            frame_positions.append(frame_data[joint_start:joint_start+3])  # 位置xyz
                            frame_rotations.append(frame_data[joint_start+3:joint_start+6])  # 旋转xyz
                    
                    positions.append(frame_positions)
                    rotations.append(frame_rotations)
                
                # 转换为numpy数组
                positions = np.array(positions)  # shape: (frames, joints, 3)
                rotations = np.array(rotations)  # shape: (frames, joints, 3)

                # 保存数据
                np.save(os.path.join(save_dir, 'positions.npy'), positions)
                np.save(os.path.join(save_dir, 'rotations.npy'), rotations)
                np.save(os.path.join(save_dir, 'timestamp.npy'), system_timestamps)
                np.save(os.path.join(save_dir, 'device_time.npy'), device_timestamps)

                # 保存数据信息
                info = {
                    'total_frames': len(self.parsed_data['values']),
                    'start_time': self.parsed_data['system_time'][0],
                    'end_time': self.parsed_data['system_time'][-1],
                    'positions_shape': positions.shape,
                    'rotations_shape': rotations.shape,
                    'timestamps': {
                        'system_time': 'time.perf_counter() - 系统高精度时间戳 (timestamp.npy)',
                        'device_time': '设备发送的原始时间戳 (device_time.npy)'
                    },
                    'data_structure': {
                        'positions': '(frames, joints, 3) - xyz坐标',
                        'rotations': '(frames, joints, 3) - xyz欧拉角',
                        'joint_order': [
                            '右手: 手腕 + 15个手指关节',
                            '左手: 手腕 + 15个手指关节'
                        ]
                    }
                }
                
                info_path = os.path.join(save_dir, 'info.txt')
                with open(info_path, 'w', encoding='utf-8') as f:
                    for key, value in info.items():
                        f.write(f"{key}: {value}\n")

                print(f"\n已保存数据到: {save_dir}")
                print(f"总帧数: {len(self.parsed_data['values'])}")
                print(f"数据形状:")
                print(f"- 位置数据: {positions.shape}")
                print(f"- 旋转数据: {rotations.shape}")

            except Exception as e:
                print(f"保存数据时出错: {str(e)}")

    def cleanup(self):
        """清理资源"""
        if self.sock:
            self.sock.close()
            self.sock = None
        self.device_ready = False

    def _parse_data(self, data_str):
        """解析数据字符串
        数据格式：每只手16个关节，每个关节6个值(位置xyz和旋转xyz)
        总共: 16*6*2 = 192个值 (两只手)
        """
        try:
            # 解析时间戳
            time_start = data_str.find("time ") + 5
            time_end = data_str.find(" pos")
            timestamp = data_str[time_start:time_end]
            
            # 查找数据起始位置
            start_idx = data_str.find("subpackage 1/1,") + len("subpackage 1/1,")
            if start_idx == -1:
                return None
                
            # 提取数值部分
            values_str = data_str[start_idx:]
            values = [float(x) for x in values_str.strip().split(',') if x]
            
            # 确保数据完整性
            if len(values) < 192:  # 16*6*2 = 192 (两只手)
                print(f"数据不完整: 只有 {len(values)} 个值")
                return None
                
            # 分离左右手数据
            right_hand = values[:96]  # 前96个值是右手
            left_hand = values[96:192]  # 后96个值是左手
            
            return {
                'timestamp': timestamp,
                'values': values[:192]  # 只取前192个值
            }
                
        except Exception as e:
            print(f"数据解析错误: {e}")
            return None

def main():
    recorder = GloveRecorder()
    recorder.run()

if __name__ == "__main__":
    main()
