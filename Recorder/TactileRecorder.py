import time
import serial
import socket
import struct
import sys
import threading
import os
import numpy as np
from datetime import datetime
from Recorder.BaseRecorder import BaseRecorder

def parse_data(data):
    """解析传感器数据"""
    if len(data) < 39:
        return None

    channels = []
    for i in range(12):
        high_byte = data[3 + i * 3]
        low_byte = data[4 + i * 3]
        value = (high_byte << 8) + low_byte
        channels.append(value)
    return channels

def check_checksum(data):
    """检查数据校验和"""
    checksum = sum(data[:38]) & 0xFF
    return checksum == data[38]

class TactileRecorder(BaseRecorder):
    def __init__(self, left_port=None, right_port=None, baudrate=115200):
        """初始化触觉传感器记录器
        Args:
            left_port: 左手串口号
            right_port: 右手串口号
            baudrate: 波特率
        """
        super().__init__()
        self.ports = {
            'left': left_port,
            'right': right_port
        }
        self.baudrate = baudrate
        self.serial_ports = {'left': None, 'right': None}
        self.buffers = {'left': bytearray(), 'right': bytearray()}
        
        # 帧率统计
        self.frame_stats = {
            'left': {'count': 0, 'total': 0, 'fps': 0.0, 'last_time': time.perf_counter()},
            'right': {'count': 0, 'total': 0, 'fps': 0.0, 'last_time': time.perf_counter()}
        }
        
        # 初始化UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = ('localhost', 12345)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
        
        # 定义手的标识符
        self.identifiers = {
            'left': 0x02,  # 左手标识符
            'right': 0x03  # 右手标识符
        }
        
        # 数据存储
        self.tactile_data = {
            'left': {'timestamps': [], 'channels': []},
            'right': {'timestamps': [], 'channels': []}
        }
        
        # 添加错误帧统计
        self.error_stats = {
            'left': {'count': 0, 'last_report_time': time.perf_counter()},
            'right': {'count': 0, 'last_report_time': time.perf_counter()}
        }
        self.report_interval = 5.0  # 每5秒报告一次错误统计
        
        # 初始化设备
        try:
            self._init_device()
            print("触觉传感器初始化成功")
            self.device_ready = True
        except Exception as ex:
            print(f"触觉传感器初始化失败: {ex}")
            raise ex

    def _init_device(self):
        """初始化串口设备"""
        for side, port in self.ports.items():
            if port:
                try:
                    self.serial_ports[side] = serial.Serial(
                        port=port,
                        baudrate=self.baudrate,
                        timeout=0
                    )
                    print(f"{side}手触觉传感器({port})初始化成功")
                except Exception as e:
                    print(f"{side}手触觉传感器({port})初始化失败: {e}")
                    raise e

    def _check_device(self):
        """检查设备状态"""
        return any(ser is not None for ser in self.serial_ports.values()) and self.device_ready

    def start_recording(self, save_dir=None):
        """开始记录"""
        if super().start_recording(save_dir):
            with self._data_lock:
                self.tactile_data = {
                    'left': {'timestamps': [], 'channels': []},
                    'right': {'timestamps': [], 'channels': []}
                }
                # 重置帧率统计
                for side in self.frame_stats:
                    self.frame_stats[side] = {
                        'count': 0, 
                        'total': 0, 
                        'fps': 0.0,
                        'last_time': time.perf_counter()
                    }
            return True
        return False

    def _send_udp_data(self, side, channels):
        """发送UDP数据包
        Args:
            side: 'left' 或 'right'
            channels: 12个通道的数据列表
        """
        try:
            # 打包数据：标识符(1字节) + 12个通道数据(每个2字节)
            identifier = self.identifiers[side]
            data = struct.pack('<B12H', identifier, *channels)
            self.sock.sendto(data, self.server_address)
        except Exception as e:
            print(f"\nUDP数据发送错误({side}手): {e}")

    def _report_errors(self, side):
        """报告错误统计"""
        current_time = time.perf_counter()
        stats = self.error_stats[side]
        
        if stats['count'] > 0 and current_time - stats['last_report_time'] >= self.report_interval:
            print(f"\n{side}手传感器统计: {stats['count']} 个无效帧 "
                  f"(过去 {self.report_interval:.1f} 秒)")
            # 重置统计
            stats['count'] = 0
            stats['last_report_time'] = current_time

    def _update_frame_stats(self, side):
        """更新帧率统计"""
        stats = self.frame_stats[side]
        stats['count'] += 1
        stats['total'] += 1
        
        current_time = time.perf_counter()
        elapsed = current_time - stats['last_time']
        
        if elapsed >= 1.0:
            stats['fps'] = stats['count'] / elapsed
            stats['count'] = 0
            stats['last_time'] = current_time

    def record_frame(self):
        """记录一帧数据"""
        if not self.is_recording():
            return False

        frame_recorded = False
        try:
            # 处理每个串口的数据
            for side, ser in self.serial_ports.items():
                if ser is None:
                    continue
                    
                # 非阻塞读取串口数据
                if ser.in_waiting > 0:
                    self.buffers[side] += ser.read(ser.in_waiting)

                    # 处理缓冲区中的数据
                    while len(self.buffers[side]) >= 39:
                        frame = self.buffers[side][:39]
                        self.buffers[side] = self.buffers[side][39:]

                        # 检查帧头和校验和
                        if frame[0] == 0x40 and frame[1] == 0x5C and check_checksum(frame):
                            channels = parse_data(frame)
                            if channels:
                                timestamp = time.perf_counter()
                                
                                with self._data_lock:
                                    self.tactile_data[side]['timestamps'].append(timestamp)
                                    self.tactile_data[side]['channels'].append(channels)
                                    self._update_frame_stats(side)
                                
                                # 发送UDP数据
                                self._send_udp_data(side, channels)
                                
                                frame_recorded = True
                        else:
                            # 统计错误帧
                            self.error_stats[side]['count'] += 1
                            self.buffers[side] = self.buffers[side][1:]
                
                # 定期报告错误统计
                self._report_errors(side)

            if frame_recorded:
                self.update_fps()
                return True

        except Exception as e:
            print(f"\n触觉数据读取错误: {e}")
        return False

    def save_data(self):
        """保存记录的数据"""
        if not self.save_dir:
            return
            
        with self._data_lock:
            try:
                # 创建保存目录
                save_dir = os.path.join(self.save_dir, 'tactile_data')
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存每个手的数据
                for side in ['left', 'right']:
                    if not self.tactile_data[side]['timestamps']:
                        continue
                        
                    # 转换为numpy数组
                    timestamps = np.array(self.tactile_data[side]['timestamps'])
                    channels = np.array(self.tactile_data[side]['channels'])
                    
                    # 保存数据（使用带有手的标识的文件名）
                    np.save(os.path.join(save_dir, f'{side}_hand_timestamp.npy'), timestamps)
                    np.save(os.path.join(save_dir, f'{side}_hand_channels.npy'), channels)
                    
                    # 保存信息
                    info = {
                        'side': side,
                        'total_frames': len(timestamps),
                        'channels': 12,
                        'port': self.ports[side],
                        'baudrate': self.baudrate,
                        'data_format': {
                            'channels': 'Tactile data on 12 channels',
                            'timestamps': 'seconds since start (time.perf_counter)'
                        }
                    }
                    
                    info_path = os.path.join(save_dir, f'{side}_hand_info.txt')
                    with open(info_path, 'w') as f:
                        for key, value in info.items():
                            f.write(f"{key}: {value}\n")
                    
                    print(f"\n已保存{side}手触觉数据:")
                    print(f"- 时间戳: {len(timestamps)} 帧")
                    print(f"- 通道数据: {channels.shape}")
                    print(f"- 保存位置: {save_dir}")
                
            except Exception as e:
                print(f"保存数据时出错: {e}")

    def cleanup(self):
        """清理资源"""
        # 在清理前打印最终的错误统计
        for side in ['left', 'right']:
            if self.error_stats[side]['count'] > 0:
                print(f"\n{side}手传感器最终统计: 共 {self.error_stats[side]['count']} 个无效帧")
        
        # 关闭串口
        for side, ser in self.serial_ports.items():
            if ser:
                ser.close()
        self.serial_ports = {'left': None, 'right': None}
        
        # 关闭UDP socket
        if hasattr(self, 'sock'):
            self.sock.close()
            
        self.device_ready = False

    def get_stats(self):
        """获取当前统计信息"""
        stats = {}
        for side in ['left', 'right']:
            stats[f'{side}_fps'] = self.frame_stats[side]['fps']
            stats[f'{side}_total'] = self.frame_stats[side]['total']
        return stats

def main():
    # 创建触觉记录器（同时支持左右手）
    recorder = TactileRecorder(
        left_port='COM12',
        right_port='COM4'
    )
    recorder.run()

if __name__ == "__main__":
    main()
