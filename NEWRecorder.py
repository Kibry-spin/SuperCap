import numpy as np
import time
import os
import json
from datetime import datetime

class GloveRecorder:
    def __init__(self):
        self.data = []  # 存储所有帧的数据
        self.timestamps = []  # 存储时间戳
        self.is_recording = False
        self.parsed_data = {'timestamp': None, 'values': []}
        self.last_print_time = time.time()
        
    def start_recording(self, save_dir=None):
        """开始记录数据"""
        if save_dir is None:
            # 使用当前时间创建文件夹名
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join("glove_data", current_time)
        
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.data = []
        self.timestamps = []
        self.is_recording = True
        print(f"开始记录数据，保存目录: {save_dir}")
        
    def stop_recording(self):
        """停止记录并保存数据"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if len(self.data) == 0:
            print("没有记录到数据")
            return
            
        # 转换为numpy数组
        data_array = np.array(self.data, dtype=np.float32)
        timestamps_array = np.array(self.timestamps)
        
        # 保存数据
        np.save(os.path.join(self.save_dir, 'glove_data.npy'), data_array)
        np.save(os.path.join(self.save_dir, 'timestamps.npy'), timestamps_array)
        
        # 保存信息
        info = {
            'total_frames': len(self.data),
            'start_time': self.timestamps[0],
            'end_time': self.timestamps[-1],
            'duration': self.timestamps[-1] - self.timestamps[0]
        }
        with open(os.path.join(self.save_dir, 'info.json'), 'w') as f:
            json.dump(info, f, indent=4)
            
        print(f"数据记录完成:")
        print(f"- 总帧数: {len(self.data)}")
        print(f"- 数据形状: {data_array.shape}")
        print(f"- 保存目录: {self.save_dir}")
        print("保存文件:")
        print("- glove_data.npy: 手套数据 (float32 array, shape=(N, 201))")
        print("- timestamps.npy: 时间戳数据")
        print("- info.json: 记录信息")
        
    def parse_data(self, data_str):
        """解析数据字符串
        数据格式: "Glove1 time {timestamp} pos quat relative gesture -1 -1 fn {frame_num} subpackage 1/1,{values}"
        """
        try:
            # 解析时间戳
            time_start = data_str.find("time ") + 5
            time_end = data_str.find(" pos")
            timestamp = data_str[time_start:time_end]
            
            # 查找数据起始位置 "0.15,0,0.2,"
            start_idx = data_str.find("0.15,0,0.2,")
            if start_idx == -1:
                return False
                
            # 提取数值部分
            values_str = data_str[start_idx:]
            values = [float(x) for x in values_str.strip().split(',')]
            
            # 更新解析后的数据
            self.parsed_data['timestamp'] = timestamp
            self.parsed_data['values'] = values
            return True
            
        except Exception as e:
            print(f"数据解析错误: {e}")
            return False
            
    def update(self):
        """更新数据"""
        current_time = time.time()
        
        # 每秒打印一次数据
        if current_time - self.last_print_time >= 1.0:  # 每秒打印一次
            if len(self.parsed_data['values']) > 0:
                self.print_frame_data()
            self.last_print_time = current_time
        
        # 记录数据
        if not self.is_recording:
            return
            
        if len(self.parsed_data['values']) > 0:
            self.data.append(self.parsed_data['values'])
            self.timestamps.append(self.parsed_data['timestamp'])
            
    def print_frame_data(self):
        """打印当前帧数据"""
        values = self.parsed_data['values']
        timestamp = self.parsed_data['timestamp']
        
        print(f"\n时间戳: {timestamp}")
        
        # 右手数据 (0-66)
        print("右手:")
        print(f"  手腕位置: [{values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f}]")
        print(f"  手腕旋转: [{values[3]:.3f}, {values[4]:.3f}, {values[5]:.3f}, {values[6]:.3f}]")
        
        # 左手数据 (从134开始)
        left_start = 134
        print("左手:")
        print(f"  手腕位置: [{values[left_start]:.3f}, {values[left_start+1]:.3f}, {values[left_start+2]:.3f}]")
        print(f"  手腕旋转: [{values[left_start+3]:.3f}, {values[left_start+4]:.3f}, {values[left_start+5]:.3f}, {values[left_start+6]:.3f}]")

def print_frame_data(frame_data):
    """以易读的格式打印帧数据"""
    values = frame_data['values']
    timestamp = frame_data['timestamp']
    
    print(f"\n时间戳: {timestamp}")
    print("右手:")
    print(f"  手腕位置 (x,y,z): [{values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f}]")
    print(f"  手腕旋转 (四元数): [{values[3]:.3f}, {values[4]:.3f}, {values[5]:.3f}, {values[6]:.3f}]")
    
    finger_names = ['拇指', '食指', '中指', '无名指', '小指']
    for i, name in enumerate(finger_names):
        base_idx = 7 + i * 12
        print(f"  {name}:")
        for j in range(3):
            idx = base_idx + j * 4
            print(f"    关节{j+1} (四元数): [{values[idx]:.3f}, {values[idx+1]:.3f}, {values[idx+2]:.3f}, {values[idx+3]:.3f}]")
    
    # 左手数据 (从134开始)
    left_start = 134
    print("\n左手:")
    print(f"  手腕位置 (x,y,z): [{values[left_start]:.3f}, {values[left_start+1]:.3f}, {values[left_start+2]:.3f}]")
    print(f"  手腕旋转 (四元数): [{values[left_start+3]:.3f}, {values[left_start+4]:.3f}, {values[left_start+5]:.3f}, {values[left_start+6]:.3f}]")
    
    for i, name in enumerate(finger_names):
        base_idx = left_start + 7 + i * 12
        print(f"  {name}:")
        for j in range(3):
            idx = base_idx + j * 4
            print(f"    关节{j+1} (四元数): [{values[idx]:.3f}, {values[idx+1]:.3f}, {values[idx+2]:.3f}, {values[idx+3]:.3f}]")

def update(self):
    """更新数据"""
    current_time = time.time()
    
    # 每秒打印一次数据
    if hasattr(self, 'last_print_time'):
        if current_time - self.last_print_time >= 1.0:  # 每秒打印一次
            if len(self.parsed_data['values']) > 0:
                print_frame_data(self.parsed_data)
            self.last_print_time = current_time
    else:
        self.last_print_time = current_time
    
    # 原有的更新逻辑
    if not self.is_recording:
        return
        
    if len(self.parsed_data['values']) > 0:
        self.data.append(self.parsed_data['values'])
        self.timestamps.append(self.parsed_data['timestamp']) 