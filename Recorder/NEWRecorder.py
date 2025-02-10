import socket
import time
import numpy as np
import os
from datetime import datetime
import keyboard
import pickle
import json

class GloveRecorder:
    def __init__(self, port=2211):
        """初始化手套数据记录器"""
        self.port = port
        self.sock = None
        self.recording = False
        self.running = True
        self.raw_data = []  # 存储原始数据字符串
        self.parsed_data = {
            'timestamp': [],  # 时间戳
            'device': [],    # 设备名称
            'wrist_pos': [], # 手腕位置(3)
            'wrist_rot': [], # 手腕旋转(4)
            'right_hand': {  # 右手数据
                'wrist': [],      # 位置(3) + 旋转(4)
                'thumb': [],      # 各关节旋转(4×3)
                'index': [],      # 各关节旋转(4×3) 
                'middle': [],     # 各关节旋转(4×3)
                'ring': [],       # 各关节旋转(4×3)
                'pinky': []       # 各关节旋转(4×3)
            },
            'left_hand': {   # 左手数据
                'wrist': [],      # 位置(3) + 旋转(4)
                'thumb': [],      # 各关节旋转(4×3) 
                'index': [],      # 各关节旋转(4×3)
                'middle': [],     # 各关节旋转(4×3)
                'ring': [],       # 各关节旋转(4×3)
                'pinky': []       # 各关节旋转(4×3)
            },
            'raw_strings': [],  # 原始数据字符串
            'values': []        # 存储数值数据
        }
        self.frame_count = 0
        self.save_count = 0
        self.last_fps_time = time.perf_counter()
        self.fps = 0.0
        self.stable_fps = False
        self.stable_count = 0
        
        try:
            self.configure_socket()
            print(f"已连接到UDP服务器，端口 {self.port}")
            print("等待数据流...")
        except Exception as ex:
            print(f"UDP服务器初始化失败: {ex}")
            raise ex

    def configure_socket(self):
        """配置UDP接收器"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16384)
        server_address = ('localhost', self.port)
        self.sock.bind(server_address)
        self.sock.settimeout(0.001)  # 非阻塞模式

    def parse_glove_data(self, data_str):
        """解析手套数据字符串"""
        try:
            # 按逗号分割数据
            parts = data_str.split(',')
            if len(parts) < 135:  # 头部信息 + 134个数值
                print(f"数据长度不足: {len(parts)}")
                return None

            # 解析头部信息
            header = parts[0].split()
            if len(header) < 12:
                print(f"头部信息不完整: {header}")
                return None

            timestamp = f"{header[2]} {header[3]}"  # 2025-01-20 13:18:52.284
            
            # 提取所有数值数据 (从parts[1]开始，确保获取全部134个值)
            try:
                values = []
                for i in range(1, 135):  # 确保获取134个值
                    if i < len(parts):
                        values.append(float(parts[i]))
                    else:
                        print(f"数据不完整，缺少值: {i}")
                        return None
                
                if len(values) != 134:
                    print(f"数值数量不正确: {len(values)}/134")
                    return None
                    
            except ValueError as e:
                print(f"数值转换错误: {e}")
                return None

            return {
                'timestamp': timestamp,
                'values': values
            }
        except Exception as e:
            print(f"\n数据解析错误: {str(e)}")
            return None

    def start_recording(self):
        """开始记录数据"""
        if not self.recording:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs('glove_data', exist_ok=True)
            
            self.recording = True
            self.raw_data = []
            self.parsed_data = {
                'timestamp': [],
                'values': []
            }
            self.save_count = 0
            print("\n开始记录手套数据")

    def stop_recording(self):
        """停止记录数据"""
        if self.recording:
            self.recording = False
            print(f"\n停止记录")
            print(f"当前帧率: {self.fps:.2f}")
            print(f"总帧数: {self.save_count}")

    def save_data(self):
        """保存记录的数据"""
        if not self.parsed_data['timestamp']:
            print("没有数据可保存")
            return

        try:
            # 创建保存目录
            save_dir = os.path.join('glove_data', self.timestamp)
            os.makedirs(save_dir, exist_ok=True)

            # 直接将values列表转换为numpy数组，保持原始顺序
            data = np.array(self.parsed_data['values'], dtype=np.float32)
            timestamps = np.array(self.parsed_data['timestamp'])

            # 保存数据
            np.save(os.path.join(save_dir, 'glove_data.npy'), data)
            np.save(os.path.join(save_dir, 'timestamps.npy'), timestamps)

            # 保存原始数据
            raw_path = os.path.join(save_dir, 'raw_data.txt')
            with open(raw_path, 'w', encoding='utf-8') as f:
                for data_str in self.raw_data:
                    f.write(data_str + '\n')

            # 更新数据结构信息
            info = {
                'total_frames': self.save_count,
                'start_time': self.parsed_data['timestamp'][0],
                'end_time': self.parsed_data['timestamp'][-1],
                'data_structure': {
                    'format': '直接保存原始数据顺序，每帧134个值',
                    'order': [
                        '0-2: 右手腕位置 (x,y,z)',
                        '3-6: 右手腕旋转 (四元数)',
                        '7-133: 手指数据',
                    ]
                }
            }
            
            with open(os.path.join(save_dir, 'info.json'), 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=4, ensure_ascii=False)

            print(f"\n已保存 {self.save_count} 帧到: {save_dir}")
            print("保存的文件包括:")
            print("- glove_data.npy: 原始顺序数据 (float32 array, shape=(N, 134))")
            print("- timestamps.npy: 时间戳数据")
            print("- raw_data.txt: 原始数据")
            print("- info.json: 数据信息和结构说明")

        except Exception as e:
            print(f"保存数据时出错: {str(e)}")

    def check_data_stream(self):
        """检查数据流并计算帧率"""
        try:
            data, _ = self.sock.recvfrom(4096)
            data_str = data.decode()
            
            if data_str.startswith("Glove1"):
                self.frame_count += 1
                current_time = time.perf_counter()
                elapsed_time = current_time - self.last_fps_time
                
                if elapsed_time >= 1.0:
                    new_fps = self.frame_count / elapsed_time
                    
                    # 检查帧率是否稳定
                    if abs(new_fps - self.fps) < 2.0:
                        self.stable_count += 1
                    else:
                        self.stable_count = 0
                    
                    self.fps = new_fps
                    print(f"\r检测数据流... 帧率: {self.fps:.2f}", end='', flush=True)
                    
                    if self.stable_count >= 3 and not self.stable_fps:
                        self.stable_fps = True
                        print("\n数据流稳定。按空格键开始记录，按'Q'退出")
                    
                    self.frame_count = 0
                    self.last_fps_time = current_time

        except socket.timeout:
            pass
        except Exception as e:
            print(f"\n检查数据流时出错: {e}")

    def record_frame(self):
        """记录一帧数据"""
        if not self.recording:
            return

        try:
            data, _ = self.sock.recvfrom(4096)
            data_str = data.decode()
            
            if data_str.startswith("Glove1"):
                # 解析数据
                parsed = self.parse_glove_data(data_str)
                if parsed is not None:
                    # 存储解析后的数据
                    self.parsed_data['timestamp'].append(parsed['timestamp'])
                    self.parsed_data['values'].append(parsed['values'])
                    
                    # 存储原始数据
                    self.raw_data.append(data_str)
                    self.save_count += 1  # 总帧数计数
                
                # FPS计算相关
                self.frame_count += 1  # 每秒帧数计数
                current_time = time.perf_counter()
                elapsed_time = current_time - self.last_fps_time
                
                if elapsed_time >= 1.0:
                    self.fps = self.frame_count / elapsed_time
                    print(f"\r记录中... 帧率: {self.fps:.2f} | 总帧数: {self.save_count}", end='', flush=True)
                    self.frame_count = 0
                    self.last_fps_time = current_time

        except socket.timeout:
            pass
        except Exception as e:
            print(f"\n记录帧时出错: {e}")

    def run(self):
        """运行记录器"""
        try:
            # 等待数据流稳定
            while self.running and not self.stable_fps:
                self.check_data_stream()
                if keyboard.is_pressed('q'):
                    return
                time.sleep(0.001)

            # 主循环
            while self.running:
                if keyboard.is_pressed('space'):
                    if not self.recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
                        self.save_data()
                    time.sleep(0.2)
                    continue
                
                if keyboard.is_pressed('q'):
                    self.running = False
                    break

                if self.recording:
                    self.record_frame()
                else:
                    self.check_data_stream()
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n用户中断记录")
        finally:
            if self.recording:
                self.stop_recording()
                self.save_data()
            self.sock.close()

def main():
    recorder = GloveRecorder()
    recorder.run()

if __name__ == "__main__":
    main()