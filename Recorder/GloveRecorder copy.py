import socket
import time
import numpy as np
import os
from datetime import datetime
import keyboard
import json

class GloveRecorder:
    def __init__(self, port=2211):
        """初始化手套数据记录器"""
        self.port = port
        self.sock = None
        self.recording = False
        self.running = True
        self.data = []  # 存储原始数据字符串
        self.frame_count = 0
        self.save_count = 0
        self.last_fps_time = time.perf_counter()
        self.fps = 0.0
        self.stable_fps = False
        self.stable_count = 0
        
        try:
            self.configure_socket()
            print(f"Connected to UDP server on port {self.port}")
            print("Waiting for data stream...")
        except Exception as ex:
            print(f"Failed to initialize UDP server: {ex}")
            raise ex

    def configure_socket(self):
        """配置UDP接收器"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16384)
        server_address = ('localhost', self.port)
        self.sock.bind(server_address)
        self.sock.settimeout(0.001)  # 非阻塞模式

    def start_recording(self):
        """开始记录数据"""
        if not self.recording:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs('glove_data', exist_ok=True)
            
            self.recording = True
            self.data = []
            self.save_count = 0
            print("\nStarted recording glove data")

    def stop_recording(self):
        """停止记录数据"""
        if self.recording:
            self.recording = False
            print(f"\nStopped recording")
            print(f"Current FPS: {self.fps:.2f}")
            print(f"Total frames saved: {self.save_count}")

    def save_data(self):
        """保存记录的数据"""
        if not self.data:
            print("No data to save")
            return

        # 保存原始数据
        raw_path = os.path.join('glove_data', f"raw_glove_data_{self.timestamp}.txt")
        with open(raw_path, 'w', encoding='utf-8') as f:
            for data_str in self.data:
                f.write(data_str + '\n')

        print(f"Saved {self.save_count} frames to {raw_path}")
 
    def check_data_stream(self):
        """检查数据流并计算帧率"""
        try:
            data, _ = self.sock.recvfrom(4096)
            data_str = data.decode()
            
            if data_str.startswith("Glove1"):
                print(f"\rReceived data: {data_str[:50]}...", end='', flush=True)  # 添加调试信息
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
                    print(f"\rDetecting data stream... FPS: {self.fps:.2f}", end='', flush=True)
                    
                    if self.stable_count >= 3 and not self.stable_fps:
                        self.stable_fps = True
                        print("\nData stream is stable. Press SPACE to start recording, 'Q' to quit")
                    
                    self.frame_count = 0
                    self.last_fps_time = current_time

        except socket.timeout:
            pass
        except Exception as e:
            print(f"\nError checking data stream: {e}")

    def record_frame(self):
        """记录一帧数据"""
        if not self.recording:
            return

        try:
            data, _ = self.sock.recvfrom(4096)
            data_str = data.decode()
            
            if data_str.startswith("Glove1"):
                print(f"\rRecording data: {data_str[:50]}...", end='', flush=True)  # 添加调试信息
                self.data.append(data_str)
                self.save_count += 1
                self.frame_count += 1
                
                current_time = time.perf_counter()
                elapsed_time = current_time - self.last_fps_time
                
                if elapsed_time >= 1.0:
                    self.fps = self.frame_count / elapsed_time
                    print(f"\rRecording... FPS: {self.fps:.2f} | Frames: {self.save_count}", end='', flush=True)
                    self.frame_count = 0
                    self.last_fps_time = current_time

        except socket.timeout:
            pass
        except Exception as e:
            print(f"\nError recording frame: {e}")

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
                    continue  # 添加continue，避免在切换状态时立即执行下面的代码
                
                if keyboard.is_pressed('q'):
                    self.running = False
                    break

                if self.recording:
                    self.record_frame()
                else:
                    self.check_data_stream()
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
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
