import socket
import time
import os
import keyboard
import argparse
from datetime import datetime

class GloveDataSender:
    def __init__(self, data_file, target_port=2211, target_ip='127.0.0.1', replay_speed=1.0):
        """初始化数据发送器
        Args:
            data_file: 数据文件路径
            target_port: 目标端口号
            target_ip: 目标IP地址
            replay_speed: 回放速度倍率（1.0为原速）
        """
        self.data_file = data_file
        self.target_port = target_port
        self.target_ip = target_ip
        self.replay_speed = replay_speed
        self.running = True
        self.sock = None
        self.frame_count = 0
        self.last_fps_time = time.perf_counter()
        self.fps = 0.0
        
        # 初始化UDP发送器
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"UDP发送器初始化成功")
        except Exception as ex:
            print(f"UDP发送器初始化失败: {ex}")
            raise ex
            
        # 加载数据
        self.load_data()

    def load_data(self):
        """加载数据文件"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.data = f.readlines()
            print(f"已加载 {len(self.data)} 帧数据")
            
            # 计算原始帧率
            if len(self.data) >= 2:
                # 从数据中提取时间戳
                first_time = self.extract_timestamp(self.data[0])
                last_time = self.extract_timestamp(self.data[-1])
                if first_time and last_time:
                    duration = last_time - first_time
                    self.original_fps = len(self.data) / duration
                    print(f"原始数据帧率: {self.original_fps:.1f} FPS")
                    print(f"目标回放帧率: {self.original_fps * self.replay_speed:.1f} FPS")
        except Exception as e:
            print(f"加载数据文件失败: {e}")
            raise e

    def extract_timestamp(self, data_str):
        """从数据字符串中提取时间戳"""
        try:
            time_start = data_str.find("time ") + 5
            time_end = data_str.find(" pos")
            if time_start > 4 and time_end > time_start:
                return float(data_str[time_start:time_end])
        except:
            pass
        return None

    def send_frame(self, frame_data):
        """发送一帧数据"""
        try:
            self.sock.sendto(frame_data.encode(), (self.target_ip, self.target_port))
            return True
        except Exception as e:
            print(f"发送数据失败: {e}")
            return False

    def update_fps(self):
        """更新并显示当前帧率"""
        self.frame_count += 1
        current_time = time.perf_counter()
        elapsed_time = current_time - self.last_fps_time
        
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            print(f"\r回放中... 帧率: {self.fps:.1f} FPS | "
                  f"已发送: {self.frame_count} 帧", end='', flush=True)
            self.frame_count = 0
            self.last_fps_time = current_time

    def run(self):
        """运行发送循环"""
        print("\n开始发送数据...")
        print("按 'Q' 退出，按空格键 暂停/继续")
        
        paused = False
        frame_interval = 1.0 / (self.original_fps * self.replay_speed)
        last_frame_time = time.perf_counter()
        
        try:
            while self.running:
                current_time = time.perf_counter()
                
                # 检查按键
                if keyboard.is_pressed('q'):
                    print("\n用户退出")
                    break
                    
                if keyboard.is_pressed('space'):
                    paused = not paused
                    if paused:
                        print("\n已暂停")
                    else:
                        print("\n继续回放")
                    time.sleep(0.2)  # 防止重复触发
                    
                if paused:
                    time.sleep(0.1)
                    continue
                
                # 控制发送速率
                if current_time - last_frame_time >= frame_interval:
                    # 发送当前帧
                    frame_data = self.data[self.frame_count % len(self.data)]
                    if self.send_frame(frame_data):
                        self.update_fps()
                        last_frame_time = current_time
                    
                # 小延迟，避免CPU占用过高
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n程序被中断")
        finally:
            if self.sock:
                self.sock.close()
            print("\n发送结束")

def main():
    parser = argparse.ArgumentParser(description='手套数据回放发送工具')
    parser.add_argument('data_file', help='数据文件路径')
    parser.add_argument('--port', type=int, default=2211, help='目标端口号')
    parser.add_argument('--ip', default='127.0.0.1', help='目标IP地址')
    parser.add_argument('--speed', type=float, default=1.0, help='回放速度倍率')
    
    args = parser.parse_args()
    
    try:
        sender = GloveDataSender(
            data_file=args.data_file,
            target_port=args.port,
            target_ip=args.ip,
            replay_speed=args.speed
        )
        sender.run()
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main() 