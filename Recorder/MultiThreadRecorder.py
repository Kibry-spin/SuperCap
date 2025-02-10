import os
import time
import threading
import keyboard
from datetime import datetime
from Recorder.RealSenseRecorder import RealSenseRecorder
from Recorder.TrackerRecorder import TrackerRecorder
from Recorder.GloveRecorder import GloveRecorder
from Recorder.TactileRecorder import TactileRecorder

class MultiThreadRecorder:
    """多线程记录器，用于同时记录多个设备的数据"""
    
    def __init__(self, tracker_serials, serial_ports=None, glove_port=2211, folder_name="test"):
        """
        初始化多线程记录器
        
        Args:
            tracker_serials (dict): Tracker序列号字典，格式为 {'tracker_name': 'serial_number'}
            serial_ports (dict): 串口配置字典，格式为 {'left_tactile': 'COM12', 'right_tactile': 'COM4'}
            glove_port (int): 数据手套的UDP端口号
            folder_name (str): 保存数据的文件夹名称前缀，默认为"test"
        """
        self.recording = False
        self.save_dir = None
        self.folder_name = folder_name  # 使用传入的folder_name
        self.recorders = {}
        self.threads = {}
        self.total_frames = {}
        self.current_fps = {}
        self.frame_counts = {}
        self.last_time = {}  # 添加时间戳记录
        
        # 创建事件对象
        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self.exit_event = threading.Event()
        
        # 创建锁
        self.stats_lock = threading.Lock()
        
        # 初始化记录器
        self._init_recorders(tracker_serials, serial_ports, glove_port)
        
        # 初始化统计数据
        for name in self.recorders:
            self.total_frames[name] = 0
            self.current_fps[name] = 0.0
            self.frame_counts[name] = 0
            self.last_time[name] = time.perf_counter()  # 初始化时间戳
        
    def _init_recorders(self, tracker_serials, serial_ports, glove_port):
        """初始化所有记录器"""
        # 检查Tracker序列号是否有重复
        used_serials = set()
        for name, serial in tracker_serials.items():
            if serial in used_serials:
                print(f"警告: 检测到重复的Tracker序列号 {serial}")
                print("每个Tracker必须使用唯一的序列号")
                raise ValueError(f"重复的Tracker序列号: {serial}")
            used_serials.add(serial)

        # 初始化RealSense相机（优先初始化，因为可能需要更多时间）
        try:
            self.recorders['realsense'] = RealSenseRecorder()
            print("RealSense相机初始化成功")
        except Exception as e:
            print(f"RealSense相机初始化失败: {e}")

        # 初始化触觉传感器记录器
        try:
            self.recorders['tactile'] = TactileRecorder(
                left_port=serial_ports['left_tactile'],
                right_port=serial_ports['right_tactile']
            )
            print("触觉传感器初始化成功")
        except Exception as e:
            print(f"触觉传感器初始化失败: {e}")
        
        # 初始化数据手套记录器
        try:
            self.recorders['glove'] = GloveRecorder(port=glove_port)
            print("数据手套初始化成功")
        except Exception as e:
            print(f"数据手套初始化失败: {e}")
        
        # 初始化Tracker记录器
        for name, serial in tracker_serials.items():
            try:
                # 确保不同的Tracker使用不同的VR实例
                self.recorders[name] = TrackerRecorder(
                    tracker_name=name,
                    tracker_serial=serial
                )
                print(f"{name} 初始化成功 (序列号: {serial})")
                time.sleep(0.1)  # 添加短暂延迟，避免VR系统初始化冲突
            except Exception as e:
                print(f"{name} 初始化失败: {e}")
        
        if not self.recorders:
            raise Exception("没有任何设备初始化成功")
            
    def _get_next_folder_number(self):
        """获取下一个可用的文件夹序号"""
        base_dir = 'MultimodalData'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            return 1
            
        # 获取所有已存在的编号
        existing_folders = []
        for folder in os.listdir(base_dir):
            if folder.startswith(f"{self.folder_name}-"):
                try:
                    number = int(folder.split('-')[1])
                    existing_folders.append(number)
                except ValueError:
                    continue
        
        if not existing_folders:
            return 1
            
        # 找到第一个可用的编号
        next_number = 1
        while next_number in existing_folders:
            next_number += 1
            
        return next_number

    def start_recording(self):
        """开始记录"""
        if not self.recording:
            # 创建保存目录
            folder_number = self._get_next_folder_number()
            self.save_dir = os.path.join('MultimodalData', f"{self.folder_name}-{folder_number}")
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"\n开始记录，数据将保存到: {self.save_dir}")
            
            # 重置统计数据和事件标志
            self.stop_event.clear()
            self.start_event.clear()
            
            current_time = time.perf_counter()
            with self.stats_lock:
                for name in self.recorders:
                    self.total_frames[name] = 0
                    self.frame_counts[name] = 0
                    self.current_fps[name] = 0.0
                    self.last_time[name] = current_time
            
            # 首先启动RealSense相机记录器
            if 'realsenseCamera' in self.recorders:
                self.recorders['realsenseCamera'].start_recording(self.save_dir)
                self.threads['realsenseCamera'] = threading.Thread(
                    target=self._record_thread,
                    args=('realsenseCamera', self.recorders['realsenseCamera']),
                    daemon=True
                )
                self.threads['realsenseCamera'].start()
                time.sleep(0.5)  # 给RealSense相机更多初始化时间
            
            # 然后启动其他记录器
            for name, recorder in self.recorders.items():
                if name != 'realsenseCamera':  # 跳过已启动的RealSense相机
                    recorder.start_recording(self.save_dir)
                    self.threads[name] = threading.Thread(
                        target=self._record_thread,
                        args=(name, recorder),
                        daemon=True
                    )
                    self.threads[name].start()
                    time.sleep(0.1)  # 每个记录器之间添加延迟
            
            # 启动状态打印线程
            self.status_thread = threading.Thread(
                target=self._print_status,
                daemon=True
            )
            self.status_thread.start()
            
            # 设置开始事件
            self.recording = True
            time.sleep(0.2)  # 确保所有线程都已准备就绪
            self.start_event.set()
            
            return True
        return False
        
    def stop_recording(self, force_stop=False):
        """停止记录
        Args:
            force_stop: 是否强制停止（不保存数据）
        """
        if self.recording:
            # 先设置停止标志
            self.stop_event.set()
            
            # 停止所有记录器
            for name, recorder in self.recorders.items():
                try:
                    recorder.stop_recording(force_stop=force_stop)
                except Exception as e:
                    print(f"停止 {name} 录制时出错: {e}")
            
            # 等待所有线程结束
            for name, thread in self.threads.items():
                if thread.is_alive():
                    thread.join(timeout=1.0)
                    if thread.is_alive():
                        print(f"警告: {name} 录制线程未能正常结束")
            
            # 只在正常停止时打印统计信息
            if not force_stop:
                print("\n正在停止记录...")
                print("\n录制统计:")
                with self.stats_lock:
                    for name in self.recorders:
                        if name in self.total_frames and name in self.current_fps:
                            print(f"{name:15}: 总帧数 {self.total_frames[name]:6d}, "
                                  f"平均帧率 {self.current_fps[name]:5.1f} fps")
            
            self.recording = False
            self.threads.clear()
            return True
        return False
        
    def _record_thread(self, name, recorder):
        """记录线程函数"""
        print(f"等待 {name} 开始录制...")
        
        # 等待开始信号
        self.start_event.wait()
        print(f"开始 {name} 录制")
        
        try:
            start_time = time.perf_counter()
            frame_count = 0
            
            while not self.stop_event.is_set():
                try:
                    # 根据设备类型设置不同的优先级和处理逻辑
                    if name == 'realsenseCamera':
                        # RealSense相机需要最高优先级和最小延迟
                        success = recorder.record_frame()
                        time.sleep(0.001)  # 1ms延迟
                    elif 'tracker' in name.lower():
                        # Tracker设备使用更保守的延迟
                        success = recorder.record_frame()
                        if success:
                            time.sleep(0.008)  # 8ms延迟，避免与RealSense相机竞争
                        else:
                            time.sleep(0.001)  # 如果记录失败，使用更短的延迟
                    else:
                        # 其他设备使用中等延迟
                        success = recorder.record_frame()
                        time.sleep(0.005)  # 5ms延迟
                    
                    if success:
                        current_time = time.perf_counter()
                        with self.stats_lock:
                            self.total_frames[name] += 1
                            frame_count += 1
                            
                            # 使用指数移动平均计算帧率
                            elapsed = current_time - start_time
                            if elapsed >= 0.1:  # 至少累积0.1秒再更新帧率
                                instant_fps = frame_count / elapsed
                                alpha = 0.3  # 平滑因子
                                if name in self.current_fps:
                                    self.current_fps[name] = alpha * instant_fps + (1 - alpha) * self.current_fps[name]
                                else:
                                    self.current_fps[name] = instant_fps
                                
                                # 重置计数器
                                frame_count = 0
                                start_time = current_time
                                
                except Exception as e:
                    print(f"\n{name} 录制线程出错: {e}")
                    time.sleep(0.1)  # 错误后等待较长时间
                    continue
                    
        except Exception as e:
            print(f"\n{name} 录制线程出错: {e}")
        finally:
            print(f"{name} 录制结束")
            
    def _print_status(self):
        """打印状态信息"""
        try:
            while not self.stop_event.is_set():
                # 只在正常录制时清屏和打印状态
                if self.recording and not any(thread.name.startswith('error_') for thread in threading.enumerate()):
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    # 打印分隔线和表头
                    print("-" * 45)
                    print("-" * 45)
                    print(f"{'设备':<15} | {'帧率':>8} | {'总帧数':>8} --")
                    print("-" * 45)
                    
                    with self.stats_lock:
                        # 特殊处理触觉传感器的显示
                        if 'tactile' in self.recorders:
                            tactile_stats = self.recorders['tactile'].get_stats()
                            print(f"left_tactile   | {tactile_stats.get('left_fps', 0.0):>8.1f} | "
                                  f"{tactile_stats.get('left_total', 0):>8}")
                            print(f"right_tactile  | {tactile_stats.get('right_fps', 0.0):>8.1f} | "
                                  f"{tactile_stats.get('right_total', 0):>8}")
                        
                        # 显示其他设备的统计信息
                        for name in self.recorders:
                            if name != 'tactile':  # 跳过触觉传感器，因为已经单独处理
                                print(f"{name:<15} | {self.current_fps.get(name, 0.0):>8.1f} | "
                                      f"{self.total_frames.get(name, 0):>8}")
                    
                    print("-" * 45)
                
                time.sleep(1.0)  # 每秒更新一次
                
        except Exception as e:
            print(f"\n状态打印线程出错: {e}")
            
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        self.exit_event.set()
        
        if self.recording:
            self.stop_recording(force_stop=True)
        
        for name, recorder in self.recorders.items():
            try:
                recorder.cleanup()
            except Exception as e:
                print(f"清理 {name} 资源时出错: {e}")
                
        self.recorders.clear()
            
    def run(self):
        """运行记录管理器"""
        print("\n所有设备就绪")
        print("按空格键开始/停止记录")
        print("按Q键退出程序")
        
        try:
            while not self.exit_event.is_set():
                if keyboard.is_pressed('space'):
                    if not self.recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
                    time.sleep(0.2)  # 防止重复触发
                    
                if keyboard.is_pressed('q'):
                    break
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n用户中断程序")
        finally:
            self.cleanup()
