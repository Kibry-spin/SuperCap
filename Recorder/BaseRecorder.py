from abc import ABC, abstractmethod
import time
from datetime import datetime
import os
import threading

class BaseRecorder(ABC):
    """记录器基类，定义统一接口"""
    def __init__(self):
        self.recording = False
        self.running = True
        self.frame_count = 0  # 每秒的帧计数
        self.total_frames = 0  # 总帧数
        self.last_fps_time = time.perf_counter()
        self.fps = 0.0
        self.save_dir = None
        self.device_ready = False
        
        # 添加线程安全锁
        self._lock = threading.Lock()
        self._data_lock = threading.Lock()
        
    @abstractmethod
    def _init_device(self):
        """初始化设备"""
        pass
        
    @abstractmethod
    def _check_device(self):
        """检查设备状态"""
        pass
        
    @abstractmethod
    def record_frame(self):
        """记录一帧数据，返回是否成功记录"""
        pass
        
    @abstractmethod
    def save_data(self):
        """保存记录的数据"""
        pass
        
    @abstractmethod
    def cleanup(self):
        """清理资源"""
        pass
        
    def create_save_dir(self, base_dir, timestamp=None):
        """创建保存目录"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(base_dir, timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        return self.save_dir
        
    def start_recording(self, save_dir=None):
        """开始记录"""
        with self._lock:
            if not self.recording and self._check_device():
                if save_dir:
                    self.save_dir = save_dir
                self.recording = True
                self.frame_count = 0
                self.total_frames = 0  # 重置总帧数
                self.last_fps_time = time.perf_counter()
                return True
            return False
            
    def stop_recording(self, force_stop=False):
        """停止记录
        Args:
            force_stop: 是否强制停止（不保存数据）
        """
        with self._lock:
            if self.recording:
                self.recording = False
                if not force_stop:
                    self.save_data()
                return True
            return False
            
    def update_fps(self):
        """更新帧率"""
        with self._lock:
            self.frame_count += 1
            self.total_frames += 1  # 更新总帧数
            current_time = time.perf_counter()
            elapsed_time = current_time - self.last_fps_time
            
            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.last_fps_time = current_time
                return True
            return False
            
    def is_recording(self):
        """检查是否正在记录"""
        with self._lock:
            return self.recording
            
    def get_fps(self):
        """获取当前帧率"""
        with self._lock:
            return self.fps 