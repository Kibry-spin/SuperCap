import numpy as np
import cv2
import os
import pickle
import pyrealsense2 as rs
import triad_openvr
from datetime import datetime
import keyboard
import time

class DataRecorder:
    def __init__(self, board_size=(9,9), square_size=0.01):
        """初始化记录器"""
        self.board_size = board_size
        self.square_size = square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.frame_count = 0
        self.init_devices()
        
    def init_devices(self):
        """初始化RealSense和Tracker"""
        # 初始化RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)
        
        # 初始化Tracker
        try:
            self.vr = triad_openvr.triad_openvr()
            print("Tracker初始化成功")
        except Exception as ex:
            print(f"Tracker初始化失败: {ex}")
            raise ex
            
    def get_tracker_pose(self):
        """获取Tracker位姿"""
        for deviceName in self.vr.devices:
            if deviceName == 'tracker_1':
                return self.vr.devices[deviceName].get_pose_euler()
        return None
        
    def save_frame(self, save_dir):
        """保存一帧数据"""
        # 获取图像
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("无法获取彩色图像")
            return False
            
        # 获取Tracker位姿
        tracker_pose = self.get_tracker_pose()
        if tracker_pose is None:
            print("无法获取Tracker位姿")
            return False
            
        # 保存图像
        color_image = np.asanyarray(color_frame.get_data())
        image_path = os.path.join(save_dir, f"{self.frame_count}.png")
        cv2.imwrite(image_path, color_image)
        
        # 保存Tracker位姿
        pose_data = {
            "x": tracker_pose[0],
            "y": tracker_pose[1],
            "z": tracker_pose[2],
            "roll": tracker_pose[3],
            "pitch": tracker_pose[4],
            "yaw": tracker_pose[5]
        }
        pose_path = os.path.join(save_dir, f"{self.frame_count}.pkl")
        with open(pose_path, 'wb') as f:
            pickle.dump(pose_data, f)
            
        self.frame_count += 1
        return True
        
    def run(self):
        """运行记录程序"""
        print("按空格键保存当前帧")
        print("按Q键退出")
        
        # 创建保存目录
        save_dir = os.path.join("./Calibration_data", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)
        print(f"数据将保存到: {save_dir}")
        
        try:
            while True:
                # 获取并显示图像
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # 显示Tracker位姿
                    tracker_pose = self.get_tracker_pose()
                    if tracker_pose:
                        pose_text = f"X:{tracker_pose[0]:.3f} Y:{tracker_pose[1]:.3f} Z:{tracker_pose[2]:.3f}"
                        angle_text = f"Roll:{tracker_pose[3]:.1f} Pitch:{tracker_pose[4]:.1f} Yaw:{tracker_pose[5]:.1f}"
                        cv2.putText(color_image, pose_text, (30, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(color_image, angle_text, (30, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.putText(color_image, f"已记录: {self.frame_count} 帧", (30, 110),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Camera View', color_image)
                    cv2.waitKey(1)
                
                if keyboard.is_pressed('space'):
                    print("\n保存当前帧...")
                    if self.save_frame(save_dir):
                        print(f"成功保存第 {self.frame_count} 帧")
                    time.sleep(0.2)
                    
                elif keyboard.is_pressed('q'):
                    break
                    
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()
            print(f"\n共记录 {self.frame_count} 帧数据")
            print(f"数据保存在: {save_dir}")

def main():
    recorder = DataRecorder(board_size=(9,9), square_size=0.01)
    recorder.run()

if __name__ == "__main__":
    main() 