import numpy as np
import triad_openvr
import transforms3d
import time

class CameraPoseTracker:
    def __init__(self):
        # 初始化VR系统
        try:
            self.vr = triad_openvr.triad_openvr()
            print("VR系统初始化成功")
        except Exception as e:
            print(f"VR系统初始化失败: {e}")
            raise
            
        # 加载手眼标定结果
        try:
            self.H_tracker_cam = np.load('hand_eye_calibration.npy')
            print("已加载手眼标定结果")
        except Exception as e:
            print(f"加载标定结果失败: {e}")
            raise

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """欧拉角转旋转矩阵"""
        return transforms3d.euler.euler2mat(roll, pitch, yaw, 'sxyz')
    
    def pose_to_matrix(self, pose):
        """将位姿转换为4x4变换矩阵"""
        x, y, z, roll, pitch, yaw = pose
        R = self.euler_to_rotation_matrix(roll, pitch, yaw)
        T = np.array([x, y, z])
        
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = T
        return H

    def get_tracker_pose(self):
        """获取Tracker位姿"""
        try:
            if 'tracker_1' in self.vr.devices:
                return self.vr.devices['tracker_1'].get_pose_euler()
            return None
        except Exception as e:
            print(f"获取Tracker位姿失败: {e}")
            return None

    def track_camera_pose(self):
        """实时追踪相机位置"""
        print("开始追踪相机位置... (按Ctrl+C退出)")
        try:
            while True:
                tracker_pose = self.get_tracker_pose()
                if tracker_pose is not None:
                    # 构建Tracker在Vive中的变换矩阵
                    H_vive_tracker = self.pose_to_matrix(tracker_pose)
                    
                    # 计算相机在Vive坐标系中的变换矩阵
                    H_vive_cam = H_vive_tracker @ self.H_tracker_cam
                    
                    # 提取相机位置和姿态
                    position = H_vive_cam[:3, 3]
                    rotation = transforms3d.euler.mat2euler(H_vive_cam[:3, :3], 'sxyz')
                    
                    # 清屏并显示位置信息
                    print("\033[H\033[J")  # 清屏
                    print("相机在Vive坐标系中的位置和姿态:")
                    print(f"位置:")
                    print(f"  X: {position[0]:.3f} m")
                    print(f"  Y: {position[1]:.3f} m")
                    print(f"  Z: {position[2]:.3f} m")
                    print(f"姿态 (弧度):")
                    print(f"  Roll : {rotation[0]:.3f}")
                    print(f"  Pitch: {rotation[1]:.3f}")
                    print(f"  Yaw  : {rotation[2]:.3f}")
                
                time.sleep(0.1)  # 刷新率10Hz
                
        except KeyboardInterrupt:
            print("\n停止追踪")

def main():
    tracker = CameraPoseTracker()
    tracker.track_camera_pose()

if __name__ == '__main__':
    main() 