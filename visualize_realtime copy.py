import numpy as np
import open3d as o3d
import triad_openvr
import time
import threading

def create_coordinate_frame(size=0.1):
    """创建坐标系显示"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

def create_camera_visualization(scale=0.05):
    """创建简化的相机模型
    相机坐标系：
    - Z轴：相机光轴方向（相机朝向），指向后方
    - X轴：向左
    - Y轴：向下
    """
    # 计算视锥体的深度和宽度
    depth = scale * 2  # 视锥体深度
    width = scale * 0.8  # 视锥体宽度
    height = scale * 0.6  # 视锥体高度
    
    points = np.array([
        [0, 0, 0],          # 相机中心（光心）
        [width, -height, -depth],   # 左下
        [-width, -height, -depth],  # 右下
        [-width, height, -depth],   # 右上
        [width, height, -depth],    # 左上
    ])
    
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],  # 从中心到四个角
        [1, 2], [2, 3], [3, 4], [4, 1],  # 框的四条边
    ])
    
    # 创建相机视锥体
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # 设置颜色：所有线条为蓝色
    colors = [[0, 0, 1] for _ in range(len(lines))]  # 蓝色
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def load_calibration_result(folder_path, method=4):
    """加载标定结果"""
    cam2ee_file = f"{folder_path}/FinalTransforms/T_cam2EE_Method_{method}.npz"
    T_cam2ee = np.load(cam2ee_file)['arr_0']
    return T_cam2ee

def get_tracker_pose(v):
    """获取Tracker的实时位姿"""
    T = np.eye(4)
    try:
        for deviceName in v.devices:
            if deviceName == 'tracker_1':
                # 获取位置和欧拉角
                [x, y, z, roll, pitch, yaw] = v.devices[deviceName].get_pose_euler()
                # 创建旋转矩阵
                Rx = np.array([[1, 0, 0],
                             [0, np.cos(roll), -np.sin(roll)],
                             [0, np.sin(roll), np.cos(roll)]])
                
                Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                             [0, 1, 0],
                             [-np.sin(pitch), 0, np.cos(pitch)]])
                
                Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                             [np.sin(yaw), np.cos(yaw), 0],
                             [0, 0, 1]])
                
                R = Rz @ Ry @ Rx
                T[:3, :3] = R
                T[:3, 3] = [x, y, z]
                break
    except Exception as e:
        print(f"Error getting tracker pose: {e}")
    return T

def visualize_realtime(folder_path, method=4):
    """实时可视化标定结果"""
    # 初始化VR
    try:
        v = triad_openvr.triad_openvr()
        print("VR系统初始化成功")
    except Exception as ex:
        print(f"VR系统初始化失败: {ex}")
        return

    # 加载标定结果
    T_cam2ee = load_calibration_result(folder_path, method)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 创建世界坐标系
    world_frame = create_coordinate_frame(size=0.1)
    vis.add_geometry(world_frame)
    
    # 创建Tracker和相机的可视化对象
    tracker_frame = create_coordinate_frame(size=0.05)
    camera = create_camera_visualization()
    
    # 创建连接原点的线
    origin_line = o3d.geometry.LineSet()
    origin_line.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 0, 0]]))
    origin_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    origin_line.colors = o3d.utility.Vector3dVector(np.array([[0.5, 0.5, 0.5]]))  # 灰色
    
    # 创建相机朝向射线
    direction_line = o3d.geometry.LineSet()
    direction_line.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 0, -0.2]]))  # 初始位置，Z轴负方向
    direction_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    direction_line.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))  # 红色
    
    vis.add_geometry(tracker_frame)
    vis.add_geometry(camera)
    vis.add_geometry(origin_line)
    vis.add_geometry(direction_line)
    
    # 设置可视化参数
    vis.get_render_option().point_size = 1
    vis.get_render_option().line_width = 2.0
    vis.get_render_option().background_color = np.array([1, 1, 1])
    
    # 设置默认视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    
    print("开始实时显示，按Ctrl+C退出...")
    
    try:
        while True:
            # 获取Tracker位姿
            T_tracker = get_tracker_pose(v)
            
            # 更新Tracker可视化
            tracker_frame_new = create_coordinate_frame(size=0.05)
            tracker_frame_new.transform(T_tracker)
            vis.remove_geometry(tracker_frame)
            vis.add_geometry(tracker_frame_new)
            tracker_frame = tracker_frame_new
            
            # 计算相机位姿
            T_cam = T_tracker @ np.linalg.inv(T_cam2ee)
            
            # 更新相机可视化
            camera_new = create_camera_visualization()
            camera_new.transform(T_cam)
            vis.remove_geometry(camera)
            vis.add_geometry(camera_new)
            camera = camera_new
            
            # 更新连接原点的线
            origin_line_new = o3d.geometry.LineSet()
            cam_position = T_cam[:3, 3]  # 获取相机位置
            origin_line_new.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], cam_position]))
            origin_line_new.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
            origin_line_new.colors = o3d.utility.Vector3dVector(np.array([[0.5, 0.5, 0.5]]))  # 灰色
            vis.remove_geometry(origin_line)
            vis.add_geometry(origin_line_new)
            origin_line = origin_line_new
            
            # 更新相机朝向射线
            direction_line_new = o3d.geometry.LineSet()
            cam_z_direction = T_cam[:3, 2]  # 获取相机Z轴方向（第三列）
            direction_end = cam_position + cam_z_direction * 0.2  # 沿Z轴方向延伸0.2米
            direction_line_new.points = o3d.utility.Vector3dVector(np.array([cam_position, direction_end]))
            direction_line_new.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
            direction_line_new.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))  # 红色
            vis.remove_geometry(direction_line)
            vis.add_geometry(direction_line_new)
            direction_line = direction_line_new
            
            # 更新显示
            vis.poll_events()
            vis.update_renderer()
            
            time.sleep(0.01)  # 控制更新频率
            
    except KeyboardInterrupt:
        print("\n停止显示")
    finally:
        vis.destroy_window()

if __name__ == "__main__":
    # 数据文件夹路径
    folder_path = "Calibration_data/20250210_152255"
    

    # 开始实时可视化
    visualize_realtime(folder_path, method=4) 