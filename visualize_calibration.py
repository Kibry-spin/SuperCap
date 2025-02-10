import numpy as np
import open3d as o3d
import glob
import pickle

def create_coordinate_frame(size=0.1):
    """创建坐标系显示"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

def create_camera_visualization(scale=0.05):
    """创建简化的相机模型"""
    # 相机模型点
    points = np.array([
        [0, 0, 0],          # 相机中心
        [-scale, -scale, scale*2],   # 左下
        [scale, -scale, scale*2],    # 右下
        [scale, scale, scale*2],     # 右上
        [-scale, scale, scale*2],    # 左上
    ])
    
    # 相机框线
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],  # 从中心到四个角
        [1, 2], [2, 3], [3, 4], [4, 1]   # 框的四条边
    ])
    
    # 创建LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # 设置颜色（蓝色）
    colors = [[0, 0, 1] for i in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def load_calibration_result(folder_path, method=4):
    """加载标定结果"""
    # 加载相机到Tracker的变换矩阵
    cam2ee_file = f"{folder_path}/FinalTransforms/T_cam2EE_Method_{method}.npz"
    T_cam2ee = np.load(cam2ee_file)['arr_0']
    return T_cam2ee

def load_tracker_poses(folder_path):
    """加载Tracker位姿数据"""
    transform_files = sorted(glob.glob(f'{folder_path}/*.pkl'))
    tracker_poses = []
    
    for fname in transform_files:
        with open(fname, 'rb') as fp:
            position = pickle.load(fp)
            
        # 创建4x4变换矩阵
        T = np.eye(4)
        T[:3, 3] = [position['x'], position['y'], position['z']]
        tracker_poses.append(T)
    
    return tracker_poses

def visualize_calibration(folder_path, method=4):
    """可视化标定结果"""
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 加载数据
    T_cam2ee = load_calibration_result(folder_path, method)
    tracker_poses = load_tracker_poses(folder_path)
    
    # 创建世界坐标系
    world_frame = create_coordinate_frame(size=0.1)
    vis.add_geometry(world_frame)
    
    # 为每个位置创建Tracker和相机的可视化
    for i, T_tracker in enumerate(tracker_poses):
        # 创建Tracker坐标系
        tracker_frame = create_coordinate_frame(size=0.05)
        tracker_frame.transform(T_tracker)
        vis.add_geometry(tracker_frame)
        
        # 计算相机位姿
        T_cam = T_tracker @ np.linalg.inv(T_cam2ee)
        
        # 创建相机可视化
        camera = create_camera_visualization()
        camera.transform(T_cam)
        vis.add_geometry(camera)
    
    # 设置可视化参数
    vis.get_render_option().point_size = 1
    vis.get_render_option().line_width = 2.0
    vis.get_render_option().background_color = np.array([1, 1, 1])  # 白色背景
    
    # 设置默认视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 数据文件夹路径
    folder_path = "Calibration_data/20250210_163043"
    

    # 可视化标定结果
    visualize_calibration(folder_path, method=4) 