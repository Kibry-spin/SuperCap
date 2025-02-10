import os
import numpy as np
import open3d as o3d
import cv2
import json

def load_intrinsics(file_path):
    intrinsics = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split()
            intrinsics[key] = float(value)
    return intrinsics

def create_point_cloud_from_images(color_image, depth_image, intrinsics):
    height, width = depth_image.shape

    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['ppx'], intrinsics['ppy']
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    depth_image_o3d = o3d.geometry.Image(depth_image)
    color_image_o3d = o3d.geometry.Image(color_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image_o3d, depth_image_o3d, convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    return pcd

def visualize_realsense_data(session_folder):
    # 从根目录加载内参
    intrinsics_file = os.path.join("realsense_data", "camera_intrinsics.txt")
    if not os.path.exists(intrinsics_file):
        raise FileNotFoundError(f"Camera intrinsics file not found: {intrinsics_file}")
    intrinsics = load_intrinsics(intrinsics_file)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 创建坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # 添加相机位置标记（使用球体）
    camera_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    camera_marker.paint_uniform_color([1, 0, 0])  # 红色
    camera_marker.translate([0, 0, 0])  # 相机在原点
    vis.add_geometry(camera_marker)

    # 添加相机视锥体（可选）
    points = np.array([
        [0, 0, 0],      # 相机位置
        [0.1, 0.075, 0.1],  # 右上
        [0.1, -0.075, 0.1], # 右下
        [-0.1, -0.075, 0.1], # 左下
        [-0.1, 0.075, 0.1]   # 左上
    ])
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],  # 从相机到四个角
        [1, 2], [2, 3], [3, 4], [4, 1]   # 连接四个角
    ])
    colors = np.array([[1, 0, 0] for _ in range(len(lines))])  # 红色线条
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

    # 设置初始视角
    ctr = vis.get_view_control()
    
    # 设置相机参数
    cam = ctr.convert_to_pinhole_camera_parameters()
    # 从相机原点出发的视角
    cam.extrinsic = np.array([
        [-1, 0,  0,  0],    # 相机坐标系的X轴取反
        [ 0, -1, 0,  0],    # 相机坐标系的Y轴取反
        [ 0, 0,  1,  0],    # 相机坐标系的Z轴保持
        [ 0, 0,  0,  1]     # 相机位置(原点)
    ])
    ctr.convert_from_pinhole_camera_parameters(cam)
    
    # 调整视野
    ctr.set_zoom(0.3)
    ctr.set_lookat([0, 0, 0])  # 设置观察点为原点

    depth_dir = os.path.join(session_folder, "depth")
    rgb_dir = os.path.join(session_folder, "rgb")

    if not os.path.exists(depth_dir) or not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"Data directories not found in: {session_folder}")

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.startswith('depth_')])
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.startswith('color_')])

    if not depth_files or not rgb_files:
        raise FileNotFoundError(f"No image files found in: {session_folder}")

    pcd = None

    # 创建保存点云的目录
    ply_dir = os.path.join(session_folder, "ply")
    os.makedirs(ply_dir, exist_ok=True)

    # 创建保存相机信息的目录
    camera_info_dir = os.path.join(session_folder, "camera_info")
    os.makedirs(camera_info_dir, exist_ok=True)

    for i, (depth_file, rgb_file) in enumerate(zip(depth_files, rgb_files)):
        depth_image_path = os.path.join(depth_dir, depth_file)
        rgb_image_path = os.path.join(rgb_dir, rgb_file)

        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        color_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        new_pcd = create_point_cloud_from_images(color_image, depth_image, intrinsics)

        # 保存点云为PLY文件
        ply_filename = f"frame_{i:04d}.ply"
        ply_path = os.path.join(ply_dir, ply_filename)
        o3d.io.write_point_cloud(ply_path, new_pcd)
        
        # 保存对应的相机信息
        camera_info = {
            'position': [0, 0, 0],  # 相机位置
            'rotation': [1, 0, 0,   # 旋转矩阵
                        0, 1, 0,
                        0, 0, 1],
            'intrinsics': {         # 相机内参
                'fx': intrinsics['fx'],
                'fy': intrinsics['fy'],
                'cx': intrinsics['ppx'],
                'cy': intrinsics['ppy']
            }
        }
        
        camera_info_path = os.path.join(camera_info_dir, f"camera_{i:04d}.json")
        with open(camera_info_path, 'w') as f:
            json.dump(camera_info, f, indent=4)
            
        print(f"已保存相机信息: {camera_info_path}")

        if pcd is None:
            pcd = new_pcd
            vis.add_geometry(pcd)
        else:
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    session_folder = "realsense_data\session_20250208_144405"
    try:
        visualize_realsense_data(session_folder)
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"未预期的错误: {e}")