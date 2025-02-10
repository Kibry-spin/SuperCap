import os
import numpy as np
import open3d as o3d
import cv2
import json
import laspy

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
    
    # 创建点云数据
    points = []
    colors = []
    
    for v in range(height):
        for u in range(width):
            z = depth_image[v, u] / 1000.0  # 转换为米
            if z > 0:
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                
                # 添加点坐标
                points.append([x, y, z])
                
                # 添加颜色
                b, g, r = color_image[v, u]
                colors.append([r, g, b])  # LAS格式需要0-255范围的RGB值
    
    return np.array(points), np.array(colors, dtype=np.uint8)

def save_las(points, colors, filename):
    """保存为LAS格式文件"""
    # 创建LAS文件头
    header = laspy.LasHeader(point_format=2, version="1.2")  # 使用点格式2，包含RGB颜色
    
    # 创建LAS文件
    las = laspy.LasData(header)
    
    # 设置点坐标
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    
    # 设置颜色 (LAS格式使用16位颜色)
    las.red = colors[:, 0].astype(np.uint16) * 256   # 扩展到16位
    las.green = colors[:, 1].astype(np.uint16) * 256
    las.blue = colors[:, 2].astype(np.uint16) * 256
    
    # 保存文件
    las.write(filename)

def visualize_realsense_data(session_folder):
    # 从根目录加载内参
    intrinsics_file = os.path.join("realsense_data", "camera_intrinsics.txt")
    if not os.path.exists(intrinsics_file):
        raise FileNotFoundError(f"Camera intrinsics file not found: {intrinsics_file}")
    intrinsics = load_intrinsics(intrinsics_file)

    depth_dir = os.path.join(session_folder, "depth")
    rgb_dir = os.path.join(session_folder, "rgb")

    if not os.path.exists(depth_dir) or not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"Data directories not found in: {session_folder}")

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.startswith('depth_')])
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.startswith('color_')])

    if not depth_files or not rgb_files:
        raise FileNotFoundError(f"No image files found in: {session_folder}")

    # 创建保存点云的目录
    las_dir = os.path.join(session_folder, "las")
    os.makedirs(las_dir, exist_ok=True)

    # 创建保存相机信息的目录
    camera_info_dir = os.path.join(session_folder, "camera_info")
    os.makedirs(camera_info_dir, exist_ok=True)

    for i, (depth_file, rgb_file) in enumerate(zip(depth_files, rgb_files)):
        print(f"处理帧 {i+1}/{len(depth_files)}")
        
        depth_image_path = os.path.join(depth_dir, depth_file)
        rgb_image_path = os.path.join(rgb_dir, rgb_file)

        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        color_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # 生成点云
        points, colors = create_point_cloud_from_images(color_image, depth_image, intrinsics)

        # 保存为LAS文件
        las_filename = f"frame_{i:04d}.las"
        las_path = os.path.join(las_dir, las_filename)
        save_las(points, colors, las_path)
        
        # 保存相机信息
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
            
        print(f"已保存点云: {las_path}")
        print(f"已保存相机信息: {camera_info_path}")

if __name__ == "__main__":
    session_folder = "realsense_data\\session_20250118_133615"
    try:
        visualize_realsense_data(session_folder)
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"未预期的错误: {e}")