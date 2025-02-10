#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: Jory.d
@contact: 707564875@qq.com
@site: 
@software: PyCharm
@file: read_realsense_stream2.py
@desc:  读取realsense流, 保存数据(包括图像数据  深度数据  点云数据)
        连接realsense相机进行录像, 按照帧序号保存 彩色图 深度图 点云图(ply & pcd)
        经过测试, 保存的点云图都可以使用pcl_viewer 打开和查看, pcd文件的纹理色彩也是没问题的
"""

import sys
sys.path.append('.')
import platform

os_name = platform.system()
if os_name == 'Linux':
    sys.path.append('/usr/local/lib')
    sys.path.append('/usr/local/lib/python3.7/pyrealsense2')
import pyrealsense2 as rs

import os, os.path as osp
import shutil
import numpy as np
import cv2
import shutil
import glob

WIDTH = 640
HEIGHT = 480
FPS = 15
SKIP_FRAME = 4
save_path = '/media/xxx/20221109/realsense_data_20221109/'

MAX_FRAME_NUM = 100


def save():
    if osp.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    ctx = rs.context()
    _devices = ctx.query_devices()
    if len(_devices) == 0:
        print("No device connected, please connect a RealSense device.")
        return -100

    # Configure depth and color streams
    pipeline = rs.pipeline(ctx)
    config = rs.config()

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)

    if config.can_resolve(pipeline):
        config.resolve(pipeline)
    else:
        print('rs.config().resolve() failed. please check rs.config.')
        return -200

    # Start streaming
    pipe_profile = pipeline.start(config)
    _device = pipe_profile.get_device()
    if _device is None:
        print('pipe_profile.get_device() is None .')
        return -400
    assert _device is not None

    depth_sensor = _device.first_depth_sensor()
    g_depth_scale = depth_sensor.get_depth_scale()  # 0.00100...

    device_product_line = str(_device.get_info(rs.camera_info.product_line))
    print(device_product_line)
    found_rgb = False
    for s in _device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    align_to = rs.stream.color
    # align_to = rs.stream.depth
    align = rs.align(align_to)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than clipping_distance_in_meters meters away
    clipping_distance_in_meters = 6  # 6 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Processing blocks
    pc = rs.pointcloud()
    g_colorizer = rs.colorizer(0)

    ###################################################################################
    # realsense-viewer 软件里的postprocess顺序是：
    """
    decimation_filter --> HDR Merge --> threshold_filter --> Depth to Disparity --> spatial_filter
    --> temporal_filter --> Disparity to Depth
    """
    # 使用rs内部的接口进行点云滤波
    g_rs_downsample_filter = rs.decimation_filter(
        magnitude=2 ** 1,
    )  # 下采样率
    g_rs_thres_filter = rs.threshold_filter(min_dist=0.1, max_dist=5.0)
    g_rs_spatical_filter = rs.spatial_filter(
        magnitude=2,
        smooth_alpha=0.5,
        smooth_delta=20,
        hole_fill=0,
    )
    g_rs_templ_filter = rs.temporal_filter(
        smooth_alpha=0.1,
        smooth_delta=40.,
        persistence_control=3
    )
    g_rs_depth2disparity_trans = rs.disparity_transform(True)
    g_rs_disparity2depth_trans = rs.disparity_transform(False)
    g_rs_depth_postprocess_list = [
        g_rs_downsample_filter,
        g_rs_thres_filter,
        g_rs_depth2disparity_trans,
        g_rs_spatical_filter,
        g_rs_templ_filter,
        g_rs_disparity2depth_trans
    ]
    ###################################################################################

    i = 0
    while i < MAX_FRAME_NUM:
        i += 1
        ret, frames = pipeline.try_wait_for_frames()
        if not ret:
            print('try_wait_for_frames() failed.')
            break

        if i > 0 and i % SKIP_FRAME != 0:
            continue

        # align
        align_frames = align.process(frames)
        frames = align_frames

        frame_num = frames.frame_number
        timestamp = frames.timestamp
        profile = frames.profile
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        depth_frame_filter = depth_frame
        for trans in g_rs_depth_postprocess_list:
            depth_frame_filter = trans.process(depth_frame_filter)
        depth_frame = depth_frame_filter

        # Convert images to numpy arrays
        depth_img = np.asanyarray(depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())
        depth_colormap = g_colorizer.colorize(depth_frame)
        depth_colormap = np.asarray(depth_colormap.get_data())

        print('timestramp: ', timestamp)
        print('frame_num: ', frame_num)
        print('color_img.shape: ', color_img.shape)
        print('depth_img.shape: ', depth_img.shape)

        # 得到 相机的内参和外参, align之后参数可能会改变
        # 当depth进行一系列滤波之后, 两者的内参会不一样, 主要是下采样导致的
        # 内参要以 color_intrin 为主
        g_intrinsics = frames.get_profile().as_video_stream_profile().get_intrinsics()
        color_intrin = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        depth_intrin = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        g_color_intrinsics_matrix = np.array([
            [color_intrin.fx, 0., color_intrin.ppx],
            [0., color_intrin.fy, color_intrin.ppy],
            [0, 0, 1.]
        ])
        g_depth_intrinsics_matrix = np.array([
            [depth_intrin.fx, 0., depth_intrin.ppx],
            [0., depth_intrin.fy, depth_intrin.ppy],
            [0, 0, 1.]
        ])

        extrinsics = depth_frame.get_profile().get_extrinsics_to(color_frame.get_profile())
        rotation, translation = extrinsics.rotation, extrinsics.translation
        ppx, ppy = depth_intrin.ppx, depth_intrin.ppy
        fx, fy = depth_intrin.fx, depth_intrin.fy
        coeffs = depth_intrin.coeffs

        # mapped_frame, color_source = depth_frame, depth_colormap
        mapped_frame, color_source = color_frame, color_img
        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)
        v, t = points.get_vertices(), points.get_texture_coordinates()
        pointcloud_xyz = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz    shape: [N,3]
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv 色彩
        cw, ch = color_img.shape[:2][::-1]
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch - 1, out=u)
        np.clip(v, 0, cw - 1, out=v)
        pointcloud_rgb = color_img[u, v]    # rgb   shape: [N,3]
        # print(f'pointcloud_xyz: {pointcloud_xyz.shape}')
        # print(f'pointcloud_rgb: {pointcloud_rgb.shape}')
        # 获取点云数据方法1
        pointcloud_data_ply = pointcloud_xyz
        pointcloud_data_pcd = np.concatenate([pointcloud_xyz, pointcloud_rgb], axis=1)

        # # 保存 realsense格式的ply点云文件 带纹理色彩
        # save_filepath = f'{save_path}/pointcloud_3d/{frame_num}.realsense.ply'
        # os.makedirs(osp.dirname(save_filepath), exist_ok=True)
        # points.export_to_ply(save_filepath, mapped_frame)

        # 保存 RGB图像数据和深度数据
        img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_img_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
        # ################################## save img ##############################
        save_filepath = f'{save_path}/color/{frame_num}.png'
        os.makedirs(osp.dirname(save_filepath), exist_ok=True)
        cv2.imwrite(save_filepath, img_bgr)
        save_filepath = f'{save_path}/color/{frame_num}.npy'
        np.save(save_filepath, depth_img)

        # 保存深度图像
        # depth_img_bgr = cv2.cvtColor(depth_colormap, cv2.COLOR_RGB2BGR)
        save_filepath = f'{save_path}/depth/{frame_num}.png'
        os.makedirs(osp.dirname(save_filepath), exist_ok=True)
        cv2.imwrite(save_filepath, depth_colormap)

        ## 保存点云
        save_filepath = f'{save_path}/pointcloud_3d/{frame_num}.ply'
        os.makedirs(osp.dirname(save_filepath), exist_ok=True)
        # 获取点云数据方法2
        pointcloud_data_ply = depth2xyz(depth_img, g_depth_intrinsics_matrix, g_depth_scale)
        save_2_ply(pointcloud_data_ply, save_filepath)
        save_filepath = f'{save_path}/pointcloud_3d/{frame_num}.pcd'
        pointcloud_data_pcd = depth2xyzrgb(color_img, depth_img, g_depth_intrinsics_matrix, g_depth_scale)
        save_2_pcd(pointcloud_data_pcd, save_filepath)

        # Show images
        if img_bgr.shape[:2] != depth_colormap.shape[:2]:
            _H,_W = depth_colormap.shape[:2]
            img_bgr = cv2.resize(img_bgr, (_W,_H))
        images = np.hstack((img_bgr, depth_colormap))
        if images.shape[1] > 1200:
            images = cv2.resize(images, dsize=None, fx=0.5, fy=0.5)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

    pipeline.stop()
    print('read stream end ...')


def depth2xyz(depth_map, depth_cam_matrix, flatten=True, depth_scale=0.0010):
    """
    https://blog.csdn.net/tycoer/article/details/106761886
    # 深度转点云 https://blog.csdn.net/FUTEROX/article/details/126128581

    depth_map = np.random.randint(0,10000,(720, 1280))
    depth_cam_matrix = np.array([[540, 0,  640],
                                 [0,   540,360],
                                 [0,   0,    1]])
    pc = depth2xyz(depth_map, depth_cam_matrix)
    """
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
    z = depth_map * depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy

    xyz = np.dstack((x, y, z)).reshape(-1, 3) if flatten else np.dstack((x, y, z))
    # xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)

    # ################ pcl 坐标轴是 右手坐标系, 需要把realsense的坐标轴转换下  ################
    # ## realsense坐标系：  X代表水平方向右  Y代表垂直方向下    Z代表深度距离内
    # ## 右手坐标系: X代表水平方向右  Y代表垂直向上   Z代表深度距离向外
    # ## 左手坐标系: X代表水平方向右  Y代表垂直向上   Z代表深度距离向内
    xyz[:, [1, 2]] *= -1.
    # ################ ################ ################ ################ ################

    return xyz  # [N,3]


def save_2_ply(pointcloud_xyz, save_filepath):
    data_ply = pointcloud_xyz
    ##################### save *.ply ###########################
    # data_ply.shape:  [N,3]  or  [N,6]
    assert isinstance(data_ply, np.ndarray)
    is_color = data_ply.shape[1] == 6
    float_formatter = lambda x: "%.4f" % x
    points = []
    for i in data_ply:
        if is_color:
            if np.all(i[:3] == 0): continue
            points.append("{} {} {} {} {} {} 0\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           int(i[3]), int(i[4]), int(i[5])))
        else:
            if np.all(i == 0): continue
            points.append("{} {} {}\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           ))

    file = open(save_filepath, "w")
    if is_color:
        file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(points)))
    else:
        file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    end_header
    %s
    ''' % (len(points), "".join(points)))

    file.close()
    ######################################################################


def depth2xyzrgb(color_map, depth_map, depth_cam_matrix, depth_scale=0.0010):
    """
    https://blog.csdn.net/tycoer/article/details/106761886
    # 深度转点云 https://blog.csdn.net/FUTEROX/article/details/126128581
    color_map = np.random.randint(0,255,(720, 1280, 3))
    depth_map = np.random.randint(0,10000,(720, 1280))
    depth_cam_matrix = np.array([[540, 0,  640],
                                 [0,   540,360],
                                 [0,   0,    1]])
    pc = depth2xyzrgb(color_map, depth_map, depth_cam_matrix)
    """
    if depth_map.shape[:2] != color_map.shape[:2]:
        _h, _w = depth_map.shape[:2]
        color_map = cv2.resize(color_map, (_w, _h))

    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]  # [H,W,2]
    z = depth_map * depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    height, width = color_map.shape[:2]
    data_ply = np.zeros((6, width * height), dtype=np.float32)
    data_ply[0] = x.reshape(-1)
    data_ply[1] = y.reshape(-1)
    data_ply[2] = z.reshape(-1)
    data_ply[3] = color_map[:, :, 0].reshape(-1)
    data_ply[4] = color_map[:, :, 1].reshape(-1)
    data_ply[5] = color_map[:, :, 2].reshape(-1)

    data_ply = data_ply.T  # [N,6]
    # ################ pcl 坐标轴是 右手坐标系, 需要把realsense的坐标轴转换下  ################
    # ## realsense坐标系：  X代表水平方向右  Y代表垂直方向下    Z代表深度距离内
    # ## 右手坐标系: X代表水平方向右  Y代表垂直向上   Z代表深度距离向外
    # ## 左手坐标系: X代表水平方向右  Y代表垂直向上   Z代表深度距离向内
    data_ply[:, [1, 2]] *= -1.
    # ################ ################ ################ ################ ################

    return data_ply


def save_2_pcd(pointcloud_data_pcd, save_filepath):
    data_pcd = pointcloud_data_pcd
    assert isinstance(data_pcd, np.ndarray)
    # [N, 6]
    is_color = data_pcd.shape[1] == 6
    float_formatter = lambda x: "%.4f" % x
    points = []
    for i in data_pcd:
        if is_color:
            r, g, b = list(map(int, i[3:]))
            points.append("{} {} {} {}\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           (int(r) << 16) | (int(g) << 8) | int(b),
                           ))
        else:
            points.append("{} {} {}\n".format
                          (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                           ))

    file = open(save_filepath, "w")
    if is_color:
        file.write('''# .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z rgb
    SIZE 4 4 4 4
    TYPE F F F U
    COUNT 1 1 1 1
    WIDTH %d
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS %d
    DATA ascii
    %s
    ''' % (len(points), len(points), "".join(points)))
    else:
        file.write('''# .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z
    SIZE 4 4 4
    TYPE F F F
    COUNT 1 1 1
    WIDTH %d
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS %d
    DATA ascii
    %s
    ''' % (len(points), len(points), "".join(points)))
    file.close()


if __name__ == "__main__":
    print('hello')
    save()

# 加载数据
data_dir = 'E:/Project/ForheartUnity/glove_data/20250120_192115'
data = np.load(os.path.join(data_dir, 'glove_data1.npy'))
first_frame = data[0]  # 获取第一帧数据

print(f"数据形状: {data.shape}")
print("\n第一帧数据详细信息:")

# 右手数据 (0-66)
print("右手:")
print("  手腕位置 (x,y,z):", first_frame[0:3])
print("  手腕旋转 (四元数):", first_frame[3:7])
print("  拇指:")
for i in range(3):
    print(f"    关节{i+1} (四元数):", first_frame[7+i*4:11+i*4])
print("  食指:")
for i in range(3):
    print(f"    关节{i+1} (四元数):", first_frame[19+i*4:23+i*4])
print("  中指:")
for i in range(3):
    print(f"    关节{i+1} (四元数):", first_frame[31+i*4:35+i*4])
print("  无名指:")
for i in range(3):
    print(f"    关节{i+1} (四元数):", first_frame[43+i*4:47+i*4])
print("  小指:")
for i in range(3):
    print(f"    关节{i+1} (四元数):", first_frame[55+i*4:59+i*4])

# 左手数据 (从134开始)
left_start = 134  # 左手数据的起始位置
print("\n左手:")
print("  手腕位置 (x,y,z):", first_frame[left_start:left_start+3])
print("  手腕旋转 (四元数):", first_frame[left_start+3:left_start+7])
print("  拇指:")
for i in range(3):
    print(f"    关节{i+1} (四元数):", first_frame[left_start+7+i*4:left_start+11+i*4])
print("  食指:")
for i in range(3):
    print(f"    关节{i+1} (四元数):", first_frame[left_start+19+i*4:left_start+23+i*4])
print("  中指:")
for i in range(3):
    print(f"    关节{i+1} (四元数):", first_frame[left_start+31+i*4:left_start+35+i*4])
print("  无名指:")
for i in range(3):
    print(f"    关节{i+1} (四元数):", first_frame[left_start+43+i*4:left_start+47+i*4])
print("  小指:")
for i in range(3):
    print(f"    关节{i+1} (四元数):", first_frame[left_start+55+i*4:left_start+59+i*4])

# 打印数组长度，用于验证
print("\n数组长度:", len(first_frame))