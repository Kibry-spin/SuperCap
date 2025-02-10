import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import pickle
from scipy.spatial.transform import Rotation as R

# global variable
g = {}


def siyuanshu2rotation_matrix(Rq):
    # 不同机器人输出的四元数顺序不同，有的是[w, x, y, z],有的是[x, y, z, w]
    # 因此这里要根据具体情况进行调整
    Rnew = [Rq[1], Rq[2], Rq[3], Rq[0]]
    
    # R.from_quat输入的四元数顺序必须为： [x, y, z, w]
    Rm = R.from_quat(Rnew)
    rotation_matrix = Rm.as_matrix()
    return rotation_matrix

def Rm_t2T(Rm, t):
    return np.array(
        [list(Rm[0]) + [t[0]], list(Rm[1]) + [t[1]], list(Rm[2]) + [t[2]], [0, 0, 0, 1]]
    )

# 计算相机的reprojection error
def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist,ShowPlot=False):
    total_error = 0
    num_points = 0
    errors = []
    for i in range(len(objpoints)):
        imgpoints_projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        imgpoints_projected = imgpoints_projected.reshape(-1, 1, 2)
        error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
        errors.append(error)
        total_error += error
        num_points += 1
    mean_error = total_error / num_points

    if ShowPlot:
        plt.figure()
        fig, ax = plt.subplots()
        fig.set_size_inches(6.14, 3.16)
        img_indices = range(1, len(errors) + 1)
        ax.bar(img_indices, errors)
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Reprojection Error')
        ax.set_title('Reprojection Error for Each Image')
        fig.tight_layout()
        # 保存图片
        plt.savefig(g['image_folder'] + '/ReprojectionError.jpg')
        # 关闭图形，避免阻塞
        plt.close(fig)
        # 非阻塞方式显示图形
        plt.ion()
        plt.show()
        plt.pause(1)  # 暂停1秒显示图形
        plt.close('all')
    return mean_error


# 1 find chessboard corners： 找每个标定板图片的角点
def find_chessboard_corners(images, pattern_size, ShowCorners=False):
    chessboard_corners = []
    IndexWithImg = []
    i = 0
    print("Finding corners...")
    for image in images:
        # 检查图像是否有效
        if image is None:
            print(f"Invalid image at index {i}, skipping...")
            i += 1
            continue
            
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size)
            
            if ret:
                # 角点检测成功
                chessboard_corners.append(corners)
                cv2.drawChessboardCorners(image, pattern_size, corners, ret)
                if ShowCorners:
                    plt.imshow(image)
                    plt.title("Detected corner in image: " + str(i))
                    plt.show()
                if not os.path.exists(g['image_folder'] + "/DetectedCorners"):
                    os.makedirs(g['image_folder'] + "/DetectedCorners")
                cv2.imwrite(g['image_folder'] + "/DetectedCorners/DetectedCorners" + str(i) + ".png", image)
                IndexWithImg.append(i)
                print(f"Successfully found corners in image {i}")
            else:
                print(f"No chessboard found in image: {i}")
            
        except Exception as e:
            print(f"Error processing image {i}: {str(e)}")
            
        i += 1
        
    if not chessboard_corners:
        raise ValueError("No valid chessboard corners found in any image!")
        
    print(f"Found corners in {len(chessboard_corners)} images out of {len(images)}")
    return chessboard_corners, IndexWithImg


# 2 calculate_intrinsics：计算相机的内参
def calculate_intrinsics(chessboard_corners, IndexWithImg, pattern_size, square_size, 
                         ImgSize, ShowProjectError=False):
    imgpoints = chessboard_corners
    objpoints = []
    for i in range(len(IndexWithImg)):
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
        objpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, ImgSize, None, None)
    print("The projection error from the calibration is: ",
          calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist,ShowProjectError))
    return mtx, dist   

# 3 calculate extrinsics： 计算每次拍照时相机的姿态，也就是相机的外参
def compute_camera_poses(chessboard_corners, pattern_size, square_size, intrinsic_matrix, dist):
    object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # Estimate the pose of the chessboard corners
    RTarget2Cam, TTarget2Cam = [], []
    i = 1
    for corners in chessboard_corners:
        _, rvec, tvec, _ = cv2.solvePnPRansac(object_points, corners, intrinsic_matrix, dist)
        i = 1 + i
        R, _ = cv2.Rodrigues(rvec)  
        RTarget2Cam.append(R)
        TTarget2Cam.append(tvec)
    return RTarget2Cam, TTarget2Cam    
    

def euler2rotation_matrix(x, y, z, roll, pitch, yaw):
    """将相机坐标系转换到Tracker坐标系
    Tracker坐标系（右手系）：
    - X轴：向右
    - Y轴：向上
    - Z轴：向后
    
    输入参数：
    x, y, z: Tracker位置（米）
    roll: 绕X轴旋转（弧度）
    pitch: 绕Y轴旋转（弧度）
    yaw: 绕Z轴旋转（弧度）
    
    返回：
    T: 4x4变换矩阵，从相机坐标系到Tracker坐标系的变换
    """
    # 注意：输入的角度已经是弧度制
    roll_rad = roll
    pitch_rad = pitch
    yaw_rad = yaw
    
    # 创建旋转矩阵
    # 绕X轴旋转（roll）
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll_rad), -np.sin(roll_rad)],
                   [0, np.sin(roll_rad), np.cos(roll_rad)]])
    
    # 绕Y轴旋转（pitch）               
    Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                   [0, 1, 0],
                   [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    
    # 绕Z轴旋转（yaw）               
    Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                   [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                   [0, 0, 1]])
    
    # Tracker使用的是roll-pitch-yaw顺序
    R = Rz @ Ry @ Rx
    
    # 创建变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    # 返回从相机坐标系到Tracker坐标系的变换
    return T

if __name__ == '__main__': 
    # 修改为你的数据目录
    image_folder = 'Calibration_data/20250210_163043'
    # image_folder="data3/20250121_100022"
    g['image_folder'] = image_folder
    
    # 根据实际标定板修改参数
    pattern_size=(7, 10)  # 标定板的格子数量
    square_size=12/1000  # 标定板格子的边长（米）
    
    # 读取所有标定板图片
    image_files = sorted(glob.glob(f'{image_folder}/*.png'))
    images = [cv2.imread(f) for f in image_files]
    images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]   
    
    # 读取Tracker位姿数据
    All_T_base2EE_list, All_T_EE2base_list = [], []
    transform_files = sorted(glob.glob(f'{image_folder}/*.pkl'))
    for fname in transform_files:
        with open(fname, 'rb') as fp:
            position = pickle.load(fp)   
        # 从pickle文件中读取Tracker的位置和欧拉角
        x = position["x"]
        y = position["y"]
        z = position["z"]
        roll = position["roll"]
        pitch = position["pitch"]
        yaw = position["yaw"]
        
        # 转换为变换矩阵
        trans_matrix = euler2rotation_matrix(x, y, z, roll, pitch, yaw)
        npz_matrix = np.array(trans_matrix)
        All_T_EE2base_list.append(npz_matrix)
        All_T_base2EE_list.append(np.linalg.inv(npz_matrix))
            
    
    # 1 find chessboard corners： 找标定板图片的角点
    chessboard_corners, IndexWithImg = find_chessboard_corners(images, pattern_size)
    
    # 2 calculate_intrinsics： 计算内参
    intrinsic_matrix, dist = calculate_intrinsics(chessboard_corners, IndexWithImg, pattern_size, 
                                            square_size, images[0].shape[:2], 
                                            ShowProjectError = True)  
    print('Intrinsicx Matrix is: ')
    print(intrinsic_matrix)
    
    # 3 calculate camera pose： 计算相机外参
    RTarget2Cam, TTarget2Cam = compute_camera_poses(chessboard_corners, pattern_size, 
                                                    square_size, intrinsic_matrix, dist)  
    
    T_base2EE_list = [All_T_base2EE_list[i] for i in IndexWithImg]  
    T_target2cam = [np.concatenate((R, T), axis=1) for R, T in zip(RTarget2Cam, TTarget2Cam)]
    
    for i in range(len(T_target2cam)):
        T_target2cam[i] = np.concatenate((T_target2cam[i], np.array([[0, 0, 0, 1]])), axis=0)
    
    T_cam2target = [np.linalg.inv(T) for T in T_target2cam] 
    R_cam2target = [T[:3, :3] for T in T_cam2target]
    R_target2cam = [T[:3, :3] for T in T_target2cam]
    
    R_vec_cam2target = [cv2.Rodrigues(R)[0] for R in R_cam2target]  # 旋转向量与旋转矩阵之间转换
    T_cam2target = [T[:3, 3] for T in T_cam2target]   
    t_target2cam = [T[:3, 3] for T in T_target2cam]   
    
    TEE2Base = [np.linalg.inv(T) for T in T_base2EE_list]   
    REE2Base = [T[:3, :3] for T in TEE2Base]
    R_vecEE2Base = [cv2.Rodrigues(R)[0] for R in REE2Base]  
    tEE2Base = [T[:3, 3] for T in TEE2Base]                
    
    # 求解最终的方程
    # Create folder to save final results
    if not os.path.exists(g['image_folder'] + "/FinalTransforms"):
        os.mkdir(g['image_folder'] + "/FinalTransforms")
        
    # solve hand-eye calibration
    for i in range(0, 5):
        print("----------------------Method:", i, '--------------------')
        R_cam2EE, t_cam2EE = cv2.calibrateHandEye(REE2Base, tEE2Base,
                                                            R_target2cam, t_target2cam,
                                                            method=i)              
        print("R_cam2EE:3*1")
        print(np.transpose(cv2.Rodrigues(R_cam2EE)[0]))
        print("t_cam2EE:")
        print(np.transpose(t_cam2EE))
        # Create 4x4 transfromation matrix
        T_cam2EE = np.concatenate((R_cam2EE, t_cam2EE), axis=1)
        T_cam2EE = np.concatenate((T_cam2EE, np.array([[0, 0, 0, 1]])), axis=0)
        
        #Save results in folder FinalTransforms
        np.savez(g['image_folder'] + "/FinalTransforms/T_cam2EE_Method_"+str(i)+".npz", T_cam2EE)

    
