import socket
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class HandJoint:
    name: str
    position: np.ndarray  # [x, y, z]
    quaternion: np.ndarray  # [x, y, z, w]

@dataclass
class Finger:
    name: str
    joints: List[HandJoint]  # 3个关节：掌指关节、近节指关节、远节指关节

@dataclass
class Hand:
    hand_type: str  # "左手" or "右手"
    wrist_position: np.ndarray  # [x, y, z]
    wrist_quaternion: np.ndarray  # [x, y, z, w]
    fingers: List[Finger]  # 5个手指

@dataclass
class GloveFrame:
    device: str  # "Glove1"
    timestamp: str
    frame_number: int
    gesture: str
    relative: int
    subpackage: str
    hands: List[Hand]
    error: Optional[str] = None

def parse_glove_data(data_str: str) -> GloveFrame:
    """解析手套数据"""
    try:
        # 分割数据
        parts = data_str.split(',')
        header = parts[0].split()
        
        # 解析头部信息
        if len(header) < 10 or header[0] != "Glove1":
            raise ValueError(f"无效的数据格式: {data_str[:100]}")
            
        # 基本信息
        device = header[0]  # Glove1
        timestamp = f"{header[2]} {header[3]}"  # 时间戳
        
        # 查找关键字的位置
        gesture_idx = header.index("gesture") + 1
        relative_idx = gesture_idx + 1
        fn_idx = header.index("fn") + 1
        subpackage_idx = header.index("subpackage") + 1
        
        gesture = int(header[gesture_idx])  # 手势
        relative = int(header[relative_idx])  # relative值
        frame_number = int(header[fn_idx])  # 帧号
        subpackage = header[subpackage_idx]  # 子包信息
        
        # 解析数值数据
        values = [float(x) for x in parts[1:] if x.strip()]
        if len(values) < 224:  # 检查数据长度
            raise ValueError(f"数据长度不足: {len(values)}")
            
        # 解析手部数据
        hands = []
        finger_names = ['拇指', '食指', '中指', '无名指', '小指']
        joint_names = ['掌指关节', '近节指关节', '远节指关节']
        
        for hand_idx, hand_type in enumerate(['左手', '右手']):
            offset = hand_idx * 112  # 每只手112个浮点数
            
            # 手腕数据
            wrist_pos = np.array(values[offset:offset+3])
            wrist_quat = np.array(values[offset+3:offset+7])
            
            # 解析手指数据
            fingers = []
            current_idx = offset + 7  # 跳过手腕数据
            
            for finger_name in finger_names:
                joints = []
                for joint_name in joint_names:
                    pos = np.array(values[current_idx:current_idx+3])
                    quat = np.array(values[current_idx+3:current_idx+7])
                    joints.append(HandJoint(
                        name=joint_name,
                        position=pos,
                        quaternion=quat
                    ))
                    current_idx += 7  # 每个关节7个值
                    
                fingers.append(Finger(
                    name=finger_name,
                    joints=joints
                ))
            
            hands.append(Hand(
                hand_type=hand_type,
                wrist_position=wrist_pos,
                wrist_quaternion=wrist_quat,
                fingers=fingers
            ))
        
        return GloveFrame(
            device=device,
            timestamp=timestamp,
            frame_number=frame_number,
            gesture=gesture,
            relative=relative,
            subpackage=subpackage,
            hands=hands
        )
        
    except Exception as e:
        return GloveFrame(
            device='unknown',
            timestamp=str(datetime.now()),
            frame_number=-1,
            gesture=-1,
            relative=-1,
            subpackage='error',
            hands=[],
            error=str(e)
        )

def print_frame(frame: GloveFrame):
    """格式化打印帧数据"""
    print("\n" + "="*50)
    print(f"设备: {frame.device}")
    print(f"时间戳: {frame.timestamp}")
    print(f"帧号: {frame.frame_number}")
    print(f"手势: {frame.gesture}")
    print(f"相对值: {frame.relative}")
    print(f"子包: {frame.subpackage}")
    
    if frame.error:
        print(f"错误: {frame.error}")
        return
        
    for hand in frame.hands:
        print(f"\n{hand.hand_type}:")
        print(f"手腕位置: [{hand.wrist_position[0]:.3f}, {hand.wrist_position[1]:.3f}, {hand.wrist_position[2]:.3f}]")
        print(f"手腕旋转: [{hand.wrist_quaternion[0]:.3f}, {hand.wrist_quaternion[1]:.3f}, "
              f"{hand.wrist_quaternion[2]:.3f}, {hand.wrist_quaternion[3]:.3f}]")
        
        for finger in hand.fingers:
            print(f"\n  {finger.name}:")
            for joint in finger.joints:
                print(f"    {joint.name}:")
                print(f"      位置: [{joint.position[0]:.3f}, {joint.position[1]:.3f}, {joint.position[2]:.3f}]")
                print(f"      旋转: [{joint.quaternion[0]:.3f}, {joint.quaternion[1]:.3f}, "
                      f"{joint.quaternion[2]:.3f}, {joint.quaternion[3]:.3f}]")

def main():
    # 创建UDP套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16384)
    
    # 绑定地址和端口
    server_address = ('localhost', 4396)
    sock.bind(server_address)
    print(f"UDP服务器正在运行，监听地址：{server_address}")

    try:
        while True:
            data, address = sock.recvfrom(4096)
            data_str = data.decode()
            
            # 解析并打印数据
            frame = parse_glove_data(data_str)
            print_frame(frame)
            
            time.sleep(0.5)  # 控制打印频率
            
    except KeyboardInterrupt:
        print("\n服务器已停止")
    finally:
        sock.close()

if __name__ == "__main__":
    main()