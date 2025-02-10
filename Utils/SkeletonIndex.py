from enum import Enum, auto

class KinemHumanHandsSkeleton32Index(Enum):
    # 右手
    RightHand = 0
    RightHandThumb1 = 1
    RightHandThumb2 = 2
    RightHandThumb3 = 3
    RightHandIndex1 = 4
    RightHandIndex2 = 5
    RightHandIndex3 = 6
    RightHandMiddle1 = 7
    RightHandMiddle2 = 8
    RightHandMiddle3 = 9
    RightHandRing1 = 10
    RightHandRing2 = 11
    RightHandRing3 = 12
    RightHandPinky1 = 13
    RightHandPinky2 = 14
    RightHandPinky3 = 15

    # 左手
    LeftHand = 16
    LeftHandThumb1 = 17
    LeftHandThumb2 = 18
    LeftHandThumb3 = 19
    LeftHandIndex1 = 20
    LeftHandIndex2 = 21
    LeftHandIndex3 = 22
    LeftHandMiddle1 = 23
    LeftHandMiddle2 = 24
    LeftHandMiddle3 = 25
    LeftHandRing1 = 26
    LeftHandRing2 = 27
    LeftHandRing3 = 28
    LeftHandPinky1 = 29
    LeftHandPinky2 = 30
    LeftHandPinky3 = 31

# 骨骼父子关系映射
JOINT_EULER_FATHER = [
    -1,  # RightHand
    0,   # RightHandThumb1 -> RightHand
    1,   # RightHandThumb2 -> RightHandThumb1
    2,   # RightHandThumb3 -> RightHandThumb2
    0,   # RightHandIndex1 -> RightHand
    4,   # RightHandIndex2 -> RightHandIndex1
    5,   # RightHandIndex3 -> RightHandIndex2
    0,   # RightHandMiddle1 -> RightHand
    7,   # RightHandMiddle2 -> RightHandMiddle1
    8,   # RightHandMiddle3 -> RightHandMiddle2
    0,   # RightHandRing1 -> RightHand
    10,  # RightHandRing2 -> RightHandRing1
    11,  # RightHandRing3 -> RightHandRing2
    0,   # RightHandPinky1 -> RightHand
    13,  # RightHandPinky2 -> RightHandPinky1
    14,  # RightHandPinky3 -> RightHandPinky2
    
    -1,  # LeftHand 
    16,  # LeftHandThumb1 -> LeftHand
    17,  # LeftHandThumb2 -> LeftHandThumb1
    18,  # LeftHandThumb3 -> LeftHandThumb2
    16,  # LeftHandIndex1 -> LeftHand
    20,  # LeftHandIndex2 -> LeftHandIndex1
    21,  # LeftHandIndex3 -> LeftHandIndex2
    16,  # LeftHandMiddle1 -> LeftHand
    23,  # LeftHandMiddle2 -> LeftHandMiddle1
    24,  # LeftHandMiddle3 -> LeftHandMiddle2
    16,  # LeftHandRing1 -> LeftHand
    26,  # LeftHandRing2 -> LeftHandRing1
    27,  # LeftHandRing3 -> LeftHandRing2
    16,  # LeftHandPinky1 -> LeftHand
    29,  # LeftHandPinky2 -> LeftHandPinky1
    30,  # LeftHandPinky3 -> LeftHandPinky2
]

# 不同骨骼命名映射
STEAMVR_SKELETON_NAMES = [
    "RightHand",
    "finger_thumb_0_r", "finger_thumb_1_r", "finger_thumb_2_r",
    "finger_index_0_r", "finger_index_1_r", "finger_index_2_r",
    "finger_middle_0_r", "finger_middle_1_r", "finger_middle_2_r", 
    "finger_ring_0_r", "finger_ring_1_r", "finger_ring_2_r",
    "finger_pinky_0_r", "finger_pinky_1_r", "finger_pinky_2_r",
    
    "LeftHand",
    "finger_thumb_0_l", "finger_thumb_1_l", "finger_thumb_2_l",
    "finger_index_0_l", "finger_index_1_l", "finger_index_2_l",
    "finger_middle_0_l", "finger_middle_1_l", "finger_middle_2_l",
    "finger_ring_0_l", "finger_ring_1_l", "finger_ring_2_l",
    "finger_pinky_0_l", "finger_pinky_1_l", "finger_pinky_2_l"
]

# 确保这些变量都被导出
__all__ = ['KinemHumanHandsSkeleton32Index', 'JOINT_EULER_FATHER']


