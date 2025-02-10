import triad_openvr
import time
import sys
import numpy as np
import math
import pdb


def vive_tracker(target_serial=None):
    deviceCount = 0

    try:
        v = triad_openvr.triad_openvr()
    except Exception as ex:
        if (type(ex).__name__ == 'OpenVRError' and ex.args[0] == 'VRInitError_Init_HmdNotFoundPresenceFailed (error number 126)'):
            print('无法找到Tracker')
            print('请检查:')
            print('1. SteamVR是否运行?')
            print('2. Vive Tracker是否开启并与SteamVR配对?')
            print('3. Lighthouse基站是否开启且Tracker在视野范围内?\n\n')
        else:
            template = "发生异常 {0}. 参数:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
        print(ex.args)
        return

    # 打印所有设备信息
    v.print_discovered_objects()
    
    print("\n开始追踪...")
    while True:
        for deviceName, device in v.devices.items():
            # 获取设备序列号
            serial = device.get_serial().decode('utf-8')
            
            # 如果指定了目标序列号，则只打印该序列号的设备
            if target_serial is None or serial == target_serial:
                [x,y,z,roll,pitch,yaw] = device.get_pose_euler()
                y_rot = math.radians(pitch)
                print(f"设备: {deviceName} (序列号: {serial})")
                print(f"位置: x={x:.3f}, y={y:.3f}, z={z:.3f}")
                print(f"旋转: roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}\n")

if __name__ == '__main__':
    try:
        # 可以在这里指定目标Tracker的序列号
        # 例如: vive_tracker("LHR-1CB8A619")
        # 不指定序列号则打印所有Tracker数据
        vive_tracker(target_serial="LHB-530B9203")
    except Exception as e:
        print(e)
