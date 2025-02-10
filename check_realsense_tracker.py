import triad_openvr
import time
import sys
import numpy as np
import math

def check_realsense_tracker(target_serial='LHR-1CB8A619'):
    """检查RealSense Tracker的状态
    Args:
        target_serial: Tracker的序列号，默认为RealSense Tracker的序列号
    """
    try:
        # 初始化VR系统
        v = triad_openvr.triad_openvr()
        print("\n正在检查VR系统状态...")
        print("已发现的设备:")
        v.print_discovered_objects()
        
        # 查找目标Tracker
        target_found = False
        for deviceName, device in v.devices.items():
            serial = device.get_serial().decode('utf-8')
            if serial == target_serial:
                target_found = True
                print(f"\n找到目标Tracker:")
                print(f"- 设备名称: {deviceName}")
                print(f"- 序列号: {serial}")
                break
        
        if not target_found:
            print(f"\n警告: 未找到序列号为 {target_serial} 的Tracker")
            print("请检查:")
            print("1. Tracker是否已开启并正确配对")
            print("2. 序列号是否正确")
            print("3. SteamVR是否正常运行")
            return
            
        print("\n开始监测Tracker状态...")
        print("按Ctrl+C退出")
        
        # 监测Tracker状态
        lost_count = 0
        total_frames = 0
        last_print_time = time.time()
        
        while True:
            for deviceName, device in v.devices.items():
                serial = device.get_serial().decode('utf-8')
                if serial == target_serial:
                    try:
                        # 获取位姿数据
                        pose = device.get_pose_euler()
                        if pose is None:
                            lost_count += 1
                            print("\rTracker 追踪丢失...", end='', flush=True)
                            continue
                            
                        [x, y, z, roll, pitch, yaw] = pose
                        total_frames += 1
                        
                        # 每秒更新一次状态
                        current_time = time.time()
                        if current_time - last_print_time >= 1.0:
                            loss_rate = (lost_count / (total_frames + lost_count)) * 100 if total_frames + lost_count > 0 else 0
                            print(f"\r位置: x={x:.3f}, y={y:.3f}, z={z:.3f} | "
                                  f"旋转: roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f} | "
                                  f"丢失率: {loss_rate:.1f}%", end='', flush=True)
                            last_print_time = current_time
                            
                    except Exception as e:
                        print(f"\n读取位姿数据时出错: {e}")
                        lost_count += 1
                        
            time.sleep(0.01)  # 限制更新频率
            
    except KeyboardInterrupt:
        print("\n\n用户中断检查")
    except Exception as ex:
        if (type(ex).__name__ == 'OpenVRError' and 
            ex.args[0] == 'VRInitError_Init_HmdNotFoundPresenceFailed (error number 126)'):
            print('\n无法找到Tracker')
            print('请检查:')
            print('1. SteamVR是否运行?')
            print('2. Vive Tracker是否开启并与SteamVR配对?')
            print('3. Lighthouse基站是否开启且Tracker在视野范围内?')
        else:
            print(f"\n程序出错: {ex}")

if __name__ == '__main__':
    # 可以通过命令行参数指定不同的序列号
    serial = sys.argv[1] if len(sys.argv) > 1 else 'LHR-1CB8A619'
    check_realsense_tracker(serial) 