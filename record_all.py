from Recorder.MultiThreadRecorder import MultiThreadRecorder

def main():
    # 指定Tracker序列号
    tracker_serials = {
        'realsense_tracker': 'LHR-1CB8A619',
        'left_tracker': 'LHR-530B9203',      # 左手 Tracker
        'right_tracker': 'LHR-2132E0A8',      # 右手 Tracker
       
    }

    # 指定触觉串口配置
    serial_ports = { 
        'left_tactile': 'COM6',    # 左手触觉传感器串口
        'right_tactile': 'COM4'     # 右手触觉传感器串口
    }
    glove_port = 2211
    recorder = None
    folder_name="newTest"
    try:
        recorder = MultiThreadRecorder(
            tracker_serials=tracker_serials,
            serial_ports=serial_ports,
            glove_port=glove_port,
            folder_name=folder_name
        )
        recorder.run()
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        if recorder:
            recorder.cleanup()
        print("程序结束")
 
if __name__ == '__main__':
    main()