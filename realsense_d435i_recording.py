import pyrealsense2 as rs
import numpy as np
import cv2
import os

def configure_realsense():
    # Create a pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure the pipeline to stream depth and color
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline_profile = pipeline.start(config)

    # Get the depth sensor's depth scale
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {depth_scale}")

    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align

def main(output_dir="realsense_data", total_frames=150):
    os.makedirs(output_dir, exist_ok=True)

    pipeline, align = configure_realsense()

    try:
        for frame_id in range(total_frames):
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("Could not acquire depth or color frames.")
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Save depth image as 16-bit PNG
            depth_image_path = os.path.join(output_dir, f"depth_{frame_id}.png")
            cv2.imwrite(depth_image_path, depth_image)

            # Save color image
            color_image_path = os.path.join(output_dir, f"color_{frame_id}.jpg")
            cv2.imwrite(color_image_path, color_image)

            print(f"Saved frame {frame_id + 1}/{total_frames}")

            # Display images
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense', images)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 