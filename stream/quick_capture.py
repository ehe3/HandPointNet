import sys

sys.path.append('/usr/local/lib')

import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

counter = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        img_height, img_width = depth_image.shape
        crop_height, crop_width = 200, 200
        img_y_upper, img_y_lower, img_x_upper, img_x_lower = (img_height + crop_height) // 2, (img_height - crop_height) // 2, (img_width + crop_width) // 2, (img_width - crop_width) // 2
        depth_image = depth_image[img_y_lower:img_y_upper, img_x_lower:img_x_upper]

        hand_depth_rgb = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))
        hand_depth_rgb[depth_image > 1e-6] = [255, 255, 255]
        hand_depth_rgb[depth_image > 1./ depth_scale] = [0, 0, 0]

        if counter % 10 == 0: 
            np.save('./samples/sample{:06d}.npy'.format(counter // 10), depth_image)
            cv2.imwrite('./samples/sample{:06d}.jpg'.format(counter // 10), depth_image)

        counter = counter + 1
        
        cv2.namedWindow('images', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('images', hand_depth_rgb)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally: 
    pipeline.stop()
