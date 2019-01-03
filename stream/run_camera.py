# Explicity append path on Mojave for pyrealsense2
import sys
import os

sys.path.append('/usr/local/lib')

# RealSense python wrapper
import pyrealsense2 as rs   

# Pytorch 
import torch
import numpy as np
import cv2

# PointNet specifics
from network import PointNet_Plus
from predictor import Predictor

# Other utilities
import argparse

# Create parser to intialize network
parser = argparse.ArgumentParser()
parser.add_argument('--SAMPLE_NUM', type=int, default = 1024,  help='number of sample points')
parser.add_argument('--JOINT_NUM', type=int, default = 21,  help='number of joints')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 6,  help='number of input point features')
parser.add_argument('--PCA_SZ', type=int, default = 42,  help='number of PCA components')
parser.add_argument('--knn_K', type=int, default = 64,  help='K for knn search')
parser.add_argument('--sample_num_level1', type=int, default = 512,  help='number of first layer groups')
parser.add_argument('--sample_num_level2', type=int, default = 128,  help='number of second layer groups')
parser.add_argument('--ball_radius', type=float, default=0.015, help='square of radius for ball query in level 1')
parser.add_argument('--ball_radius2', type=float, default=0.04, help='square of radius for ball query in level 2')

opt = parser.parse_args()

# Load pretrained model
net = PointNet_Plus(opt)
net.load_state_dict(torch.load('./results/P0/pretrained_net.pth', map_location='cpu'))
net.eval()

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
# Create predictor to handle prediction outputs
predictor = Predictor(net=net, fx = intrinsics.fx, fy = intrinsics.fy, sample_num=1024, depth_scale=depth_scale)

# Crop size (middle crop)
img_height, img_width = 200, 200

# Streaming loop
try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # See that the frame is valid
        if not depth_frame:
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Perform crop before the prediction
        curr_height, curr_width = depth_image.shape
        img_y_lower, img_y_upper, img_x_lower, img_x_upper = (curr_height - img_height) // 2, (curr_height + img_height) // 2, (curr_width - img_width) // 2, (curr_width + img_width) // 2
        depth_image = depth_image[img_y_lower:img_y_upper, img_x_lower:img_x_upper]
        
        outputs_xy = predictor.make_prediction(depth_image)
        
        # Render images
        hand_depth_rgb = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))
        hand_depth_rgb[depth_image > 1e-6] = [255, 255, 255]
        hand_depth_rgb[depth_image > 1./ depth_scale] = [0, 0, 0]

        for i in range(21):
            joint_y, joint_x = int(outputs_xy[i, 1]), int(outputs_xy[i, 0])
            if joint_x < 1 or joint_x > img_width - 2 or joint_y < 1 or joint_y > img_height - 2: continue
            
            # Plot center point and the points around it
            hand_depth_rgb[joint_y, joint_x] = [0, 0, 255]
            hand_depth_rgb[joint_y, joint_x + 1] = [0, 0, 255]
            hand_depth_rgb[joint_y, joint_x - 1] = [0, 0, 255]
            hand_depth_rgb[joint_y + 1, joint_x] = [0, 0, 255]
            hand_depth_rgb[joint_y + 1, joint_x + 1] = [0, 0, 255]
            hand_depth_rgb[joint_y + 1, joint_x - 1] = [0, 0, 255]
            hand_depth_rgb[joint_y - 1, joint_x] = [0, 0, 255]
            hand_depth_rgb[joint_y - 1, joint_x + 1] = [0, 0, 255]
            hand_depth_rgb[joint_y - 1, joint_x - 1] = [0, 0, 255]

        cv2.namedWindow('Predicting Hand Poses', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth Image with Prediction', hand_depth_rgb)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()

