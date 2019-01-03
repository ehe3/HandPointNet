from predictor import Predictor
from network import PointNet_Plus
import argparse
import torch
import numpy as np
import struct
import cv2

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

net = PointNet_Plus(opt)
net.load_state_dict(torch.load('./results/P0/pretrained_net.pth', map_location='cpu'))
net.eval()

with open('../data/cvpr15_MSRAHandGestureDB/P0/1/000089_depth.bin', 'rb') as f:
    # variables are int32
    img_width = int.from_bytes(f.read(4), byteorder='little')
    img_height = int.from_bytes(f.read(4), byteorder='little')
    bb_left = int.from_bytes(f.read(4), byteorder='little')
    bb_top = int.from_bytes(f.read(4), byteorder='little')
    bb_right = int.from_bytes(f.read(4), byteorder='little')
    bb_bottom = int.from_bytes(f.read(4), byteorder='little')
    bb_width = bb_right - bb_left
    bb_height = bb_bottom - bb_top
    valid_pixel_num = bb_width * bb_height
    # variables in array are float32
    hand_depth = np.asarray(struct.unpack('{}f'.format(valid_pixel_num), f.read(4 * valid_pixel_num))).reshape((bb_height, bb_width))

predictor = Predictor(net=net, focal_msra=241.42, sample_num=1024)
prediction = predictor.make_prediction(hand_depth)

hand_depth_rgb = np.zeros((hand_depth.shape[0], hand_depth.shape[1], 3))
hand_depth_rgb[hand_depth > 1e-6] = [255, 255, 255]

np.set_printoptions(threshold=np.nan)

images = np.hstack((hand_depth_rgb, prediction))
cv2.namedWindow('test pred', cv2.WINDOW_AUTOSIZE)
cv2.imshow('depth image with prediction', images)
cv2.waitKey(0)
cv2.destroyAllWindows()
