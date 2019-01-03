import numpy as np 
import cv2
from sklearn.decomposition import PCA
import scipy.io as sio
import math
import struct
import pptk # only works with python 3.6

import torch
from torch.autograd import Variable
import argparse

from network import PointNet_Plus
from utils import group_points

'''
net: trained PyTorch model
focal_msra: focal length of camera 
sample_num: number of point cloud elements to sample
'''
class Predictor():
    def __init__(self, net, focal_msra, sample_num):
        self.net = net
        self.focal_msra = focal_msra #241.42
        self.sample_num = sample_num

    # image is a numpy array representing the depth image
    def make_prediction(self, image):
        # 2.1 Read Binary File
        # bin_arr has shape (4827,) - different bin files have different shapes
        with open(image, 'rb') as f:
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
        np.set_printoptions(threshold=np.nan)        
        # 2.2 Convert Depth to XYZ
        hand_3d = np.zeros((valid_pixel_num, 3))
        for i in range(0, bb_height):
          for j in range(0, bb_width):
            idx = j * bb_height + i
            hand_3d[idx][0] = -1 * (img_width / 2. - (j + bb_left)) * hand_depth[i][j] / self.focal_msra
            hand_3d[idx][1] = (img_height / 2. - (i + bb_top)) * hand_depth[i][j] / self.focal_msra
            hand_3d[idx][2] = hand_depth[i][j]

        # remove entries where all 3D values are equal to 0
        hand_points = hand_3d[(hand_3d[:,0] != 0) | (hand_3d[:,1] != 0) | (hand_3d[:,2] != 0)]
        
        # 2.3 Create OBB
        pca = PCA(n_components=3)
        pca.fit(hand_points)
        orig_coeff = np.transpose(pca.components_)
        coeff = np.transpose(pca.components_)
        if coeff[1][0] < 0 : coeff[:,0] = -1 * coeff[:,0]
        if coeff[2][2] < 0 : coeff[:,2] = -1 * coeff[:,2]
        coeff[:,1] = np.cross(coeff[:,2], coeff[:,0])
        hand_points_rotate = np.matmul(hand_points, coeff)
        
        # 2.4 Sampling
        NUM_POINTS = hand_points.shape[0]
        if NUM_POINTS < self.sample_num : # repeat points if the total number is less than sampled
          tmp = math.floor(self.sample_num/ NUM_POINTS)
          rand_ind = []
          for i in range(tmp):
            rand_ind.append([i for i in range(NUM_POINTS)])
          rand_ind = np.append(np.asarray(rand_ind), np.random.choice(NUM_POINTS, self.sample_num % NUM_POINTS))
        else:
          rand_ind = np.random.choice(NUM_POINTS, self.sample_num)
        
        hand_points_sampled = hand_points[rand_ind,:]
        hand_points_rotate_sampled = hand_points_rotate[rand_ind, :]
        
        # 2.5 Compute Surface Normal
        normals = pptk.estimate_normals(points=hand_points, k=30, r=np.inf)
        normals_sampled = normals[rand_ind]
        sensor_center = np.array([0, 0, 0])
        
        for k in range(self.sample_num):
          p1 = sensor_center - hand_points_sampled[k, :]
          angle = np.arctan2(np.linalg.norm(np.cross(p1, normals_sampled[k, :])), np.dot(p1, normals_sampled[k, :]))
          if angle > math.pi / 2 or angle < -1 * math.pi / 2: normals_sampled[k, :] = -1 * normals_sampled[k, :]
        
        normals_sampled_rotate = np.matmul(normals_sampled, coeff)
        
        # 2.6 Normalize Point Cloud
        x_min, x_max = np.min(hand_points_rotate[:, 0]), np.max(hand_points_rotate[:, 0])
        y_min, y_max = np.min(hand_points_rotate[:, 1]), np.max(hand_points_rotate[:, 1])
        z_min, z_max = np.min(hand_points_rotate[:, 2]), np.max(hand_points_rotate[:, 2])
        
        SCALE = 1.2
        bb3d_x_len = SCALE * (x_max - x_min)
        bb3d_y_len = SCALE * (y_max - y_min)
        bb3d_z_len = SCALE * (z_max - z_min)
        max_bb3d_len = max(bb3d_x_len, bb3d_y_len, bb3d_z_len)
        hand_points_normalized_sampled = hand_points_rotate_sampled / max_bb3d_len
        
        if NUM_POINTS < self.sample_num: offset = np.mean(hand_points_rotate, axis=0) / max_bb3d_len
        else : offset = np.mean(hand_points_normalized_sampled, axis=0)
        hand_points_normalized_sampled -= offset
        
        # 2.7 FPS Sampling
        # Takes in a point cloud and a sample number K and approximates the K furthest points in the PC
        def farthest_point_sampling_fast(point_cloud, sample_num):
          pc_num = point_cloud.shape[0]
          if pc_num < sample_num:
            sampled_idx = np.append(np.asarray([i for i in range(pc_num)]), np.random.randint(pc_num, size=sample_num-pc_num))
          else:
            sampled_idx = np.zeros(sample_num, dtype=np.int)
            sampled_idx[0] = np.random.randint(pc_num, size=1)[0]
            diff = point_cloud - point_cloud[sampled_idx[0], :]
            min_dist = np.sum(np.multiply(diff, diff), axis=1)
        
            for i in range(1, sample_num):
              sampled_idx[i] = np.argmax(min_dist)
              # update the minimum distance
              if i < sample_num:
                valid_idx = np.where(min_dist > 1e-8)[0]
                diff = point_cloud[valid_idx, :] - point_cloud[sampled_idx[i]]
                min_dist[valid_idx] = np.minimum(min_dist[valid_idx], np.sum(np.multiply(diff, diff), axis=1))
          
          return np.unique(sampled_idx)
        
        pc = np.hstack((hand_points_normalized_sampled, normals_sampled_rotate))
        
        # 1st level
        sampled_idx_l1 = farthest_point_sampling_fast(hand_points_normalized_sampled, 512)
        other_idx = np.setdiff1d(np.asarray([i for i in range(self.sample_num)]), sampled_idx_l1)
        new_idx = np.append(sampled_idx_l1, other_idx)
        pc = pc[new_idx, :]
        
        # 2nd level
        sampled_idx_l2 = farthest_point_sampling_fast(pc[0:512, 0:2], 128)
        other_idx = np.setdiff1d(np.asarray([i for i in range(512)]), sampled_idx_l2)
        new_idx = np.append(sampled_idx_l2, other_idx)
        pc[0:512, :] = pc[new_idx, :]
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--SAMPLE_NUM', type=int, default = 1024,  help='number of sample points')
        parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 6,  help='number of input point features')
        parser.add_argument('--knn_K', type=int, default = 64,  help='K for knn search')
        parser.add_argument('--sample_num_level1', type=int, default = 512,  help='number of first layer groups')
        parser.add_argument('--sample_num_level2', type=int, default = 128,  help='number of second layer groups')
        parser.add_argument('--ball_radius', type=float, default=0.015, help='square of radius for ball query in level 1')
        parser.add_argument('--ball_radius2', type=float, default=0.04, help='square of radius for ball query in level 2')

        opt = parser.parse_args()

        # Test HandPointNet After Preprocess
        inputs = torch.unsqueeze(torch.from_numpy(pc), 0)
        inputs_level1, inputs_level1_center = group_points(inputs.float(), opt)
        with torch.no_grad():
          inputs_level1, inputs_level1_center = Variable(inputs_level1), Variable(inputs_level1_center)
        estimation = self.net(inputs_level1, inputs_level1_center)
        
        '''
        estimation: torch.Tensor, [1, 42]
        '''
        outputs_xyz_mat = sio.loadmat('../preprocess/P0/PCA_mean_xyz.mat')
        outputs_xyz = torch.from_numpy(outputs_xyz_mat['PCA_mean_xyz'].astype(np.float32))
        PCA_coeff_mat = sio.loadmat('../preprocess/P0/PCA_coeff.mat')
        PCA_coeff = torch.from_numpy(PCA_coeff_mat['PCA_coeff'][:, 0:42].astype(np.float32)).transpose(0, 1)
        outputs_xyz = torch.addmm(outputs_xyz, estimation, PCA_coeff)
        
        # Unnormalize Output
        outputs_xyz = np.matmul(max_bb3d_len * (outputs_xyz.view(-1, 3).detach().numpy() + offset), np.linalg.inv(coeff)) 
        outputs_xy = outputs_xyz[:, 0:2]
        outputs_xy[:, 0] = outputs_xy[:, 0] + img_width / 2
        outputs_xy[:, 1] = -1 * outputs_xy[:, 1] + img_height / 2
        prediction_image = np.zeros((img_height, img_width, 3), np.uint8)
       
        for i in range(21):
            prediction_image[int(outputs_xy[i, 1]), int(outputs_xy[i, 0])] = [255, 0, 0]
        
        # crop the prediction image to match the bounding box
        prediction_image = prediction_image[bb_top:bb_bottom, bb_left:bb_right]
        
        # Plot Outputs
        hand_depth_rgb = np.zeros((hand_depth.shape[0], hand_depth.shape[1], 3))
        hand_depth_rgb[hand_depth > 1e-6] = [255, 255, 255]
        images = np.hstack((hand_depth_rgb, prediction_image))
        cv2.imshow('prediction', images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
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
    
    predictor = Predictor(net=net, focal_msra=241.42, sample_num=1024)
    predictor.make_prediction('../data/cvpr15_MSRAHandGestureDB/P0/1/000000_depth.bin')
