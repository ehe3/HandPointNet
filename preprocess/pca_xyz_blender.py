import numpy as np 
from sklearn.decomposition import PCA
import math
import struct
import pptk
import cv2
import os

root_dir  = '/Volumes/AndrewJayZhou/Dev/HandPointNet/data/blender_v2'
joint_num = 13

gt_xyz_file = 'Volume_GT_XYZ.npy'
jnt_xyz_all = []
for dir, _, fnames in sorted(os.walk(root_dir)):

    for fname in fnames:
        if gt_xyz_file not in fname:
            continue

        # load all gt coordinates into MxN array where M = number of dataset images, and N = joint_num * 3
        jnt_xyz = np.load(os.path.join(dir, gt_xyz_file))  
        jnt_xyz = jnt_xyz.reshape(jnt_xyz.shape[0], joint_num * 3)
        if len(jnt_xyz_all) == 0:
            jnt_xyz_all = jnt_xyz
        else:
            jnt_xyz_all = np.concatenate((jnt_xyz_all, jnt_xyz), axis=0)


pca = PCA()
pca.fit(jnt_xyz_all)
coeff    = np.transpose(pca.components_)
latent   = np.transpose(pca.explained_variance_)
xyz_mean = np.transpose(pca.mean_)

np.save(os.path.join(root_dir, 'PCA_coeff.npy'), coeff)
np.save(os.path.join(root_dir, 'PCA_latent_weight.npy'), latent)
np.save(os.path.join(root_dir, 'PCA_mean_xyz.npy'), xyz_mean)








































