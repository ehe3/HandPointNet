import numpy as np 
from sklearn.decomposition import PCA
import math
import struct
import pptk
import cv2
import os

# np.set_printoptions(threshold=np.nan)

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

        for i in range(sample_num):
            sampled_idx[i] = np.argmax(min_dist)
            # update the minimum distance
            if i < sample_num:
                valid_idx = np.where(min_dist > 1e-8)[0]
                diff = point_cloud[valid_idx, :] - point_cloud[sampled_idx[i]]
                min_dist[valid_idx] = np.minimum(min_dist[valid_idx], np.sum(np.multiply(diff, diff), axis=1))
  
    return np.unique(sampled_idx)

def plot(points, gt):
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(points[:,0], points[:,1], points[:,2], c='b')

    # plot 3d gt
    ax.scatter(gt[:,0], gt[:,1], gt[:,2], c='r')
    ax.view_init(90,-90)
    

def preprocess(depth_map_path, gt_path, add_noise=False):
    # 1. Read depth files 
    img_width = 256
    img_height = 256
    bb_left = 0
    bb_top = 0
    bb_right = img_width
    bb_bottom = img_height
    bb_width = bb_right - bb_left
    bb_height = bb_bottom - bb_top
    valid_pixel_num = bb_width * bb_height
    # variables in array are float32
    # hand_depth = np.asarray(struct.unpack('{}f'.format(valid_pixel_num), f.read(4 * valid_pixel_num))).reshape((bb_height, bb_width))
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
    depth_map = depth_map[:,:,0] #rgb channels have same information. take one

    # create synthetic noise
    if add_noise: 
        mean = np.random.uniform(-0.05, 0.05)
        std = np.random.uniform(0.005,0.12)
        noise = np.random.normal(0,std, depth_map.shape)
        depth_map += noise

    # 2. Convert Depth to XYZ
    PIXEL_SIZE_mm = 1/8
    FOCAL_BLENDER_mm = 35 # focal length
    MAX_DEPTH = 20 # z clip
    foot_3d = np.zeros((valid_pixel_num, 3))
    for i in range(0, bb_height):
      for j in range(0, bb_width):
        idx = j * bb_height + i
        z = depth_map[i][j]
        foot_3d[idx][0] = -1 * (img_width / 2. - (j + bb_left)) * PIXEL_SIZE_mm / FOCAL_BLENDER_mm * z
        foot_3d[idx][1] = (img_height / 2. - (i + bb_top)) * PIXEL_SIZE_mm / FOCAL_BLENDER_mm * z 
        foot_3d[idx][2] = z

    # remove entries where all 3D values are equal to 0
    foot_points = foot_3d[(foot_3d[:,0] != 0) | (foot_3d[:,1] != 0) | (foot_3d[:,2] != 0)]
    # remove entries where z value is above MAX_Z
    foot_points = foot_points[abs(foot_points[:,2])<MAX_DEPTH]
    # plot(foot_points)

    # 3. Create OBB
    pca = PCA(n_components=3)
    pca.fit(foot_points)
    coeff = np.transpose(pca.components_)

    # Check if this rotation is what we want
    if coeff[1][0] < 0 : coeff[:,0] = -1 * coeff[:,0]
    if coeff[2][2] < 0 : coeff[:,2] = -1 * coeff[:,2]
    coeff[:,1] = np.cross(coeff[:,2], coeff[:,0])
    foot_points_rotate = np.matmul(foot_points, coeff)


    # 4. Sampling
    NUM_POINTS = foot_points.shape[0]
    SAMPLE_NUM = 1024

    if NUM_POINTS < SAMPLE_NUM: # repeat points if the total number is less than sampled
      tmp = math.floor(SAMPLE_NUM / NUM_POINTS)
      rand_ind = []
      for i in range(tmp):
        rand_ind.append([i for i in range(NUM_POINTS)])
      rand_ind = np.append(np.asarray(rand_ind), np.random.choice(NUM_POINTS, SAMPLE_NUM % NUM_POINTS))
    else:
      rand_ind = np.random.choice(NUM_POINTS, SAMPLE_NUM)

    foot_points_sampled = foot_points[rand_ind,:]
    foot_points_rotate_sampled = foot_points_rotate[rand_ind, :]


    # 5. Compute Surface Normal
    normals = pptk.estimate_normals(points=foot_points, k=30, r=np.inf, verbose=False)
    normals_sampled = normals[rand_ind]
    sensor_center = np.array([0, 0, 0])

    for k in range(SAMPLE_NUM):
      p1 = sensor_center - foot_points_sampled[k, :]
      angle = np.arctan2(np.linalg.norm(np.cross(p1, normals_sampled[k, :])), np.dot(p1, normals_sampled[k, :]))
      if angle > math.pi / 2 or angle < -1 * math.pi / 2: normals_sampled[k, :] = -1 * normals_sampled[k, :]

    normals_sampled_rotate = np.matmul(normals_sampled, coeff)

    # 6. Normalize Point Cloud
    x_min, x_max = np.min(foot_points_rotate[:, 0]), np.max(foot_points_rotate[:, 0])
    y_min, y_max = np.min(foot_points_rotate[:, 1]), np.max(foot_points_rotate[:, 1])
    z_min, z_max = np.min(foot_points_rotate[:, 2]), np.max(foot_points_rotate[:, 2])

    SCALE = 1.2
    bb3d_x_len = SCALE * (x_max - x_min)
    bb3d_y_len = SCALE * (y_max - y_min)
    bb3d_z_len = SCALE * (z_max - z_min)
    max_bb3d_len = max(bb3d_x_len, bb3d_y_len, bb3d_z_len)
    foot_points_normalized_sampled = foot_points_rotate_sampled / max_bb3d_len

    if NUM_POINTS < SAMPLE_NUM: offset = np.mean(foot_points_rotate, axis=0) / max_bb3d_len
    else: offset = np.mean(foot_points_normalized_sampled, axis=0)

    foot_points_normalized_sampled -= offset

    # 7. FPS Sampling
    pc = np.hstack((foot_points_normalized_sampled, normals_sampled_rotate))

    # 1st level
    sampled_idx_l1 = farthest_point_sampling_fast(foot_points_normalized_sampled, 512)
    other_idx = np.setdiff1d(np.asarray([i for i in range(SAMPLE_NUM)]), sampled_idx_l1)
    new_idx = np.append(sampled_idx_l1, other_idx)
    pc = pc[new_idx, :]

    # 2nd level
    sampled_idx_l2 = farthest_point_sampling_fast(pc[0:512, 0:2], 128)
    other_idx = np.setdiff1d(np.asarray([i for i in range(512)]), sampled_idx_l2)
    new_idx = np.append(sampled_idx_l2, other_idx)
    pc[0:512, :] = pc[new_idx, :]

    # 8. Ground Truth
    with open(gt_path, 'rt') as f:
        jnt_xyz = f.read().split(',')
        jnt_xyz = np.asarray(jnt_xyz).astype('float').reshape((-1,3))
    # !!!! make z value positive !!!!
    jnt_xyz[:,2] = -jnt_xyz[:,2] 

    # normalize gt
    jnt_xyz_normalized = np.matmul(jnt_xyz, coeff) / max_bb3d_len
    jnt_xyz_normalized -= offset

    return pc, coeff, max_bb3d_len, offset, jnt_xyz_normalized


if __name__ == "__main__":
    data_root = '/Volumes/AndrewJayZhou/Dataset/FootPoseDepth/syn/v3_40k'
    save_root = '/Volumes/AndrewJayZhou/Dev/HandPointNet/data/v3_40k'
  
    depth_ext = '.exr'
    jnt_gt_ext = '_joint_pos.txt'

    sample_num = 1024
    joint_num  = 13

    for dir, _, fnames in sorted(os.walk(data_root)):
        if dir == data_root:
            continue

        # save directory
        set_folder = os.path.basename(dir)
        save_dir   = os.path.join(save_root, set_folder)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # filter fnames. keep only depth map files
        fnames = [f for f in fnames if depth_ext in f]
        frame_num = len(fnames)

        # set specific arrays initialization
        point_cloud_FPS = np.zeros((frame_num, sample_num, 6))
        volume_rotate = np.zeros((frame_num,3,3))
        volume_length = np.zeros((frame_num,1))
        volume_offset = np.zeros((frame_num,3))
        volume_gt_xyz = np.zeros((frame_num,joint_num,3))
      
        index = 0
        for fname in fnames:
            if index % 10 == 0:
                print('processing set {} image {} ... '.format(set_folder, index))
            # preprocess
            depth_map_path = os.path.join(dir, fname)
            gt_path        = depth_map_path.replace(depth_ext, jnt_gt_ext)
            pc, coeff, max_bb3d_len, offset, jnt_xyz_normalized = preprocess(depth_map_path, gt_path, add_noise=True)

            # store image information in set specific arrays
            point_cloud_FPS[index] = pc
            volume_rotate[index]   = coeff
            volume_length[index]   = max_bb3d_len
            volume_offset[index]   = offset
            volume_gt_xyz[index]   = jnt_xyz_normalized

            index+=1

        # save set specific arrays 
        np.save(os.path.join(save_dir, 'Point_Cloud_FPS.npy'), point_cloud_FPS)
        np.save(os.path.join(save_dir, 'Volume_rotate.npy'), volume_rotate)
        np.save(os.path.join(save_dir, 'Volume_length.npy'), volume_length)
        np.save(os.path.join(save_dir, 'Volume_offset.npy'), volume_offset)
        np.save(os.path.join(save_dir, 'Volume_GT_XYZ.npy'), volume_gt_xyz)
          


























