import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import scipy.io as sio
import pdb

SAMPLE_NUM = 1024
JOINT_NUM = 13

class FootPointDataset(data.Dataset):
    def __init__(self, root_path, opt, train=True):
        self.device = torch.device('cuda:{}'.format(opt.main_gpu)) if opt.ngpu>0 else torch.device('cpu')
        self.root_path = root_path
        self.train = train
        self.test_suffix = opt.test_suffix

        self.PCA_SZ = opt.PCA_SZ
        self.SAMPLE_NUM = opt.SAMPLE_NUM
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
        self.JOINT_NUM = opt.JOINT_NUM

        self.total_frame_num = self.__total_frmae_num()

        self.point_clouds = np.empty(shape=[self.total_frame_num, self.SAMPLE_NUM, self.INPUT_FEATURE_NUM],
                                     dtype=np.float32)
        self.volume_length = np.empty(shape=[self.total_frame_num, 1], dtype=np.float32)
        self.gt_xyz = np.empty(shape=[self.total_frame_num, self.JOINT_NUM, 3], dtype=np.float32)

        self.start_index = 0
        self.end_index = 0

        for cur_data_dir, _, _ in sorted(os.walk(self.root_path)):
            if cur_data_dir == self.root_path:
                continue
            
            if self.test_suffix in cur_data_dir and not self.train: # Testing
                print("Testing: " + cur_data_dir)
                self.__loaddata(cur_data_dir)
            elif not self.test_suffix in cur_data_dir and self.train: # Training
                print("Training: " + cur_data_dir)
                self.__loaddata(cur_data_dir)
            else:
                continue

        self.point_clouds = torch.from_numpy(self.point_clouds)
        self.volume_length = torch.from_numpy(self.volume_length)
        self.gt_xyz = torch.from_numpy(self.gt_xyz)

        self.gt_xyz = self.gt_xyz.view(self.total_frame_num, -1)

        # load PCA coeff
        PCA_data_path = self.root_path
        PCA_coeff_mat = np.load(os.path.join(PCA_data_path, 'PCA_coeff.npy'))

        self.PCA_coeff = torch.from_numpy(PCA_coeff_mat[:, 0:self.PCA_SZ].astype(np.float32))
        PCA_mean_mat = np.load(os.path.join(PCA_data_path, 'PCA_mean_xyz.npy'))
        self.PCA_mean = torch.from_numpy(PCA_mean_mat.astype(np.float32))

        tmp = self.PCA_mean.expand(self.total_frame_num, self.JOINT_NUM * 3)
        tmp_demean = self.gt_xyz - tmp
        self.gt_pca = torch.mm(tmp_demean, self.PCA_coeff)

        self.PCA_coeff = self.PCA_coeff.transpose(0, 1).to(self.device)
        self.PCA_mean = self.PCA_mean.view(1,-1).to(self.device)

    def __getitem__(self, index):
        return self.point_clouds[index, :, :], self.volume_length[index], self.gt_pca[index, :], self.gt_xyz[index, :]

    def __len__(self):
        return self.point_clouds.size(0)

    def __loaddata(self, data_dir):
        point_cloud   = np.load(os.path.join(data_dir, 'Point_Cloud_FPS.npy'))
        gt_data       = np.load(os.path.join(data_dir, "Volume_GT_XYZ.npy"))
        volume_length = np.load(os.path.join(data_dir, "Volume_length.npy"))

        self.start_index = self.end_index + 1
        self.end_index = self.end_index + len(point_cloud)

        self.point_clouds[(self.start_index - 1):self.end_index, :, :] = point_cloud.astype(np.float32)
        self.gt_xyz[(self.start_index - 1):self.end_index, :, :] = gt_data.astype(np.float32)
        self.volume_length[(self.start_index - 1):self.end_index, :] = volume_length.astype(np.float32)

    def __total_frmae_num(self):
        frame_num = 0

        for cur_data_dir, _, _ in sorted(os.walk(self.root_path)):
            if cur_data_dir == self.root_path:
                continue
            
            if self.test_suffix in cur_data_dir and not self.train: # Testing
                frame_num = frame_num + self.__get_frmae_num(cur_data_dir)
            elif not self.test_suffix in cur_data_dir and self.train: # Training
                frame_num = frame_num + self.__get_frmae_num(cur_data_dir)
            else:
                continue

        return frame_num

    def __get_frmae_num(self, data_dir):
        volume_length = np.load(os.path.join(data_dir, "Volume_length.npy"))
        return len(volume_length)