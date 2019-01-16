import matplotlib as mpl 
mpl.use('TkAgg')

import sys
sys.path.append('../train_eval')
sys.path.append('../stream')

import os
import argparse
from tqdm import tqdm

import torch
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from network import PointNet_Plus
from predictor import Predictor

num_images = 545
sample_dir = './samples'
depth_scale = 0.0010000000474974513
clipping_distance_in_meters = 1.0
playback_fps = 10.
cloud_file_name = os.path.join(sample_dir, 'clouds{}.npy'.format(num_images))
predictions_file_name = os.path.join(sample_dir, 'predictions{}.npy'.format(num_images))

# Set up network for predictions
parser = argparse.ArgumentParser()
parser.add_argument('--SAMPLE_NUM', type=int, default = 1024,  help='number of sample points')
parser.add_argument('--JOINT_NUM', type=int, default = 13,  help='number of joints')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 6,  help='number of input point features')
parser.add_argument('--PCA_SZ', type=int, default = 26,  help='number of PCA components')
parser.add_argument('--knn_K', type=int, default = 64,  help='K for knn search')
parser.add_argument('--sample_num_level1', type=int, default = 512,  help='number of first layer groups')
parser.add_argument('--sample_num_level2', type=int, default = 128,  help='number of second layer groups')
parser.add_argument('--ball_radius', type=float, default=0.015, help='square of radius for ball query in level 1')
parser.add_argument('--ball_radius2', type=float, default=0.04, help='square of radius for ball query in level 2')

opt = parser.parse_args()

device = torch.device('cpu')
net = PointNet_Plus(opt).to(device)
net.netR_1 = torch.nn.DataParallel(net.netR_1, range(4))
net.netR_2 = torch.nn.DataParallel(net.netR_2, range(4))
net.netR_3 = torch.nn.DataParallel(net.netR_3, range(4))
net.load_state_dict(torch.load('../train_eval/results/P10/netR_45.pth', map_location='cpu'))
net.eval()

fx, fy, depth_scale = 320.8097839355469, 320.8097839355469, 0.0010000000474974513

predictor = Predictor(net=net, fx=fx, fy=fy, sample_num=1024, depth_scale=depth_scale)

if os.path.isfile(cloud_file_name) and os.path.isfile(predictions_file_name):
    clouds = np.load(cloud_file_name)
    predictions = np.load(predictions_file_name)
else:
    clouds, predictions = [], []
    for i in tqdm(range(num_images)):
        file_name = '{:06d}.npy'.format(i)
        depth_image = np.load(os.path.join(sample_dir, file_name))
        try:
            pc, outputs_xyz = predictor.make_prediction(depth_image)
            clouds.append(pc)
            predictions.append(outputs_xyz)
        except:
            # Add dummy array with no joints
            clouds.append(np.zeros((3, 0)))
            predictions.append(np.zeros((3, 0)))

    np.save(cloud_file_name, clouds)
    np.save(predictions_file_name, predictions)

def update_graph(num):
    pc_data, pd_data = clouds[num], predictions[num]
    t1, t1c, t2, t2c, t3, t3c, t4, t4c, t5, t5c, ai, ao, h = pd_data[0, :], pd_data[1, :], pd_data[2, :], pd_data[3, :], pd_data[4, :] ,pd_data[5, :], pd_data[6, :], pd_data[7, :], pd_data[8, :], pd_data[9, :], pd_data[10, :], pd_data[11, :], pd_data[12, :]   
    am = (ai + ao) / 2.
    point_cloud_graph.set_data(pc_data[:, 1].flatten(), pc_data[:, 0].flatten())
    point_cloud_graph.set_3d_properties(pc_data[:, 2].flatten())
    l1.set_data([t1[1], t1c[1]], [t1[0], t1c[0]])
    l1.set_3d_properties([t1[2], t1c[2]])
    l2.set_data([t2[1], t2c[1]], [t2[0], t2c[0]])
    l2.set_3d_properties([t2[2], t2c[2]])
    l3.set_data([t3[1], t3c[1]], [t3[0], t3c[0]])
    l3.set_3d_properties([t3[2], t3c[2]])
    l4.set_data([t4[1], t4c[1]], [t4[0], t4c[0]])
    l4.set_3d_properties([t4[2], t4c[2]])
    l5.set_data([t5[1], t5c[1]], [t5[0], t5c[0]])
    l5.set_3d_properties([t5[2], t5c[2]])
    l6.set_data([t1c[1], am[1]], [t1c[0], am[0]])
    l6.set_3d_properties([t1c[2], am[2]])
    l7.set_data([t2c[1], am[1]], [t2c[0], am[0]])
    l7.set_3d_properties([t2c[2], am[2]])
    l8.set_data([t3c[1], am[1]], [t3c[0], am[0]])
    l8.set_3d_properties([t3c[2], am[2]])
    l9.set_data([t4c[1], am[1]], [t4c[0], am[0]])
    l9.set_3d_properties([t4c[2], am[2]])
    l10.set_data([t5c[1], am[1]], [t5c[0], am[0]])
    l10.set_3d_properties([t5c[2], am[2]])
    l11.set_data([am[1], h[1]], [am[0], h[0]])
    l11.set_3d_properties([am[2], h[2]])
    predictions_graph.set_data(pd_data[:, 1].flatten(), pd_data[:, 0].flatten()) 
    predictions_graph.set_3d_properties(pd_data[:, 2].flatten())
    title.set_text('{:06d}.npy'.format(num))

fig = plt.figure(figsize=plt.figaspect(0.5))
fig.subplots_adjust(hspace=0.05, wspace=0.05)
point_cloud_ax = fig.add_subplot(121, projection='3d')
point_cloud_ax.view_init(90, -90)
point_cloud_ax.set_xlim3d([-0.5, 0.5])
point_cloud_ax.set_ylim3d([-0.5, 0.5])
point_cloud_ax.set_zlim3d([-0.5, 0.5])
predictions_ax = fig.add_subplot(122, projection='3d')
predictions_ax.view_init(90, -90)
predictions_ax.set_xlim3d([-0.5, 0.5])
predictions_ax.set_ylim3d([-0.5, 0.5])
predictions_ax.set_zlim3d([-0.5, 0.5])
title = point_cloud_ax.set_title('000000.npy')

pc_data = clouds[0]
pd_data = predictions[0]
t1, t1c, t2, t2c, t3, t3c, t4, t4c, t5, t5c, ai, ao, h = pd_data[0, :], pd_data[1, :], pd_data[2, :], pd_data[3, :], pd_data[4, :] ,pd_data[5, :], pd_data[6, :], pd_data[7, :], pd_data[8, :], pd_data[9, :], pd_data[10, :], pd_data[11, :], pd_data[12, :]   
am = (ai + ao) / 2.
point_cloud_graph, = point_cloud_ax.plot(pc_data[:, 1].flatten(), pc_data[:, 0].flatten(), pc_data[:, 2].flatten(), linestyle='', marker='o', markersize=1)
l1, = predictions_ax.plot([t1[1], t1c[1]], [t1[0], t1c[0]], [t1[2], t1c[2]], color='green')
l2, = predictions_ax.plot([t2[1], t2c[1]], [t2[0], t2c[0]], [t2[2], t2c[2]], color='green')
l3, = predictions_ax.plot([t3[1], t3c[1]], [t3[0], t3c[0]], [t3[2], t3c[2]], color='green')
l4, = predictions_ax.plot([t4[1], t4c[1]], [t4[0], t4c[0]], [t4[2], t4c[2]], color='green')
l5, = predictions_ax.plot([t5[1], t5c[1]], [t5[0], t5c[0]], [t5[2], t5c[2]], color='green')
l6, = predictions_ax.plot([t1c[1], am[1]], [t1c[0], am[0]], [t1c[2], am[2]], color='orange')
l7, = predictions_ax.plot([t2c[1], am[1]], [t2c[0], am[0]], [t2c[2], am[2]], color='orange')
l8, = predictions_ax.plot([t3c[1], am[1]], [t3c[0], am[0]], [t3c[2], am[2]], color='orange')
l9, = predictions_ax.plot([t4c[1], am[1]], [t4c[0], am[0]], [t4c[2], am[2]], color='orange')
l10, = predictions_ax.plot([t5c[1], am[1]], [t5c[0], am[0]], [t5c[2], am[2]], color='orange')
l11, = predictions_ax.plot([am[1], h[1]], [am[0], h[0]], [am[2], h[2]], color='blue')

predictions_graph, = predictions_ax.plot(pd_data[:, 1].flatten(), pd_data[:, 0].flatten(), pd_data[:, 2].flatten(), linestyle='', marker='o', color='red', markersize=2)

animation_running = True

def on_click(event):
    global animation_running
    if animation_running: 
        ani.event_source.stop()
        animation_running = False
    else:
        ani.event_source.start()
        animation_running = True

fig.canvas.mpl_connect('button_press_event', on_click)
ani = animation.FuncAnimation(fig, update_graph, num_images, interval=int(1000. / playback_fps))

plt.show()

