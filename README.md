ssh -i ML.pem ubuntu@ec2-18-218-10-68.us-east-2.compute.amazonaws.com

nvidia-docker run -p 8097:8097 --shm-size 500G -v /home/ubuntu/dataset:/workspace/data -v /home/ubuntu/res/FootPointNet:/workspace/results -it andrewjzhou/foot-point-net:train bash

python train_eval/train_foot.py --data_root data/blender_v2 --save_root_dir results/iter_001


# cpu
python3 train_eval/train_foot.py --data_root data/blender_v2 --ngpu 0 