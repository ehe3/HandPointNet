### Example Commands

ssh -i ML_Northern_Virginia.pem ubuntu@ec2-18-220-136-141.us-east-2.compute.amazonaws.com

aws s3 sync s3://foot-point-net dataset

nvidia-docker run -p 8097:8097 --shm-size 500G -v /home/ubuntu/dataset:/workspace/data -v /home/ubuntu/res/FootPointNet:/workspace/results -it andrewjzhou/foot-point-net:train bash

python train_eval/train_foot.py --data_root data/blender_v2 --save_root_dir results/iter_001 --nepoch 45


# cpu
python3 train_eval/train_foot.py --data_root data/blender_v2 --ngpu 0 --save_root_dir 