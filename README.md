### Example Commands

ssh -i ML_Northern_Virginia.pem ubuntu@ec2-35-175-113-199.compute-1.amazonaws.com

aws s3 cp s3://foot-point-net/v3_40k dataset

nvidia-docker run -p 8097:8097 --shm-size 500G -v /home/ubuntu/dataset:/workspace/data -v /home/ubuntu/res/FootPointNet:/workspace/results -it andrewjzhou/foot-point-net:train bash
 
python train_eval/train_foot.py --data_root data/v3_40k --save_root_dir results/iter_002 --nepoch 50 --save_freq 5 --learning_rate 0.001 --step_size 20

scp -i ML_Northern_Virginia.pem ubuntu@ec2-35-175-113-199.compute-1.amazonaws.com:/home/ubuntu/res/FootPointNet/iter_002/* /Volumes/AndrewJayZhou/Dev/HandPointNet/results/iter_002

# cpu
python3 train_eval/train_foot.py --data_root data/v3_40k --ngpu 0 --save_root_dir results/testing 