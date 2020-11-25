#cd ../../github_resources/mmdetection3d
#cd /home/lz/data/Projects/Vision/3D/github_resources/mmdetection3d
RANK=0 python -m tools.test ../../code/working_code/configs/votenet_partnet_eval_config.py log/latest.pth --out train.pkl

