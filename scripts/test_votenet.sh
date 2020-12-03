#cd ../../github_resources/mmdetection3d

cd /home/lz/data/Projects/Vision/3D/github_resources/mmdetection3d
#RANK=0 python -m tools.test ../../code/working_code/configs/votenet_partnet_eval_config.py log/Laptop-1/latest.pth --out results/train_Laptop-1.pkl
#RANK=0 python -m tools.test ../../code/working_code/configs/votenet_partnet_eval_config.py log/Keyboard-1/latest.pth --out results/train_Keyboard-1.pkl
RANK=0 python -m tools.test ../../code/working_code/configs/votenet_partnet_eval_config.py log/Bed-3/latest.pth --out results/train_Bed-3.pkl
cd ../../code/working_code
