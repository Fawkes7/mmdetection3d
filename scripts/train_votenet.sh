#cd ../../github_resources/mmdetection3d
cd /home/lz/data/Projects/Vision/3D/github_resources/mmdetection3d
#RANK=0 python -m tools.train ../../code/working_code/configs/votenet_partnet_config.py --work-dir=log/Laptop-1
#RANK=0 python -m tools.train ../../code/working_code/configs/votenet_partnet_config.py --work-dir=log/Keyboard-1
RANK=0 python -m tools.train ../../code/working_code/configs/votenet_partnet_config.py --work-dir=log/Bed-1

#RANK=0 python -m tools.train ../../code/working_code/old_config/votenet_partnet_config.py --work-dir=log
cd ../../code/working_code