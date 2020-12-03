cd /home/lz/data/Projects/Vision/3D/github_resources/mmdetection3d
RANK=0 python -m tools.train ../../code/working_code/configs/votenet_partnet_config.py --work-dir=log/&lformat log_name &rformat
cd ../../code/working_code