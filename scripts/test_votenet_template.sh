cd /home/lz/data/Projects/Vision/3D/github_resources/mmdetection3d
RANK=0 python -m tools.test ../../code/working_code/configs/votenet_partnet_eval_config.py log/&lformat log_name &rformat/latest.pth --out results/&lformat save_name &rformat.pkl
cd ../../code/working_code
