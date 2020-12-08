#cd ../../github_resources/mmdetection3d
<<<<<<< HEAD
BASE=$PWD
CONFIG="/configs/votenet_partnet_config.py"
MMDETPATH="/home/haoyuan/mmdet/mmdetection3d"

cd $MMDETPATH

RANK=0 python3.6 -m tools.train ${BASE}${CONFIG} --work-dir=log
=======
cd /home/lz/data/Projects/Vision/3D/github_resources/mmdetection3d
#RANK=0 python -m tools.train ../../code/working_code/configs/votenet_partnet_config.py --work-dir=log/Laptop-1
#RANK=0 python -m tools.train ../../code/working_code/configs/votenet_partnet_config.py --work-dir=log/Keyboard-1
RANK=0 python -m tools.train ../../code/working_code/configs/votenet_partnet_config.py --work-dir=log/Bed-1

#RANK=0 python -m tools.train ../../code/working_code/old_config/votenet_partnet_config.py --work-dir=log
cd ../../code/working_code
>>>>>>> acd8ddf40dc7fd917025f70cfcf847a53493fa28
