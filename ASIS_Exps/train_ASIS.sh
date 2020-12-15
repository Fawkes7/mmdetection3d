BASE=$PWD
CONFIG="/configs/votenet_partnet_config.py"
MMDETPATH="/home/haoyuan/mmdet/mmdetection3d"

cd $MMDETPATH


RANK=0 python -m tools.train ../../code/working_code/configs/votenet_partnet_config.py --work-dir=log/Bed-1