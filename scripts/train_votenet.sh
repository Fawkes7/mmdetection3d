#cd ../../github_resources/mmdetection3d
BASE=$PWD
CONFIG="/configs/votenet_partnet_config.py"
MMDETPATH="/home/haoyuan/mmdet/mmdetection3d"

cd $MMDETPATH

RANK=0 python3.6 -m tools.train ${BASE}${CONFIG} --work-dir=log
