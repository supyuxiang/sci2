set -x

nvidia-smi
conda activate fyx_sci
cd /home/yxfeng/project2/sci925


echo 'Start running'
CUDA_VISIBLE_DEVICES=7 python main.py
echo 'Running completed'
