set -x

cd /home/yxfeng/project2/sci925


echo 'Start running'
CUDA_VISIBLE_DEVICES=8 python main.py --model_name 'agtmlp' --epochs 500
echo 'Running completed'
