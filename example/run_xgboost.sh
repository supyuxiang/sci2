set -x

cd /home/yxfeng/project2/sci925


echo 'Start running XGBoost (CPU/GPU independent)'
# If you have GPU-enabled xgboost, you can export below for consistency, though script itself doesn't need CUDA var
CUDA_VISIBLE_DEVICES=7 python example/run_xgboost.py
echo 'Running completed'


