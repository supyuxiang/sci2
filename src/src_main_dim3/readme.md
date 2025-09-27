python data_preprocessing.py

python train_T3.py

python train_atm.py

train_try.py

python train.py

python train_CARNet.py

python train_model_main.py     实现了CARNet＋self-attention



python data_preprocessing_1.py

python data_preprocessing_0.py



###814



python /root/sci/src/src_main/train_model_main_0_adaptive.py  # 使用自适应优化器(AdamW + ReduceLROnPlateau)


python /root/sci/src/src_main/train_model_main_1_base_adaptive.py



#############
config.py
data_preprocessing.py
model_CARNet_enhance.py
train_model_main_0_adaptive.py
train_model_main_1_base_adaptive.py
#############



815

python /root/sci/src/src_main/novel_architectures/train_multi_scale_st.py

python /root/sci/src/src_main/novel_architectures/train_physics_aware_gnn.py

python /root/sci/src/src_main/novel_architectures/train_mixture_of_experts.py

python /root/sci/src/src_main/novel_architectures/train_adaptive_scale_decomposition.py

python /root/sci/src/src_main/novel_architectures/train_hierarchical_scale_aware.py

python /root/sci/src/src_main/novel_architectures/train_dynamic_scale_adaptation.py




####825
python /data1/chzhang/sci824/src/src_main/train_model_main_0_adaptive.py --model_0  "EnhancedCARNet" --scheduler_0  "ReduceLROnPlateau" --device_0 'cuda:8' --gpu_device_ids_0 '8'
python /data1/chzhang/sci824/src/src_main/train_model_main_0_adaptive.py --model_0  "EnhancedCARNet" --scheduler_0  "CosineAnnealingLR" --device_0 'cuda:8' --gpu_device_ids_0 '8'
python /data1/chzhang/sci824/src/src_main/train_model_main_0_adaptive.py --model_0  "EnhancedCARNet" --scheduler_0  "CosineAnnealingWarmRestarts" --device_0 'cuda:8' --gpu_device_ids_0 '8'
python /data1/chzhang/sci824/src/src_main/train_model_main_0_adaptive.py --model_0  "EnhancedCARNet" --scheduler_0  "MultiStepLR" --device_0 'cuda:8' --gpu_device_ids_0 '8'
python /data1/chzhang/sci824/src/src_main/train_model_main_0_adaptive.py --model_0  "EnhancedCARNet" --scheduler_0  "CyclicLR" --device_0 'cuda:8' --gpu_device_ids_0 '8'
python /data1/chzhang/sci824/src/src_main/train_model_main_0_adaptive.py --model_0  "'EnhancedCARNet_v2'" --scheduler_0  "ReduceLROnPlateau" --device_0 'cuda:8' --gpu_device_ids_0 '8'


python /data1/chzhang/sci824/src/src_main/train_model_main_1_base_adaptive.py --model_1  "EnhancedCARNet" --scheduler_1  "ReduceLROnPlateau" --device_1 'cuda:8' --gpu_device_ids_1 '8'
python /data1/chzhang/sci824/src/src_main/train_model_main_1_base_adaptive.py --model_1  "EnhancedCARNet" --scheduler_1  "CosineAnnealingLR" --device_1 'cuda:8' --gpu_device_ids_1 '8'
python /data1/chzhang/sci824/src/src_main/train_model_main_1_base_adaptive.py --model_1  "EnhancedCARNet" --scheduler_1  "CosineAnnealingWarmRestarts" --device_1 'cuda:8' --gpu_device_ids_1 '8'
python /data1/chzhang/sci824/src/src_main/train_model_main_1_base_adaptive.py --model_1  "EnhancedCARNet" --scheduler_1  "MultiStepLR" --device_1 'cuda:8' --gpu_device_ids_1 '8'
python /data1/chzhang/sci824/src/src_main/train_model_main_1_base_adaptive.py --model_1  "EnhancedCARNet" --scheduler_1  "CyclicLR" --device_1 'cuda:8' --gpu_device_ids_1 '8'
python /data1/chzhang/sci824/src/src_main/train_model_main_1_base_adaptive.py --model_1  "'EnhancedCARNet_v2'" --scheduler_1  "ReduceLROnPlateau" --device_1 'cuda:8' --gpu_device_ids_1 '8'


#物理约束,散度
python /data1/chzhang/sci824/src/src_main/train_model_main_1_physics_ds.py --device_1 'cuda:8' --gpu_device_ids_1 '8'



#0
python /data1/chzhang/sci824/src/src_main/train_model_main_0_adaptive.py --model_0  "EnhancedCARNet" --scheduler_0  "ReduceLROnPlateau" --device_0 'cuda:8' --gpu_device_ids_0 '8'
