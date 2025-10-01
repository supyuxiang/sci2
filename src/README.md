Two-stage regression (T, then T,u,v,p)

Structure:
- data/data_module.py: CSV loader and phase splits
- models/: base interface and sklearn wrappers
- trainer/two_stage_trainer.py: training orchestration
- utils/: IO and metrics helpers
- main.py: CLI entry

Usage:
```bash
CUDA_VISIBLE_DEVICES=8 python main.py --config config.yaml --excel '/home/yxfeng/project2/sci925/data/正确的二维（无裂解,3MPa).xlsx'
```

