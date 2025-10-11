Two-stage regression (T, then T,u,v,p)

Structure:
- data/data_module.py: CSV loader and phase splits
- models/: base interface and sklearn wrappers
- trainer/two_stage_trainer.py: training orchestration
- utils/: IO and metrics helpers
- main.py: CLI entry

Usage:
```bash
conda activate fyx_sci
cd /home/yxfeng/project2/sci2
CUDA_VISIBLE_DEVICES=5 python main.py
```

