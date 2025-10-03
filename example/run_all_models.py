import os
import sys
from pathlib import Path
import yaml

# Ensure project src on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logger import Logger
from src.models.models import ModelFactory
from src.data.data_manager import DataManager
from src.trainer.train import Trainer


def main():
    # Load config
    config_path = ROOT / 'src' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Force 1 epoch quick run
    config['Train']['epochs'] = 500

    logger = Logger('run_all_models', config)

    # Build dataloaders once
    dm = DataManager(config)
    dl1_train, dl1_test = dm.dataloader_phase1_train, dm.dataloader_phase1_test
    dl2_train, dl2_test = dm.dataloader_phase2_train, dm.dataloader_phase2_test

    # Iterate all model names declared in config
    model_names = config.get('Model', {}).get('name_list', [])
    if not model_names:
        raise ValueError('Model.name_list is empty in config.yaml')

    for name in model_names:
        print(f"===== Running model: {name} (phase 1) =====")
        try:
            m1 = ModelFactory(phase=1, logger=logger, model_name=name).model
            Trainer(config=config, model=m1, dataloader_train=dl1_train, logger=logger, save_model_dir=config['Train']['save_model_path1']).validate(dl1_test)
        except Exception as e:
            print(f"Model {name} phase 1 failed: {e}")

        print(f"===== Running model: {name} (phase 2) =====")
        try:
            m2 = ModelFactory(phase=2, logger=logger, model_name=name).model
            Trainer(config=config, model=m2, dataloader_train=dl2_train, logger=logger, save_model_dir=config['Train']['save_model_path2']).validate(dl2_test)
        except Exception as e:
            print(f"Model {name} phase 2 failed: {e}")


if __name__ == '__main__':
    main()
