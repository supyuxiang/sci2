import sys
from pathlib import Path
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.data_manager import DataManager

try:
    import xgboost as xgb
except ImportError:
    print("Please install xgboost: pip install xgboost")
    raise


def to_numpy_loader(dataloader):
    X_list, y_list = [], []
    for xb, yb in dataloader:
        X_list.append(xb.numpy())
        y_list.append(yb.numpy())
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    return X, y


def main():
    # load config
    config_path = ROOT / 'src' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dm = DataManager(config)

    dl1_tr, dl1_te = dm.dataloader_phase1_train, dm.dataloader_phase1_test
    dl2_tr, dl2_te = dm.dataloader_phase2_train, dm.dataloader_phase2_test

    X1_tr, y1_tr = to_numpy_loader(dl1_tr)
    X1_te, y1_te = to_numpy_loader(dl1_te)

    X2_tr, y2_tr = to_numpy_loader(dl2_tr)
    X2_te, y2_te = to_numpy_loader(dl2_te)

    # Phase 1: single-target
    print('Training XGBoost for phase 1 ...')
    model1 = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=-1, random_state=42)
    model1.fit(X1_tr, y1_tr)
    print('Phase 1 score:', model1.score(X1_te, y1_te))

    # Phase 2: may be multi-target -> train per target
    print('Training XGBoost for phase 2 ...')
    if y2_tr.ndim == 1:
        model2 = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=-1, random_state=42)
        model2.fit(X2_tr, y2_tr)
        print('Phase 2 score:', model2.score(X2_te, y2_te))
    else:
        models = []
        scores = []
        for i in range(y2_tr.shape[1]):
            m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=-1, random_state=42)
            m.fit(X2_tr, y2_tr[:, i])
            s = m.score(X2_te, y2_te[:, i])
            print(f'Phase 2 target {i} score:', s)
            models.append(m)
            scores.append(s)
        print('Phase 2 avg score:', float(np.mean(scores)))


if __name__ == '__main__':
    main()
