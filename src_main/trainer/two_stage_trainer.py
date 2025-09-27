from __future__ import annotations

import json
import os
from typing import Dict, Any

import numpy as np

from data.data_module import DataModule
from models.models import build_regressor, BaseRegressor
from utils.metrics import regression_metrics
from utils.io import ensure_dir


class TwoStageTrainer:
    """Stage 1: predict T on first part of data.
    Stage 2: predict T,u,v,p on the remaining data.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.dm = DataModule(config)
        self.output_dir: str = config["training"]["output_dir"]
        ensure_dir(self.output_dir)

        self.model_phase1: BaseRegressor | None = None
        self.model_phase2: BaseRegressor | None = None

    def _build_models(self) -> None:
        m1 = self.config["models"]["phase1"]
        m2 = self.config["models"]["phase2"]
        self.model_phase1 = build_regressor(m1.get("type"), m1.get("params", {}), multi_output=False)
        self.model_phase2 = build_regressor(m2.get("type"), m2.get("params", {}), multi_output=True)

    def train(self) -> Dict[str, Any]:
        self.dm.load()
        self._build_models()

        # Phase 1
        ph1 = self.dm.get_phase1()
        assert self.model_phase1 is not None
        self.model_phase1.fit(ph1.X_train, ph1.y_train)
        preds1 = self.model_phase1.predict(ph1.X_test)
        metrics1 = regression_metrics(ph1.y_test, preds1)

        # Phase 2
        ph2 = self.dm.get_phase2()
        assert self.model_phase2 is not None
        self.model_phase2.fit(ph2.X_train, ph2.y_train)
        preds2 = self.model_phase2.predict(ph2.X_test)
        metrics2 = regression_metrics(ph2.y_test, preds2)

        # Save
        ensure_dir(os.path.join(self.output_dir, "models"))
        ensure_dir(os.path.join(self.output_dir, "reports"))
        # Use joblib within model.save
        self.model_phase1.save(os.path.join(self.output_dir, "models", "phase1_model.joblib"))
        self.model_phase2.save(os.path.join(self.output_dir, "models", "phase2_model.joblib"))

        report = {
            "phase1": {"metrics": metrics1},
            "phase2": {"metrics": metrics2},
        }
        with open(os.path.join(self.output_dir, "reports", "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

