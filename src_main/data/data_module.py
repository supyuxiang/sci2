from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PhaseDatasets:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class DataModule:
    """Load CSV and produce phase-specific datasets.

    - Phase 1: rows [0:index_phase1_end) => predict T
    - Phase 2: rows [index_phase1_end: ) => predict T,u,v,p
    """

    def __init__(self, config: Dict):
        self.config = config
        self.excel_path: str = config["dataset"]["excel_path"]
        self.index_phase1_end: int = int(config["dataset"]["index_phase1_end"])
        self.feature_cols: List[str] = list(config["dataset"]["features"])
        self.phase1_targets: List[str] = list(config["dataset"]["phase1_targets"])
        self.phase2_targets: List[str] = list(config["dataset"]["phase2_targets"])
        self.test_ratio: float = float(config["training"]["test_ratio"])
        self.random_state: int = int(config["training"]["random_state"])

        self.df: pd.DataFrame | None = None

    def load(self) -> None:
        # 自动根据扩展名选择加载方式（支持 .csv / .xlsx / .xls），并在Excel失败时回退到CSV解析
        fmt = str(self.config.get("dataset", {}).get("format", "auto")).lower()
        lower = self.excel_path.lower()
        if fmt == "csv" or (fmt == "auto" and lower.endswith(".csv")):
            # Try robust CSV reading
            self.df = self._read_csv_robust(self.excel_path)
        elif fmt == "excel" or (fmt == "auto" and (lower.endswith(".xlsx") or lower.endswith(".xls"))):
            sheet = self.config.get("dataset", {}).get("sheet", 0)
            try:
                self.df = pd.read_excel(self.excel_path, sheet_name=sheet)
            except Exception:
                # 某些文件扩展名为.xlsx但不是标准Excel，尝试按CSV读取
                self.df = self._read_csv_robust(self.excel_path)
        else:
            raise ValueError(f"Unsupported file extension: {self.excel_path}")
        missing_features = [c for c in self.feature_cols if c not in self.df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns in CSV: {missing_features}")
        # targets are optional to exist depending on phase; we'll check in getters

    def _read_csv_robust(self, path: str) -> pd.DataFrame:
        """Try user-provided hints first, then multiple encodings and separators."""
        ds = self.config.get("dataset", {})
        sep_hint = ds.get("csv_sep")
        enc_hint = ds.get("csv_encoding")
        encodings = [enc_hint] + [None, "utf-8", "utf-8-sig", "gbk", "gb2312", "big5", "latin1"]
        seps = [sep_hint] + [None, ",", ";", "\t", "\s+"]
        # First try fast engine with common encodings
        for enc in encodings:
            if enc is None and sep_hint is not None:
                # Let pandas sniff with default engine
                try:
                    return pd.read_csv(path, sep=sep_hint)
                except Exception:
                    pass
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                pass
        # Then try python engine and different separators (including regex sep)
        for enc in encodings:
            for sep in seps:
                try:
                    return pd.read_csv(path, encoding=enc, engine="python", sep=sep)
                except Exception:
                    continue
        # If all attempts failed, raise a clear error
        raise ValueError(
            "Failed to parse file as Excel and CSV with multiple encodings/separators. "
            f"Please verify the file at: {path}"
        )

    def _split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return train_test_split(
            X, y, test_size=self.test_ratio, random_state=self.random_state, shuffle=True
        )

    def get_phase1(self) -> PhaseDatasets:
        if self.df is None:
            raise RuntimeError("Call load() before accessing datasets")
        df_phase1 = self.df.iloc[: self.index_phase1_end]
        for t in self.phase1_targets:
            if t not in df_phase1.columns:
                raise ValueError(f"Missing target column for phase1: {t}")
        X = df_phase1[self.feature_cols].to_numpy(dtype=float)
        y = df_phase1[self.phase1_targets].to_numpy(dtype=float).squeeze()
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        X_train, X_test, y_train, y_test = self._split(X, y)
        return PhaseDatasets(X_train, X_test, y_train, y_test)

    def get_phase2(self) -> PhaseDatasets:
        if self.df is None:
            raise RuntimeError("Call load() before accessing datasets")
        df_phase2 = self.df.iloc[self.index_phase1_end :]
        for t in self.phase2_targets:
            if t not in df_phase2.columns:
                raise ValueError(f"Missing target column for phase2: {t}")
        X = df_phase2[self.feature_cols].to_numpy(dtype=float)
        y = df_phase2[self.phase2_targets].to_numpy(dtype=float)
        X_train, X_test, y_train, y_test = self._split(X, y)
        return PhaseDatasets(X_train, X_test, y_train, y_test)

