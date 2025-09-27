from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import yaml

from trainer.two_stage_trainer import TwoStageTrainer
from utils.io import ensure_dir


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage regression trainer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--excel", type=str, default=None, help="Override Excel path")
    parser.add_argument("--format", type=str, default=None, choices=["auto", "excel", "csv"], help="Force file format")
    parser.add_argument("--sheet", type=str, default=None, help="Excel sheet index or name")
    parser.add_argument("--csv-sep", dest="csv_sep", type=str, default=None, help="CSV separator, e.g. ',' or '\t'")
    parser.add_argument("--csv-encoding", dest="csv_encoding", type=str, default=None, help="CSV encoding, e.g. 'gbk' or 'utf-8-sig'")
    parser.add_argument("--out", type=str, default=None, help="Override output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.excel:
        config.setdefault("dataset", {})["excel_path"] = args.excel
    if args.format:
        config.setdefault("dataset", {})["format"] = args.format
    if args.sheet is not None:
        # allow numeric index
        try:
            sheet_val = int(args.sheet)
        except Exception:
            sheet_val = args.sheet
        config.setdefault("dataset", {})["sheet"] = sheet_val
    if args.csv_sep is not None:
        config.setdefault("dataset", {})["csv_sep"] = args.csv_sep
    if args.csv_encoding is not None:
        config.setdefault("dataset", {})["csv_encoding"] = args.csv_encoding
    if args.out:
        config.setdefault("training", {})["output_dir"] = args.out
    ensure_dir(config["training"]["output_dir"])

    trainer = TwoStageTrainer(config)
    report = trainer.train()
    print("Training done. Metrics summary:")
    print(report)


if __name__ == "__main__":
    main()

