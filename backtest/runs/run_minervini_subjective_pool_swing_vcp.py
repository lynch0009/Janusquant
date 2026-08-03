# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.runs import run_minervini_subjective_pool as rolling_pool


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "minervini_subjective_pool_swing_vcp"


def main() -> None:
    args = rolling_pool.parse_args()
    if args.output_dir == str(rolling_pool.DEFAULT_OUTPUT_DIR):
        args.output_dir = str(DEFAULT_OUTPUT_DIR)
    args.vcp_mode = "swing"
    rolling_pool.run_subjective_pool(
        args,
        strategy_builder=rolling_pool.build_strategy,
        script_name=Path(__file__).name,
    )


if __name__ == "__main__":
    main()
