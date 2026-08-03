# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
'''
python backtest\runs\run_minervini_subjective_pool.py `
  --start-date 2026-01-01 `
  --end-date 2026-05-19 `
  --min-rps 90 `
  --min-revenue-yoy 0.20 `
  --min-net-profit-yoy 0.25 `
  --platform-window 40 `
  --watchlist-top-pct 0.25 `
  --price-mode hfq
'''
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.data import DuckDBDataPortal, ResearchDailyHistoryStore
from backtest.db import DuckDBConfig
from backtest.execution import EngineConfig
from backtest.strategies import MinerviniAshareStrategy
from backtest.runs.minervini_profiles import (
    MINERVINI_PROFILES,
    STRICT_SUBJECTIVE_PROFILE,
    resolve_minervini_profile,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "minervini_subjective_pool"
TRADEABLE_SETUP_TYPES = (
    "vcp_breakout",
    "platform_breakout",
    "leader_continuation",
)
VCP_WATCH_DISTANCE_RANGE = (-0.03, 0.01)
PLATFORM_WATCH_DISTANCE_RANGE = (-0.03, 0.01)
LEADER_WATCH_DISTANCE_RANGE = (-0.02, 0.01)
VCP_WATCH_VOLUME_RATIO = 1.03
PLATFORM_WATCH_VOLUME_RATIO = 1.3
LEADER_WATCH_VOLUME_RATIO = 1.1


def build_run_output_dir(base_dir: Path, start_date: datetime, end_date: datetime) -> Path:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}_{run_timestamp}"
    return base_dir / run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export daily Minervini subjective candidate pools.")
    parser.add_argument("--profile", choices=sorted(MINERVINI_PROFILES), default=STRICT_SUBJECTIVE_PROFILE)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-05-19")
    parser.add_argument("--benchmark-code", default=EngineConfig().benchmark_code)
    parser.add_argument("--scan-every", "--rebalance-every", dest="scan_every", type=int, default=None)
    parser.add_argument("--min-listing-trade-days", type=int, default=None)
    parser.add_argument("--min-liqa-mv", type=float, default=None)
    parser.add_argument("--max-liqa-mv", type=float, default=None)
    parser.add_argument("--min-revenue-yoy", type=float, default=None)
    parser.add_argument("--min-net-profit-yoy", type=float, default=None)
    parser.add_argument("--min-ttm-net-profit-yoy", type=float, default=None)
    parser.add_argument("--min-eps-yoy-floor", type=float, default=None)
    parser.add_argument("--min-eps-ttm-yoy", type=float, default=None)
    parser.add_argument("--min-rps", type=float, default=None)
    parser.add_argument("--min-close-to-high-250", type=float, default=None)
    parser.add_argument("--min-above-low-250", type=float, default=None)
    parser.add_argument("--price-mode", default="hfq", choices=["raw", "qfq", "hfq"])
    parser.add_argument("--breakout-buffer-pct", type=float, default=None)
    parser.add_argument("--min-breakout-volume-ratio", type=float, default=None)
    parser.add_argument("--platform-window", type=int, default=None)
    parser.add_argument("--max-platform-depth", type=float, default=None)
    parser.add_argument("--vcp-short-window", type=int, default=None)
    parser.add_argument("--vcp-mid-window", type=int, default=None)
    parser.add_argument("--vcp-base-window", type=int, default=None)
    parser.add_argument("--vcp-long-window", type=int, default=None)
    parser.add_argument("--max-vcp-depth", type=float, default=None)
    parser.add_argument("--vcp-breakout-volume-ratio", type=float, default=None)
    parser.add_argument("--watchlist-top-pct", type=float, default=0.25)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return resolve_minervini_profile(parser.parse_args(), default_profile=STRICT_SUBJECTIVE_PROFILE)


def build_strategy(args: argparse.Namespace) -> MinerviniAshareStrategy:
    args = resolve_minervini_profile(args, default_profile=STRICT_SUBJECTIVE_PROFILE)
    return MinerviniAshareStrategy(
        benchmark_code=args.benchmark_code,
        top_k=9999,
        hold_days=20,
        rebalance_every_n_trade_days=args.scan_every,
        min_listing_trade_days=args.min_listing_trade_days,
        min_liqa_mv=args.min_liqa_mv,
        max_liqa_mv=args.max_liqa_mv,
        min_revenue_yoy=args.min_revenue_yoy,
        min_net_profit_yoy=args.min_net_profit_yoy,
        min_ttm_net_profit_yoy=args.min_ttm_net_profit_yoy,
        min_eps_yoy_floor=args.min_eps_yoy_floor,
        min_eps_ttm_yoy=args.min_eps_ttm_yoy,
        allow_missing_revenue_yoy=False,
        min_rps=args.min_rps,
        min_close_to_high_250=args.min_close_to_high_250,
        min_above_low_250=args.min_above_low_250,
        price_mode=args.price_mode,
        ma_micro_window=5,
        ma_pullback_window=20,
        ma_short_window=50,
        ma_mid_window=150,
        ma_long_window=200,
        ma_long_rise_window=20,
        breakout_buffer_pct=args.breakout_buffer_pct,
        min_breakout_volume_ratio=args.min_breakout_volume_ratio,
        vcp_mode=getattr(args, "vcp_mode", "rolling"),
        platform_window=args.platform_window,
        max_platform_depth=args.max_platform_depth,
        vcp_short_window=args.vcp_short_window,
        vcp_mid_window=args.vcp_mid_window,
        vcp_base_window=args.vcp_base_window,
        vcp_long_window=args.vcp_long_window,
        max_vcp_depth=args.max_vcp_depth,
        vcp_breakout_volume_ratio=args.vcp_breakout_volume_ratio,
        breakout_as_filter=False,
        vcp_breakout_bonus=12.0,
        platform_breakout_bonus=6.0,
        build_execution_raw_bridge=False,
    )


def enrich_with_basic_info(frame: pd.DataFrame, data_portal: DuckDBDataPortal) -> pd.DataFrame:
    if frame.empty:
        return frame
    codes = sorted(frame["code"].astype(str).unique().tolist())
    basic = data_portal.get_stock_basic(codes, fields=["code", "code_name"])
    if basic.empty:
        enriched = frame.copy()
        if "code_name" not in enriched.columns:
            enriched["code_name"] = ""
        enriched["code_name"] = enriched["code_name"].fillna("")
        return enriched
    enriched = frame.drop(columns=["code_name"], errors="ignore").merge(basic, on="code", how="left")
    enriched["code_name"] = enriched["code_name"].fillna("")
    return enriched


def remove_new_high_breakout_signal(
    frame: pd.DataFrame,
    strategy: MinerviniAshareStrategy,
) -> pd.DataFrame:
    """主观池模式里显式移除 new_high_breakout。

    这里不改策略主链，只在主观池导出前做一次轻量重构：
    1. new_high_breakout / new_high_breakout_55 不再作为可交易 setup。
    2. 重新生成 setup_type / pivot / stop / risk。
    3. 基于新 setup 重新计算 pool_score / execution_score。
    """
    if frame.empty:
        return frame

    adjusted = frame.copy()
    adjusted["new_high_breakout"] = False
    adjusted["new_high_breakout_55"] = False

    vcp_mask = adjusted["vcp_breakout"].fillna(False)
    platform_mask = adjusted["platform_breakout"].fillna(False)
    leader_mask = adjusted["leader_continuation"].fillna(False)

    adjusted["setup_type"] = np.select(
        [vcp_mask, platform_mask, leader_mask],
        [
            "vcp_breakout",
            "platform_breakout",
            "leader_continuation",
        ],
        default=None,
    )

    platform_pivot_col = f"platform_high_{strategy.platform_window}"
    platform_low_col = f"platform_low_{strategy.platform_window}"
    vcp_pivot_col = f"vcp_pivot_{strategy.vcp_long_window}"
    vcp_low_col = f"vcp_low_{strategy.vcp_long_window}"
    platform_pivot_raw_col = f"{platform_pivot_col}_raw"
    platform_low_raw_col = f"{platform_low_col}_raw"
    vcp_pivot_raw_col = f"{vcp_pivot_col}_raw"
    vcp_low_raw_col = f"{vcp_low_col}_raw"

    adjusted["pivot_price"] = np.where(
        vcp_mask,
        adjusted[vcp_pivot_col],
        np.where(
            platform_mask,
            adjusted[platform_pivot_col],
            np.where(leader_mask, adjusted["new_high_pivot_price"], np.nan),
        ),
    )
    adjusted["base_low_price"] = np.where(
        vcp_mask,
        adjusted[vcp_low_col],
        np.where(
            platform_mask,
            adjusted[platform_low_col],
            np.where(leader_mask, adjusted["new_high_base_low_price"], np.nan),
        ),
    )
    adjusted["pivot_price_raw"] = np.where(
        vcp_mask,
        adjusted[vcp_pivot_raw_col],
        np.where(
            platform_mask,
            adjusted[platform_pivot_raw_col],
            np.where(leader_mask, adjusted["new_high_pivot_price_raw"], np.nan),
        ),
    )
    adjusted["base_low_price_raw"] = np.where(
        vcp_mask,
        adjusted[vcp_low_raw_col],
        np.where(
            platform_mask,
            adjusted[platform_low_raw_col],
            np.where(leader_mask, adjusted["new_high_base_low_price_raw"], np.nan),
        ),
    )

    adjusted["signal_pivot_price"] = adjusted["pivot_price"]
    adjusted["signal_base_low_price"] = adjusted["base_low_price"]

    signal_stop_floor = np.maximum(
        adjusted["base_low_price"],
        adjusted["pivot_price"] - adjusted["atr_14"] * strategy.stop_atr_multiple,
    )
    raw_stop_floor = np.maximum(
        adjusted["base_low_price_raw"],
        adjusted["pivot_price_raw"] - adjusted["atr_14_raw"] * strategy.stop_atr_multiple,
    )
    adjusted["initial_stop_loss"] = np.where(adjusted["pivot_price"].notna(), signal_stop_floor, np.nan)
    adjusted["risk_per_share"] = adjusted["pivot_price"] - adjusted["initial_stop_loss"]
    adjusted["signal_initial_stop_loss"] = adjusted["initial_stop_loss"]
    adjusted["signal_risk_per_share"] = adjusted["risk_per_share"]
    adjusted["initial_stop_loss_raw"] = np.where(adjusted["pivot_price_raw"].notna(), raw_stop_floor, np.nan)
    adjusted["risk_per_share_raw"] = adjusted["pivot_price_raw"] - adjusted["initial_stop_loss_raw"]
    adjusted["has_breakout"] = adjusted["setup_type"].notna()

    adjusted = strategy._apply_selection_score(adjusted)
    return adjusted


def collect_daily_pool_frame(
    strategy: MinerviniAshareStrategy,
    trade_dates: list[datetime],
    data_portal: DuckDBDataPortal,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for trade_date in trade_dates:
        daily_frame = strategy.signal_table.get(trade_date, pd.DataFrame()).copy()
        if daily_frame.empty:
            continue
        daily_frame["trade_date"] = pd.Timestamp(trade_date)
        rows.append(daily_frame)
    if not rows:
        return pd.DataFrame(columns=["trade_date", "code", "pool_score", "execution_score"])

    pool = pd.concat(rows, ignore_index=True)
    pool = remove_new_high_breakout_signal(pool, strategy)
    pool = enrich_with_basic_info(pool, data_portal)
    pool = (
        pool.sort_values(
            ["trade_date", "pool_score", "execution_score", "growth_leadership_score", "code"],
            ascending=[True, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    return pool


def build_research_watchlist(pool: pd.DataFrame, strategy: MinerviniAshareStrategy, top_pct: float) -> pd.DataFrame:
    if pool.empty:
        return pool.copy()

    top_pct = min(max(float(top_pct), 0.0), 1.0)
    if top_pct <= 0.0:
        return pool.iloc[0:0].copy()

    watch = pool.copy()
    watch["research_watch_score"] = (
        watch["pool_score"] * 0.60
        + watch["growth_leadership_score"] * 0.25
        + watch["vcp_maturity_score"].fillna(0.0) * 0.15
    )

    selected_frames: list[pd.DataFrame] = []
    for _, daily in watch.groupby("trade_date", sort=True):
        keep_count = max(1, int(np.ceil(len(daily) * top_pct)))
        picked = (
            daily.sort_values(
                [
                    "research_watch_score",
                    "pool_score",
                    "growth_leadership_score",
                    "vcp_maturity_score",
                    "code",
                ],
                ascending=[False, False, False, False, True],
            )
            .head(keep_count)
            .copy()
        )
        selected_frames.append(picked)
    return pd.concat(selected_frames, ignore_index=True) if selected_frames else watch.iloc[0:0].copy()


def _compute_watch_distance_score(distance_pct: pd.Series, lower_bound: float, upper_bound: float) -> pd.Series:
    """把“距离 pivot 的远近”转成 0~100 分，0% 距离附近最高。"""
    distance = pd.to_numeric(distance_pct, errors="coerce")
    score = pd.Series(0.0, index=distance.index, dtype=float)
    valid = distance.notna() & distance.ge(lower_bound) & distance.le(upper_bound)
    if not valid.any():
        return score

    below_mask = valid & distance.le(0.0)
    if below_mask.any():
        score.loc[below_mask] = 100.0 * (distance.loc[below_mask] - lower_bound) / (0.0 - lower_bound)

    above_mask = valid & distance.gt(0.0)
    if above_mask.any():
        score.loc[above_mask] = 100.0 * (upper_bound - distance.loc[above_mask]) / upper_bound

    return score.clip(lower=0.0, upper=100.0)


def build_trigger_watchlist(
    pool: pd.DataFrame,
    strategy: MinerviniAshareStrategy,
    trade_dates: list[datetime],
) -> pd.DataFrame:
    """生成下一交易日使用的盘前触发清单。

    输出按信号生成日归档：`trade_date` / `signal_date` 都保留为信号日，
    `next_trade_date` 表示这些观察项映射到哪一个下一交易日盘前使用。

    这里不做“单选 setup_plan_type”，而是同一行同时保留：
    1. VCP watch
    2. Platform watch
    3. Leader continuation watch

    这样主观交易时可以直接看到一只股票同时满足哪些观察条件，
    并拿到对应的 raw pivot / stop / risk / 所需放量阈值。
    """
    if pool.empty or len(trade_dates) < 2:
        return pool.iloc[0:0].copy()

    trigger = pool.copy()
    next_trade_date_map = {
        pd.Timestamp(trade_dates[idx]): pd.Timestamp(trade_dates[idx + 1])
        for idx in range(len(trade_dates) - 1)
    }
    trigger["signal_date"] = trigger["trade_date"]
    trigger["next_trade_date"] = trigger["signal_date"].map(next_trade_date_map)
    trigger = trigger[trigger["next_trade_date"].notna()].copy()
    if trigger.empty:
        return trigger

    close_raw = pd.to_numeric(trigger["close_raw"], errors="coerce")
    atr_raw = pd.to_numeric(trigger["atr_14_raw"], errors="coerce")
    volume_ma_50 = pd.to_numeric(trigger["volume_ma_50"], errors="coerce")

    def attach_watch_block(
        frame: pd.DataFrame,
        *,
        prefix: str,
        pivot_col: str,
        low_col: str,
        base_mask: pd.Series,
        distance_range: tuple[float, float],
        required_volume_ratio: float,
    ) -> None:
        pivot = pd.to_numeric(frame[pivot_col], errors="coerce")
        base_low = pd.to_numeric(frame[low_col], errors="coerce")
        stop = pd.concat(
            [
                base_low,
                pivot - atr_raw * strategy.stop_atr_multiple,
            ],
            axis=1,
        ).max(axis=1, skipna=True)
        risk = pivot - stop
        risk_pct = risk / pivot.replace(0, np.nan)
        distance_pct = close_raw / pivot.replace(0, np.nan) - 1.0
        distance_score = _compute_watch_distance_score(distance_pct, *distance_range)
        lower_bound, upper_bound = distance_range
        watch_mask = (
            base_mask.fillna(False)
            & pivot.notna()
            & base_low.notna()
            & stop.notna()
            & risk.gt(0.0)
            & risk_pct.le(strategy.max_initial_stop_pct)
            & distance_pct.ge(lower_bound)
            & distance_pct.le(upper_bound)
        )

        frame[f"{prefix}_pivot_price_raw"] = pivot
        frame[f"{prefix}_base_low_price_raw"] = base_low
        frame[f"{prefix}_initial_stop_loss_raw"] = stop
        frame[f"{prefix}_risk_per_share_raw"] = risk
        frame[f"{prefix}_risk_pct"] = risk_pct
        frame[f"{prefix}_distance_to_pivot_pct"] = distance_pct
        frame[f"{prefix}_distance_score"] = distance_score
        frame[f"{prefix}_required_volume_ratio"] = required_volume_ratio
        frame[f"{prefix}_required_day_volume"] = volume_ma_50 * required_volume_ratio
        frame[f"{prefix}_watch"] = watch_mask

    vcp_mask = (
        trigger["vcp_ready"].fillna(False)
        & pd.to_numeric(trigger["vcp_maturity_score"], errors="coerce").ge(50.0)
    )
    attach_watch_block(
        trigger,
        prefix="vcp",
        pivot_col=f"vcp_pivot_{strategy.vcp_long_window}_raw",
        low_col=f"vcp_low_{strategy.vcp_long_window}_raw",
        base_mask=vcp_mask,
        distance_range=VCP_WATCH_DISTANCE_RANGE,
        required_volume_ratio=VCP_WATCH_VOLUME_RATIO,
    )

    platform_mask = (
        trigger[f"platform_high_{strategy.platform_window}_raw"].notna()
        & trigger[f"platform_low_{strategy.platform_window}_raw"].notna()
        & pd.to_numeric(trigger["platform_depth"], errors="coerce").le(strategy.max_platform_depth)
    )
    attach_watch_block(
        trigger,
        prefix="platform",
        pivot_col=f"platform_high_{strategy.platform_window}_raw",
        low_col=f"platform_low_{strategy.platform_window}_raw",
        base_mask=platform_mask,
        distance_range=PLATFORM_WATCH_DISTANCE_RANGE,
        required_volume_ratio=PLATFORM_WATCH_VOLUME_RATIO,
    )

    leader_watch_base = (
        pd.to_numeric(trigger[f"rps_{strategy.rps_window_long}"], errors="coerce").ge(90.0)
        & pd.to_numeric(trigger[f"rps_{strategy.rps_window_short}"], errors="coerce").ge(92.0)
        & pd.to_numeric(trigger["close_to_high_250"], errors="coerce").ge(0.85)
        & (pd.to_numeric(trigger[f"ma_{strategy.ma_micro_window}"], errors="coerce") > pd.to_numeric(trigger[f"ma_{strategy.ma_pullback_window}"], errors="coerce"))
        & (pd.to_numeric(trigger[f"ma_{strategy.ma_pullback_window}"], errors="coerce") > pd.to_numeric(trigger[f"ma_{strategy.ma_short_window}"], errors="coerce"))
        & pd.to_numeric(trigger["pullback_depth_20"], errors="coerce").le(0.18)
        & trigger["high_20_prev_raw"].notna()
        & trigger["low_20_prev_raw"].notna()
    )
    trigger["leader_pivot_source"] = np.where(leader_watch_base, "high_20_prev_raw", "")
    trigger["leader_pivot_price_raw"] = np.where(leader_watch_base, trigger["high_20_prev_raw"], np.nan)
    trigger["leader_base_low_price_raw"] = np.where(leader_watch_base, trigger["low_20_prev_raw"], np.nan)
    attach_watch_block(
        trigger,
        prefix="leader",
        pivot_col="leader_pivot_price_raw",
        low_col="leader_base_low_price_raw",
        base_mask=leader_watch_base,
        distance_range=LEADER_WATCH_DISTANCE_RANGE,
        required_volume_ratio=LEADER_WATCH_VOLUME_RATIO,
    )

    trigger["watch_flag_count"] = (
        trigger["vcp_watch"].astype(int)
        + trigger["platform_watch"].astype(int)
        + trigger["leader_watch"].astype(int)
    )
    trigger["best_watch_distance_score"] = trigger[
        ["vcp_distance_score", "platform_distance_score", "leader_distance_score"]
    ].max(axis=1, skipna=True)
    trigger["trigger_watch_score"] = (
        trigger["pool_score"] * 0.35
        + trigger["best_watch_distance_score"] * 0.25
        + trigger["vcp_maturity_score"].fillna(0.0) * 0.25
        + trigger["vcp_volume_stability_score"].fillna(0.0) * 0.15
    )
    trigger["trigger_watch_priority"] = (
        trigger["vcp_watch"].astype(int) * 3
        + trigger["platform_watch"].astype(int) * 2
        + trigger["leader_watch"].astype(int) * 1
    )
    trigger = trigger[
        trigger["vcp_watch"] | trigger["platform_watch"] | trigger["leader_watch"]
    ].copy()
    if trigger.empty:
        return trigger

    trigger = trigger.sort_values(
        [
            "trade_date",
            "trigger_watch_priority",
            "watch_flag_count",
            "trigger_watch_score",
            "best_watch_distance_score",
            "pool_score",
            "code",
        ],
        ascending=[True, False, False, False, False, False, True],
    ).reset_index(drop=True)
    return trigger


def build_tradeable_setup_list(pool: pd.DataFrame) -> pd.DataFrame:
    if pool.empty:
        return pool.copy()

    tradeable = pool[pool["setup_type"].isin(TRADEABLE_SETUP_TYPES)].copy()
    if tradeable.empty:
        return tradeable

    tradeable["tradeable_priority"] = np.select(
        [
            tradeable["setup_type"] == "vcp_breakout",
            tradeable["setup_type"] == "platform_breakout",
            tradeable["setup_type"] == "leader_continuation",
        ],
        [3, 2, 1],
        default=0,
    )
    tradeable["tradeable_setup_score"] = tradeable["execution_score"]
    tradeable = tradeable.sort_values(
        [
            "trade_date",
            "tradeable_priority",
            "tradeable_setup_score",
            "pivot_proximity_score",
            "pool_score",
            "code",
        ],
        ascending=[True, False, False, False, False, True],
    ).reset_index(drop=True)
    return tradeable


def build_diagnostic_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    for column in ("trade_date", "pubDate", "statDate", "latest_feature_pubDate", "latest_feature_statDate"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    sort_columns = [column for column in ("trade_date", "code", "pubDate", "statDate") if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
    return frame


def build_text_report(pool: pd.DataFrame, trade_dates: list[datetime], *, report_kind: str) -> str:
    grouped: dict[pd.Timestamp, pd.DataFrame] = {
        pd.Timestamp(trade_date): group.copy()
        for trade_date, group in pool.groupby("trade_date", sort=True)
    }

    lines: list[str] = []
    for trade_date in trade_dates:
        ts = pd.Timestamp(trade_date)
        daily = grouped.get(ts)
        if daily is None or daily.empty:
            lines.append(f"{ts:%Y-%m-%d} [0]")
            continue

        lines.append(f"{ts:%Y-%m-%d} [{len(daily)}]")
        for row in daily.itertuples(index=False):
            code_name = str(getattr(row, "code_name", "") or "")
            if report_kind == "research_watchlist":
                lines.append(
                    ",".join(
                        [
                            row.code,
                            code_name,
                            f"{float(getattr(row, 'research_watch_score')):.4f}",
                            f"{float(getattr(row, 'pool_score')):.4f}",
                            f"{float(getattr(row, 'growth_leadership_score')):.4f}",
                            f"{float(getattr(row, 'vcp_maturity_score')):.4f}",
                            str(getattr(row, "setup_type", "") or ""),
                        ]
                    )
                )
            elif report_kind == "trigger_watchlist":
                lines.append(
                    ",".join(
                        [
                            row.code,
                            code_name,
                            f"{float(getattr(row, 'trigger_watch_score')):.4f}",
                            f"{float(getattr(row, 'pool_score')):.4f}",
                            f"{float(getattr(row, 'watch_flag_count')):.0f}",
                            str(getattr(row, "setup_type", "") or ""),
                            "1" if bool(getattr(row, "vcp_watch", False)) else "0",
                            "1" if bool(getattr(row, "platform_watch", False)) else "0",
                            "1" if bool(getattr(row, "leader_watch", False)) else "0",
                            f"{float(getattr(row, 'vcp_contraction_count')):.0f}" if pd.notna(getattr(row, "vcp_contraction_count", np.nan)) else "",
                            f"{float(getattr(row, 'vcp_maturity_score')):.4f}" if pd.notna(getattr(row, "vcp_maturity_score", np.nan)) else "",
                            f"{float(getattr(row, 'vcp_volume_stability_score')):.4f}" if pd.notna(getattr(row, "vcp_volume_stability_score", np.nan)) else "",
                            f"{float(getattr(row, 'vcp_breakout_score')):.4f}" if pd.notna(getattr(row, "vcp_breakout_score", np.nan)) else "",
                            f"{float(getattr(row, 'volume_ratio_50')):.4f}" if pd.notna(getattr(row, "volume_ratio_50", np.nan)) else "",
                            f"{float(getattr(row, 'vcp_required_volume_ratio')):.4f}" if pd.notna(getattr(row, "vcp_required_volume_ratio", np.nan)) else "",
                            f"{float(getattr(row, 'vcp_required_day_volume')):.0f}" if pd.notna(getattr(row, "vcp_required_day_volume", np.nan)) else "",
                            f"{float(getattr(row, 'close_location_score')):.4f}" if pd.notna(getattr(row, "close_location_score", np.nan)) else "",
                            f"{float(getattr(row, 'vcp_pivot_price_raw')):.4f}" if pd.notna(getattr(row, "vcp_pivot_price_raw", np.nan)) else "",
                            f"{float(getattr(row, 'platform_pivot_price_raw')):.4f}" if pd.notna(getattr(row, "platform_pivot_price_raw", np.nan)) else "",
                            f"{float(getattr(row, 'leader_pivot_price_raw')):.4f}" if pd.notna(getattr(row, "leader_pivot_price_raw", np.nan)) else "",
                            f"{float(getattr(row, 'vcp_distance_to_pivot_pct')):.4f}" if pd.notna(getattr(row, "vcp_distance_to_pivot_pct", np.nan)) else "",
                            f"{float(getattr(row, 'platform_distance_to_pivot_pct')):.4f}" if pd.notna(getattr(row, "platform_distance_to_pivot_pct", np.nan)) else "",
                            f"{float(getattr(row, 'leader_distance_to_pivot_pct')):.4f}" if pd.notna(getattr(row, "leader_distance_to_pivot_pct", np.nan)) else "",
                        ]
                    )
                )
            elif report_kind == "tradeable_setup_list":
                lines.append(
                    ",".join(
                        [
                            row.code,
                            code_name,
                            str(getattr(row, "setup_type", "") or ""),
                            f"{float(getattr(row, 'tradeable_priority')):.0f}",
                            f"{float(getattr(row, 'tradeable_setup_score')):.4f}",
                            f"{float(getattr(row, 'pivot_proximity_score')):.4f}",
                            f"{float(getattr(row, 'stop_distance_score')):.4f}",
                            f"{float(getattr(row, 'pool_score')):.4f}",
                            f"{float(getattr(row, 'vcp_contraction_count')):.0f}" if pd.notna(getattr(row, "vcp_contraction_count", np.nan)) else "",
                            f"{float(getattr(row, 'vcp_maturity_score')):.4f}" if pd.notna(getattr(row, "vcp_maturity_score", np.nan)) else "",
                            f"{float(getattr(row, 'vcp_volume_stability_score')):.4f}" if pd.notna(getattr(row, "vcp_volume_stability_score", np.nan)) else "",
                            f"{float(getattr(row, 'vcp_breakout_score')):.4f}" if pd.notna(getattr(row, "vcp_breakout_score", np.nan)) else "",
                            f"{float(getattr(row, 'volume_ratio_50')):.4f}" if pd.notna(getattr(row, "volume_ratio_50", np.nan)) else "",
                            f"{float(getattr(row, 'close_location_score')):.4f}" if pd.notna(getattr(row, "close_location_score", np.nan)) else "",
                        ]
                    )
                )
            else:
                lines.append(
                    ",".join(
                        [
                            row.code,
                            code_name,
                            f"{float(getattr(row, 'pool_score')):.4f}",
                            f"{float(getattr(row, 'execution_score')):.4f}",
                            str(getattr(row, "setup_type", "") or ""),
                        ]
                    )
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_results(
    output_dir: Path,
    pool: pd.DataFrame,
    research_watchlist: pd.DataFrame,
    trigger_watchlist: pd.DataFrame,
    tradeable_setup_list: pd.DataFrame,
    fundamental_missing_records: pd.DataFrame,
    full_pool_report: str,
    research_watchlist_report: str,
    trigger_watchlist_report: str,
    tradeable_setup_report: str,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pool.to_csv(output_dir / "daily_pool.csv", index=False)
    research_watchlist.to_csv(output_dir / "research_watchlist.csv", index=False)
    trigger_watchlist.to_csv(output_dir / "trigger_watchlist.csv", index=False)
    tradeable_setup_list.to_csv(output_dir / "tradeable_setup_list.csv", index=False)
    fundamental_missing_records.to_csv(output_dir / "fundamental_missing_records.csv", index=False)
    (output_dir / "daily_pool.txt").write_text(full_pool_report, encoding="utf-8")
    (output_dir / "research_watchlist.txt").write_text(research_watchlist_report, encoding="utf-8")
    (output_dir / "trigger_watchlist.txt").write_text(trigger_watchlist_report, encoding="utf-8")
    (output_dir / "tradeable_setup_list.txt").write_text(tradeable_setup_report, encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def run_subjective_pool(
    args: argparse.Namespace,
    *,
    strategy_builder=build_strategy,
    script_name: str | None = None,
) -> dict[str, object]:
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    db_client = DuckDBConfig(read_only=True)
    data_portal = DuckDBDataPortal(db_client)
    trade_dates = data_portal.get_trade_calendar(start_date, end_date)
    if not trade_dates:
        raise ValueError("No trade dates available in the requested date range.")

    strategy = strategy_builder(args)
    research_store = ResearchDailyHistoryStore(data_portal)
    strategy.prepare(data_portal, trade_dates, research_store=research_store)

    pool = collect_daily_pool_frame(strategy, trade_dates, data_portal)
    fundamental_missing_records = build_diagnostic_frame(strategy.fundamental_missing_records)
    research_watchlist = build_research_watchlist(pool, strategy, top_pct=args.watchlist_top_pct)
    trigger_watchlist = build_trigger_watchlist(pool, strategy, trade_dates)
    tradeable_setup_list = build_tradeable_setup_list(pool)
    full_pool_report = build_text_report(pool, trade_dates, report_kind="daily_pool")
    research_watchlist_report = build_text_report(
        research_watchlist,
        trade_dates,
        report_kind="research_watchlist",
    )
    trigger_watchlist_report = build_text_report(
        trigger_watchlist,
        trade_dates[:-1],
        report_kind="trigger_watchlist",
    )
    tradeable_setup_report = build_text_report(
        tradeable_setup_list,
        trade_dates,
        report_kind="tradeable_setup_list",
    )

    output_dir = build_run_output_dir(Path(args.output_dir), trade_dates[0], trade_dates[-1])
    export_results(
        output_dir,
        pool,
        research_watchlist,
        trigger_watchlist,
        tradeable_setup_list,
        fundamental_missing_records,
        full_pool_report,
        research_watchlist_report,
        trigger_watchlist_report,
        tradeable_setup_report,
        metadata={
            "script_name": script_name or Path(__file__).name,
            "date_range": f"{trade_dates[0]:%Y-%m-%d} -> {trade_dates[-1]:%Y-%m-%d}",
            "trade_date_count": len(trade_dates),
            "pool_row_count": int(len(pool)),
            "research_watchlist_row_count": int(len(research_watchlist)),
            "trigger_watchlist_row_count": int(len(trigger_watchlist)),
            "tradeable_setup_row_count": int(len(tradeable_setup_list)),
            "fundamental_feature_collection": getattr(strategy, "fundamental_feature_collection", ""),
            "fundamental_feature_version": getattr(strategy, "fundamental_feature_version", ""),
            "fundamental_missing_row_count": int(len(fundamental_missing_records)),
            "watchlist_top_pct": float(args.watchlist_top_pct),
            "dropped_setup_type": "new_high_breakout",
            "parameters": vars(args),
        },
    )

    summary = {
        "output_dir": str(output_dir),
        "trade_date_count": len(trade_dates),
        "non_empty_trade_date_count": int(pool["trade_date"].nunique()) if not pool.empty else 0,
        "pool_row_count": int(len(pool)),
        "research_watchlist_row_count": int(len(research_watchlist)),
        "trigger_watchlist_row_count": int(len(trigger_watchlist)),
        "tradeable_setup_row_count": int(len(tradeable_setup_list)),
        "fundamental_missing_row_count": int(len(fundamental_missing_records)),
        "vcp_mode": getattr(args, "vcp_mode", "rolling"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    run_subjective_pool(parse_args())


if __name__ == "__main__":
    main()
