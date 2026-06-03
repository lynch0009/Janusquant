"""回测结果分析模块。

这里负责把订单、成交、权益曲线和已平仓持仓整理成结构化分析结果，
并导出成 JSON、CSV 和 Markdown 报告。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.execution.config import EngineConfig

from .reporter import BacktestMarkdownReporter


def _safe_float(value: Any) -> float | None:
    """把 pandas / numpy 标量安全转换成 Python float。"""

    if value is None or pd.isna(value):
        return None
    return float(value)


@dataclass
class BacktestAnalyticsReport:
    """承载回测汇总指标和各类明细表。"""

    summary: dict[str, float | int | None]
    equity: pd.DataFrame
    benchmark: pd.DataFrame
    daily_returns: pd.DataFrame
    order_stats: pd.DataFrame
    skip_reason_stats: pd.DataFrame
    exit_reason_stats: pd.DataFrame
    position_stats: pd.DataFrame
    monthly_returns: pd.DataFrame
    monthly_excess_returns: pd.DataFrame
    pnl_by_code: pd.DataFrame
    score_bucket_stats: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """导出成适合 JSON 序列化的普通字典。"""

        return {
            "summary": self.summary,
            "equity": self.equity.to_dict(orient="records"),
            "benchmark": self.benchmark.to_dict(orient="records"),
            "daily_returns": self.daily_returns.to_dict(orient="records"),
            "order_stats": self.order_stats.to_dict(orient="records"),
            "skip_reason_stats": self.skip_reason_stats.to_dict(orient="records"),
            "exit_reason_stats": self.exit_reason_stats.to_dict(orient="records"),
            "position_stats": self.position_stats.to_dict(orient="records"),
            "monthly_returns": self.monthly_returns.to_dict(orient="records"),
            "monthly_excess_returns": self.monthly_excess_returns.to_dict(orient="records"),
            "pnl_by_code": self.pnl_by_code.to_dict(orient="records"),
            "score_bucket_stats": self.score_bucket_stats.to_dict(orient="records"),
        }

    def export(self, output_dir: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """把分析结果整体导出到目录，便于复盘和落盘。"""

        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        with (target_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(self.summary, file, ensure_ascii=False, indent=2, default=str)

        exports = {
            "equity.csv": self.equity,
            "benchmark.csv": self.benchmark,
            "daily_returns.csv": self.daily_returns,
            "order_stats.csv": self.order_stats,
            "skip_reason_stats.csv": self.skip_reason_stats,
            "exit_reason_stats.csv": self.exit_reason_stats,
            "position_stats.csv": self.position_stats,
            "monthly_returns.csv": self.monthly_returns,
            "monthly_excess_returns.csv": self.monthly_excess_returns,
            "pnl_by_code.csv": self.pnl_by_code,
            "score_bucket_stats.csv": self.score_bucket_stats,
        }
        for filename, frame in exports.items():
            frame.to_csv(target_dir / filename, index=False)

        BacktestMarkdownReporter().generate(self, target_dir, metadata=metadata)


class BacktestAnalyzer:
    """把回测账本结果整理成策略评估报告。"""

    def __init__(self, *, annual_trading_days: int = 252, risk_free_rate: float | None = None):
        """初始化年化参数和无风险利率。"""

        self.annual_trading_days = annual_trading_days
        self.risk_free_rate = EngineConfig().risk_free_rate if risk_free_rate is None else risk_free_rate

    def analyze(self, result) -> BacktestAnalyticsReport:
        """对回测结果做一次完整分析。"""

        equity = result.equity_frame().copy()
        benchmark = result.benchmark_frame().copy()
        orders = result.orders_frame().copy()
        positions = result.closed_positions_frame().copy()

        equity = self._prepare_equity_frame(equity)
        benchmark = self._prepare_benchmark_frame(benchmark, equity)
        positions = self._prepare_positions_frame(positions)
        daily_returns = self._build_daily_returns(equity)
        summary = self._build_summary(result, equity, benchmark, daily_returns, orders, positions)
        order_stats = self._build_order_stats(orders)
        skip_reason_stats = self._build_skip_reason_stats(orders)
        exit_reason_stats = self._build_exit_reason_stats(positions)
        position_stats = self._build_position_stats(positions)
        monthly_returns = self._build_monthly_returns(equity)
        monthly_excess_returns = self._build_monthly_excess_returns(benchmark)
        pnl_by_code = self._build_pnl_by_code(positions)
        score_bucket_stats = self._build_score_bucket_stats(positions)

        return BacktestAnalyticsReport(
            summary=summary,
            equity=equity,
            benchmark=benchmark,
            daily_returns=daily_returns,
            order_stats=order_stats,
            skip_reason_stats=skip_reason_stats,
            exit_reason_stats=exit_reason_stats,
            position_stats=position_stats,
            monthly_returns=monthly_returns,
            monthly_excess_returns=monthly_excess_returns,
            pnl_by_code=pnl_by_code,
            score_bucket_stats=score_bucket_stats,
        )

    def _prepare_equity_frame(self, equity: pd.DataFrame) -> pd.DataFrame:
        """标准化权益曲线，并补充回撤列。"""

        if equity.empty:
            return equity

        frame = equity.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.sort_values("trade_date").reset_index(drop=True)
        rolling_peak = frame["total_equity"].cummax()
        frame["drawdown"] = frame["total_equity"] / rolling_peak - 1
        return frame

    def _prepare_positions_frame(self, positions: pd.DataFrame) -> pd.DataFrame:
        """标准化已平仓持仓表，并统一持有交易日口径。"""

        if positions.empty:
            return positions

        frame = positions.copy()
        if "entry_trade_date" in frame.columns:
            frame["entry_trade_date"] = pd.to_datetime(frame["entry_trade_date"])
        if "exit_trade_date" in frame.columns:
            frame["exit_trade_date"] = pd.to_datetime(frame["exit_trade_date"])

        if "holding_trade_days" not in frame.columns:
            if {"entry_trade_index", "exit_trade_index"}.issubset(frame.columns):
                frame["holding_trade_days"] = frame["exit_trade_index"] - frame["entry_trade_index"]
            elif {"entry_trade_date", "exit_trade_date"}.issubset(frame.columns):
                frame["holding_trade_days"] = (frame["exit_trade_date"] - frame["entry_trade_date"]).dt.days

        frame["holding_trade_days"] = frame["holding_trade_days"].fillna(0).astype(int)
        frame["holding_days"] = frame["holding_trade_days"]
        return frame

    def _prepare_benchmark_frame(self, benchmark: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
        """标准化基准指数曲线，并换算成可对比的权益线。"""

        columns = [
            "trade_date",
            "benchmark_code",
            "benchmark_close",
            "benchmark_daily_return",
            "benchmark_equity",
            "benchmark_drawdown",
            "strategy_return",
            "benchmark_return",
            "excess_return",
            "relative_equity",
            "relative_return",
            "excess_drawdown",
        ]
        if benchmark.empty or equity.empty:
            return pd.DataFrame(columns=columns)

        frame = benchmark.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["benchmark_close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["benchmark_code"] = frame["code"].astype(str)
        frame = frame.dropna(subset=["trade_date", "benchmark_close"]).sort_values("trade_date")
        if frame.empty:
            return pd.DataFrame(columns=columns)

        aligned = (
            equity[["trade_date"]]
            .drop_duplicates()
            .sort_values("trade_date")
            .merge(
                frame[["trade_date", "benchmark_code", "benchmark_close"]],
                on="trade_date",
                how="left",
            )
        )
        benchmark_code = frame["benchmark_code"].dropna()
        if not benchmark_code.empty:
            aligned["benchmark_code"] = aligned["benchmark_code"].fillna(str(benchmark_code.iloc[0]))
        aligned["benchmark_close"] = aligned["benchmark_close"].ffill()
        aligned = aligned.dropna(subset=["benchmark_close"]).reset_index(drop=True)
        if aligned.empty:
            return pd.DataFrame(columns=columns)

        initial_equity = float(equity["total_equity"].iloc[0])
        initial_close = float(aligned["benchmark_close"].iloc[0])
        if initial_close <= 0:
            return pd.DataFrame(columns=columns)

        aligned["benchmark_daily_return"] = aligned["benchmark_close"].pct_change().fillna(0.0)
        aligned["benchmark_equity"] = initial_equity * aligned["benchmark_close"] / initial_close
        rolling_peak = aligned["benchmark_equity"].cummax()
        aligned["benchmark_drawdown"] = aligned["benchmark_equity"] / rolling_peak - 1
        strategy_equity = equity[["trade_date", "total_equity"]].copy()
        aligned = aligned.merge(strategy_equity, on="trade_date", how="left")
        aligned["strategy_return"] = aligned["total_equity"] / initial_equity - 1
        aligned["benchmark_return"] = aligned["benchmark_equity"] / initial_equity - 1
        aligned["excess_return"] = aligned["strategy_return"] - aligned["benchmark_return"]
        aligned["relative_equity"] = aligned["total_equity"] / aligned["benchmark_equity"]
        aligned["relative_return"] = aligned["relative_equity"] - 1
        relative_peak = aligned["relative_equity"].cummax()
        aligned["excess_drawdown"] = aligned["relative_equity"] / relative_peak - 1
        return aligned[columns]

    def _build_daily_returns(self, equity: pd.DataFrame) -> pd.DataFrame:
        """由权益曲线推导逐日收益率。"""

        if equity.empty:
            return pd.DataFrame(columns=["trade_date", "daily_return"])

        frame = equity[["trade_date", "total_equity"]].copy()
        frame["daily_return"] = frame["total_equity"].pct_change().fillna(0.0)
        return frame[["trade_date", "daily_return"]]

    def _build_summary(
        self,
        result,
        equity: pd.DataFrame,
        benchmark: pd.DataFrame,
        daily_returns: pd.DataFrame,
        orders: pd.DataFrame,
        positions: pd.DataFrame,
    ) -> dict[str, float | int | None]:
        """生成一页式汇总指标。"""

        summary: dict[str, float | int | None] = {
            "order_count": int(len(result.orders)),
            "trade_count": int(len(result.trades)),
            "closed_position_count": int(len(result.closed_positions)),
            "open_position_count": int(len(result.final_positions)),
            "filled_order_count": 0,
            "skipped_order_count": 0,
            "fill_rate": None,
            "final_equity": None,
            "total_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "benchmark_total_return": None,
            "benchmark_annualized_return": None,
            "benchmark_max_drawdown": None,
            "excess_return": None,
            "excess_annualized_return": None,
            "relative_return": None,
            "excess_max_drawdown": None,
            "sharpe": None,
            "max_drawdown": None,
            "calmar": None,
            "win_rate": None,
            "avg_holding_days": None,
            "avg_position_return": None,
            "avg_position_pnl": None,
            "profit_factor": None,
            "payoff_ratio": None,
            "commission_total": None,
            "tax_total": None,
        }

        if not orders.empty:
            filled_count = int((orders["status"] == "FILLED").sum())
            skipped_count = int((orders["status"] == "SKIPPED").sum())
            summary["filled_order_count"] = filled_count
            summary["skipped_order_count"] = skipped_count
            summary["fill_rate"] = filled_count / len(orders) if len(orders) > 0 else None
            summary["commission_total"] = _safe_float(orders["commission"].fillna(0.0).sum())
            summary["tax_total"] = _safe_float(orders["tax"].fillna(0.0).sum())

        if not equity.empty:
            initial_equity = float(equity["total_equity"].iloc[0])
            final_equity = float(equity["total_equity"].iloc[-1])
            total_return = final_equity / initial_equity - 1 if initial_equity > 0 else None
            daily = daily_returns["daily_return"] if not daily_returns.empty else pd.Series(dtype=float)
            daily_std = float(daily.std(ddof=0)) if not daily.empty else 0.0
            daily_mean = float(daily.mean()) if not daily.empty else 0.0
            annualized_return = None
            if initial_equity > 0 and len(equity) > 1:
                annualized_return = (final_equity / initial_equity) ** (self.annual_trading_days / len(equity)) - 1
            annualized_volatility = daily_std * np.sqrt(self.annual_trading_days) if daily_std > 0 else 0.0
            sharpe = None
            if annualized_volatility and annualized_volatility > 0:
                sharpe = (daily_mean * self.annual_trading_days - self.risk_free_rate) / annualized_volatility
            max_drawdown = float(equity["drawdown"].min()) if "drawdown" in equity.columns else None
            calmar = None
            if annualized_return is not None and max_drawdown is not None and max_drawdown < 0:
                calmar = annualized_return / abs(max_drawdown)

            summary.update(
                {
                    "final_equity": final_equity,
                    "total_return": total_return,
                    "annualized_return": annualized_return,
                    "annualized_volatility": annualized_volatility,
                    "sharpe": sharpe,
                    "max_drawdown": max_drawdown,
                    "calmar": calmar,
                }
            )

        if not benchmark.empty:
            benchmark_initial_equity = float(benchmark["benchmark_equity"].iloc[0])
            benchmark_final_equity = float(benchmark["benchmark_equity"].iloc[-1])
            benchmark_total_return = (
                benchmark_final_equity / benchmark_initial_equity - 1
                if benchmark_initial_equity > 0
                else None
            )
            benchmark_annualized_return = None
            if benchmark_initial_equity > 0 and len(benchmark) > 1:
                benchmark_annualized_return = (
                    benchmark_final_equity / benchmark_initial_equity
                ) ** (self.annual_trading_days / len(benchmark)) - 1
            benchmark_max_drawdown = (
                float(benchmark["benchmark_drawdown"].min())
                if "benchmark_drawdown" in benchmark.columns
                else None
            )
            excess_return = None
            if summary.get("total_return") is not None and benchmark_total_return is not None:
                excess_return = float(summary["total_return"]) - benchmark_total_return
            excess_annualized_return = None
            relative_return = None
            excess_max_drawdown = None
            if "relative_equity" in benchmark.columns and not benchmark["relative_equity"].empty:
                relative_final_equity = float(benchmark["relative_equity"].iloc[-1])
                relative_return = relative_final_equity - 1
                if relative_final_equity > 0 and len(benchmark) > 1:
                    excess_annualized_return = relative_final_equity ** (
                        self.annual_trading_days / len(benchmark)
                    ) - 1
            if "excess_drawdown" in benchmark.columns:
                excess_max_drawdown = float(benchmark["excess_drawdown"].min())

            summary.update(
                {
                    "benchmark_total_return": benchmark_total_return,
                    "benchmark_annualized_return": benchmark_annualized_return,
                    "benchmark_max_drawdown": benchmark_max_drawdown,
                    "excess_return": excess_return,
                    "excess_annualized_return": excess_annualized_return,
                    "relative_return": relative_return,
                    "excess_max_drawdown": excess_max_drawdown,
                }
            )

        if not positions.empty:
            realized_return = positions["realized_return"].dropna()
            realized_pnl = positions["realized_pnl"].dropna()
            wins = positions[positions["realized_pnl"] > 0]
            losses = positions[positions["realized_pnl"] < 0]
            profit_factor = None
            if not losses.empty:
                total_loss = abs(losses["realized_pnl"].sum())
                profit_factor = wins["realized_pnl"].sum() / total_loss if total_loss > 0 else None
            payoff_ratio = None
            if not wins.empty and not losses.empty:
                avg_win = wins["realized_return"].mean()
                avg_loss = losses["realized_return"].abs().mean()
                payoff_ratio = avg_win / avg_loss if avg_loss and avg_loss > 0 else None

            summary.update(
                {
                    "win_rate": float((realized_return > 0).mean()) if not realized_return.empty else None,
                    "avg_holding_days": float(positions["holding_trade_days"].mean()) if not positions["holding_trade_days"].empty else None,
                    "avg_position_return": float(realized_return.mean()) if not realized_return.empty else None,
                    "avg_position_pnl": float(realized_pnl.mean()) if not realized_pnl.empty else None,
                    "profit_factor": _safe_float(profit_factor),
                    "payoff_ratio": _safe_float(payoff_ratio),
                }
            )

        return summary

    def _build_order_stats(self, orders: pd.DataFrame) -> pd.DataFrame:
        """按买卖方向和订单状态聚合订单表现。"""

        columns = ["side", "status", "count", "avg_filled_price", "avg_filled_quantity", "commission_total", "tax_total"]
        if orders.empty:
            return pd.DataFrame(columns=columns)

        grouped = (
            orders.groupby(["side", "status"], dropna=False)
            .agg(
                count=("order_id", "count"),
                avg_filled_price=("filled_price", "mean"),
                avg_filled_quantity=("filled_quantity", "mean"),
                commission_total=("commission", "sum"),
                tax_total=("tax", "sum"),
            )
            .reset_index()
            .sort_values(["side", "status"])
            .reset_index(drop=True)
        )
        return grouped

    def _build_skip_reason_stats(self, orders: pd.DataFrame) -> pd.DataFrame:
        """统计订单被跳过的原因分布。"""

        columns = ["skip_reason", "count"]
        if orders.empty or "skip_reason" not in orders.columns:
            return pd.DataFrame(columns=columns)

        skipped = orders[orders["status"] == "SKIPPED"].copy()
        if skipped.empty:
            return pd.DataFrame(columns=columns)

        return (
            skipped["skip_reason"]
            .fillna("UNKNOWN")
            .value_counts()
            .rename_axis("skip_reason")
            .reset_index(name="count")
        )

    def _build_position_stats(self, positions: pd.DataFrame) -> pd.DataFrame:
        """输出逐笔持仓明细，便于逐单复盘。"""

        columns = [
            "position_id",
            "code",
            "entry_trade_date",
            "exit_trade_date",
            "holding_days",
            "holding_trade_days",
            "exit_reason",
            "entry_price",
            "exit_price",
            "quantity",
            "realized_pnl",
            "realized_return",
            "score",
            "open_order_id",
            "close_order_id",
        ]
        if positions.empty:
            return pd.DataFrame(columns=columns)

        available_columns = [column for column in columns if column in positions.columns]
        return positions[available_columns].sort_values(["exit_trade_date", "code"]).reset_index(drop=True)

    def _build_exit_reason_stats(self, positions: pd.DataFrame) -> pd.DataFrame:
        """按平仓原因聚合胜率、收益和持有交易日数。"""

        columns = ["exit_reason", "trade_count", "win_rate", "avg_return", "total_pnl", "avg_holding_days"]
        if positions.empty or "exit_reason" not in positions.columns:
            return pd.DataFrame(columns=columns)

        frame = positions.copy()
        frame["exit_reason"] = frame["exit_reason"].fillna("UNKNOWN")

        grouped = (
            frame.groupby("exit_reason")
            .agg(
                trade_count=("position_id", "count"),
                win_rate=("realized_return", lambda s: (s > 0).mean()),
                avg_return=("realized_return", "mean"),
                total_pnl=("realized_pnl", "sum"),
                avg_holding_days=("holding_trade_days", "mean"),
            )
            .reset_index()
            .sort_values(["trade_count", "total_pnl"], ascending=[False, False])
            .reset_index(drop=True)
        )
        return grouped

    def _build_monthly_returns(self, equity: pd.DataFrame) -> pd.DataFrame:
        """按自然月聚合月度收益率。"""

        columns = ["month", "month_start_equity", "month_end_equity", "monthly_return"]
        if equity.empty:
            return pd.DataFrame(columns=columns)

        frame = equity[["trade_date", "total_equity"]].copy()
        frame["month"] = frame["trade_date"].dt.to_period("M").astype(str)
        monthly = (
            frame.groupby("month")
            .agg(
                month_start_equity=("total_equity", "first"),
                month_end_equity=("total_equity", "last"),
            )
            .reset_index()
        )
        monthly["monthly_return"] = monthly["month_end_equity"] / monthly["month_start_equity"] - 1
        return monthly

    def _build_monthly_excess_returns(self, benchmark: pd.DataFrame) -> pd.DataFrame:
        """按自然月聚合策略、基准和超额收益。"""

        columns = [
            "month",
            "strategy_month_start_equity",
            "strategy_month_end_equity",
            "strategy_monthly_return",
            "benchmark_month_start_equity",
            "benchmark_month_end_equity",
            "benchmark_monthly_return",
            "excess_monthly_return",
            "month_end_relative_return",
            "month_end_excess_drawdown",
        ]
        if benchmark.empty:
            return pd.DataFrame(columns=columns)

        required_columns = {"trade_date", "total_equity", "benchmark_equity", "relative_return", "excess_drawdown"}
        if not required_columns.issubset(benchmark.columns):
            return pd.DataFrame(columns=columns)

        frame = benchmark.copy()
        frame["month"] = pd.to_datetime(frame["trade_date"]).dt.to_period("M").astype(str)
        monthly = (
            frame.groupby("month")
            .agg(
                strategy_month_start_equity=("total_equity", "first"),
                strategy_month_end_equity=("total_equity", "last"),
                benchmark_month_start_equity=("benchmark_equity", "first"),
                benchmark_month_end_equity=("benchmark_equity", "last"),
                month_end_relative_return=("relative_return", "last"),
                month_end_excess_drawdown=("excess_drawdown", "last"),
            )
            .reset_index()
        )
        monthly["strategy_monthly_return"] = (
            monthly["strategy_month_end_equity"] / monthly["strategy_month_start_equity"] - 1
        )
        monthly["benchmark_monthly_return"] = (
            monthly["benchmark_month_end_equity"] / monthly["benchmark_month_start_equity"] - 1
        )
        monthly["excess_monthly_return"] = (
            monthly["strategy_monthly_return"] - monthly["benchmark_monthly_return"]
        )
        return monthly[columns]

    def _build_pnl_by_code(self, positions: pd.DataFrame) -> pd.DataFrame:
        """按股票代码聚合收益表现。"""

        columns = ["code", "trade_count", "win_rate", "total_pnl", "avg_return", "avg_holding_days"]
        if positions.empty:
            return pd.DataFrame(columns=columns)

        grouped = (
            positions.groupby("code")
            .agg(
                trade_count=("position_id", "count"),
                win_rate=("realized_return", lambda s: (s > 0).mean()),
                total_pnl=("realized_pnl", "sum"),
                avg_return=("realized_return", "mean"),
                avg_holding_days=("holding_trade_days", "mean"),
            )
            .reset_index()
            .sort_values(["total_pnl", "avg_return"], ascending=[False, False])
            .reset_index(drop=True)
        )
        return grouped

    def _build_score_bucket_stats(self, positions: pd.DataFrame) -> pd.DataFrame:
        """按信号分数分桶，观察高分组是否更稳定。"""

        columns = ["score_bucket", "trade_count", "win_rate", "avg_return", "total_pnl"]
        if positions.empty or "score" not in positions.columns:
            return pd.DataFrame(columns=columns)

        frame = positions.copy()
        frame = frame[frame["score"].notna()].copy()
        if frame.empty or len(frame) < 4:
            return pd.DataFrame(columns=columns)

        bucket_count = min(5, frame["score"].nunique())
        if bucket_count < 2:
            return pd.DataFrame(columns=columns)

        frame["score_bucket"] = pd.qcut(frame["score"], q=bucket_count, duplicates="drop")
        grouped = (
            frame.groupby("score_bucket", observed=True)
            .agg(
                trade_count=("position_id", "count"),
                win_rate=("realized_return", lambda s: (s > 0).mean()),
                avg_return=("realized_return", "mean"),
                total_pnl=("realized_pnl", "sum"),
            )
            .reset_index()
        )
        grouped["score_bucket"] = grouped["score_bucket"].astype(str)
        return grouped
