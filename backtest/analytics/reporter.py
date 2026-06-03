from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import colors as mcolors
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mdates = None
    sns = None
    mcolors = None
    font_manager = None
    FuncFormatter = None
    PercentFormatter = None


def _safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _format_number(value: Any, digits: int = 2) -> str:
    value = _safe_value(value)
    if value is None:
        return "-"
    return f"{float(value):,.{digits}f}"


def _format_percent(value: Any, digits: int = 2) -> str:
    value = _safe_value(value)
    if value is None:
        return "-"
    return f"{float(value):.{digits}%}"


def _format_int(value: Any) -> str:
    value = _safe_value(value)
    if value is None:
        return "-"
    return f"{int(round(float(value))):,}"


def _format_date(value: Any) -> str:
    value = _safe_value(value)
    if value is None:
        return "-"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    text = str(value)
    return text[:10] if len(text) >= 10 and text[4] == "-" else text


def _to_markdown_table(
    frame: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    renames: dict[str, str] | None = None,
    formatters: dict[str, Any] | None = None,
    max_rows: int | None = None,
) -> str:
    if frame.empty:
        return "_无数据_"

    working = frame.copy()
    if columns is not None:
        available = [column for column in columns if column in working.columns]
        working = working[available]
    if max_rows is not None:
        working = working.head(max_rows)

    if renames:
        working = working.rename(columns=renames)
    if formatters:
        for column, formatter in formatters.items():
            if column in working.columns:
                working[column] = working[column].map(formatter)

    for column in working.columns:
        if pd.api.types.is_datetime64_any_dtype(working[column]):
            working[column] = working[column].map(_format_date)
        else:
            working[column] = working[column].map(lambda value: "-" if _safe_value(value) is None else str(_safe_value(value)))

    header = "| " + " | ".join(working.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(working.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in working.astype(str).values.tolist()]
    return "\n".join([header, divider, *rows])


class BacktestMarkdownReporter:
    def __init__(self) -> None:
        self._configure_matplotlib()

    def generate(self, report, output_dir: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        charts_dir = target_dir / "charts"
        metadata = metadata or {}

        chart_refs = self._build_charts(report, charts_dir)
        markdown = self._build_markdown(report, chart_refs, metadata)
        report_path = target_dir / "report.md"
        report_path.write_text(markdown, encoding="utf-8")
        return report_path

    def _configure_matplotlib(self) -> None:
        if not MATPLOTLIB_AVAILABLE:
            return

        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        preferred_fonts = [
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        chosen_fonts = [font for font in preferred_fonts if font in available_fonts]
        if not chosen_fonts:
            chosen_fonts = ["DejaVu Sans"]

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.sans-serif"] = chosen_fonts
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["figure.dpi"] = 140
        plt.rcParams["savefig.bbox"] = "tight"

    def _build_charts(self, report, charts_dir: Path) -> dict[str, str]:
        if not MATPLOTLIB_AVAILABLE:
            return {}

        charts_dir.mkdir(parents=True, exist_ok=True)
        chart_refs: dict[str, str] = {}

        for key, path in (
            ("equity_drawdown", self._plot_equity_and_drawdown(report.equity, report.benchmark, charts_dir / "equity_drawdown.png")),
            ("benchmark_relative", self._plot_benchmark_relative(report.benchmark, charts_dir / "benchmark_relative.png")),
            ("daily_returns", self._plot_daily_returns(report.daily_returns, charts_dir / "daily_returns.png")),
            ("monthly_returns", self._plot_monthly_returns(report.monthly_returns, charts_dir / "monthly_returns.png")),
            ("pnl_by_code", self._plot_pnl_by_code(report.pnl_by_code, charts_dir / "pnl_by_code.png")),
            ("exit_reason_stats", self._plot_exit_reason_stats(report.exit_reason_stats, charts_dir / "exit_reason_stats.png")),
            ("score_bucket_stats", self._plot_score_bucket_stats(report.score_bucket_stats, charts_dir / "score_bucket_stats.png")),
            ("return_distribution", self._plot_return_distribution(report.position_stats, charts_dir / "return_distribution.png")),
            ("holding_vs_return", self._plot_holding_vs_return(report.position_stats, charts_dir / "holding_vs_return.png")),
        ):
            if path:
                chart_refs[key] = path
        return chart_refs

    def _plot_equity_and_drawdown(self, equity: pd.DataFrame, benchmark: pd.DataFrame, output_path: Path) -> str | None:
        if equity.empty:
            return None

        frame = equity.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        axes[0].plot(frame["trade_date"], frame["total_equity"], color="#1565c0", linewidth=2.2, label="Strategy")
        if not benchmark.empty:
            benchmark_frame = benchmark.copy()
            benchmark_frame["trade_date"] = pd.to_datetime(benchmark_frame["trade_date"])
            axes[0].plot(
                benchmark_frame["trade_date"],
                benchmark_frame["benchmark_equity"],
                color="#ef6c00",
                linewidth=2.0,
                linestyle="--",
                label="Benchmark",
            )
            axes[0].legend(loc="best")
        axes[0].set_title("Equity Curve vs Benchmark")
        axes[0].set_ylabel("Equity")
        axes[0].ticklabel_format(style="plain", axis="y")

        axes[1].fill_between(frame["trade_date"], frame["drawdown"], 0, color="#ef5350", alpha=0.35)
        axes[1].plot(frame["trade_date"], frame["drawdown"], color="#c62828", linewidth=1.5, label="Strategy Drawdown")
        if not benchmark.empty and "benchmark_drawdown" in benchmark.columns:
            benchmark_frame = benchmark.copy()
            benchmark_frame["trade_date"] = pd.to_datetime(benchmark_frame["trade_date"])
            axes[1].plot(
                benchmark_frame["trade_date"],
                benchmark_frame["benchmark_drawdown"],
                color="#ef6c00",
                linewidth=1.6,
                linestyle="--",
                label="Benchmark Drawdown",
            )
            axes[1].legend(loc="lower left")
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        fig.savefig(output_path)
        plt.close(fig)
        return output_path.name

    def _plot_benchmark_relative(self, benchmark: pd.DataFrame, output_path: Path) -> str | None:
        required_columns = {"trade_date", "excess_return", "relative_return", "excess_drawdown"}
        if benchmark.empty or not required_columns.issubset(benchmark.columns):
            return None

        frame = benchmark.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.dropna(subset=["trade_date", "excess_return", "relative_return", "excess_drawdown"])
        if frame.empty:
            return None

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})

        excess_colors = np.where(frame["excess_return"] >= 0, "#2e7d32", "#c62828")
        axes[0].bar(frame["trade_date"], frame["excess_return"], color=excess_colors, width=0.8, alpha=0.75)
        axes[0].axhline(0, color="#546e7a", linewidth=1)
        axes[0].set_title("Excess Return vs Benchmark")
        axes[0].set_ylabel("Excess Return")
        axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))

        axes[1].plot(frame["trade_date"], frame["relative_return"], color="#6a1b9a", linewidth=2.0)
        axes[1].axhline(0, color="#546e7a", linewidth=1)
        axes[1].set_title("Relative Return Curve")
        axes[1].set_ylabel("Relative Return")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))

        axes[2].fill_between(frame["trade_date"], frame["excess_drawdown"], 0, color="#8e24aa", alpha=0.25)
        axes[2].plot(frame["trade_date"], frame["excess_drawdown"], color="#6a1b9a", linewidth=1.8)
        axes[2].set_title("Excess Drawdown")
        axes[2].set_ylabel("Excess Drawdown")
        axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        fig.savefig(output_path)
        plt.close(fig)
        return output_path.name

    def _plot_daily_returns(self, daily_returns: pd.DataFrame, output_path: Path) -> str | None:
        if daily_returns.empty:
            return None

        frame = daily_returns.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        colors = np.where(frame["daily_return"] >= 0, "#2e7d32", "#c62828")
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.bar(frame["trade_date"], frame["daily_return"], color=colors, width=0.8)
        ax.axhline(0, color="#546e7a", linewidth=1)
        ax.set_title("Daily Returns")
        ax.set_ylabel("Return")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        fig.savefig(output_path)
        plt.close(fig)
        return output_path.name

    def _plot_monthly_returns(self, monthly_returns: pd.DataFrame, output_path: Path) -> str | None:
        if monthly_returns.empty:
            return None

        frame = monthly_returns.copy()
        frame["month"] = pd.to_datetime(frame["month"], format="%Y-%m")
        frame["year"] = frame["month"].dt.year
        frame["month_num"] = frame["month"].dt.month

        heatmap = (
            frame.pivot(index="year", columns="month_num", values="monthly_return")
            .reindex(columns=range(1, 13))
            .sort_index(ascending=False)
        )
        if heatmap.empty:
            return None

        annot = heatmap.apply(lambda column: column.map(lambda value: "" if pd.isna(value) else f"{value:.1%}"))
        mask = heatmap.isna()

        max_abs = float(np.nanmax(np.abs(heatmap.to_numpy(dtype=float)))) if not mask.all().all() else 0.0
        if max_abs == 0.0:
            max_abs = 0.01

        cmap = sns.diverging_palette(130, 15, s=90, l=55, center="light", as_cmap=True)
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

        figure_height = max(3.6, 1.6 + 0.58 * len(heatmap.index))
        fig, ax = plt.subplots(figsize=(11.5, figure_height))
        sns.heatmap(
            heatmap,
            mask=mask,
            cmap=cmap,
            norm=norm,
            annot=annot,
            fmt="",
            linewidths=0.6,
            linecolor="#f3f4f6",
            cbar_kws={"format": FuncFormatter(lambda value, _: f"{value:.0%}")},
            annot_kws={"fontsize": 9},
            ax=ax,
        )

        ax.set_title("Monthly Returns Heatmap")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        ax.set_xticklabels([f"{month:02d}" for month in heatmap.columns], rotation=0)
        ax.set_yticklabels([str(year) for year in heatmap.index], rotation=0)

        for text in ax.texts:
            if not text.get_text():
                continue
            x_pos, y_pos = text.get_position()
            row_index = int(round(y_pos - 0.5))
            column_index = int(round(x_pos - 0.5))
            value = heatmap.iloc[row_index, column_index]
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text.set_color("#111827" if luminance > 0.62 else "#f9fafb")

        fig.savefig(output_path)
        plt.close(fig)
        return output_path.name

    def _plot_pnl_by_code(self, pnl_by_code: pd.DataFrame, output_path: Path) -> str | None:
        if pnl_by_code.empty:
            return None

        winners = pnl_by_code.sort_values("total_pnl", ascending=False).head(8)
        losers = pnl_by_code.sort_values("total_pnl", ascending=True).head(8)
        frame = pd.concat([winners, losers], ignore_index=True).drop_duplicates(subset=["code"]).sort_values("total_pnl")
        if frame.empty:
            return None

        colors = np.where(frame["total_pnl"] >= 0, "#2e7d32", "#c62828")
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(frame["code"], frame["total_pnl"], color=colors)
        ax.axvline(0, color="#546e7a", linewidth=1)
        ax.set_title("P&L by Code")
        ax.set_xlabel("Total P&L")

        fig.savefig(output_path)
        plt.close(fig)
        return output_path.name

    def _plot_exit_reason_stats(self, exit_reason_stats: pd.DataFrame, output_path: Path) -> str | None:
        if exit_reason_stats.empty:
            return None

        frame = exit_reason_stats.copy().sort_values("total_pnl")
        colors = np.where(frame["total_pnl"] >= 0, "#2e7d32", "#c62828")
        fig, ax1 = plt.subplots(figsize=(11, 5.5))
        ax1.bar(frame["exit_reason"], frame["total_pnl"], color=colors, alpha=0.85)
        ax1.axhline(0, color="#546e7a", linewidth=1)
        ax1.set_ylabel("Total P&L")
        ax1.set_title("Exit Reason Breakdown")

        ax2 = ax1.twinx()
        ax2.plot(frame["exit_reason"], frame["trade_count"], color="#1565c0", marker="o", linewidth=2)
        ax2.set_ylabel("Trade Count")

        fig.savefig(output_path)
        plt.close(fig)
        return output_path.name

    def _plot_score_bucket_stats(self, score_bucket_stats: pd.DataFrame, output_path: Path) -> str | None:
        if score_bucket_stats.empty:
            return None

        frame = score_bucket_stats.copy()
        fig, ax1 = plt.subplots(figsize=(11, 5.5))
        bars = ax1.bar(frame["score_bucket"], frame["avg_return"], color="#42a5f5", alpha=0.85)
        ax1.axhline(0, color="#546e7a", linewidth=1)
        ax1.set_ylabel("Avg Return")
        ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax1.set_title("Score Bucket Performance")

        ax2 = ax1.twinx()
        ax2.plot(frame["score_bucket"], frame["win_rate"], color="#ef6c00", marker="o", linewidth=2)
        ax2.set_ylabel("Win Rate")
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))

        for bar, trade_count in zip(bars, frame["trade_count"]):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(trade_count)}", ha="center", va="bottom", fontsize=9)

        fig.savefig(output_path)
        plt.close(fig)
        return output_path.name

    def _plot_return_distribution(self, position_stats: pd.DataFrame, output_path: Path) -> str | None:
        if position_stats.empty or "realized_return" not in position_stats.columns:
            return None

        frame = position_stats[position_stats["realized_return"].notna()].copy()
        if frame.empty:
            return None

        fig, ax = plt.subplots(figsize=(10.5, 5))
        ax.hist(frame["realized_return"], bins=min(20, max(5, len(frame))), color="#5c6bc0", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="#c62828", linewidth=1)
        ax.set_title("Realized Return Distribution")
        ax.set_xlabel("Return")
        ax.set_ylabel("Trade Count")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))

        fig.savefig(output_path)
        plt.close(fig)
        return output_path.name

    def _plot_holding_vs_return(self, position_stats: pd.DataFrame, output_path: Path) -> str | None:
        required_columns = {"holding_trade_days", "realized_return"}
        if position_stats.empty or not required_columns.issubset(position_stats.columns):
            return None

        frame = position_stats.dropna(subset=["holding_trade_days", "realized_return"]).copy()
        if frame.empty:
            return None

        fig, ax = plt.subplots(figsize=(10.5, 5))
        scatter = ax.scatter(
            frame["holding_trade_days"],
            frame["realized_return"],
            c=frame["realized_pnl"] if "realized_pnl" in frame.columns else frame["realized_return"],
            cmap="RdYlGn",
            alpha=0.8,
            s=70,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.axhline(0, color="#546e7a", linewidth=1)
        ax.set_title("Holding Days vs Realized Return")
        ax.set_xlabel("Holding Trade Days")
        ax.set_ylabel("Realized Return")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        fig.colorbar(scatter, ax=ax, label="P&L / Return Intensity")

        fig.savefig(output_path)
        plt.close(fig)
        return output_path.name

    def _build_markdown(self, report, chart_refs: dict[str, str], metadata: dict[str, Any]) -> str:
        summary = report.summary
        position_stats = report.position_stats
        auto_insights = self._build_auto_insights(report)
        top_winners = self._select_top_positions(position_stats, ascending=False)
        top_losers = self._select_top_positions(position_stats, ascending=True)

        lines: list[str] = []
        lines.append("# 回测分析报告")
        lines.append("")
        lines.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        strategy_name = metadata.get("strategy_name")
        if strategy_name:
            lines.append(f"- 策略名称: {strategy_name}")
        if not report.benchmark.empty and "benchmark_code" in report.benchmark.columns:
            benchmark_code = report.benchmark["benchmark_code"].dropna()
            if not benchmark_code.empty:
                lines.append(f"- 对比基准: {benchmark_code.iloc[0]}")
        if "script_name" in metadata:
            lines.append(f"- 生成脚本: {metadata['script_name']}")
        if "date_range" in metadata:
            lines.append(f"- 回测区间: {metadata['date_range']}")
        elif not report.equity.empty:
            lines.append(
                f"- 回测区间: {_format_date(report.equity['trade_date'].min())} 至 {_format_date(report.equity['trade_date'].max())}"
            )
        lines.append("")

        lines.append("## 核心指标")
        lines.append("")
        summary_frame = pd.DataFrame(
            [
                {"指标": "期末权益", "值": _format_number(summary.get("final_equity"))},
                {"指标": "总收益率", "值": _format_percent(summary.get("total_return"))},
                {"指标": "年化收益率", "值": _format_percent(summary.get("annualized_return"))},
                {"指标": "年化波动率", "值": _format_percent(summary.get("annualized_volatility"))},
                {"指标": "基准总收益率", "值": _format_percent(summary.get("benchmark_total_return"))},
                {"指标": "基准年化收益率", "值": _format_percent(summary.get("benchmark_annualized_return"))},
                {"指标": "基准最大回撤", "值": _format_percent(summary.get("benchmark_max_drawdown"))},
                {"指标": "超额收益", "值": _format_percent(summary.get("excess_return"))},
                {"指标": "超额年化收益", "值": _format_percent(summary.get("excess_annualized_return"))},
                {"指标": "相对收益", "值": _format_percent(summary.get("relative_return"))},
                {"指标": "超额最大回撤", "值": _format_percent(summary.get("excess_max_drawdown"))},
                {"指标": "夏普比率", "值": _format_number(summary.get("sharpe"), 3)},
                {"指标": "最大回撤", "值": _format_percent(summary.get("max_drawdown"))},
                {"指标": "Calmar", "值": _format_number(summary.get("calmar"), 3)},
                {"指标": "胜率", "值": _format_percent(summary.get("win_rate"))},
                {"指标": "平均持有天数", "值": _format_number(summary.get("avg_holding_days"), 2)},
                {"指标": "平均单笔收益率", "值": _format_percent(summary.get("avg_position_return"))},
                {"指标": "平均单笔盈亏", "值": _format_number(summary.get("avg_position_pnl"))},
                {"指标": "Profit Factor", "值": _format_number(summary.get("profit_factor"), 3)},
                {"指标": "Payoff Ratio", "值": _format_number(summary.get("payoff_ratio"), 3)},
                {"指标": "成交订单数", "值": _format_int(summary.get("filled_order_count"))},
                {"指标": "跳过订单数", "值": _format_int(summary.get("skipped_order_count"))},
                {"指标": "手续费合计", "值": _format_number(summary.get("commission_total"))},
                {"指标": "印花税合计", "值": _format_number(summary.get("tax_total"))},
            ]
        )
        lines.append(_to_markdown_table(summary_frame))
        lines.append("")

        if metadata.get("parameters"):
            lines.append("## 运行参数")
            lines.append("")
            params = pd.DataFrame(
                [{"参数": key, "取值": self._stringify_metadata_value(value)} for key, value in metadata["parameters"].items()]
            )
            lines.append(_to_markdown_table(params))
            lines.append("")

        if auto_insights:
            lines.append("## 自动摘要")
            lines.append("")
            for item in auto_insights:
                lines.append(f"- {item}")
            lines.append("")

        if chart_refs:
            lines.append("## 资金与收益图表")
            lines.append("")
            for title, key in (
                ("资金曲线与回撤", "equity_drawdown"),
                ("超额收益与相对收益", "benchmark_relative"),
                ("日收益分布", "daily_returns"),
                ("月度收益", "monthly_returns"),
                ("单笔收益率分布", "return_distribution"),
                ("持仓天数与收益关系", "holding_vs_return"),
            ):
                if key in chart_refs:
                    lines.append(f"### {title}")
                    lines.append("")
                    lines.append(f"![{key}](charts/{chart_refs[key]})")
                    lines.append("")

            lines.append("## 交易结构图表")
            lines.append("")
            for title, key in (
                ("个股盈亏贡献", "pnl_by_code"),
                ("退出原因表现", "exit_reason_stats"),
                ("信号分数分桶表现", "score_bucket_stats"),
            ):
                if key in chart_refs:
                    lines.append(f"### {title}")
                    lines.append("")
                    lines.append(f"![{key}](charts/{chart_refs[key]})")
                    lines.append("")
        else:
            lines.append("> 当前环境未安装 `matplotlib`，本次仅生成 Markdown 表格报告，未生成图片。")
            lines.append("")

        lines.append("## 关键统计表")
        lines.append("")

        if not report.order_stats.empty:
            lines.append("### 订单状态统计")
            lines.append("")
            lines.append(
                _to_markdown_table(
                    report.order_stats,
                    renames={
                        "side": "方向",
                        "status": "状态",
                        "count": "数量",
                        "avg_filled_price": "平均成交价",
                        "avg_filled_quantity": "平均成交数量",
                        "commission_total": "手续费合计",
                        "tax_total": "税费合计",
                    },
                    formatters={
                        "数量": _format_int,
                        "平均成交价": _format_number,
                        "平均成交数量": _format_number,
                        "手续费合计": _format_number,
                        "税费合计": _format_number,
                    },
                )
            )
            lines.append("")

        if not report.skip_reason_stats.empty:
            lines.append("### 跳过原因统计")
            lines.append("")
            lines.append(
                _to_markdown_table(
                    report.skip_reason_stats,
                    renames={"skip_reason": "跳过原因", "count": "次数"},
                    formatters={"次数": _format_int},
                )
            )
            lines.append("")

        if not report.exit_reason_stats.empty:
            lines.append("### 平仓原因统计")
            lines.append("")
            lines.append(
                _to_markdown_table(
                    report.exit_reason_stats,
                    renames={
                        "exit_reason": "平仓原因",
                        "trade_count": "交易数",
                        "win_rate": "胜率",
                        "avg_return": "平均收益率",
                        "total_pnl": "总盈亏",
                        "avg_holding_days": "平均持有天数",
                    },
                    formatters={
                        "交易数": _format_int,
                        "胜率": _format_percent,
                        "平均收益率": _format_percent,
                        "总盈亏": _format_number,
                        "平均持有天数": _format_number,
                    },
                )
            )
            lines.append("")

        if not report.monthly_returns.empty:
            lines.append("### 月度收益明细")
            lines.append("")
            lines.append(
                _to_markdown_table(
                    report.monthly_returns,
                    renames={
                        "month": "月份",
                        "month_start_equity": "月初权益",
                        "month_end_equity": "月末权益",
                        "monthly_return": "月收益率",
                    },
                    formatters={
                        "月初权益": _format_number,
                        "月末权益": _format_number,
                        "月收益率": _format_percent,
                    },
                )
            )
            lines.append("")

        if not report.monthly_excess_returns.empty:
            lines.append("### 月度超额收益明细")
            lines.append("")
            lines.append(
                _to_markdown_table(
                    report.monthly_excess_returns,
                    renames={
                        "month": "月份",
                        "strategy_month_start_equity": "策略月初权益",
                        "strategy_month_end_equity": "策略月末权益",
                        "strategy_monthly_return": "策略月收益率",
                        "benchmark_month_start_equity": "基准月初权益",
                        "benchmark_month_end_equity": "基准月末权益",
                        "benchmark_monthly_return": "基准月收益率",
                        "excess_monthly_return": "超额月收益率",
                        "month_end_relative_return": "月末相对收益",
                        "month_end_excess_drawdown": "月末超额回撤",
                    },
                    formatters={
                        "策略月初权益": _format_number,
                        "策略月末权益": _format_number,
                        "策略月收益率": _format_percent,
                        "基准月初权益": _format_number,
                        "基准月末权益": _format_number,
                        "基准月收益率": _format_percent,
                        "超额月收益率": _format_percent,
                        "月末相对收益": _format_percent,
                        "月末超额回撤": _format_percent,
                    },
                )
            )
            lines.append("")

        if not report.pnl_by_code.empty:
            lines.append("### 个股盈亏排行")
            lines.append("")
            lines.append(
                _to_markdown_table(
                    report.pnl_by_code,
                    columns=["code", "trade_count", "win_rate", "total_pnl", "avg_return", "avg_holding_days"],
                    renames={
                        "code": "代码",
                        "trade_count": "交易数",
                        "win_rate": "胜率",
                        "total_pnl": "总盈亏",
                        "avg_return": "平均收益率",
                        "avg_holding_days": "平均持有天数",
                    },
                    formatters={
                        "交易数": _format_int,
                        "胜率": _format_percent,
                        "总盈亏": _format_number,
                        "平均收益率": _format_percent,
                        "平均持有天数": _format_number,
                    },
                    max_rows=12,
                )
            )
            lines.append("")

        if not report.score_bucket_stats.empty:
            lines.append("### 分数分桶统计")
            lines.append("")
            lines.append(
                _to_markdown_table(
                    report.score_bucket_stats,
                    renames={
                        "score_bucket": "分桶",
                        "trade_count": "交易数",
                        "win_rate": "胜率",
                        "avg_return": "平均收益率",
                        "total_pnl": "总盈亏",
                    },
                    formatters={
                        "交易数": _format_int,
                        "胜率": _format_percent,
                        "平均收益率": _format_percent,
                        "总盈亏": _format_number,
                    },
                )
            )
            lines.append("")

        if not top_winners.empty:
            lines.append("### 单笔盈利前列")
            lines.append("")
            lines.append(self._position_table(top_winners))
            lines.append("")

        if not top_losers.empty:
            lines.append("### 单笔亏损前列")
            lines.append("")
            lines.append(self._position_table(top_losers))
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _build_auto_insights(self, report) -> list[str]:
        summary = report.summary
        insights: list[str] = []

        if summary.get("total_return") is not None:
            insights.append(f"区间总收益率为 {_format_percent(summary.get('total_return'))}，期末权益 {_format_number(summary.get('final_equity'))}。")
        if summary.get("max_drawdown") is not None:
            insights.append(f"最大回撤为 {_format_percent(summary.get('max_drawdown'))}，夏普比率为 {_format_number(summary.get('sharpe'), 3)}。")
        if summary.get("benchmark_total_return") is not None:
            insights.append(
                f"同期基准收益率为 {_format_percent(summary.get('benchmark_total_return'))}，基准最大回撤为 {_format_percent(summary.get('benchmark_max_drawdown'))}。"
            )
        if summary.get("excess_return") is not None:
            insights.append(
                f"相对基准的超额收益为 {_format_percent(summary.get('excess_return'))}，相对收益曲线期末为 {_format_percent(summary.get('relative_return'))}。"
            )
        if summary.get("excess_max_drawdown") is not None:
            insights.append(
                f"超额回撤为 {_format_percent(summary.get('excess_max_drawdown'))}，反映策略相对基准回吐幅度。"
            )

        if not report.exit_reason_stats.empty:
            best_exit = report.exit_reason_stats.sort_values("total_pnl", ascending=False).iloc[0]
            insights.append(
                f"贡献最高的退出方式是 `{best_exit['exit_reason']}`，总盈亏 {_format_number(best_exit['total_pnl'])}，交易 {int(best_exit['trade_count'])} 笔。"
            )

        if not report.pnl_by_code.empty:
            best_code = report.pnl_by_code.sort_values("total_pnl", ascending=False).iloc[0]
            worst_code = report.pnl_by_code.sort_values("total_pnl", ascending=True).iloc[0]
            insights.append(
                f"个股维度上，贡献最高的是 `{best_code['code']}`({_format_number(best_code['total_pnl'])})，拖累最大的是 `{worst_code['code']}`({_format_number(worst_code['total_pnl'])})。"
            )

        if not report.position_stats.empty and "realized_return" in report.position_stats.columns:
            position_frame = report.position_stats.dropna(subset=["realized_return"]).copy()
            if not position_frame.empty:
                best_trade = position_frame.sort_values("realized_return", ascending=False).iloc[0]
                worst_trade = position_frame.sort_values("realized_return", ascending=True).iloc[0]
                insights.append(
                    f"单笔最佳交易为 `{best_trade['code']}`，收益率 {_format_percent(best_trade['realized_return'])}；单笔最差交易为 `{worst_trade['code']}`，收益率 {_format_percent(worst_trade['realized_return'])}。"
                )

        return insights

    def _select_top_positions(self, frame: pd.DataFrame, *, ascending: bool) -> pd.DataFrame:
        if frame.empty or "realized_pnl" not in frame.columns:
            return pd.DataFrame()
        return frame.sort_values("realized_pnl", ascending=ascending).head(8).copy()

    def _position_table(self, frame: pd.DataFrame) -> str:
        return _to_markdown_table(
            frame,
            columns=[
                "code",
                "entry_trade_date",
                "exit_trade_date",
                "holding_trade_days",
                "exit_reason",
                "entry_price",
                "exit_price",
                "realized_pnl",
                "realized_return",
                "score",
            ],
            renames={
                "code": "代码",
                "entry_trade_date": "开仓日",
                "exit_trade_date": "平仓日",
                "holding_trade_days": "持有交易日",
                "exit_reason": "平仓原因",
                "entry_price": "开仓价",
                "exit_price": "平仓价",
                "realized_pnl": "实现盈亏",
                "realized_return": "实现收益率",
                "score": "信号分数",
            },
            formatters={
                "开仓日": _format_date,
                "平仓日": _format_date,
                "持有交易日": _format_int,
                "开仓价": _format_number,
                "平仓价": _format_number,
                "实现盈亏": _format_number,
                "实现收益率": _format_percent,
                "信号分数": _format_number,
            },
        )

    def _stringify_metadata_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return _format_number(value, 6).rstrip("0").rstrip(".")
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value) if value else "-"
        if value is None or value == "":
            return "-"
        return str(value)
