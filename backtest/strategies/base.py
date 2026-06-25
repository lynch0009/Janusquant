"""策略抽象定义。

策略层的职责是：准备信号所需数据、声明依赖字段、在给定日期生成候选股。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence

import pandas as pd

from backtest.utils.datetime_utils import to_pydatetime

from .models import DailyCandidate


class BaseSelectionStrategy(ABC):
    """选股策略抽象基类。"""

    def prepare(self, data_portal, trade_dates: Sequence[datetime], *, research_store=None) -> None:
        """在回测正式开始前做一次预处理，默认无需处理。"""

        return None

    def target_exposure(
        self,
        trade_date: datetime,
        portfolio_snapshot: dict,
    ) -> float | None:
        """返回某个交易日的目标组合暴露比例。"""

        return None

    def should_rebalance_exposure(
        self,
        trade_date: datetime,
        previous_trade_date: datetime | None,
        portfolio_snapshot: dict,
    ) -> bool:
        """是否需要在当天立即把组合仓位拉回目标暴露。"""

        return False

    def uses_target_portfolio_rebalance(self) -> bool:
        """是否启用“目标组合净额调仓”模式。"""

        return False

    @abstractmethod
    def required_feature_fields(self) -> Sequence[str]:
        """声明策略生成信号时必须读取的特征字段。"""

        raise NotImplementedError

    @abstractmethod
    def generate_candidates(
        self,
        signal_date: datetime,
        feature_slice: pd.DataFrame,
        portfolio_snapshot: dict,
    ) -> list[DailyCandidate]:
        """在某个交易日根据特征截面和当前组合生成候选股列表。"""

        raise NotImplementedError


class SignalTableSelectionStrategy(BaseSelectionStrategy):
    """为依赖 signal_table 的策略提供统一模板方法。"""

    def __init__(self) -> None:
        self.signal_table: dict[datetime, pd.DataFrame] = {}

    def reset_signal_table(self) -> None:
        self.signal_table = {}

    def set_signal_table(self, signal_table: dict[datetime, pd.DataFrame]) -> None:
        self.signal_table = signal_table

    def signal_frame(self, signal_date: datetime) -> pd.DataFrame | None:
        return self.signal_table.get(to_pydatetime(signal_date))

    def group_signal_table(
        self,
        frame: pd.DataFrame,
        *,
        date_col: str = "trade_date",
        limit: int | None = None,
    ) -> dict[datetime, pd.DataFrame]:
        """把按日期组织的信号明细表标准化成 signal_table。"""

        if frame.empty:
            return {}

        signal_table: dict[datetime, pd.DataFrame] = {}
        for trade_date, group in frame.groupby(date_col, sort=True):
            selected = group.head(limit) if limit is not None else group
            if selected.empty:
                continue
            signal_table[to_pydatetime(trade_date)] = selected.reset_index(drop=True)
        return signal_table


class SmallCapSignalTableStrategy(SignalTableSelectionStrategy):
    """为小票轮动类策略提供统一的股票池预处理模板。"""

    smallcap_code_prefixes: tuple[str, ...] = ("sh.60", "sh.68", "sz.00", "sz.30")

    def _prepare_smallcap_feature_frame(
        self,
        data_portal,
        trade_dates: Sequence[datetime],
        *,
        feature_fields: Sequence[str],
        cap_field: str,
        rebalance_every_n_trade_days: int,
        min_listing_trade_days: int,
        min_cap_value: float | None = None,
        max_cap_value: float | None = None,
        rebalance_only: bool = True,
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        """准备小票策略共用的基础股票池和完整交易日历。"""

        if not trade_dates:
            return pd.DataFrame(), pd.DatetimeIndex([])

        signal_start = to_pydatetime(trade_dates[0])
        signal_end = to_pydatetime(trade_dates[-1])

        feature_history = data_portal.get_feature_history(
            signal_start,
            signal_end,
            fields=list(feature_fields),
        )
        if feature_history.empty:
            return pd.DataFrame(), pd.DatetimeIndex([])

        frame = feature_history.rename(columns={"date": "trade_date"}).copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame[
            frame["code"].map(
                lambda code: str(code).startswith(self.smallcap_code_prefixes)
            )
        ].copy()
        frame = frame[frame[cap_field].notna() & (frame[cap_field] > 0)].copy()
        if min_cap_value is not None:
            frame = frame[frame[cap_field] >= min_cap_value].copy()
        if max_cap_value is not None:
            frame = frame[frame[cap_field] <= max_cap_value].copy()
        if frame.empty:
            return pd.DataFrame(), pd.DatetimeIndex([])

        trade_date_frame = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(trade_dates),
                "trade_index": list(range(len(trade_dates))),
            }
        )
        frame = frame.merge(trade_date_frame, on="trade_date", how="inner")
        if rebalance_only:
            rebalance_dates = pd.to_datetime(trade_dates[:: rebalance_every_n_trade_days])
            frame = frame[frame["trade_date"].isin(rebalance_dates)].copy()
        if frame.empty:
            return pd.DataFrame(), pd.DatetimeIndex([])

        basic_info = data_portal.get_stock_basic(
            sorted(frame["code"].unique().tolist()),
            fields=["code", "ipoDate", "outDate"],
        )
        if basic_info.empty:
            return pd.DataFrame(), pd.DatetimeIndex([])

        basic_info = basic_info.copy()
        basic_info["ipoDate"] = pd.to_datetime(basic_info["ipoDate"], errors="coerce")
        basic_info["outDate"] = pd.to_datetime(basic_info["outDate"], errors="coerce")
        earliest_ipo_date = basic_info["ipoDate"].dropna().min()
        if pd.isna(earliest_ipo_date):
            return pd.DataFrame(), pd.DatetimeIndex([])

        full_trade_calendar = pd.DatetimeIndex(
            pd.to_datetime(
                data_portal.get_trade_calendar(
                    to_pydatetime(earliest_ipo_date),
                    signal_end,
                )
            )
        )
        if full_trade_calendar.empty:
            return pd.DataFrame(), pd.DatetimeIndex([])

        basic_info["ipo_trade_index"] = basic_info["ipoDate"].map(
            lambda value: full_trade_calendar.searchsorted(value, side="left") if pd.notna(value) else pd.NA
        )
        frame["trade_index"] = frame["trade_date"].map(
            lambda value: full_trade_calendar.searchsorted(value, side="left")
        )

        frame = frame.merge(
            basic_info[["code", "ipoDate", "outDate", "ipo_trade_index"]],
            on="code",
            how="left",
        )
        frame = frame[frame["ipo_trade_index"].notna()].copy()
        frame["ipo_trade_index"] = frame["ipo_trade_index"].astype(int)
        frame = frame[(frame["trade_index"] - frame["ipo_trade_index"]) >= min_listing_trade_days].copy()
        frame = frame[frame["outDate"].isna() | (frame["trade_date"] < frame["outDate"])].copy()
        return frame, full_trade_calendar
