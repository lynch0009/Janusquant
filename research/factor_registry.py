from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from backtest.utils.price_limits import is_limit_down_close_series

from .errors import ResearchDataContractError, ResearchFactorError
from .validation import require_columns, validate_unique_keys


FEATURE_FORMULA_VERSION = "smallcap_reversal_factor_v3"


class FactorContext:
    """Shared, memoized calculations for a single ordered history frame."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self._numeric: dict[str, pd.Series] = {}
        self._shift: dict[tuple[str, int], pd.Series] = {}
        self._rolling: dict[tuple[str, str, int, int], pd.Series] = {}

    def numeric(self, column: str) -> pd.Series:
        if column not in self.frame.columns:
            raise ResearchDataContractError(f"factor history missing required column: {column}")
        if column not in self._numeric:
            self._numeric[column] = pd.to_numeric(self.frame[column], errors="coerce")
        return self._numeric[column]

    def shift(self, column: str, periods: int) -> pd.Series:
        key = (column, int(periods))
        if key not in self._shift:
            self._shift[key] = self.numeric(column).groupby(
                self.frame["code"], sort=False, observed=True
            ).shift(periods)
        return self._shift[key]

    def rolling(self, column: str, window: int, operation: str, *, min_periods: int | None = None) -> pd.Series:
        minimum = int(window if min_periods is None else min_periods)
        key = (column, operation, int(window), minimum)
        if key not in self._rolling:
            grouped = self.numeric(column).groupby(self.frame["code"], sort=False, observed=True)
            rolling = grouped.rolling(int(window), min_periods=minimum)
            method = getattr(rolling, operation)
            self._rolling[key] = method().reset_index(level=0, drop=True)
        return self._rolling[key]

    def rolling_series(
        self,
        series: pd.Series,
        *,
        cache_key: str,
        window: int,
        operation: str,
        min_periods: int | None = None,
    ) -> pd.Series:
        minimum = int(window if min_periods is None else min_periods)
        key = (cache_key, operation, int(window), minimum)
        if key not in self._rolling:
            rolling = series.groupby(self.frame["code"], sort=False, observed=True).rolling(
                int(window), min_periods=minimum
            )
            self._rolling[key] = getattr(rolling, operation)().reset_index(level=0, drop=True)
        return self._rolling[key]


@dataclass(frozen=True)
class FactorSpec:
    """单个因子的注册协议。

    新增因子时优先只补一个 FactorSpec：声明依赖字段、预热窗口、默认方向和计算函数。
    runner/dataset/metrics 不应该因为普通因子新增而改动。
    """

    name: str
    required_fields: tuple[str, ...]
    warmup_window: int
    default_direction: int
    compute: Callable[[FactorContext], pd.Series]
    dependencies: tuple[str, ...] = ()
    version: str = "v1"


class FactorRegistry:
    """Validated factor registry with deterministic dependency resolution."""

    def __init__(self, specs: Iterable[FactorSpec] = ()):
        self._specs: dict[str, FactorSpec] = {}
        for spec in specs:
            self.register(spec)

    @property
    def specs(self) -> dict[str, FactorSpec]:
        return dict(self._specs)

    def register(self, spec: FactorSpec) -> None:
        name = str(spec.name).strip().lower()
        if not name:
            raise ResearchFactorError("factor name cannot be empty")
        if name in self._specs:
            raise ResearchFactorError(f"factor already registered: {name}")
        if int(spec.warmup_window) < 0:
            raise ResearchFactorError(f"factor warmup_window must be >= 0: {name}")
        if int(spec.default_direction) not in (-1, 1):
            raise ResearchFactorError(f"factor direction must be 1 or -1: {name}")
        self._specs[name] = FactorSpec(
            name=name,
            required_fields=tuple(dict.fromkeys(spec.required_fields)),
            warmup_window=int(spec.warmup_window),
            default_direction=int(spec.default_direction),
            compute=spec.compute,
            dependencies=tuple(str(value).strip().lower() for value in spec.dependencies),
            version=str(spec.version),
        )

    def validate_names(self, names: Iterable[str]) -> tuple[str, ...]:
        resolved: list[str] = []
        for value in names:
            name = str(value).strip().lower()
            if name not in self._specs:
                raise ResearchFactorError(f"unsupported feature: {value}")
            if name not in resolved:
                resolved.append(name)
        return tuple(resolved)

    def resolve(self, names: Iterable[str]) -> tuple[FactorSpec, ...]:
        requested = self.validate_names(names)
        ordered: list[FactorSpec] = []
        state: dict[str, int] = {}

        def visit(name: str, path: tuple[str, ...]) -> None:
            status = state.get(name, 0)
            if status == 2:
                return
            if status == 1:
                raise ResearchFactorError(f"factor dependency cycle: {' -> '.join((*path, name))}")
            spec = self._specs.get(name)
            if spec is None:
                raise ResearchFactorError(f"unknown factor dependency: {name}")
            state[name] = 1
            for dependency in spec.dependencies:
                if dependency not in self._specs:
                    raise ResearchFactorError(f"factor {name} has unknown dependency: {dependency}")
                visit(dependency, (*path, name))
            state[name] = 2
            ordered.append(spec)

        for name in requested:
            visit(name, ())
        return tuple(ordered)

    def required_fields(self, names: Iterable[str]) -> tuple[str, ...]:
        fields = set()
        for spec in self.resolve(names):
            fields.update(spec.required_fields)
        preferred = ("open", "high", "low", "close", "preclose", "amount", "volume", "turn", "isST")
        return tuple(field for field in preferred if field in fields) + tuple(sorted(fields - set(preferred)))

    def max_warmup_window(self, names: Iterable[str]) -> int:
        return max((spec.warmup_window for spec in self.resolve(names)), default=0)

    def version_signature(self, names: Iterable[str]) -> str:
        payload = "|".join(f"{spec.name}:{spec.version}" for spec in self.resolve(names))
        digest = sha1(payload.encode("utf-8")).hexdigest()[:12]
        return f"{FEATURE_FORMULA_VERSION}:{digest}"

    def default_directions(self, names: Iterable[str]) -> dict[str, int]:
        requested = self.validate_names(names)
        return {name: self._specs[name].default_direction for name in requested}


class FactorEngine:
    def __init__(self, registry: FactorRegistry):
        self.registry = registry

    def compute(self, daily_frame: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
        requested = self.registry.validate_names(features)
        if daily_frame.empty:
            return daily_frame.copy()
        require_columns(daily_frame, ("code", "trade_date"), context="factor history")
        result = daily_frame.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
        if result["trade_date"].isna().any():
            raise ResearchDataContractError("factor history contains invalid trade_date values")
        validate_unique_keys(result, context="factor history")
        result = result.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)
        required = self.registry.required_fields(requested)
        require_columns(result, required, context="factor history")
        context = FactorContext(result)
        for spec in self.registry.resolve(requested):
            values = spec.compute(context)
            if len(values) != len(result):
                raise ResearchFactorError(
                    f"factor {spec.name} returned {len(values)} rows for {len(result)} input rows"
                )
            result[spec.name] = pd.Series(values, index=result.index)
        return result


def _rolling_mean(context: FactorContext, column: str, window: int) -> pd.Series:
    return context.rolling(column, window, "mean")


def _ma(window: int) -> Callable[[FactorContext], pd.Series]:
    def compute(context: FactorContext) -> pd.Series:
        return _rolling_mean(context, "close", window)

    return compute


def _ret(window: int) -> Callable[[FactorContext], pd.Series]:
    def compute(context: FactorContext) -> pd.Series:
        shifted = context.shift("close", window)
        return context.numeric("close") / shifted.replace(0, np.nan) - 1.0

    return compute


def _ma_slope(feature_name: str, lookback: int) -> Callable[[FactorContext], pd.Series]:
    def compute(context: FactorContext) -> pd.Series:
        base = context.numeric(feature_name)
        shifted = context.shift(feature_name, lookback)
        return base / shifted.replace(0, np.nan) - 1.0

    return compute


def _distance_to_hhv60(context: FactorContext) -> pd.Series:
    hhv = context.rolling("close", 60, "max")
    return context.numeric("close") / hhv.replace(0, np.nan) - 1.0


def _pullback_from_hhv(window: int) -> Callable[[FactorContext], pd.Series]:
    def compute(context: FactorContext) -> pd.Series:
        hhv = context.rolling("close", window, "max")
        return context.numeric("close") / hhv.replace(0, np.nan) - 1.0

    return compute


def _max_drawdown(window: int) -> Callable[[FactorContext], pd.Series]:
    def compute(context: FactorContext) -> pd.Series:
        close = context.numeric("close")

        def rolling_drawdown(values: np.ndarray) -> float:
            if values.size == 0 or np.isnan(values).all():
                return np.nan
            running_peak = np.maximum.accumulate(values)
            drawdowns = values / np.where(running_peak == 0, np.nan, running_peak) - 1.0
            return float(np.nanmin(drawdowns))

        return (
            close.groupby(context.frame["code"], sort=False, observed=True)
            .rolling(window, min_periods=window)
            .apply(rolling_drawdown, raw=True)
            .reset_index(level=0, drop=True)
        )

    return compute


def _up_day_ratio(window: int) -> Callable[[FactorContext], pd.Series]:
    def compute(context: FactorContext) -> pd.Series:
        close = context.numeric("close")
        prev_close = context.shift("close", 1)
        up_day = (close > prev_close).astype(float).where(prev_close.notna())
        return context.rolling_series(
            up_day,
            cache_key=f"up_day:{window}",
            window=window,
            operation="mean",
        )

    return compute


def _up_amount_ratio(window: int) -> Callable[[FactorContext], pd.Series]:
    def compute(context: FactorContext) -> pd.Series:
        close = context.numeric("close")
        prev_close = context.shift("close", 1)
        amount = context.numeric("amount")
        up_amount = amount.where(close > prev_close, 0.0).where(prev_close.notna())
        down_amount = amount.where(close < prev_close, 0.0).where(prev_close.notna())
        up_count = (close > prev_close).astype(float).where(prev_close.notna())
        down_count = (close < prev_close).astype(float).where(prev_close.notna())
        up_sum = context.rolling_series(up_amount, cache_key=f"up_amount:{window}", window=window, operation="sum")
        down_sum = context.rolling_series(down_amount, cache_key=f"down_amount:{window}", window=window, operation="sum")
        up_days = context.rolling_series(up_count, cache_key=f"up_count:{window}", window=window, operation="sum")
        down_days = context.rolling_series(down_count, cache_key=f"down_count:{window}", window=window, operation="sum")
        up_avg = up_sum / up_days.replace(0, np.nan)
        down_avg = down_sum / down_days.replace(0, np.nan)
        return up_avg / down_avg.replace(0, np.nan)

    return compute


def _down_shrink(window: int) -> Callable[[FactorContext], pd.Series]:
    def compute(context: FactorContext) -> pd.Series:
        close = context.numeric("close")
        prev_close = context.shift("close", 1)
        amount = context.numeric("amount")
        down_amount = amount.where(close < prev_close, 0.0).where(prev_close.notna())
        down_count = (close < prev_close).astype(float).where(prev_close.notna())
        down_sum = context.rolling_series(down_amount, cache_key=f"down_amount:{window}", window=window, operation="sum")
        down_days = context.rolling_series(down_count, cache_key=f"down_count:{window}", window=window, operation="sum")
        down_avg = down_sum / down_days.replace(0, np.nan)
        amount_avg = _rolling_mean(context, "amount", window)
        return 1.0 - down_avg / amount_avg.replace(0, np.nan)

    return compute


def _amount_ratio(fast_window: int, slow_window: int) -> Callable[[FactorContext], pd.Series]:
    def compute(context: FactorContext) -> pd.Series:
        fast = _rolling_mean(context, "amount", fast_window)
        slow = _rolling_mean(context, "amount", slow_window)
        return fast / slow.replace(0, np.nan)

    return compute


def _down_shrink_10d(context: FactorContext) -> pd.Series:
    # 只奖励“下跌后缩量”的组合，单纯上涨缩量或放量下跌都不会得到高分。
    ret_10d = context.numeric("ret_10d")
    amount_expand = context.numeric("amount_expand")
    return (-ret_10d).clip(lower=0.0) * (1.0 - amount_expand).clip(lower=0.0)


def _range_ratio_3_20(context: FactorContext) -> pd.Series:
    high = context.numeric("high")
    low = context.numeric("low")
    daily_range = high / low.replace(0, np.nan) - 1.0
    fast = context.rolling_series(daily_range, cache_key="daily_range", window=3, operation="mean")
    slow = context.rolling_series(daily_range, cache_key="daily_range", window=20, operation="mean")
    return fast / slow.replace(0, np.nan)


def _limit_down_count_5d(context: FactorContext) -> pd.Series:
    is_limit_down = is_limit_down_close_series(
        context.frame["code"],
        context.numeric("preclose"),
        context.numeric("close"),
        is_st=context.frame["isST"],
    ).astype(float)
    return context.rolling_series(
        is_limit_down,
        cache_key="is_limit_down",
        window=5,
        operation="sum",
    )


FACTOR_REGISTRY: dict[str, FactorSpec] = {
    # 默认方向表示“原始值乘以方向后，数值越大越好”。
    # 例如 ret_5d 的方向为 -1，代表短期跌幅越大，研究排序值越靠前。
    "distance_to_hhv60": FactorSpec(
        name="distance_to_hhv60",
        required_fields=("close",),
        warmup_window=60,
        default_direction=-1,
        compute=_distance_to_hhv60,
    ),
    "ret_5d": FactorSpec(
        name="ret_5d",
        required_fields=("close",),
        warmup_window=5,
        default_direction=-1,
        compute=_ret(5),
    ),
    "ret_1d": FactorSpec(
        name="ret_1d",
        required_fields=("close",),
        warmup_window=1,
        default_direction=1,
        compute=_ret(1),
    ),
    "ret_10d": FactorSpec(
        name="ret_10d",
        required_fields=("close",),
        warmup_window=10,
        default_direction=-1,
        compute=_ret(10),
    ),
    "ret_20d": FactorSpec(
        name="ret_20d",
        required_fields=("close",),
        warmup_window=20,
        default_direction=1,
        compute=_ret(20),
    ),
    "ret_60d": FactorSpec(
        name="ret_60d",
        required_fields=("close",),
        warmup_window=60,
        default_direction=1,
        compute=_ret(60),
    ),
    "ma10": FactorSpec(
        name="ma10",
        required_fields=("close",),
        warmup_window=10,
        default_direction=1,
        compute=_ma(10),
    ),
    "ma20": FactorSpec(
        name="ma20",
        required_fields=("close",),
        warmup_window=20,
        default_direction=1,
        compute=_ma(20),
    ),
    "ma20_slope_5d": FactorSpec(
        name="ma20_slope_5d",
        required_fields=("close",),
        warmup_window=25,
        default_direction=1,
        compute=_ma_slope("ma20", 5),
        dependencies=("ma20",),
    ),
    "pullback_from_hhv20": FactorSpec(
        name="pullback_from_hhv20",
        required_fields=("close",),
        warmup_window=20,
        default_direction=1,
        compute=_pullback_from_hhv(20),
    ),
    "max_drawdown_20d": FactorSpec(
        name="max_drawdown_20d",
        required_fields=("close",),
        warmup_window=20,
        default_direction=1,
        compute=_max_drawdown(20),
    ),
    "up_day_ratio_20d": FactorSpec(
        name="up_day_ratio_20d",
        required_fields=("close",),
        warmup_window=20,
        default_direction=1,
        compute=_up_day_ratio(20),
    ),
    "up_amount_ratio_20d": FactorSpec(
        name="up_amount_ratio_20d",
        required_fields=("close", "amount"),
        warmup_window=20,
        default_direction=1,
        compute=_up_amount_ratio(20),
    ),
    "down_shrink_20d": FactorSpec(
        name="down_shrink_20d",
        required_fields=("close", "amount"),
        warmup_window=20,
        default_direction=1,
        compute=_down_shrink(20),
    ),
    "amount_ratio_3_20": FactorSpec(
        name="amount_ratio_3_20",
        required_fields=("amount",),
        warmup_window=20,
        default_direction=-1,
        compute=_amount_ratio(3, 20),
    ),
    "amount_expand": FactorSpec(
        name="amount_expand",
        required_fields=("amount",),
        warmup_window=20,
        default_direction=-1,
        compute=_amount_ratio(5, 20),
    ),
    "down_shrink_10d": FactorSpec(
        name="down_shrink_10d",
        required_fields=("close", "amount"),
        warmup_window=20,
        default_direction=1,
        compute=_down_shrink_10d,
        dependencies=("ret_10d", "amount_expand"),
    ),
    "range_ratio_3_20": FactorSpec(
        name="range_ratio_3_20",
        required_fields=("high", "low"),
        warmup_window=20,
        default_direction=-1,
        compute=_range_ratio_3_20,
    ),
    "limit_down_count_5d": FactorSpec(
        name="limit_down_count_5d",
        required_fields=("code", "preclose", "close", "isST"),
        warmup_window=5,
        default_direction=-1,
        compute=_limit_down_count_5d,
    ),
}

DEFAULT_FACTOR_REGISTRY = FactorRegistry(FACTOR_REGISTRY.values())
