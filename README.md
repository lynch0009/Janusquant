# Janusquant

面向 A 股的本地量化研究与回测框架：从数据准备、因子验证和策略选股，到交易约束、组合记账、绩效分析与结果导出。

Janusquant 关注的不是“把历史信号按收盘价成交”这一类理想化回测，而是 A 股研究中容易影响结果可信度的实际问题：涨跌停与停牌、ST 和异常状态、流动性约束、复权口径、手续费与滑点、分红送转，以及信号、成交和组合权益之间的一致性。

> 本项目用于量化研究、策略验证和工程学习，不构成投资建议。历史回测无法保证未来表现。

## 为什么使用 Janusquant

- **面向 A 股交易规则**：回测链路可处理涨跌停、停牌、ST、流动性过滤、交易成本和滑点等约束，减少理想化成交。
- **统一数据与价格口径**：支持原始价格、前复权和后复权研究口径，并将行情、特征、财务、复权因子和分红数据纳入同一研究流程。
- **覆盖完整研究链路**：提供数据抓取、特征构建、策略回测、参数批跑、组合记账、风险退出和报告导出。
- **结果可追溯**：输出订单、成交、持仓、权益曲线、基准对比、绩效指标和图表，便于定位信号、执行或记账问题。
- **适合扩展**：数据访问、策略、执行、组合、风控和分析分层组织，可替换或新增组件，而不必重写整条回测链路。

## 核心能力

- DuckDB 数据访问：日线、特征、财务、复权因子、分红事件和研究面板。
- 策略实现：小市值轮动、流动性过滤、成交额冲击、反转策略、Minervini A 股选股等。
- 回测执行：信号驱动引擎、执行模型、组合记账、仓位管理和退出规则。
- 数据工具：面向 DuckDB 表的数据抓取、清洗、标准化和增量写入。

## 快速开始

### 环境准备

建议使用 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
```

`xtquant`、`TA-Lib` 等本地交易或技术分析依赖没有放进默认必装项，需要时按本地环境单独安装。

### DuckDB 配置

复制示例配置：

```powershell
Copy-Item config/duckdb.example.conf config/duckdb.conf
```

macOS / Linux：

```bash
cp config/duckdb.example.conf config/duckdb.conf
```

## 项目结构

```text
backtest/             回测主框架和数据工具
backtest/data/        DuckDB 数据门面与缓存
backtest/db/          DuckDB 连接与写入工具
backtest/strategies/  选股策略与 regime 过滤器
backtest/execution/   回测引擎和执行模型
backtest/portfolio/   订单、持仓、组合记账和结果对象
backtest/risk/        退出策略和风控规则
backtest/runs/        回测入口和数据维护脚本
backtest/utils/       通用工具函数
research/             研究框架
config/               本地配置示例
```

## 推荐入口

最小运行示例：

```bash
python backtest/runs/run_smallcap_liquidity_backtest.py --start-date 2025-01-01 --end-date 2025-04-30
```

常用入口：

- `backtest/runs/run_smallcap_liquidity_backtest.py`
- `backtest/runs/run_smallcap_liquidity_batch.py`
- `backtest/runs/run_smallcap_amount_shock_event_backtest.py`
- `backtest/runs/run_minervini_ashare_backtest.py`

运行结果会输出到 `backtest/runs/output/` 或研究脚本对应的输出目录。

## 数据依赖

仓库不包含生产行情和财务数据。运行主线回测前，需要准备本地 DuckDB 数据库，常用表包括：

- `A_stock_market_basic_info`
- `A_stock_market_day_kline`
- `A_stock_market_feature`
- `A_stock_market_adjust_factor`
- `A_stock_market_dividend_data`
- `A_stock_market_finance_data`

具体字段依赖以 `backtest/data/duckdb_portal.py` 和各策略模块为准。

## 关注作者

- 知乎：[lord_jun](https://www.zhihu.com/people/lord_jun)
- 小红书：[作者主页](https://www.xiaohongshu.com/user/profile/67ea02740000000006010c87)

## License

本项目采用 MIT License，详见 [`LICENSE`](LICENSE)。
