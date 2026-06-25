# Janusquant

面向 A 股的本地量化研究与回测框架：从数据准备、因子验证和策略选股，到交易约束、组合记账、绩效分析与结果导出。

Janusquant 关注的不是“把历史信号按收盘价成交”这一类理想化回测，而是 A 股研究中容易影响结果可信度的实际问题：涨跌停与停牌、ST 和异常状态、流动性约束、复权口径、手续费与滑点、分红送转，以及信号、成交和组合权益之间的一致性。

> 本项目用于量化研究、策略验证和工程学习，不构成投资建议。历史回测无法保证未来表现。

## 为什么使用 Janusquant

- **面向 A 股交易规则**：回测链路可处理涨跌停、停牌、ST、流动性过滤、交易成本和滑点等约束，减少理想化成交。
- **统一数据与价格口径**：支持原始价格、前复权和后复权研究口径，并将行情、特征、财务、复权因子和分红数据纳入同一研究流程。
- **覆盖完整研究链路**：提供数据抓取、横截面因子研究、策略回测、参数批跑、组合记账、风险退出和报告导出。
- **结果可追溯**：输出订单、成交、持仓、权益曲线、基准对比、绩效指标和图表，便于定位信号、执行或记账问题。
- **适合扩展**：数据访问、策略、执行、组合、风控和分析分层组织，可替换或新增组件，而不必重写整条回测链路。

## 快速导航

- [5 分钟跑通 Demo](#5-分钟跑通-demo)
- [项目适合做什么](#项目适合做什么)
- [环境与数据准备](#环境与数据准备)
- [运行策略回测](#运行策略回测)
- [因子研究](#因子研究)
- [输出与报告](#输出与报告)
- [项目结构](#项目结构)
- [开发与扩展](#开发与扩展)
- [使用 MiniQMT 进行实盘交易](#使用-miniqmt-进行实盘交易)
- [关注作者](#关注作者)

## 5 分钟跑通 Demo

Demo 会启动本地 MongoDB、导入一组小型数据，并依次运行两条小市值策略。建议先通过 Demo 验证环境，再接入自己的数据。

### 1. 安装依赖

建议使用 Python 3.10 及以上版本。

```powershell
python -m pip install -r requirements.txt
```

### 2. 下载 Demo 数据

- 下载地址：[Janusquant Demo 数据](https://pan.quark.cn/s/c5741fd8d4b4)
- 提取码：`Vtk6`

将数据包解压后，确认目录结构如下：

```text
examples/
  data/
    A_stock_market_basic_info.jsonl
    A_stock_market_day_kline.jsonl
    A_stock_market_feature.jsonl
    A_stock_market_dividend_data.jsonl
```

如果数据文件位于其他目录，可在导入时通过 `--data-dir` 指定。

### 3. 启动 MongoDB

需要本机已安装 Docker Desktop 或兼容的 Docker Compose 环境。

```powershell
docker compose -f examples/demo_mongo/docker-compose.yml up -d
```

### 4. 导入并检查数据

```powershell
python examples/demo_mongo/import_demo_mongo.py --drop-existing
python examples/demo_mongo/check_demo_data.py
```

检查成功时会输出 `status: PASS`，并列出集合记录数、股票数量和交易日数量。

### 5. 运行 Demo

Windows PowerShell：

```powershell
.\examples\demo_mongo\run_demo_smallcap.ps1 -Python python
```

macOS / Linux：

```bash
PYTHON_PATH=python bash examples/demo_mongo/run_demo_smallcap.sh
```

脚本会运行：

1. 小市值流动性轮动策略；
2. 小市值成交额冲击反转策略。

结果写入：

```text
backtest/runs/output/demo_smallcap_liquidity/
backtest/runs/output/demo_smallcap_reversal/
```

更完整的 Demo 参数和自定义 MongoDB 用法见 [`examples/demo_mongo/README.md`](examples/demo_mongo/README.md)。

## 项目适合做什么

Janusquant 主要面向具备 Python 基础、希望在本地完成 A 股研究的开发者和量化研究者。

适合的场景包括：

- 横截面因子分组、双排序、收益周期和稳定性研究；
- A 股中低频选股、轮动、反转、趋势和成长策略回测；
- 在涨跌停、停牌、ST、流动性和交易成本约束下验证策略；
- 对比不同参数、过滤规则、持仓数量和调仓周期；
- 分析订单跳过原因、成交结果、持仓盈亏、回撤和相对基准表现；
- 基于现有数据、执行、组合和风控接口开发自定义策略。

## 项目不适合做什么

明确边界比堆叠功能更重要。当前项目不以以下场景为目标：

- **高频或超低延迟交易**：Python 回测链路和当前数据结构不面向逐笔、微秒级执行。
- **开箱即用的图形化交易终端**：项目以 Python 脚本和结构化输出为主，不提供桌面 GUI。
- **托管式在线量化平台**：行情、数据库、策略代码和结果由使用者在本地维护。
- **无需数据准备的完整数据服务**：仓库不附带生产级全量行情，完整研究需自行构建或接入数据。
- **对未来收益的承诺**：示例策略用于展示研究和回测链路，不代表可直接用于真实资金。

## 核心能力

### A 股约束与组合记账

- 涨跌停、停牌和证券状态检查；
- ST、新股上市天数、价格和流动性过滤；
- 买卖滑点、佣金与印花税；
- 订单、成交、持仓、现金和权益记录；
- 现金分红、送股等公司行为处理；
- 固定止损、均线退出、低价退出和组合退出规则。

具体支持程度取决于所使用的策略、执行器及底层数据是否包含相应字段。

### 数据准备与访问

- MongoDB 配置、仓储层和统一数据访问门面；
- XtQuant 日线同步及 Baostock 指定区间补数；
- AkShare 财务数据、复权因子和分红数据获取脚本；
- Parquet 本地缓存，支持批量实验复用数据；
- Demo Mongo JSONL 数据导入、校验和导出工具。

### 策略与批量实验

仓库当前提供以下完整示例：

- 小市值流动性轮动；
- 小市值成交额冲击反转；
- Minervini 风格 A 股趋势成长；
- 对应的小市值参数批量实验入口。

### 分析与报告

- 总收益率、年化收益率、波动率、夏普比率、最大回撤和 Calmar；
- 胜率、盈亏比、Profit Factor、平均持有天数和单笔盈亏；
- 基准收益、超额收益、相对收益和超额回撤；
- 月度收益、订单状态、跳过原因、退出原因和证券盈亏统计；
- Markdown 报告、JSON/CSV 明细和 PNG 图表。

## 环境与数据准备

### Python 依赖

```powershell
python -m pip install -r requirements.txt
```

主要依赖包括 pandas、NumPy、PyMongo、Matplotlib、Plotly、AkShare 和 Baostock。以下依赖需要按本地环境单独安装：

- `xtquant`：使用 MiniQMT/XtQuant 数据接口时需要；
- `TA-Lib`：仅相关指标或本地功能需要。

### 配置 MongoDB

复制示例配置：

```powershell
Copy-Item config/mongodb.example.conf config/mongodb.conf
```

macOS / Linux：

```bash
cp config/mongodb.example.conf config/mongodb.conf
```

然后编辑 `config/mongodb.conf`：

```ini
[mongodb]
host = 127.0.0.1
port = 27017
db_name = your_db_name
username =
password =
```

也可以通过环境变量覆盖配置：

```text
MONGO_HOST
MONGO_PORT
MONGO_DB_NAME
MONGO_USERNAME
MONGO_PASSWORD
MONGO_AUTH_MECHANISM
MONGO_CONNECT_TIMEOUT_MS
MONGO_SERVER_SELECTION_TIMEOUT_MS
MONGO_SOCKET_TIMEOUT_MS
MONGO_MAX_POOL_SIZE
MONGO_MIN_POOL_SIZE
MONGO_RETRY_READS
MONGO_RETRY_WRITES
MONGO_CONNECT_EAGERLY
```

配置读取逻辑位于 `backtest/db/mongodb.py`。

### 生产数据要求

完整回测通常需要 MongoDB 中具备：

- 交易日历；
- 日线行情；
- 股票基础信息；
- 特征或因子截面；
- 复权因子与分红数据；
- Minervini 等特定策略需要的财务特征。

不同策略的数据要求不同。分钟行情不是当前示例策略跑通的必要条件。

### 同步日线行情

XtQuant 每日同步：

```powershell
python backtest/fetch_data/xtquant_day_kline_fetch.py daily
```

补指定股票和日期区间：

```powershell
python backtest/fetch_data/xtquant_day_kline_fetch.py fetch `
  --source xtquant `
  --stocks 600000.SH,000001.SZ `
  --start-date 2026-05-01 `
  --end-date 2026-05-31 `
  --dry-run
```

使用 Baostock：

```powershell
python backtest/fetch_data/xtquant_day_kline_fetch.py fetch `
  --source baostock `
  --stocks sh.600000,sz.000001 `
  --start-date 2026-05-01 `
  --end-date 2026-05-31 `
  --dry-run
```

`--dry-run` 只检查同步计划和缺失数据，不写入 MongoDB。当前日线入口不提供本地 CSV 导入模式。

## 运行策略回测

所有命令均在仓库根目录执行。建议先使用 `--help` 查看完整参数。

### 小市值流动性轮动

```powershell
python backtest/runs/run_smallcap_liquidity_backtest.py `
  --start-date 2025-01-01 `
  --end-date 2025-04-30
```

参数批量实验：

```powershell
python backtest/runs/run_smallcap_liquidity_batch.py --help
```

### 小市值成交额冲击反转

```powershell
python backtest/runs/run_smallcap_amount_shock_reversal_backtest.py `
  --start-date 2025-01-01 `
  --end-date 2025-04-30
```

参数批量实验：

```powershell
python backtest/runs/run_smallcap_amount_shock_reversal_batch.py --help
```

### Minervini 风格 A 股趋势成长

```powershell
python backtest/runs/run_minervini_ashare_backtest.py `
  --start-date 2025-01-01 `
  --end-date 2025-04-30
```

该策略依赖日线、流通市值及财务成长特征。策略说明见 [`backtest/strategies/MinerviniAshareStrategy.md`](backtest/strategies/MinerviniAshareStrategy.md)。

## 因子研究

`research` 提供与具体证券池解耦的横截面研究流水线：

```text
ResearchRequest
  → DatasetBuilder
  → FactorEngine
  → LabelBuilder
  → PanelAssembler
  → PanelTransformer
  → SampleSelector
  → Grouping
  → MetricSuite
  → Reporter
```

小市值因子研究入口：

```powershell
python -m research.smallcap_factor_research --help
```

或直接运行脚本：

```powershell
python research/smallcap_factor_research/run_research.py --help
```

周度市场状态研究：

```powershell
python -m research.smallcap_factor_research.weekly_regime --help
```

研究框架支持：

- 单因子分组与双排序；
- 多个未来收益周期；
- 结构化样本过滤；
- 批任务共享底层 Panel 和缓存；
- 可选图表与分析 Panel 导出；
- 输出清单、运行元数据和文件校验信息；
- staging 目录生成完成后再原子发布，避免留下半成品结果。

详细接口和扩展约定见 [`research/README.md`](research/README.md)。

## 输出与报告

普通策略回测默认写入：

```text
backtest/runs/output/
```

每次运行会按回测区间和运行时间创建独立目录，避免覆盖历史结果。常见文件包括：

```text
orders.csv
trades.csv
equity.csv
closed_positions.csv
analytics/
  summary.json
  report.md
  equity.csv
  benchmark.csv
  daily_returns.csv
  monthly_returns.csv
  order_stats.csv
  skip_reason_stats.csv
  exit_reason_stats.csv
  position_stats.csv
  charts/
```

实际生成的表格或图表取决于本次回测是否具备对应数据。建议同时检查：

- `orders.csv`：信号是否形成订单、订单为何跳过；
- `trades.csv`：实际成交价格、数量和交易成本；
- `equity.csv`：现金、持仓市值和总权益变化；
- `analytics/report.md`：核心指标、基准对比和图表；
- `closed_positions.csv`：单笔持仓收益和退出原因。

## 项目结构

```text
backtest/
  analytics/       绩效分析、统计表和 Markdown/图表报告
  data/            数据访问门面与本地缓存
  db/              MongoDB 配置、仓储层与同步工具
  execution/       回测引擎、执行器与交易配置
  feature/         特征与财务因子构建
  fetch_data/      行情、财务、复权和分红数据获取
  portfolio/       订单、成交、持仓、公司行为与组合账本
  risk/            止损和退出规则
  runs/            单策略回测与批量实验入口
  strategies/      策略抽象、模型与示例策略
  utils/           证券代码、状态、涨跌停和通用工具

config/            MongoDB 示例配置
examples/          Demo 数据工作流
research/          通用因子研究框架和小市值研究实例
tests/             单元测试
trading/           交易意图模型及执行环境衔接代码
```

## 开发与扩展

### 新增策略

从 `backtest/strategies/base.py` 的策略抽象开始，定义候选池、过滤、排序和目标持仓逻辑。执行和记账应尽量复用现有组件，避免在策略内部直接修改现金或持仓。

### 新增数据访问

优先扩展 `backtest/db/repository.py` 和 `backtest/data/` 的访问接口，使策略和研究代码不直接依赖 MongoDB 查询细节。

### 新增执行或风控规则

- 订单成交规则位于 `backtest/execution/`；
- 组合与公司行为记账位于 `backtest/portfolio/`；
- 止损和退出策略位于 `backtest/risk/`。

新增规则时应同时覆盖成功、无法成交、数据缺失和边界日期等测试场景。

### 新增因子研究组件

研究框架允许替换或新增：

- `DatasetBuilder`
- `FactorSpec` / `FactorRegistry`
- `LabelBuilder`
- `PanelTransformer`
- `SampleSelector`
- `ResearchMetric`
- `Reporter`

具体数据契约和稳定接口以 [`research/README.md`](research/README.md) 为准。

## 使用 MiniQMT 进行实盘交易

Janusquant 可以沿现有的交易意图模型扩展 MiniQMT 实盘执行。当前仓库已经提供：

- `SignalIntent`：记录策略生成的买卖信号；
- `OrderIntent`：统一表达开仓、平仓、加仓、减仓和调仓意图；
- `SizingMode`：支持按数量、金额、目标市值或目标权重描述仓位；
- XtQuant 行情数据同步和证券代码转换工具。

`trading/` 目前是策略与真实券商执行之间的接口层，仓库尚未提供可直接用于真实资金的完整 MiniQMT 下单适配器。接入实盘时，建议保持以下数据流：

```text
策略信号
  → SignalIntent
  → 仓位计算与风控检查
  → OrderIntent
  → MiniQMT 执行适配器
  → 委托与撤单
  → 成交回报
  → 实盘账户、订单和持仓状态同步
```

### 接入前提

- 已安装并登录券商提供的 MiniQMT 客户端；
- 当前 Python 环境能够导入 `xtquant`；
- 已确认 QMT 用户目录、资金账号和账户类型；
- 已在券商测试环境或小资金账户中验证行情、委托、撤单和回报接口；
- MiniQMT 客户端、交易时段和网络状态均正常。

### 建议的适配器职责

实盘适配器应负责将 `OrderIntent` 转换为 MiniQMT 委托，并隔离券商接口细节：

- 连接交易会话，订阅委托、成交、持仓和资产回报；
- 将项目证券代码转换为 XtQuant 格式；
- 根据可用资金、可卖数量和 A 股整手规则计算实际委托数量；
- 支持限价委托、撤单、委托查询和成交查询；
- 保存 `intent_id`、券商委托编号和成交编号之间的映射；
- 对网络中断、重复回报、部分成交和委托状态乱序进行幂等处理；
- 定时用 MiniQMT 账户数据校准本地订单、持仓和现金状态。

### 实盘风控要求

不要直接把回测订单发送到真实账户。至少应增加：

- 交易日、交易时段、停牌、ST 和涨跌停检查；
- 单笔金额、单股仓位、组合仓位和每日成交额限制；
- 可用资金、可卖数量及 T+1 限制检查；
- 重复信号与重复委托拦截；
- 委托超时、撤单失败、部分成交和断线恢复处理；
- 当日亏损、组合回撤和异常状态的全局停止开关；
- 独立的模拟模式，默认只记录拟下单内容，不发送真实委托；
- 完整保存信号、风控决策、请求参数、回报和人工操作日志。

### 推荐上线顺序

1. 只读取 MiniQMT 行情、账户和持仓，不发送委托；
2. 运行模拟模式，对比策略意图与人工预期；
3. 在测试账户验证委托、撤单、部分成交和重启恢复；
4. 使用单标的、最小仓位和严格限额进行真实资金验证；
5. 对账稳定后，再逐步扩大标的范围和资金规模。

> 实盘交易涉及真实资金、券商权限和运行环境差异。使用前应自行审查策略、执行逻辑和风险控制，并承担相应交易风险。

## 测试

运行全部单元测试：

```powershell
python -m unittest discover -s tests -v
```

只运行研究框架测试：

```powershell
python -m unittest tests.test_factor_research_framework -v
```

涉及真实 MongoDB、XtQuant、AkShare 或 Baostock 的功能可能需要额外本地服务、客户端或网络环境。

## 关注作者

- 知乎：[lord_jun](https://www.zhihu.com/people/lord_jun)
- 小红书：[作者主页](https://www.xiaohongshu.com/user/profile/67ea02740000000006010c87)

## License

本项目采用 MIT License，详见 [`LICENSE`](LICENSE)。
