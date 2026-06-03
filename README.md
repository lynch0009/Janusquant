# Janusquant

Janusquant 是一个面向 A 股量化研究的 Python 项目，覆盖数据接入、因子研究、策略回测、组合分析与结果导出等常用工作流。项目适合用于验证选股逻辑、比较参数组合、复盘交易过程，并沉淀可重复运行的研究实验。

## 项目功能

- 数据访问：封装 MongoDB 连接、仓储层、行情与特征数据读取接口。
- 数据构建：提供日线、复权因子、分红、财务数据、xtdata 日线等数据获取与同步脚本。
- 因子研究：支持截面分组、指标计算、缓存、报告生成与小市值因子实验。
- 策略实现：内置小市值流动性轮动、小市值成交额冲击反转、Minervini 风格 A 股趋势成长策略。
- 回测执行：包含回测执行器、订单模型、成交模型、持仓记账、退出规则和风控规则。
- 结果分析：输出订单、成交、持仓、净值、绩效指标和分析报告，便于复盘与二次分析。
- 参数实验：提供批量回测入口，可用于策略参数网格搜索和稳定性比较。

## 项目优势

- 模块边界清晰：数据、策略、执行、组合、风控和研究模块分层组织，便于扩展。
- 面向真实研究流程：从数据准备、策略开发、批量实验到结果分析都有对应代码入口。
- 策略样例完整：内置策略不是孤立片段，而是可以串联数据读取、回测执行和报表输出的完整示例。
- 配置可本地化：敏感数据库连接信息集中在 `config/` 管理，并支持环境变量覆盖。
- 适合继续开发：策略抽象、仓储层和交易意图层都预留了扩展空间，便于加入新的数据源、因子和策略。

## 项目结构

```text
backtest/             回测与研究核心代码
backtest/analytics/   回测分析相关工具
backtest/data/        数据访问门面
backtest/db/          MongoDB 配置、仓储层与同步工具
backtest/execution/   回测执行器、订单执行与配置
backtest/feature/     特征与因子处理
backtest/fetch_data/  行情、财务、复权、分红等数据获取脚本
backtest/portfolio/   持仓、订单、成交与绩效分析
backtest/risk/        风控与退出规则
backtest/runs/        回测入口与批量实验入口
backtest/strategies/  策略抽象与策略实现
backtest/utils/       通用工具函数

config/               本地配置与示例配置
examples/             Demo 数据与示例工作流
research/             因子研究框架与研究实例
trading/              交易意图层与执行环境衔接代码
tests/                单元测试
```

## 环境准备

建议使用 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
```

项目依赖包括 pandas、numpy、pymongo、matplotlib、plotly、akshare、baostock、xtquant 等常用研究、数据与报告库。`TA-Lib` 作为可选本地依赖，可按自己的环境单独安装。

## 配置 MongoDB

回测和研究脚本默认从本地 MongoDB 读取行情、特征、财务和股票基础信息等数据。先复制示例配置：

```powershell
Copy-Item config/mongodb.example.conf config/mongodb.conf
```

macOS / Linux：

```bash
cp config/mongodb.example.conf config/mongodb.conf
```

然后按本地环境填写 `config/mongodb.conf`：

```ini
[mongodb]
host = 127.0.0.1
port = 27017
db_name = quant
username =
password =
auth_mechanism = SCRAM-SHA-256
connect_timeout_ms = 10000
server_selection_timeout_ms = 10000
socket_timeout_ms = 30000
max_pool_size = 100
min_pool_size = 0
retry_reads = true
retry_writes = true
connect_eagerly = false
```

配置读取逻辑位于 `backtest/db/mongodb.py`。除配置文件外，也可以用环境变量覆盖连接参数：

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

## 准备数据

仓库不内置生产行情数据。运行完整回测通常需要本地 MongoDB 中具备以下数据：

- 交易日历数据
- 日线行情数据
- 分钟线行情数据
- 股票基础信息数据
- 复权因子、分红、财务或特征截面数据

可以通过 `backtest/fetch_data/` 中的数据脚本构建本地数据，也可以使用 `examples/demo_mongo/` 的小型 Demo 数据先跑通流程。

## 运行 Demo

Demo 工作流会导入一组小型 Mongo JSONL 数据，并运行两条小市值策略示例。

```powershell
cd examples/demo_mongo
docker compose up -d
```

导入 Demo 数据：

```powershell
python import_demo_mongo.py --drop-existing
```

检查数据：

```powershell
python check_demo_data.py
```

运行小市值 Demo：

```powershell
.\run_demo_smallcap.ps1 -Python python
```

macOS / Linux：

```bash
PYTHON_PATH=python bash examples/demo_mongo/run_demo_smallcap.sh
```

更多说明见 `examples/demo_mongo/README.md`。

## 常用回测入口

基础小市值流动性轮动：

```bash
python backtest/runs/run_smallcap_liquidity_backtest.py --start-date 2025-01-01 --end-date 2025-04-30
```

小市值流动性轮动参数实验：

```bash
python backtest/runs/run_smallcap_liquidity_batch.py
```

小市值成交额冲击反转：

```bash
python backtest/runs/run_smallcap_amount_shock_reversal_backtest.py --start-date 2025-01-01 --end-date 2025-04-30
```

小市值成交额冲击反转参数实验：

```bash
python backtest/runs/run_smallcap_amount_shock_reversal_batch.py
```

Minervini 风格 A 股趋势成长策略：

```bash
python backtest/runs/run_minervini_ashare_backtest.py --start-date 2025-01-01 --end-date 2025-04-30
```

回测结果默认输出到：

```text
backtest/runs/output/
```

常见输出包括订单记录、成交记录、持仓变化、净值曲线、绩效指标和分析报告。

## 开发建议

- 新增策略时，可以继承 `backtest/strategies/base.py` 中的策略抽象。
- 新增数据访问逻辑时，优先复用 `backtest/db/repository.py` 和 `backtest/data/` 的门面接口。
- 批量实验可以参考 `backtest/runs/*_batch.py` 的参数组织方式。
- 因子研究可以从 `research/smallcap_factor_research/` 和 `research/runner.py` 开始阅读。

## License

This project is licensed under the MIT License. See `LICENSE` for details.
