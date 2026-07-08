# Flight Watch Agent

一个基于 LangGraph 的机票价格关注 agent 框架。当前默认使用 `MockFlightPriceProvider` 跑通查价流程，并新增 LLM 自然语言入口，用来把用户需求解析成结构化监控任务。

## 能力边界

- 用自然语言创建监控任务，例如“帮我盯一下 SHA 到 NRT，2026-09-20 出发，低于 1800 CNY 提醒我”。
- 输入出发地、目的地、出发日期、价格阈值，创建长期监控任务。
- 周期性查询票价。
- 票价低于或等于阈值时发送通知。
- 使用 SQLite 保存监控任务、历史价格和通知记录。
- 用 LangGraph 编排两类流程：
  - LLM 需求解析流程：自然语言 -> 结构化意图 -> 本地校验 -> 创建任务或要求补充信息。
  - 价格检查流程：监控任务 -> 查询票价 -> 判断阈值 -> 通知 -> 记录结果。

## 环境

项目按 `agent_env` 虚拟环境使用：

```powershell
python -m venv agent_env
.\agent_env\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## LLM 配置

默认会读取项目根目录的 `.env` 文件。可以从 `.env.example` 复制一份：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填写：

```dotenv
OPENAI_API_KEY=your-api-key
FLIGHT_WATCH_LLM_MODEL=openai:gpt-4.1-mini
```

也可以继续用系统环境变量；系统环境变量优先于 `.env`：

```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:FLIGHT_WATCH_LLM_MODEL="openai:gpt-4.1-mini"
```

如果要指定其他 env 文件：

```powershell
$env:FLIGHT_WATCH_ENV_FILE="config/local.env"
```

也可以在命令行临时指定：

```powershell
flight-watch ask --model openai:gpt-4.1-mini "帮我盯一下 SHA 到 NRT，2026-09-20 出发，低于 1800 CNY 提醒我"
```

## 快速开始

用自然语言添加一个监控任务：

```powershell
flight-watch ask "帮我盯一下 SHA 到 NRT，2026-09-20 出发，低于 1800 CNY 提醒我"
```

用结构化参数添加一个监控任务：

```powershell
flight-watch add --origin SHA --destination NRT --depart-date 2026-09-20 --threshold 1800 --interval 3600
```

查看任务：

```powershell
flight-watch list
```

执行一次检查：

```powershell
flight-watch run-once
```

持续监控：

```powershell
flight-watch watch
```

使用 ReAct 机制搜索公开机票价格候选：

```powershell
flight-watch plan-flight --origin BJS --destination SHA --travel-date 2026-07-09 --time-preference morning --budget 1200
```

`plan-flight` 会执行最多 3 轮“生成搜索词 -> Web Search -> 页面抽取 -> 证据验证”的 ReAct 循环。机票候选至少需要 2 个独立公开来源验证后才会进入推荐结果；价格会标注为公开页面估算价，不保证最终可购价。

默认数据库文件是 `data/flight_watch.sqlite3`。可以用环境变量覆盖：

```powershell
$env:FLIGHT_WATCH_DB="data/dev.sqlite3"
```

## 后续接真实接口

在 `src/flight_watch_agent/providers.py` 中实现 `FlightPriceProvider` 协议：

```python
class MyProvider:
    def get_lowest_price(self, request: FlightSearchRequest) -> FlightQuote:
        ...
```

然后在 `src/flight_watch_agent/app.py` 的 `build_default_agent` 中替换 provider 构造即可。
