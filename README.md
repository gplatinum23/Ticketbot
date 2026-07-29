# Flight Watch Agent

Flight Watch Agent 是一个基于 LangGraph 的综合出行规划 Agent。它能够理解自然语言需求，查询火车与机票，并比较直达及中转组合，给出兼顾价格、时间和换乘体验的出行建议。

## 功能

- 自然语言解析出发地、目的地、日期、时间偏好和预算。
- 通过 `12306-mcp` 查询火车车次、余票和票价。
- 通过携程公开页面获取机票信息，不依赖官方机票 API。
- 支持直达火车、机票，以及火车和飞机的两段式组合。
- 结合本地机场、火车站索引与 LLM 生成中转城市。
- 对候选路线进行可行性检查和综合排序，输出 Top 5。
- 携程出现验证码时支持人工完成验证后继续运行。

## 系统流程

```mermaid
flowchart TD
    User[自然语言或结构化输入] --> Intent[LLM 解析出行需求]
    Intent --> Complete{信息完整?}
    Complete -->|否| Clarify[返回追问]
    Complete -->|是| Strategy[区域判断与策略选择]

    Strategy --> Hubs[生成并校验中转 Hub]
    Strategy --> Direct[并行查询直达机票]
    Hubs --> Plan[构建查询计划]
    Direct --> Plan

    Plan --> Train[12306 MCP 火车查询]
    Plan --> Init

    subgraph React[Flight ReAct 机票搜索]
        Init[初始化单航线搜索] --> Ctrip[访问携程页面]
        Ctrip --> Observe{检查页面结果}
        Observe -->|获得航班| Evidence[LLM 判断并验证证据]
        Observe -->|页面异常| Decide[LLM 选择下一步动作]
        Observe -->|人工验证码| Human[等待人工验证]
        Decide --> Ctrip
        Human --> Ctrip
        Decide -->|结束或采用备选| Evidence
    end

    Train --> Routes[组合候选路线]
    Evidence --> Routes
    Routes --> Rank[综合排序与确定性校验]
    Rank --> Result[Top 5 出行方案]
```

系统可组合以下路线：

- 直达飞机或直达火车
- 火车 + 飞机
- 飞机 + 火车
- 火车 + 火车
- 飞机 + 飞机

## 环境要求

- Python 3.11+
- Node.js LTS，且 `node`、`npm`、`npx` 可用
- Microsoft Edge 或 Chrome

```powershell
python -m venv agent_env
.\agent_env\Scripts\Activate.ps1
python -m pip install -e ".[dev,ctrip]"
```

火车查询默认使用：

```powershell
npx -y 12306-mcp
```

## 配置

```powershell
Copy-Item .env.example .env
```

常用配置：

```dotenv
OPENAI_API_KEY=your-api-key
FLIGHT_WATCH_LLM_MODEL=openai:gpt-4.1-mini
FLIGHT_WATCH_FAST_LLM_MODEL=openai:gpt-4.1-mini
FLIGHT_WATCH_ROUTE_LLM_MODEL=

FLIGHT_WATCH_12306_MCP_COMMAND=
FLIGHT_WATCH_12306_MCP_ARGS=

FLIGHT_WATCH_CTRIP_BROWSER=edge
FLIGHT_WATCH_CTRIP_HEADLESS=false
FLIGHT_WATCH_CTRIP_LOGIN_ALLOWED=true
FLIGHT_WATCH_CTRIP_USERNAME=
FLIGHT_WATCH_CTRIP_PASSWORD=
FLIGHT_WATCH_CTRIP_COOKIES_FILE=data/ctrip_cookies.json
FLIGHT_WATCH_CTRIP_REUSE_BROWSER=true
```

程序默认读取项目根目录的 `.env`，系统环境变量优先。需要人工处理携程验证码时，应使用非无头浏览器。

## 领域查询工具

规划图通过两个稳定的领域工具访问外部数据：

- `FlightSearchTool`：接收 `FlightSearchRequest`，返回 `ToolResult[FlightSearchOutput]`。
- `TrainSearchTool`：接收 `TrainSearchRequest`，返回 `ToolResult[TrainSearchOutput]`。

工具提供单次 `search()` 和批量 `search_many()` 接口。批量接口保持输入顺序、自动消除重复请求，并复用线程安全的 L1 内存缓存与 SQLite L2 持久缓存。成功结果默认缓存 300 秒、空结果默认缓存 60 秒；错误、验证码、登录和取消结果不会缓存。

```python
from datetime import date

from flight_watch_agent.app import build_default_flight_query_tool
from flight_watch_agent.travel_tools import FlightSearchRequest

tool = build_default_flight_query_tool()
result = tool.search(
    FlightSearchRequest(
        origin="CTU",
        destination="CJU",
        travel_date=date(2026, 7, 31),
    )
)
if result.ok and result.data:
    print(result.data.options)
elif result.error:
    print(result.error.code, result.error.retryable, result.error.message)
```

统一状态为 `success`、`no_results`、`human_action_required` 和 `error`。错误码包括：

```text
invalid_input, timeout, captcha_required, login_required, rate_limited,
route_mismatch, parse_failed, tool_unavailable, cancelled, internal_error
```

每个结果同时返回 `request_id`、耗时、缓存命中、尝试次数、终止原因、退避记录和后端名称。可通过以下配置调整缓存：

```dotenv
FLIGHT_WATCH_TOOL_CACHE_TTL_SECONDS=300
FLIGHT_WATCH_TOOL_CACHE_NO_RESULT_TTL_SECONDS=60
FLIGHT_WATCH_TOOL_CACHE_MAX_ENTRIES=512
FLIGHT_WATCH_TOOL_CACHE_NO_RESULTS=true
FLIGHT_WATCH_TOOL_CACHE_PATH=data/tool_cache.sqlite3
```

缓存键包含工具名、规范化请求、后端版本、解析版本、输出 schema 版本和 scope。机票缓存只保存结构化领域结果和脱敏证据，绝不保存浏览器 `raw_state`、Cookie、Token 或 URL 查询参数。需要手动清理时：

```powershell
.\agent_env\Scripts\flight-watch.exe clear-cache
```

需要注册给 LangChain Agent 时，可直接使用：

```python
from flight_watch_agent.app import build_default_agent_tools

tools = build_default_agent_tools()
# names: search_flights, search_trains
```

Agent 适配器只返回精简、可序列化的公开字段，不暴露浏览器状态或原始抓包。

### 工具执行 Runtime

`search()` 与 `search_many()` 可选接收 `ToolExecutionContext`，用于提供共享的绝对 deadline、取消信号、追踪标识和幂等键；旧调用方式保持兼容。`ToolRuntime` 是火车和机票工具唯一的重试边界：只会对 `timeout`、`rate_limited`、`tool_unavailable` 执行有限指数退避重试，其余错误（包括验证码、登录和解析失败）立即返回。同步后端调用无法被强行中断，但 deadline 或取消一旦被观察到，Runtime 不会再启动未开始的子请求或重试，并保留已经完成的批次结果。

### 携程工具组件边界

携程查询由 `CtripFlightBackend` 编排：`CtripNavigator` 负责入口导航，`BrowserSessionManager` 管理浏览器生命周期和登录会话，`CtripCaptureBackend` 只抓取 `batchSearch` 原始响应，`CtripPayloadParser` 纯解析 payload，`FlightEvidenceMapper` 转换为领域证据。`CtripSeleniumWirePageExtractor` 仅保留为旧 `PageExtractor` 协议的兼容适配器。

原始响应通过 `RawResponseStore` 生成脱敏引用；工具、规划图和 LangChain 适配器不会取得浏览器请求或 Cookie。当前内存存储用于本进程追踪，持久录制/回放和 SQLite 缓存将在后续 P1 阶段完成。

### 路线正确性边界

- 查询地点会保留原始输入，并分别解析为城市、实际机场或火车站。城市代码（如 `BJS`）可以匹配 `PEK/PKX`，明确指定 `PEK` 时不会接受 `PKX` 证据。
- 机票结果同时返回 `requested_origin/requested_destination` 与 `actual_origin/actual_destination`，展示和组合路线使用证据中的实际机场。
- 内部行程时间均为带时区的绝对 `datetime`。用户日期表示首段在出发地的本地出发日期；跨日和跨时区计算不再只比较 `"HH:MM"`。
- 火车转飞机、飞机转火车和同城跨机场必须存在地面接驳证据。接驳边包含耗时、缓冲、费用、来源和可靠性；没有接驳证据的跨城组合不会成为候选路线。
- 当前地面接驳数据是保守的内置估算，最终下单前仍应核验实时交通情况。
- 候选路线在评分和排序前统一经过确定性可行性引擎。引擎校验实际端点连通、带时区的绝对时间顺序、地面接驳以及国内/国际、同机场/跨机场、火车转飞机等不同换乘缓冲。
- 可行性结果分为 `feasible`、`infeasible` 和 `uncertain`，并附带结构化原因及可用/必需换乘分钟数。`infeasible` 不进入 Top 5；`uncertain` 会在推荐文本中标出风险代码。
- 排序完成后会按交通方式、实际端点、核心班次和换乘 Hub 生成路线骨架，移除完全重复骨架，并限制同一后半程航班的轻微火车变体最多占两个推荐名额。
- 最终候选通过确定性多样性选择覆盖 `best_overall`、`lowest_price`、`shortest_duration`、`lower_transfer_risk` 和 `shortest_wait` 等价值取向；可用的差异化路线不足 5 条时会明确说明实际数量。

## 使用

### 自然语言规划

```powershell
.\agent_env\Scripts\flight-watch.exe ask `
  "帮我查一下 2026-08-15 成都到新加坡，上午出发，预算 3000 以内的方案"
```

仅查询机票：

```powershell
.\agent_env\Scripts\flight-watch.exe ask `
  "帮我查一下 2026-08-15 成都到新加坡的方案" `
  --flight-only
```

### 结构化参数规划

```powershell
.\agent_env\Scripts\flight-watch.exe plan-flight `
  --origin CTU `
  --destination SIN `
  --travel-date 2026-08-15 `
  --time-preference morning `
  --budget 3000 `
  --currency CNY
```

### 调试机票搜索

```powershell
.\agent_env\Scripts\flight-watch.exe debug-flight-search `
  --origin CTU `
  --destination SIN `
  --travel-date 2026-08-15
```

`ask` 和 `plan-flight` 支持 `--quiet` 关闭进度提示，也可以使用 `--show-flight-raw` 查看机票搜索状态。`debug-flight-search` 支持 `--no-llm-judge` 跳过 LLM 证据判断。

## 本地数据

- `resources/station_name.js`：12306 火车站名称与代码。
- `resources/airports_normalized_with_flight_potential.csv`：机场、城市、IATA 和航班潜力。
- `resources/airports.csv`、`resources/airports.json`：机场索引补充数据。

## 测试

```powershell
.\agent_env\Scripts\python.exe -m pytest -q
```

已完成的 P1.0–P1.2 可使用独立门槛验证：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify_p1_foundation.ps1
```

该命令验证工具契约、携程组件边界、Runtime 重试/取消/部分成功、SQLite 持久缓存和公开 LangChain 输出；它不会访问真实 12306 或携程。并发治理、脱敏回放和工具注册属于后续补充性能与稳定性计划。

## 当前限制

- 携程页面、登录风控和验证码变化可能影响机票抓取。
- 12306 只能查询预售期内的日期。
- 地面接驳目前使用保守的静态估算，尚未接入实时路况和动态费用。
- 当前只支持直达和两段式路线。
- 页面票价是查询时的公开价格，不保证最终可购。
