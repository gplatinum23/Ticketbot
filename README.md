# Flight Watch Agent

涓€涓熀浜?LangGraph 鐨勫疄鏃跺嚭琛岃鍒掓鏋躲€傚綋鍓嶄繚鐣欎袱绉嶅叆鍙ｏ細

- `plan-flight`: 缁撴瀯鍖栧弬鏁版煡璇?12306 鐏溅鍊欓€夛紝骞堕€氳繃 Web Search 鎼滅储鍏紑鏈虹エ鍊欓€夈€?- `ask`: 浣跨敤 LLM 浠庤嚜鐒惰瑷€涓В鏋愬嚭琛岄渶姹傦紝鍐嶈皟鐢ㄥ悓涓€鏉¤鍒掓祦绋嬨€?
闀挎湡鏈虹エ浠锋牸鐩戞帶銆丼QLite 鐩戞帶浠诲姟銆侀槇鍊奸€氱煡鍜?mock 闀挎湡鏌ヤ环鍔熻兘宸茬粡绉婚櫎銆?
## 鐜

椤圭洰鎸?`agent_env` 铏氭嫙鐜浣跨敤锛?
```powershell
python -m venv agent_env
.\agent_env\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## 閰嶇疆

榛樿浼氳鍙栭」鐩牴鐩綍鐨?`.env` 鏂囦欢銆傚彲浠ヤ粠 `.env.example` 澶嶅埗涓€浠斤細

```powershell
Copy-Item .env.example .env
```

鐒跺悗鍦?`.env` 涓～鍐欙細

```dotenv
OPENAI_API_KEY=your-api-key
FLIGHT_WATCH_LLM_MODEL=openai:gpt-4.1-mini
FLIGHT_WATCH_12306_MCP_COMMAND=
FLIGHT_WATCH_12306_MCP_ARGS=
FLIGHT_WATCH_12306_DEBUG=false
```

12306 默认通过 `npx -y 12306-mcp` 启动，需要本机可用 Node.js/npm/npx。通常不需要填写 `FLIGHT_WATCH_12306_MCP_COMMAND` 和 `FLIGHT_WATCH_12306_MCP_ARGS`；只有当你想使用自定义 MCP 启动命令时才需要配置。
绯荤粺鐜鍙橀噺浼樺厛浜?`.env`銆傚鏋滆鎸囧畾鍏朵粬 env 鏂囦欢锛?
```powershell
$env:FLIGHT_WATCH_ENV_FILE="config/local.env"
```

## 浣跨敤

缁撴瀯鍖栧弬鏁版煡璇?12306 鐏溅鍊欓€夊拰鍏紑鏈虹エ鍊欓€夛細

```powershell
flight-watch plan-flight --origin BJP --destination SHH --travel-date 2026-07-09 --time-preference morning --budget 1200
```

鑷劧璇█瑙勫垝锛?
```powershell
flight-watch ask "甯垜鏌ヤ竴涓?2026-07-09 鍖椾含鍒颁笂娴凤紝涓婂崍鍑哄彂锛岄绠?1200 浠ュ唴鐨勬柟妗?
```

鍗曠嫭璋冭瘯鍏紑椤甸潰鏈虹エ鎼滅储锛屼笉鏌ヨ鐏溅銆佷笉鍋氳矾绾挎帓搴忥細

```powershell
flight-watch debug-flight-search --origin SIN --destination TFU --travel-date 2026-07-09 --max-iterations 1 --no-llm-judge
```

鍘绘帀 `--no-llm-judge` 鍚庝細鎺ュ叆 LLM 鍒ゆ柇缃戦〉鎶藉彇缁撴灉锛涜皟璇曡緭鍑哄寘鍚?`search_queries`銆乣raw_results`銆乣extracted_evidence`銆乣judged_evidence`銆乣verified_flight_options` 鍜?`warnings`銆?
鍗曠嫭璋冭瘯鎼虹▼ SeleniumWire 鐖櫕璺緞锛?
```powershell
python -m pip install -e ".[ctrip]"
flight-watch debug-flight-search --origin SIN --destination TFU --travel-date 2026-07-09 --max-iterations 1 --no-llm-judge
```

鎼虹▼璺緞鍙傝€?`Suysker/Ctrip-Crawler` 鐨勫仛娉曪細鍚姩娴忚鍣ㄣ€佽闂惡绋嬫満绁ㄩ〉闈€佹崟鑾?`/international/search/api/search/batchSearch` 鍝嶅簲锛屽啀瑙ｆ瀽 `flightItineraryList` 鍜?`priceList`銆傝繖鏉¤矾寰勪緷璧栨湰鏈烘祻瑙堝櫒/椹卞姩銆丼eleniumWire 璇佷功浠ｇ悊鍜屾惡绋嬮〉闈㈤鎺э紱濡傞渶瑙傚療椤甸潰浜や簰锛屽彲璁剧疆 `FLIGHT_WATCH_CTRIP_HEADLESS=false`銆?
`plan-flight` 会先通过本地 `12306-mcp` 调用 12306 的 `get-tickets`，一次性获取火车余票和官网展示价；随后执行最多 3 轮“生成搜索词 -> Web Search -> 页面抽取 -> LLM 证据判断 -> 候选归一化”的 ReAct 机票搜索循环。机票价格不调用官方机票 API，只来自公开网页搜索、页面抽取或携程 SeleniumWire 捕获；LLM 只负责判断网页抽取结果是否像有效机票证据、过滤错误价格并做字段归一化。当前阶段单个可用来源即可进入推荐结果；价格会标注为公开页面估算价，不保证最终可购价。
鏈虹エ鎼滅储浼氫紭鍏堝皾璇曟瀯閫?Skyscanner route 椤甸潰锛屼緥濡?`https://www.skyscanner.com.sg/routes/sin/tfu/singapore-changi-to-chengdu-tianfu-international.html`锛屽啀琛ュ厖 DuckDuckGo 鎼滅储缁撴灉銆係kyscanner route 椤甸潰閫氬父鏄?JavaScript 搴旂敤澹筹紝褰撳墠 HTTP 椤甸潰鎶藉彇鍣ㄤ笉淇濊瘉鑳界洿鎺ヨ鍒颁环鏍笺€?
## 褰撳墠杈圭晫

- 火车查询来自本地 `12306-mcp`，通过 `get-tickets` 返回余票和官网展示价。- 机票搜索来自公开 Web Search、页面抽取和 LLM 证据判断，不是官方机票 API。- 当前输出火车候选和 verified flight candidate；完整“火车 + 飞机”多段组合与路线评分后续再补。- `ask` 和默认 `plan-flight` 都需要配置可用的 LLM API key；`ask` 用于解析自然语言，`plan-flight` 的机票 ReAct 节点用于判断网页证据。
