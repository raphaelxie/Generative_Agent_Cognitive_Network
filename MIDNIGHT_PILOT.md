# 三日 network-cognition pilot

本入口实现 Day 1 00:00 → Day 3 16:00 的 64 小时未干预 pilot。与旧版实验入口分开，默认创建全新的输出目录，不续跑 GPT-4o-mini 的历史，不启动 hub/broker 干预分支。

当前执行任务是在本机续跑已到 Day 1 14:00 的 `smoke_mobility_v5`，并独立执行每个快照三次问卷，命令见 [README.md](README.md)。[MAIN_10202026_RUN_GUIDE.md](MAIN_10202026_RUN_GUIDE.md) 是 10 月 20 日限时运行的历史记录，其中的截止时间不适用于本次续跑。

## 启动

使用已安装项目依赖的 Python 环境，再安装 `requirements-pilot.txt` 中缺少的依赖。从项目根目录执行：

```bash
python3 reverie/backend_server/run_midnight_pilot.py describe
python3 reverie/backend_server/run_midnight_pilot.py preflight
python3 reverie/backend_server/run_midnight_pilot.py run
```

`describe` 不写数据、不调用 API。`preflight` 会发出五个短聊天请求和一个 embedding 请求，验证 Luna 的 JSON、短输出和 embedding 接口；预检失败不会开始 simulation。`run` 没有预检记录时自动预检，有记录时要求配置相符。两者均读取项目 `.env` 中的凭据，明确将聊天模型锁定为 `gpt-6-luna`、reasoning effort 为 `none`，embedding 保持 `text-embedding-3-small`。不自动降级到其他模型。

默认 seed 是 104729，每个测量状态独立测量三次，输出在 `pilot_runs/midnight_luna_seed104729/`。`--output PATH --seed N --repeats N` 可为另一条预先指定的 pilot 设置新目录和 seed；重复次数至少为 2。恢复同一条轨迹时使用原有参数。预检与运行配置包含源码摘要、Python/依赖版本、模型、seed、重复次数和 API endpoint。运行开始后不要修改这些配置；源码变化需要新的输出目录。

## 时间和 checkpoint

- 10 秒一步，终点 step 23040；夜间仍执行正常 engine ticks。
- 初始测量在 step 0、首次感知或互动之前；使用人物设定作为背景，不对空历史计算准确率。
- Day 1/2 的 10:00、14:00、18:00，以及 Day 3 的 10:00、14:00、16:00 测量。
- Day 1 20:00 至 Day 2 08:00，每 30 分钟检查启动以来累计对话图，共 25 次。对话按参与者和开始时间去重，统计区间为 `[simulation_start, snapshot_time)`，保留全体 25 个节点。
- hub 为最大 degree；broker 为无权、归一化 betweenness 最大且大于零者。最大值集合不得相交；betweenness 并列容差为绝对值 `1e-12`。
- 并列时使用 seed 与姓名的 SHA-256 优先顺序，不消耗 simulation 的随机数。连续两次合格且两名目标相同，在第二次锁定；最早 Day 1 20:30。锁定后继续记录整个窗口，绝不重新选点。
- 无合格状态就记录 `no_eligible_checkpoint`，仍跑到固定终点。不得自动延长窗口或换用第二名。
- checkpoint 与其后 +2、+4、+24 小时都保存完整状态；同一时刻复用一个快照。后面三个快照默认仅存档，不额外触发夜间测量。

晚睡修复保留逐日生成的起床时间：Hailey 在 02:00 前、Ryan 在 01:00 前不再被早晨睡眠前缀强制设为 sleeping，小时计划提示也明确说明跨午夜的晚间活动。晚间活动仍由 agent 生成，没有统一强制就寝时间。计划重复生成不会丢失早晨睡眠段。跨日动作结束比较使用完整日期时间。

## Survey 隔离和统计

每个快照、每次重复均启动新的子进程。该进程只读取快照，在自己的状态副本上检索和回答，最终退出；它不能把问题、答案、访问时间或 RNG 变化写回 live simulation。每次测量前后检查完整 live state/RNG 摘要与源快照文件摘要。之前的答卷不会作为下一次测量的上下文。

四项问卷均明确询问最近 120 分钟，使用同一份带记录时间的个人记忆上下文，并区分 thought 的记录时间与事件发生时间：

1. 全部 300 个无序人对的互动概率，Brier score。
2. degree 排名，Kendall tau-b，保留真值并列。
3. broker 排名，对照无权 betweenness，Kendall tau-b。
4. 社区划分，对照对话次数加权的 Louvain（seed 42），ARI，另报 NMI。

重复测量噪声是每个状态各次测量平均分的样本标准差，不是独立 simulation 之间的方差。解析失败最多重试三次，仍失败则记为缺失，不生成随机排名或默认零分。初始准确率及无定义的 rank correlation 留空。模型/连接故障会中断运行，不能被旧代码的默认回答悄悄替代；从上一恢复点重启时重复消耗的 API 用量仍保留在账本中。

先运行、之后再测量：

```bash
python3 reverie/backend_server/run_midnight_pilot.py run --defer-surveys
python3 reverie/backend_server/run_midnight_pilot.py measure
python3 reverie/backend_server/run_midnight_pilot.py report
```

跳过测量时状态标为 `simulation_complete_surveys_pending`。`measure` 完成全部指定测量（含最终测量）后才标为 `complete`。被中断的问卷保留已经完成的 respondent，同一重复的剩余 respondent 从原快照恢复；已有回答不进入模型上下文。

若需要保留旧轨迹配置、另外对同一批快照执行三次问卷，使用 `measure_saved_pilot.py --source-run SOURCE --output SEPARATE_OUTPUT --repeats 3`。该入口单独锁定测量设置、串行执行独立 worker、核对源文件完整性，并把分析写到独立输出目录；不改变原轨迹的重复次数或完成状态。可在后续快照生成后再次运行以补齐测量。本机续跑及三次测量命令见 [README.md](README.md)。

## 恢复与未来分支

每 30 个模拟分钟保存恢复点，涵盖记忆、scratch、空间记忆、计划、隔离标记、场景事件、待清理对象事件、下一帧位置、时钟及 Python/NumPy RNG。重启时重新执行同一个 `run` 命令。恢复会重做上次未完成的区间；远端模型本身具有随机性，因此无法保证故障后的重试逐字重现此前未提交的回答。

现在还会在距上次保存超过 60 个现实秒后的第一个完整 tick 保存恢复点。`run` / `measure` 支持 `--survey-cutoff`、`--stop-at`（均为带 UTC offset 的现实日期时间）和 `--stop-file`。正常暂停状态为 `paused`，同一输出目录的 `run` 可继续；时间限制不进入实验配置锁。正在进行的 survey 可安全结束子进程，已完成 respondent 保留。`verify --output PATH` 不调用 API，读取完整快照并核对 manifest。`pilot_session.py` 提供独立监督器、caffeinate 和处理卡住进程的现实时间兜底，具体命令见本次运行指南。

完整状态采用本地 pickle/gzip 存储并检查结构摘要，**只能加载本项目自己生成、可信的快照**，不要导入下载或别人提供的 pickle。

未来分支准备命令（只准备状态，不运行干预实验）：

```bash
python3 reverie/backend_server/run_midnight_pilot.py fork --arm hub --destination pilot_runs/hub_prepared
python3 reverie/backend_server/run_midnight_pilot.py fork --arm broker --destination pilot_runs/broker_prepared
python3 reverie/backend_server/run_midnight_pilot.py fork --arm control --destination pilot_runs/control_prepared
```

三者均加载锁定的同一 checkpoint；干预分支只修改相应目标的 isolation 标记。记录干预时刻和已在进行的聊天，已有聊天保留。此 runner 专用于未干预 pilot，不把准备好的干预状态当作新 pilot 自动运行。正式分支 runner 应读取 `branch.pkl.gz`，记录首次受阻的新互动，并按相同干预后时距比较。

## 输出与验收

- `manifest.json`：配置、进度、checkpoint 状态及快照索引。
- `checkpoint_candidates.json`：整个窗口的 degree、betweenness、并列集合和资格判定。
- `network_history.json`：累计/近期网络密度、边数、参与人数、连通分量及当前活动。
- `snapshots/`、`latest.pkl.gz`：永久测量/分支快照和最近恢复点。
- `surveys/`：各次独立答卷、检索诊断、状态隔离检查和完成标记。
- `usage.jsonl`：实际 API token usage、请求标识、阶段和模拟时刻。报告按 06:00–20:00 / 20:00–06:00 分昼夜，这只是费用分组，不是作息规则。
- `REPORT.md`、`report.json`、`perception_scores.csv`：测量噪声、缺失数据和昼夜费用。美元数是公开 Standard 价格的估算，原始 usage 保留；供应商账单及区域附加费可能不同。

验证：

```bash
python3 -m pytest -q reverie/backend_server/tests
```

测试覆盖边界时间、并列与零 betweenness、不合格窗口、首次锁定、同一时刻快照复用、夜间排程、跨日动作、完整恢复、问卷进程隔离、四项测量验证及费用归集。真实场景测试使用项目的 25-agent 初始模板；可用 `NCN_TEST_PROJECT` 指向提供模板和场景资源的项目目录。测试中的认知响应由固定夹具替代，不调用付费 API。

科学解释限定于“初步形成且 hub/broker 可分离的网络”。一条 pilot 不支持总体因果结论。正式样本数量、是否保留 64 小时、目标分离的通过率以及正式干预时距应在查看 pilot 诊断后另行确定，不能按显著性选择停止点。

## Action contract v4：智能体能够离开住所（2026-10-24 修复）

**发现的缺陷。** 此前 25 人、64 模拟小时的轨迹（`main_10212026_v3_1_continuation`）中累计网络只有 3 条边（Tom–Jane Moreno、John–Mei Lin、Sam–Jennifer Moore），所有节点 betweenness 为 0，角色窗口 25 次检查全部不合格。原因不在网络指标，而在行为：选择活动地点的 `action_location_sector_v1` / `action_location_object_vMar11` 是补全式提示，以未闭合的 `{` 结尾，旧校验器要求回复含 `}`。`gpt-6-luna` 按聊天方式回答（如 `Hobbs Cafe`），5 次重试全部被拒后，旧 fail-safe 把智能体送回住所，arena 退回自己的房间。该轨迹中约 3,800 次地点决定 100% 落在自己的住所，没有任何人去过咖啡馆、公园、酒吧、学院或商店，同一路径上的事件三元组（19/25 的 `act_event` 为 `(name, name, "is")`）和 `decide_to_talk`（解析失败回退为 "yes"）也在静默失败。只有共用一间卧室的三对夫妻能相互感知，恰好得到那 3 条边。该轨迹的网络结果反映的是此缺陷，不能解释为“该模型下不会形成网络”。

**修复。** 新增 `pilot_actions.py`，在 `NCN_PILOT=1` 下为地点（sector / arena / object）、事件三元组、`decide_to_talk`、`decide_to_react` 使用严格 JSON 合约，复用 `pilot_tasks.request_contract`：保留原模板的说明与示例，仅把补全桩换成 JSON 指令并列出逐字选项；最多 3 次，失败后带校验反馈重试；仍失败则抛出 `PilotScheduleError`，不再用默认值替代模型失败。只有一个可选项（例如咖啡馆只有 `cafe`）时不调用模型。非 pilot 路径不变，仅修正 `run_gpt_prompt_event_triple` 把 fail-safe 三元组重复包装的错误。

**防回归。** `preflight` 现在用真实模板、真实 API 检查：Isabella 的咖啡馆班次必须解析为 `Hobbs Cafe`，并验证 arena、三元组与 `decide_to_talk` 合约；失败则不开始模拟。`tests/test_pilot_actions.py` 覆盖解析器、反馈重试、耗尽即停、以及用真实迷宫与空间记忆验证智能体会进入咖啡馆或共用房间，并断言 pilot 路径不会调用旧的 `safe_generate_response`。

**运行限制。** `run --stop-at-step N`（N 为 180 的倍数）在协议边界保存并暂停，属于操作性限制，不进入配置锁。源码摘要已变化，因此此后任何轨迹都必须使用新的输出目录；旧轨迹不能用新源码续跑。

**未处理的相关问题（范围外）：** survey 概率项约 14.6% 缺失，以及 degree 排名受名单字母顺序影响。

## Schedule revision：priority-fill-v1

聊天或等待插入日程后，模型只返回 `{"priority":[2,0,1]}`：全部 source 索引按保留优先级排列，各出现一次。代码按优先级保留原时长，边界活动截短，剩余活动省略；最终仍按原日程顺序执行。例如 `[30,30,30]` 分钟、预算 70、优先级 `[2,0,1]`，分配结果为原顺序的 `[30,10,30]`。活动文字、终结标记、来源、过去区间和窗口后的日程保持不变。

非法排序最多反馈重试三次，耗尽仍停止，不猜测缺失索引。预算错误是本地状态错误，直接停止。无剩余区间无需模型；只有一个 source 时记录 `only_feasible_choice` 并直接分配。此分配方式改变了时长决策机制：最多一个活动被部分缩短，而不再让模型分别压缩多个活动。

配置增加 `reaction_revision_version=priority-fill-v1`，日程状态仍为 `standalone-tasks-v3`。源码变化后必须使用新目录，禁止将 v4 轨迹迁移后继续充当原轨迹。

诊断文件：

- `schedule_diagnostics.jsonl` 保持每次模型尝试的摘要，并为重排增加 revision ID、被重排角色和版本。
- `schedule_revision_audit.jsonl` 保存 source、预算、优先级、分配向量和最终 suffix。它记录已验证的单次分配；故障 tick 中的记录不等于已经提交的模拟状态，应以完整恢复点为准。
- 设置 `NCN_REACTION_DEBUG_LOG=/absolute/path.jsonl` 才保存重排的完整 prompt、raw 和 error，默认关闭。下面的 smoke 入口自动将它指向本次输出目录。不要把本地调试日志批量提交；筛选实际失败样例后再形成测试夹具。原 v4 失败正文未捕获，不能将人工样例标成历史真实回复。

两阶段真实验收（付费 API；目录必须尚不存在）：

```bash
PYTHONHASHSEED=104729 OPENAI_MODEL=gpt-6-luna /usr/bin/caffeinate -i -s \
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -u \
  reverie/backend_server/run_schedule_smoke.py --output pilot_runs/smoke_mobility_v5
```

入口固定 seed 104729、repeats 2、defer surveys，先暂停在 step 3600。完整快照、角色日程、重排排序和分配、调试捕获全部通过后，才以同一源码续到 5040。任一运行或验收失败就停止，不自动恢复，不删除失败证据。`smoke_status.json` 记录子进程退出码，`acceptance_003600.json` / `acceptance_005040.json` 保存各阶段与 v4 同时点的比较。

可独立重查（只读取本地可信快照，不调用 API）：

```bash
python3 reverie/backend_server/validate_schedule_smoke.py \
  --output pilot_runs/smoke_mobility_v5 --baseline pilot_runs/smoke_mobility_v4 --step 5040
```

移动检查遍历所有完成 tick 的实际坐标；行动描述 `@` 后的地址另记为目的地。v4 在 14:00 的基线是实际到访 19/25、目的地 20/25、6 个公共地点、30 条边、16 个非零 betweenness 节点、最大连通分量 21。14:00 最低功能门槛为有实际公共地点到访、边数大于 3、存在非零 betweenness；这些指标不要求随机轨迹复现旧数值，也不替代正式实验推断。
