# main_10202026：夜间运行与模型交接指南

本指南用于交给能够访问本机终端和项目文件的下一位执行模型。用户希望由模型启动、看护、暂停及恢复 simulation，避免手动操作出错。本次准备指南时尚未启动 simulation，也未启动 caffeinate；下一模型收到用户的启动指令后执行以下流程。

## 已确定的任务

- 运行名称固定为 **`main_10202026`**，输出目录固定为：
  `/Users/mac/Documents/PythonNote/SocResearch/generative_agents-main/pilot_runs/main_10202026`
- 项目根目录是 `/Users/mac/Documents/PythonNote/SocResearch/generative_agents-main`。不要在 ChatGPT 的 staging 副本中启动。
- 运行一条全新、25-agent、未干预轨迹，模拟区间为 **Day 1 00:00 → Day 3 16:00**，终点 step 23040。每步 10 个模拟秒。
- 模拟使用 `gpt-6-luna`，embedding 使用 `text-embedding-3-small`；seed 104729，每个测量状态重复 survey 三次。负责看护的模型可以更便宜，但不得因此更换 simulation 的模型。
- 保留原方案的所有测量与角色选择规则，见 [MIDNIGHT_PILOT.md](MIDNIGHT_PILOT.md)。
- 现实中的运行时间只决定何时暂停，不改变模拟时钟，不重新初始化 agent、不跳过夜间，也不重置累计互动网络。
- 本次优先保证用户出门前有可验证、可恢复的完整存档。无法事先保证八个现实小时能推进多少模拟小时；报告实际到达位置即可。

原模板中的模拟日期仍从 2023-02-13 开始，Day 1/2/3 是相对标签；现实截止日期是 **2026-10-20**。不要为了匹配目录名称修改模拟日期。

## 本次现实时间安排

以下均为 **2026-10-20，America/Los_Angeles，UTC−07:00**：

| 时间 | 必须执行的行为 |
|---|---|
| 用户指示启动后 | 启动监督器和 caffeinate；正常做初始 survey，再推进 simulation |
| 09:00 | 已在 survey 中就结束其独立子进程，并保存、暂停整个运行；此后遇到待做 survey 也直接保存、暂停 |
| 09:00–09:20 | 若仍在普通 simulation 阶段，可继续推进；不得新开 survey |
| 09:20 | 请求正常暂停；当前完整 tick 结束后保存并退出 |
| 09:40 | 若进程仍卡住，监督器终止**它自己启动的进程组**，验证最近一次完整存档；如有回退必须报告 |
| 09:45 前 | 看护模型确认退出、验证存档，并向用户报告进度及未完成 survey |
| 09:50 | 用户出门；本次运行应已结束，防休眠已释放 |

`09:20` 和 `09:40` 是为了给保存与验证留出余量而设定的操作时间。不会把现实中的“09:00 survey 截止”误认为 agent 世界里的 09:00。

时间参数必须带日期和时区。监督器会拒绝已经过期或顺序不正确的日期。**如果执行时已经超过本次窗口，就不要自动顺延到明天或取消时间限制；让用户指定下一段可运行时间。**

## 启动：执行模型照做

1. 读取本指南及 `MIDNIGHT_PILOT.md`，确认系统当前洛杉矶时间仍在本次窗口内。
2. 只检查凭据是否已配置，不打印 `.env` 或 API key。使用下方 Python 3.11；不要换成缺少依赖的旧 conda 环境。
3. 查看输出目录是否已有 `session.json`、`manifest.json` 和 `latest.pkl.gz`。如果已经在运行，接管看护，不启动第二份。如果已有存档，按恢复部分处理，绝不删目录重来。
4. 运行期间接通电源、保持机盖打开。监督器调用 `/usr/bin/caffeinate -i -s -w <supervisor_pid>`；它阻止闲置睡眠，屏幕可以熄灭。监督器退出后解除防休眠。
5. 执行下方命令。首次 `run` 会自动完成短 API 预检；不要复制另一条轨迹的预检文件。本次补入了暂停控制，旧目录的源码摘要不能替代当前目录的预检。

```bash
cd /Users/mac/Documents/PythonNote/SocResearch/generative_agents-main
mkdir -p pilot_runs/main_10202026
nohup /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -u \
  reverie/backend_server/pilot_session.py \
  --output /Users/mac/Documents/PythonNote/SocResearch/generative_agents-main/pilot_runs/main_10202026 \
  --survey-cutoff 2026-10-20T09:00:00-07:00 \
  --stop-at 2026-10-20T09:20:00-07:00 \
  --hard-stop-at 2026-10-20T09:40:00-07:00 \
  --seed 104729 --repeats 3 \
  >> pilot_runs/main_10202026/session_launcher.log 2>&1 < /dev/null &
printf '%s\n' "$!" > pilot_runs/main_10202026/session.pid
```

启动后要看到 `session.json` 中有 `status: running`、runner PID 和 caffeinate PID，并确认对应进程仍存在。shell 返回一个 PID 不能单独证明 simulation 已正常开始。

监督器和 runner 均有文件锁。遇到锁冲突先检查原进程，不能删除锁文件来强行重复启动。

本次命令没有 `--defer-surveys`：凌晨仍正常做 survey。到了 09:00，程序会停止当前 survey 或在下一次 survey 前暂停整个运行。**不要为了赶进度擅自跳过初始测量、减少重复次数，或改成全天延后 survey。**

## 看护与停止

监督器在本机独立执行截止时间，不依赖模型恰好在 09:00 发出停止命令。看护模型仍负责发现故障、核实保存结果并告知用户。

- 启动后及时检查一次；随后每约 10–15 分钟检查进程、文件更新时间、最新 API 记录及 survey 完成人数。临近 09:00、09:20、09:40 时重点检查。
- 工具单次等待保持短且可中断。若需要跨聊天轮次继续看护，使用当前应用提供的 thread heartbeat/automation 工具安排本线程回访；不要仅在回复里承诺稍后回来。任务暂停并报告后取消该回访，避免第二天误续跑。
- 只根据 `usage.jsonl` 中的实际 usage 和报告估算费用，不能拿现实运行时长直接换算 token，也不承诺八小时完成 64 个模拟小时。
- API/model 错误、源码配置不符、存档损坏或空间不足时，保留现场并报告。不要换模型、修改 seed、删状态或在未诊断前连续重试。
- 普通暂停保留全部已完成模拟步。完整 tick 涉及 25 个 agent，可能持续较长时间；程序不把只执行了一半的 tick 保存成可续跑状态。
- 除原有每 30 个模拟分钟的恢复点外，现在还会在距离上次保存超过 60 个现实秒后的**第一个完整 tick**保存恢复点。它不是每 60 秒无条件保存，因此不能承诺突发中断最多只丢 60 秒工作。

随时提前要求正常停止：

```bash
touch /Users/mac/Documents/PythonNote/SocResearch/generative_agents-main/pilot_runs/main_10202026/STOP_REQUESTED
```

然后等待 `session.json` 记录结束结果。正在 survey 时只终止独立 survey 子进程，原始 agent 快照已保存；已完成 respondent 的答卷保留，未完成 respondent 下次重做。正在 simulation 时，在完整 tick 后保存并退出。

不要使用 `killall python`、`pkill -f python` 或系统范围的 caffeinate 清理命令。09:40 的强制兜底仅针对监督器创建的独立进程组；它不代表最后一个未完成 tick 已保存。

关键文件：

| 文件 | 含义 |
|---|---|
| `session.json` | 本次监督器的截止时间、PID、退出结果及存档验证结果 |
| `manifest.json` | 最近完整存档的 step、模拟时间、状态、角色 checkpoint 和快照索引 |
| `latest.pkl.gz` | 恢复这条轨迹的权威完整状态 |
| `live_progress.json` | 最近完成的 tick；可能领先于持久存档，不能单独作为“已经保存”的依据 |
| `run.log` / `session_launcher.log` | runner / 监督器日志 |
| `surveys/step_*/repeat_*/complete.json` | 某次 survey 重复已完成的标记；仅有答卷文件不代表整次完成 |
| `usage.jsonl` | 实际请求用量和阶段；可能包含因中断而需重做的请求 |

## 保存验收与向用户报告

监督器停止 runner 后自动运行一次 `verify`，完整读取本项目产生的可信快照并验证状态摘要。也可在确认 runner 已退出后手动检查：

```bash
cd /Users/mac/Documents/PythonNote/SocResearch/generative_agents-main
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
  reverie/backend_server/run_midnight_pilot.py verify \
  --output /Users/mac/Documents/PythonNote/SocResearch/generative_agents-main/pilot_runs/main_10202026 \
  --seed 104729 --repeats 3
```

`verify` 不调用 LLM。只有 `snapshot_valid: true` 才确认完整存档可读取；正常结果还应有 `manifest_matches: true`、`agents: 25`。

- `saved_and_verified`：进程正常退出且存档验证成功；查看内部状态是 `paused` 还是已全部 `complete`。
- `recovery_verified_after_interruption`：被兜底停止或出现错误，但最近完整存档验证成功。报告实际可恢复 step，明确未保存区间可能要重做；不要称为正常暂停。
- `recovery_verified_manifest_stale`：完整快照有效，但 manifest 在中断时没同步完成。以 `verified_progress` / `latest.pkl.gz` 为准；报告这个差异，不手工篡改快照。原 runner 续跑会重新写 manifest。
- `verification_failed` / `supervisor_error`：不能宣称已经安全保存。保留日志与文件，诊断后再继续。

随后用同一输出目录生成报告，命令不调用 LLM：

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
  reverie/backend_server/run_midnight_pilot.py report \
  --output /Users/mac/Documents/PythonNote/SocResearch/generative_agents-main/pilot_runs/main_10202026
```

向用户发送这样的交接结果，填写真实数据：

> `main_10202026` 已于洛杉矶时间 … 停止。可恢复进度：Day … HH:MM:SS，step … / 23040；已验证完整存档 … 。  
> 停止原因：survey 截止 / 正常时间截止 / 手动请求 / 故障或兜底中断。  
> Survey：已完成 …；待继续 …（第几次重复、已完成多少 respondent）。  
> 角色 checkpoint：尚未进入窗口 / 搜索中 / 已锁定 … / 窗口内不合格。  
> 已记录 API 用量与估算费用：…；caffeinate 已退出。  
> 下次从同一目录恢复，先完成 …，然后继续 simulation。

如果实际使用了强制兜底、manifest 不一致、无法计费的请求或存档验证失败，必须在交接结果中说明。

## 下次恢复同一条 simulation

用户有时间并授权继续后：

1. 确认原 runner、监督器和 survey 子进程都已结束；检查 `session.json` 及完整存档。
2. 如有 `STOP_REQUESTED`，确认它属于已经结束的上一段运行，再只删除这个停止请求文件。保留所有快照、答卷、usage 和日志。
3. 使用相同项目、Python、代码、依赖、模型、seed、survey repeats 和**同一输出目录**。不要创建另一个 `main_10202026_*` 目录当作续跑，不要从旧实验启动。
4. 有新的离开时间时，按启动模板使用监督器，并将三项现实截止时间全部替换为用户批准的新日期和时区。旧的 2026-10-20 参数不得继续使用。
5. 用户在场、无需限定本次结束时间时，可用下面命令继续；它会恢复原状态，遇到未完成 survey 时从同一快照补齐。初始状态或最终 survey 尚未完成时也使用同一入口。

```bash
cd /Users/mac/Documents/PythonNote/SocResearch/generative_agents-main
/usr/bin/caffeinate -i -s \
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -u \
  reverie/backend_server/run_midnight_pilot.py run \
  --output /Users/mac/Documents/PythonNote/SocResearch/generative_agents-main/pilot_runs/main_10202026 \
  --seed 104729 --repeats 3 \
  --stop-file /Users/mac/Documents/PythonNote/SocResearch/generative_agents-main/pilot_runs/main_10202026/STOP_REQUESTED
```

这条无截止时间的恢复命令只用于用户已安排继续运行的时段，不能替代本次夜间的限时命令。暂停原因和时间限制不属于实验配置，下一段运行可换新截止时间；源码、模型等实验配置保持锁定。

## 两种 checkpoint 不要混淆

**运行恢复点**：用于今晚暂停、下一次续跑，可以在任何完整 tick 保存到 `latest.pkl.gz`。

**未来干预用的角色 checkpoint**：必须按既定科学规则，在**模拟 Day 1 20:00–Day 2 08:00**每 30 分钟检查一次累计互动网络，最高 degree 和最高正 betweenness 的并列集合不相交，连续两次选中同一 hub/broker 才在第二次锁定。

“以后再找合适的 checkpoint”在操作上解释为：**恢复后继续完成原定的自动检查，并随后查看结果**。如果今晚已经进入这个窗口，runner 当时就会检查和保存；不能等整条轨迹结束后根据 survey 结果回头挑选，也不能暂停后重新计算优先顺序或忘记前一次稳定性检查。

最早 Day 1 20:30 锁定；到 Day 2 08:00 仍不合格则保留无合格结果。被锁定的完整快照及其 +2、+4、+24 小时状态会自动保存。本次不启动 control/hub/broker 干预分支。

## 给下一模型的简短启动提示词

> 请先阅读项目中的 `MAIN_10202026_RUN_GUIDE.md` 和 `MIDNIGHT_PILOT.md`。按照指南启动并看护唯一轨迹 `main_10202026`，使用其中的限时监督器和 caffeinate。以洛杉矶现实时间 2026-10-20 09:00 作为 survey 截止，09:20 正常暂停，09:40 兜底停止，09:45 前验证存档并报告。若日期已经过期，先让我指定下一段运行窗口。运行多久就保存多少，不改变实验设计，不重复创建轨迹。普通操作已获授权，请完成启动、监控、保存验证和交接，不只提供命令。遇到错误保留现场并说明实际可恢复进度。


## 本次运行的 step 0 排程修复记录

2026-10-20 02:15（洛杉矶时间）发现旧 completion 风格排程与 Luna 输出不兼容：每小时请求可能返回从午夜开始的整天日程，旧解析器取第一行后把白天也当成睡觉；每日计划还可能被接受为空列表。已停止未完成的首个 tick，保留全部原始数据。此前接口预检覆盖了基本格式，未覆盖这一真实排程路径；现已增加回归与真实角色验证。

02:23 已完成一次仅限于本次未推进过的 step 0 的修复迁移：旧源码、配置、存档及 session 已完整归档到 `pilot_runs/main_10202026/run_config_migrations/schedule_contract_v2/`；`migration.json` 记录前后源码摘要和状态摘要。38 项离线测试、9 次独立真实排程请求及修复后接口预检均通过。初始 agent 状态和 RNG 摘要完全一致，原始初始快照、75 份答卷及完成/隔离标记未改动。只丢弃了未提交的错误首步；108 次相关 API 请求及费用仍保留在账本。

这是有归档和状态验证的 step 0 配置版本迁移，作为本次“源码变更使用新配置版本”的明确记录；没有创建第二条实验轨迹、导入旧经历或更改模型、seed、测量与 checkpoint 规则。一般的源码锁定规则继续适用，已有非零 step 时不能援引这个处理方式绕过配置校验。后续看护仍从同一个 `main_10202026` 目录恢复，核对当前 source digest；不得回滚到旧排程代码，也不得重新运行已完成的初始 survey。
