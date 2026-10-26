# Repository cleanup — 2026-10-26

用户确认删除 **2026 年 8 月 1 日之前**的旧实验记录，改按当前 25 人、64 模拟小时、每个测量快照三次问卷的方案执行。

本次移除 33 个旧实验目录或日志文件，共 360,250 个文件（其中 360,037 个曾被 Git 跟踪），文件内容合计 10,151,334,096 字节，约 10.15GB / 9.45GiB。该数值是文件逻辑大小，不等于精确释放的磁盘空间。

## 判定与边界

按实验目录、Git 历史和文件修改时间交叉核对；所有待删除文件的最后修改时间均早于截止日。`ncn_baseline_n25_full` 的少数派生图表在 2026-10-25 才提交，但属于旧实验，文件修改时间仍在 7 月之前，随该实验一起删除。文件内部的 2023 年模拟时间不用于判断实验生成日期。

删除包括旧 N15 校准与 hub/broker 干预、N25 baseline/control/hub/broker 与 identification 重复实验、旧 `_archive`、压缩回放、旧 LLM 请求日志及 preflight/identification 运行日志。

保留：

- 当前 `pilot_runs/` 全部内容，包括 v4/v5 证据、原始失败记录、14:00 恢复点及 10 月其他记录。
- 三个 `base_*` 初始模板、完整地图与人物资源、测试夹具。
- 模拟与分析源码、提示词、依赖配置；旧脚本保留但不作为当前运行入口。
- 凭据及本地配置、编辑器数据和 `.git` 历史。

README 已更新为当前方案入口；历史 10 月 20 日运行指南保留并标明用途。新增忽略规则防止压缩回放和请求日志再次混入代码仓库。

## 删除清单

| 路径 | 文件数 | MiB | 最近 Git 提交 |
|---|---:|---:|---|
| `environment/frontend_server/storage/4omini_n15_trunk_v1` | 4,914 | 181.54 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/_archive` | 228,418 | 582.95 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/bridge_probe_n15_baseline_t1800-1` | 3,707 | 89.32 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/bridge_probe_n15_broker_t2400-1` | 4,930 | 109.45 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/formal_n15_full_instruments-1` | 4,324 | 133.11 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/ncn_baseline_n25` | 3,871 | 161.33 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/ncn_baseline_n25_full` | 4,388 | 507.75 | 4d0ec88fc 2026-10-25 |
| `environment/frontend_server/storage/ncn_broker_n25` | 4,057 | 231.86 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/ncn_broker_n25_full` | 8,565 | 541.53 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/ncn_control_n25` | 4,056 | 232.34 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/ncn_control_n25_full` | 8,563 | 555.04 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/ncn_hub_n25` | 4,057 | 231.90 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/ncn_hub_n25_full` | 8,564 | 548.36 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/ncn_id_broker_r1` | 6,780 | 382.64 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/ncn_id_broker_r2` | 6,780 | 369.57 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/ncn_id_control_r1` | 6,779 | 364.19 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/ncn_id_control_r2` | 6,779 | 395.41 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/ncn_id_hub_r1` | 6,780 | 389.06 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/ncn_id_hub_r2` | 6,780 | 378.95 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/preflight_the_ville_n15-1` | 1,292 | 16.87 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/preflight_the_ville_n25-1` | 3,748 | 155.00 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/prepost_n15_calibration-1` | 3,726 | 111.96 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/prepost_n15_calibration-1_post_broker_t2400` | 4,917 | 103.49 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/prepost_n15_calibration-1_post_hub_t2400` | 4,920 | 103.59 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/storage/prepost_the_ville_n15_brokerprobe-1` | 8,526 | 67.86 | 801aae6fe 2026-07-01 |
| `environment/frontend_server/compressed_storage/July1_the_ville_isabella_maria_klaus-step-3-20` | 22 | 23.28 | 987d81f40 2026-04-20 |
| `reverie/backend_server/runs/run` | 1 | 52.24 | 801aae6fe 2026-07-01 |
| `reverie/backend_server/logs/extend_post_long.log` | 1 | 1115.58 | 未跟踪，本地旧日志 |
| `reverie/backend_server/logs/identification_w1.log` | 1 | 1364.43 | 未跟踪，本地旧日志 |
| `reverie/backend_server/logs/identification_w2.log` | 1 | 170.64 | 未跟踪，本地旧日志 |
| `preflight_n25_burnin.log` | 1 | 6.82 | 未跟踪，本地旧日志 |
| `preflight_n25_run.log` | 1 | 0.00 | 未跟踪，本地旧日志 |
| `preflight_n25_survey.log` | 1 | 3.01 | 未跟踪，本地旧日志 |

## 验证与历史恢复

清理前后逐文件摘要比对确认：v4/v5 整个运行目录、地图资源和全部三个初始模板未变。模拟源码摘要仍为 `29cfd946e0d44df4cddc0f3938bb74a3ac12875ec3e81f5d27c379c8b1cd2813`。本地详细清单保存在 `pilot_runs/repository_cleanup_20261026.json`。

清理后全套 **162 项离线测试通过**（Python 3.11、`PYTHONHASHSEED=104729`、`OPENAI_MODEL=gpt-6-luna`）。现有 14:00 快照复核结果为 step 5040、25 人、`snapshot_valid=true`、`manifest_matches=true`。保留的已追踪文件共 1,872 个，内容约 45.38MB，另有本轮新增代码与文档待提交；这不包括本地运行数据、编辑器数据和 Git 历史。

本次没有重写 Git 历史。用户随后已授权提交并推送清理及实现。曾跟踪的旧数据可从清理前提交 `4d0ec88fc` 恢复；未跟踪的旧日志没有 Git 恢复副本。删除后完整 clone 仍会包含历史数据；未来新 checkout 可使用浅克隆减少下载量。用户已决定继续使用本机，另行删除了旧电脑迁移说明、专用依赖清单和私有传输包；当前运行目录及三次问卷入口保持保留。
