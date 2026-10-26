# Generative Agents：网络认知实验

当前方案使用 **25 个智能体、64 个模拟小时、每个测量快照 3 次独立问卷**。主链路是模拟引擎推进 → 保存行为与互动网络快照 → 独立测量网络认知 → 分析结果。

本仓库保留模拟引擎、地图、初始模板和离线测试。2026 年 8 月 1 日之前的旧实验记录已从当前工作区移除；清理范围见 [清理记录](REPOSITORY_CLEANUP_20261026.md)，旧数据仍可从 Git 历史恢复。

## 当前应使用的文档

- [当前实验协议](MIDNIGHT_PILOT.md)：64 小时时间表、checkpoint 选择、问卷、诊断与分析。
- [修复及验证交接](HANDOFF_SCHEDULE_REVISION.md)：实现、真实验收结果、历史中止和剩余问题。

当前行为轨迹 `smoke_mobility_v5` 已保存至 Day 1 14:00（step 5040）。它曾因任务拆分输出无效而中止一次，经用户授权恢复后完成 14:00 验收；完整 64 小时尚未完成，不能称作全程无中止。日程重排已改为模型决定保留优先级、代码分配分钟；任务拆分的偶发失败尚未修复。

## 继续现有轨迹

继续使用本机已经验证的 Python 3.11.5 环境、现有运行目录及本地配置；不要重新安装或更改锁定依赖。不要将凭据、快照或完整模型日志上传 GitHub。

从项目根目录校验现有进度（不调用付费 API）：

```bash
PYTHONHASHSEED=104729 OPENAI_MODEL=gpt-6-luna /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
  reverie/backend_server/run_midnight_pilot.py verify \
  --output pilot_runs/smoke_mobility_v5 --seed 104729 --repeats 2
```

这里的 `--repeats 2` 是旧轨迹冻结配置，续跑保持原值并使用 `--defer-surveys`。实际问卷通过独立入口执行三次，不必为更改测量次数重跑行为轨迹：

```bash
PYTHONHASHSEED=104729 OPENAI_MODEL=gpt-6-luna /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
  reverie/backend_server/measure_saved_pilot.py \
  --source-run pilot_runs/smoke_mobility_v5 \
  --output pilot_runs/smoke_mobility_v5_measurements_r3 --repeats 3
```

问卷命令会调用付费 API。模拟与问卷按顺序执行；同一时刻只运行一个任务。目前初始、10:00、14:00 三个快照已保存，但尚无问卷回答或已完成测量。

在本机继续行为轨迹至既定终点（付费 API）：

```bash
PYTHONHASHSEED=104729 OPENAI_MODEL=gpt-6-luna /usr/bin/caffeinate -i -s \
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -u \
  reverie/backend_server/run_midnight_pilot.py run \
  --output pilot_runs/smoke_mobility_v5 --seed 104729 --repeats 2 --defer-surveys
```

运行中冻结源码；如再次中止，保留日志并检查原因，不自动反复恢复。

## 目录与输出

| 路径 | 用途 |
|---|---|
| `reverie/backend_server/run_midnight_pilot.py` | 模拟、保存与恢复 |
| `reverie/backend_server/pilot_*.py` | 当前协议、日程、行为、问卷、状态与分析 |
| `reverie/backend_server/measure_saved_pilot.py` | 独立测量已有快照 |
| `reverie/backend_server/tests/` | 不调用付费 API 的回归测试 |
| `environment/frontend_server/static_dirs/` | 地图与人物初始化资源 |
| `environment/frontend_server/storage/base_the_ville_n25/` | 当前 25 人初始模板 |
| `pilot_runs/` | 本地私有轨迹、快照、验收与测量结果，不进入 Git |

当前测量输出位于独立目录中的 `REPORT.md`、`report.json` 和 `perception_scores.csv`。三次问卷用于估计测量噪声，不等于三条独立模拟轨迹。

旧版 N15 / N25 干预及分析脚本仍保留作为代码历史参考，不是当前执行入口。保留的其他 `base_*` 目录是初始化模板，不是待删除的实验结果。当前恢复依赖源码摘要，勿随意删除或改写引擎及 prompt 文件。

## 离线验证

```bash
PYTHONHASHSEED=104729 OPENAI_MODEL=gpt-6-luna /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
  -m pytest -q reverie/backend_server/tests
```

## Upstream Attribution

This project builds on the research paper "[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)." It contains the core simulation module for generative agents—computational agents that simulate believable human behaviors—and their game environment.

**Authors:** Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein

Please cite the upstream paper if you use the simulation infrastructure:

```
@inproceedings{Park2023GenerativeAgents,
  author    = {Park, Joon Sung and O'Brien, Joseph C. and Cai, Carrie J. and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S.},
  title     = {Generative Agents: Interactive Simulacra of Human Behavior},
  year      = {2023},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  booktitle = {In the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23)},
  keywords  = {Human-AI interaction, agents, generative AI, large language models},
  location  = {San Francisco, CA, USA},
  series    = {UIST '23}
}
```

Game asset credits: background art by [PixyMoon (@_PixyMoon)](https://twitter.com/_PixyMoon_), furniture/interior design by [LimeZu (@lime_px)](https://twitter.com/lime_px), character design by [ぴぽ (@pipohi)](https://twitter.com/pipohi).
