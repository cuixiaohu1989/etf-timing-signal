# ETF 每日择时信号

每个交易日收盘后自动重算，给出**下一交易日应执行的目标仓位**。打开网页看一眼即可，无需本地开机、无需手动跑脚本。

> **研究教学用途，不构成投资建议，不承诺任何收益。**
> 本页面展示的是规则化模型的输出，不是买卖指令。据此操作风险自负。

---

## 这是一个公开仓库，请先确认

仓库设为 Public，以下内容**全部对互联网开放**，任何人可查看、克隆、转载、二次分发：

| 内容 | 路径 |
| --- | --- |
| 完整策略规则与状态机实现 | `scripts/core/` |
| 全部参数快照（策略版本号、阈值、上限） | `scripts/strategy_params.json` |
| 14 只 ETF 自 2010 年起的后复权日线 | `data/daily_hfq/` |
| 每日信号与逐日历史存档 | `site/signal.json`、`site/history/` |
| 取数、校验、部署的全部代码 | `scripts/`、`.github/workflows/` |

选择公开是把它当作课程/内容资产。若日后要保密，改为「私有计算仓 + 公开 Pages 展示仓 + PAT 跨仓推送」即可，展示层不用动。

行情数据版权归原数据提供方所有，此处仅用于研究教学。仓库未声明开源许可证，默认保留所有权利。

## 页面地址

```
https://<你的GitHub用户名>.github.io/<仓库名>/
```
默认建议仓库名 `etf-timing-signal`。页面为纯静态、无 CDN、无外部依赖，可离线打开。

## 每天什么时候跑

| 北京时间 | UTC cron | 作用 |
| --- | --- | --- |
| 17:40 | `40 9 * * 1-5` | 主跑，收盘后行情已稳定 |
| 18:50 | `50 10 * * 1-5` | 当天二次兜底，防 cron 排队被丢弃 |
| 08:15 | `15 0 * * 1-5` | 次日早间补跑，防前一晚两次都失败 |

三次**幂等**：已有数据直接跳过，重复运行不会产生重复行。全部避开 `:00` 整点（GitHub 官方说明整点负载最高，排队任务可能被延迟或丢弃）。非交易日不写数据，作业正常成功退出。

链路是：`新浪/腾讯取数 → 双源核验 → 原子写回 14 只 CSV → 全量重算状态机 → signal.json → commit 回仓库 → 部署 Pages`。规则带记忆，所以每次都从 2010 年全量递推，而不是只看最近一天。

---

## 首次上线（约 10 分钟）

### 1. 新建空公开仓库

GitHub → New repository：**Public**，名字建议 `etf-timing-signal`。
**不要勾选** Add a README / .gitignore / License，否则首次 push 会冲突。

### 2. 推送本目录

在项目根目录（本 `deploy/` 的上一级）用 Git Bash 执行：

```bash
GITHUB_USER=你的用户名 GIT_EMAIL=你的GitHub绑定邮箱 bash push_to_github.sh
```

或手动：

```bash
cd deploy
git init -b main
git config user.name "你的用户名"
git config user.email "你的GitHub绑定邮箱"
git add -A && git commit -m "init: ETF 每日择时信号云端服务"
git remote add origin https://github.com/你的用户名/etf-timing-signal.git
git push -u origin main
```

推送时若弹出浏览器，登录 GitHub 授权即可（本机 Git Credential Manager 会记住）。

### 3. 打开两项仓库设置

- **Settings → Pages → Build and deployment → Source：选 `GitHub Actions`**
- **Settings → Actions → General → Workflow permissions：选 `Read and write`**

### 4. 【必跑】连通性预检

Actions → **「00 连通性预检（上线前必跑一次）」** → Run workflow → 看 Job Summary。

这一步验证 GitHub 境外出口能否访问新浪 / 腾讯 / 东财行情接口，是全云端方案唯一的生死线。国内本机测通**不能**代表境外能通。

| 判定 | 处理 |
| --- | --- |
| `PASS` | 双源可用，继续第 5 步 |
| `PARTIAL` | 只有单源可用，把日常任务的 strict 设为 `0`，页面会显示黄色「单源降级」 |
| `BLOCKED` | **不要启用日常任务**。需要中转/代理或改国内托管 CI，否则每天只会失败 |

### 5. 手动跑一次正式任务

Actions → **「10 每日 ETF 择时信号」** → Run workflow（strict 保持 `1`）→ 等作业变绿 → 打开 Pages 链接。

页面底部「手动重新运行」按钮的链接由 CI 自动注入的 `GITHUB_REPOSITORY` 生成；本地打开 `index.html` 时该按钮会隐藏，属正常。

---

## 长期免维护：60 天自动禁用

GitHub 官方规定：**公开仓库 60 天无仓库活动时，定时工作流会被自动禁用**。官方文档未定义「仓库活动」是否包含 `GITHUB_TOKEN` 发起的 push，所以不能只靠每日自动 commit 打包票。

本仓库有两层防线：

1. **可选保活（推荐）**：创建一个 fine-grained PAT，权限只给本仓库的 `Contents: Read and write`，存为仓库 Secret `KEEPALIVE_TOKEN`。workflow 的 checkout 会自动优先使用它，每日 commit 即以你的真实账号产生，确定算活动。不配置也能跑，功能完全一致。
2. **页面哨兵**：按「上次运行之后漏掉了几个预定时点」判断停摆（与 cron 时点表一致，周末与长假不计入），连续漏 4 个即整页标红，并提示去 Actions 手动补跑。取数失败、双源降级、数据未跟上也都各有醒目状态。

> 别人 Fork 本仓库时，GitHub 默认会禁用 Fork 仓库里的定时工作流，不会跟着一起打行情接口。

## 故障排查

| 页面状态 | 含义 | 处理 |
| --- | --- | --- |
| 红色「定时任务停摆」 | 连续错过 4 个预定时点，多为工作流被禁用或 cron 被丢弃 | Actions 里看 workflow 是否被 Disable，手动 Run 一次 |
| 红色「数据异常」 | 行情取数失败，页面保留上一份信号 | 看 `site/update_status.json` 与作业日志，手动补跑 |
| 红色「数据未跟上」 | 日历里已有新交易日，但信号还停在旧日期 | 手动补跑；多为接口临时异常 |
| 黄色「单源降级」 | 腾讯不可用或手动设了 strict=0 | 人工留意异常价格 |
| 黄色「需要留意」 | 今天已过 18:00 但数据还是前一交易日 | 交易日就手动补跑；法定节假日属正常 |
| 绿色「数据正常」 | 收盘信号已生成，明日仓位已锁定 | 无需操作，隔夜消息不改变该目标 |

## 目录结构

```
.github/workflows/00-probe-network.yml   连通性预检（手动）
.github/workflows/10-daily-signal.yml   每日任务（三时点 cron + 手动）
scripts/fetch_and_update.py             14 只增量取数、真实交易日历、双源核验、原子写回
scripts/build_signal.py                 全量重算状态机，产 JSON / CSV / 每日存档
scripts/core/                           两份研究脚本副本，仅把 matplotlib 改为可选导入
scripts/strategy_params.json            参数快照与策略版本号
data/daily_hfq/                         14 只后复权日线
site/index.html                         展示页（新鲜度哨兵、双组合切换、免责声明）
site/signal.json / signal.csv            每日覆盖式产物
site/update_status.json                 取数状态，供页面判断新鲜度
site/history/<date>.json                每日信号存档，用于事后复盘与参数对比
```

## 本地复现

```bash
pip install -r requirements.txt
python scripts/probe_network.py       # 本机连通性对照
python scripts/fetch_and_update.py    # 增量取数
python scripts/build_signal.py        # 重算信号
```

CI 不生成 PNG 图表（ubuntu-latest 无中文字体），回测与绘图仍留在本地研究脚本中。

云端每天会把更新后的 CSV commit 回**本仓库**；本地研究目录 `04-数据获得/` 不会自动同步，本地重跑研究脚本前先 `git pull`。

---

**数据来源**：新浪财经、腾讯证券、东方财富公开行情接口。
**用途声明**：研究教学，不构成投资建议；策略参数可能调整，每次调整请同步更新 `strategy_params.json` 的版本号，否则前后信号无法比较。
