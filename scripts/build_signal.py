# -*- coding: utf-8 -*-
"""云端每日信号：全量重算状态机，产出网站唯一数据源 signal.json。

为什么必须全量重算：五规则是带记忆的状态机（w_curr 逐日递推），
不能只用最近 N 天算；14 只约 4 千行，pandas 数秒即可完成。

与本地 05-实验代码/analyze_new_combo_position.py 的差异：
  1. 去掉 resolve_execution_context 的 BDay(1) 校验——长假后会误判并整批停止，
     改为直接使用 prices.index[-2] / [-1]，即真实交易日历
  2. 去掉全部「11-06 / 11-09」硬编码字段名与文案
  3. 产出固定文件名（覆盖式）+ 每日 JSON 存档，不再每天堆 CSV 与 PNG
  4. 附策略版本号、参数快照、数据质量与新鲜度信息，供页面做陈旧标红
"""
import os
import sys
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "core"))

import run_etf_timing_backtest as bt          # noqa: E402
import run_portfolio_comparison as pc          # noqa: E402

DATA_DIR = os.path.join(ROOT, "data", "daily_hfq")
SITE_DIR = os.path.join(ROOT, "site")
STATUS_PATH = os.path.join(SITE_DIR, "update_status.json")
PARAMS_PATH = os.path.join(HERE, "strategy_params.json")
SHANGHAI = ZoneInfo("Asia/Shanghai")

DISCLAIMER = "本页面为量化研究与教学用途的策略信号展示，不构成任何投资建议、要约或承诺。历史回测不代表未来收益，据此操作风险自负。"


def pct_change_n(series, n):
    if len(series) <= n:
        return np.nan
    return round(float((series.iloc[-1] / series.iloc[-1 - n] - 1.0) * 100.0), 2)


def f(x, nd=2):
    return round(float(x), nd)


def read_status():
    if os.path.exists(STATUS_PATH):
        try:
            with open(STATUS_PATH, encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            pass
    return {"result": "unknown", "last_success_at": None, "last_attempt_at": None,
            "data_quality": "unknown", "errors": [], "recent_trade_days": []}


def main():
    with open(PARAMS_PATH, encoding="utf-8") as fp:
        params = json.load(fp)

    prices, raw_prices, adj_factors = pc.load_prices(DATA_DIR)
    if len(prices.index) < 3:
        raise RuntimeError("行情样本不足 3 个交易日")

    last_dt = prices.index[-1]           # 最新收盘日 = 新信号日
    prev_dt = prices.index[-2]           # 上一交易日 = 本交易日已执行仓位所依据的信号日
    status = read_status()

    ind = bt.compute_indicators(prices)
    rets = ind["rets"].fillna(0.0)
    all_syms = [s for s, _, _, _ in pc.ETF_META]
    me_dates = bt.infer_month_end_dates(prices.index, last_dt)
    sig_w, sig_r = bt.generate_five_rule_signals(prices, ind, all_syms, month_end_dates=me_dates)

    new = pc.combo_returns(prices, ind, rets, sig_w, pc.NEW_RISK, pc.NEW_BASE_CAP)
    old = pc.combo_returns(prices, ind, rets, sig_w, pc.OLD_RISK, pc.OLD_BASE_CAP)

    w_today = new["w_raw"].loc[prev_dt]      # 本交易日已执行
    w_next = new["w_raw"].loc[last_dt]       # 下一交易日应执行
    w_old_next = old["w_raw"].loc[last_dt]
    w_old_today = old["w_raw"].loc[prev_dt]

    # 数据自检：权重和必须为 100%，关键指标必须已预热
    for label, w in [("新组合明日", w_next), ("新组合今日", w_today),
                     ("原组合明日", w_old_next)]:
        total = float(w.sum())
        if abs(total - 1.0) > 0.0001:
            raise RuntimeError(f"{label} 权重合计 {total:.6f} ≠ 1.0，停止产出")
    needed = pc.NEW_RISK + pc.OLD_RISK + [pc.BOND_CODE]
    for sym in sorted(set(needed)):
        for key in ["ma20", "ma60", "ma200", "vol20", "q80", "vol60"]:
            if pd.isna(ind[key].loc[last_dt, sym]):
                raise RuntimeError(f"{sym} 在 {last_dt:%Y-%m-%d} 的 {key} 尚未预热，停止产出")

    name_map = {s: n for s, _, n, _ in pc.ETF_META}
    cat_map = {s: c for s, _, _, c in pc.ETF_META}

    instruments = []
    for sym in all_syms:
        s = prices[sym]
        vol_v = float(ind["vol20"].loc[last_dt, sym])
        q80_v = float(ind["q80"].loc[last_dt, sym])
        is_new = sym in w_next.index
        n_next = f(w_next[sym] * 100.0) if is_new else 0.0
        n_today = f(w_today[sym] * 100.0) if is_new else 0.0
        d = n_next - n_today
        is_old = sym in w_old_next.index
        o_next = f(w_old_next[sym] * 100.0) if is_old else 0.0
        o_today = f(w_old_today[sym] * 100.0) if is_old else 0.0
        o_d = o_next - o_today
        o_dir = "加仓" if o_d > 0.005 else ("减仓" if o_d < -0.005 else "不变")
        instruments.append({
            "code": sym,
            "name": name_map[sym],
            "category": cat_map[sym],
            "in_new_combo": is_new or sym == pc.BOND_CODE,
            "in_old_combo": sym in w_old_next.index,
            "close_raw": f(raw_prices.loc[last_dt, sym], 4),
            "close_hfq": f(s.loc[last_dt], 4),
            "adj_factor": f(adj_factors.loc[last_dt, sym], 6),
            "ret_1d_pct": pct_change_n(s, 1),
            "ret_5d_pct": pct_change_n(s, 5),
            "ret_20d_pct": pct_change_n(s, 20),
            "ret_60d_pct": pct_change_n(s, 60),
            "ma20": f(ind["ma20"].loc[last_dt, sym], 4),
            "ma60": f(ind["ma60"].loc[last_dt, sym], 4),
            "ma200": f(ind["ma200"].loc[last_dt, sym], 4),
            "above_ma60": bool(s.loc[last_dt] >= ind["ma60"].loc[last_dt, sym]),
            "above_ma200": bool(s.loc[last_dt] >= ind["ma200"].loc[last_dt, sym]),
            "vol20_pct": f(vol_v),
            "q80_pct": f(q80_v),
            "vol_above_q80": bool(vol_v > q80_v),
            "five_rule_weight_pct": f(sig_w.loc[last_dt, sym] * 100.0, 1),
            "rule": str(sig_r.loc[last_dt, sym]),
            "new_current_pct": n_today,
            "new_next_pct": n_next,
            "new_change_pp": f(d),
            "new_direction": "加仓" if d > 0.005 else ("减仓" if d < -0.005 else "不变"),
            "old_current_pct": o_today,
            "old_next_pct": o_next,
            "old_change_pp": f(o_d),
            "old_direction": o_dir,
        })

    def combo_payload(name, risk_codes, today, nxt, today_old=None):
        return {
            "name": name,
            "risk_codes": risk_codes,
            "bond_code": pc.BOND_CODE,
            "current_holdings_pct": {c: f(today[c] * 100.0) for c in today.index},
            "next_target_pct": {c: f(nxt[c] * 100.0) for c in nxt.index},
            "changes_pp": {c: f((nxt[c] - today[c]) * 100.0) for c in nxt.index},
            "next_risk_total_pct": f(nxt[risk_codes].sum() * 100.0),
            "current_risk_total_pct": f(today[risk_codes].sum() * 100.0),
        }

    next_hint = (last_dt + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    payload = {
        "schema_version": "1.0",
        "strategy_version": params["strategy_version"],
        "params": params,
        # GitHub Actions 自动注入；本地运行为 None，页面会隐藏手动刷新入口
        "repo": os.environ.get("GITHUB_REPOSITORY"),
        "generated_at": datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S"),
        "data_end_date": last_dt.strftime("%Y-%m-%d"),
        "signal_date": last_dt.strftime("%Y-%m-%d"),
        "current_execution_date": last_dt.strftime("%Y-%m-%d"),
        "current_signal_date": prev_dt.strftime("%Y-%m-%d"),
        "next_execution_hint": next_hint,
        "next_execution_note": "按工作日推算的预计值，遇法定节假日自动顺延，以交易所公告为准",
        "execution_rule": "T 日收盘信号锁定 T+1 应执行仓位；隔夜消息不改变该目标，只影响执行日净值",
        "combos": {
            "new": combo_payload(pc.NEW_NAME, pc.NEW_RISK, w_today, w_next),
            "old": combo_payload(pc.OLD_NAME, pc.OLD_RISK, w_old_today, w_old_next),
        },
        "instruments": instruments,
        "freshness": {
            "update_result": status.get("result", "unknown"),
            "last_success_at": status.get("last_success_at"),
            "last_attempt_at": status.get("last_attempt_at"),
            "data_quality": status.get("data_quality", "unknown"),
            "calendar_source": status.get("calendar_source"),
            "recent_trade_days": status.get("recent_trade_days", []),
            "updated_dates": status.get("updated_dates", []),
            "errors": status.get("errors", []),
            # 2026-11-09 事故后新增：便于从 signal.json 直接看出「盘中等待 / 休市 / 已收盘」
            "market_state": status.get("market_state"),
            "cutoff_ok": status.get("cutoff_ok"),
            "expected_last_trade_day": status.get("expected_last_trade_day"),
        },
        "disclaimer": DISCLAIMER,
    }

    os.makedirs(SITE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SITE_DIR, "history"), exist_ok=True)
    with open(os.path.join(SITE_DIR, "signal.json"), "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    # 每日存档：单份约 8 KB，一年约 2 MB，用于参数调整前后的信号可比与复盘
    with open(os.path.join(SITE_DIR, "history", f"{last_dt:%Y-%m-%d}.json"), "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    rows = []
    for it in instruments:
        if not it["in_new_combo"]:
            continue
        rows.append({
            "数据截止日期": payload["data_end_date"],
            "信号日期": payload["signal_date"],
            "本交易日": payload["current_execution_date"],
            "下一交易日(预计)": next_hint,
            "标的代码": it["code"], "标的名称": it["name"], "资产类别": it["category"],
            "不复权收盘": it["close_raw"], "后复权收盘": it["close_hfq"],
            "MA20": it["ma20"], "MA60": it["ma60"], "MA200": it["ma200"],
            "20日年化波动率(%)": it["vol20_pct"], "500日80%分位阈值(%)": it["q80_pct"],
            "五规则信号仓位(%)": it["five_rule_weight_pct"], "触发规则": it["rule"],
            "本交易日实际权重(%)": it["new_current_pct"],
            "下一交易日应执行权重(%)": it["new_next_pct"],
            "权重变动(百分点)": it["new_change_pp"], "调仓方向": it["new_direction"],
            "原组合下一交易日权重(%)": it["old_next_pct"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(SITE_DIR, "signal.csv"), index=False, encoding="utf-8-sig")

    print(f"[OK] 数据末日 {last_dt:%Y-%m-%d} → 信号日 {last_dt:%Y-%m-%d} → 下一交易日(预计) {next_hint}")
    print(f"     本交易日依据信号日 {prev_dt:%Y-%m-%d}")
    print(f"     新组合明日目标：{payload['combos']['new']['next_target_pct']}")
    print(f"     原组合明日目标：{payload['combos']['old']['next_target_pct']}")
    print(f"     数据质量：{payload['freshness']['data_quality']} / 更新结果：{payload['freshness']['update_result']}")
    print("     产出：site/signal.json, site/signal.csv, site/history/<date>.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
