# -*- coding: utf-8 -*-
"""云端每日信号：全量重算状态机，产出网站唯一数据源 signal.json。

2026-11-23 升级：两个组合整体切换到 P6.2 的 6ETF 完整策略（p6etf_engine）。
  - old 原股债金升级版 = 518880 + 510500 + 510300 + 511010/511260/511360
  - new 海外升级版     = 518880 + 513500 + 513100 + 511010/511260/511360
信号层与分配层均来自 fof_sdk（五规则状态机 + ERC/下行半协方差/TIPP/三阶护盾/缓冲带）。

为什么必须全量重算：五规则是带记忆的状态机（w_curr 逐日递推），TIPP 回撤垫也
依赖组合级净值路径，不能只用最近 N 天算。8 只约 3.5 千行，数秒即可完成。

与旧版 build_signal.py 的差异：
  1. 权重来源由 pc.combo_returns（第一课简单 inv_vol）换成 P6ETF 分配器
  2. 组合从 3 风险 + 1 国债 扩为 3 风险 + 3 阶固收（共 6 只）
  3. 观察池从 14 只收敛为 8 只（两组合并集）
  4. signal.json schema 保持不变（combos / instruments / freshness），前端零改动
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

import p6etf_engine as E          # noqa: E402

DATA_DIR = os.path.join(ROOT, "data", "daily_hfq")
SITE_DIR = os.path.join(ROOT, "site")
STATUS_PATH = os.path.join(SITE_DIR, "update_status.json")
PARAMS_PATH = os.path.join(HERE, "strategy_params.json")
SHANGHAI = ZoneInfo("Asia/Shanghai")

DISCLAIMER = "本页面为量化研究与教学用途的策略信号展示，不构成任何投资建议、要约或承诺。历史回测不代表未来收益，据此操作风险自负。"

# 展示用名称映射（覆盖两组合全部标的）
NAME_MAP = {
    "518880.SH": "华安黄金ETF",
    "510500.SH": "中证500ETF",
    "510300.SH": "沪深300ETF",
    "513500.SH": "标普500ETF",
    "513100.SH": "纳指ETF",
    "511010.SH": "5年国债ETF",
    "511260.SH": "十年国债ETF",
    "511360.SH": "短融ETF",
}
CAT_MAP = {
    "518880.SH": "贵金属",
    "510500.SH": "A股宽基",
    "510300.SH": "A股宽基",
    "513500.SH": "海外宽基",
    "513100.SH": "海外宽基",
    "511010.SH": "利率债",
    "511260.SH": "利率债",
    "511360.SH": "现金管理",
}


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

    R = E.run_all(DATA_DIR)
    prices = R["prices"]
    prices_raw = R["prices_raw"]
    sig = R["sig"]
    lbl = R["lbl"]
    ind = R["indicators"]
    W = R["weights"]

    if len(prices.index) < 3:
        raise RuntimeError("行情样本不足 3 个交易日")

    last_dt = prices.index[-1]           # 最新收盘日 = 新信号日
    prev_dt = prices.index[-2]           # 上一交易日 = 本交易日已执行仓位所依据的信号日
    status = read_status()

    # T+1 切分：w_raw 即目标权重，本交易日已执行取上一日目标，下一交易日取最新目标
    w_next = {k: W[k].loc[last_dt] for k in ("new", "old")}
    w_today = {k: W[k].loc[prev_dt] for k in ("new", "old")}

    # 数据自检：权重和必须为 100%，关键指标必须已预热
    for key in ("new", "old"):
        for label, w in [(f"{key}明日", w_next[key]), (f"{key}今日", w_today[key])]:
            total = float(w.sum())
            if abs(total - 1.0) > 0.0001:
                raise RuntimeError(f"{label} 权重合计 {total:.6f} ≠ 1.0，停止产出")
    for sym in E.UNIVERSE:
        for key in ["vol20", "q80", "ma60", "ma200"]:
            if key in ind and sym in ind[key].columns and pd.isna(ind[key].loc[last_dt, sym]):
                # 预热期（如短融上市较晚）允许缺失，但风险标的必须预热
                if sym in E.COMBO_RISK["old"] + E.COMBO_RISK["new"]:
                    raise RuntimeError(f"{sym} 在 {last_dt:%Y-%m-%d} 的 {key} 尚未预热，停止产出")
        for key in ["ma20", "ma60", "ma200", "vol20", "q80"]:
            if key in ind and sym in ind[key].columns and pd.isna(ind[key].loc[last_dt, sym]):
                pass

    universe = E.UNIVERSE
    instruments = []
    for sym in universe:
        s = prices[sym]
        vol_v = float(ind["vol20"].loc[last_dt, sym]) if sym in ind["vol20"].columns else np.nan
        q80_v = float(ind["q80"].loc[last_dt, sym]) if sym in ind["q80"].columns else np.nan
        # 组合归属
        in_new = sym in w_next["new"].index
        in_old = sym in w_next["old"].index
        n_next = f(w_next["new"][sym] * 100.0) if in_new else 0.0
        n_today = f(w_today["new"][sym] * 100.0) if in_new else 0.0
        d = n_next - n_today
        o_next = f(w_next["old"][sym] * 100.0) if in_old else 0.0
        o_today = f(w_today["old"][sym] * 100.0) if in_old else 0.0
        o_d = o_next - o_today
        o_dir = "加仓" if o_d > 0.005 else ("减仓" if o_d < -0.005 else "不变")
        ma60_v = float(ind["ma60"].loc[last_dt, sym]) if sym in ind["ma60"].columns else np.nan
        ma200_v = float(ind["ma200"].loc[last_dt, sym]) if sym in ind["ma200"].columns else np.nan
        instruments.append({
            "code": sym,
            "name": NAME_MAP.get(sym, sym),
            "category": CAT_MAP.get(sym, "其他"),
            "in_new_combo": in_new,
            "in_old_combo": in_old,
            "close_raw": f(prices_raw.loc[last_dt, sym], 4) if sym in prices_raw.columns else f(s.loc[last_dt], 4),
            "close_hfq": f(s.loc[last_dt], 4),
            "adj_factor": f(prices_raw.loc[last_dt, sym] / s.loc[last_dt], 6) if sym in prices_raw.columns and s.loc[last_dt] != 0 else 0.0,
            "ret_1d_pct": pct_change_n(s, 1),
            "ret_5d_pct": pct_change_n(s, 5),
            "ret_20d_pct": pct_change_n(s, 20),
            "ret_60d_pct": pct_change_n(s, 60),
            "ma20": f(ind["ma20"].loc[last_dt, sym], 4) if sym in ind["ma20"].columns else 0.0,
            "ma60": f(ma60_v, 4),
            "ma200": f(ma200_v, 4),
            "above_ma60": bool(s.loc[last_dt] >= ma60_v) if not np.isnan(ma60_v) else False,
            "above_ma200": bool(s.loc[last_dt] >= ma200_v) if not np.isnan(ma200_v) else False,
            "vol20_pct": f(vol_v),
            "q80_pct": f(q80_v),
            "vol_above_q80": bool(vol_v > q80_v) if not (np.isnan(vol_v) or np.isnan(q80_v)) else False,
            "five_rule_weight_pct": f(sig.loc[last_dt, sym] * 100.0, 1),
            "rule": str(lbl.loc[last_dt, sym]),
            "new_current_pct": n_today,
            "new_next_pct": n_next,
            "new_change_pp": f(d),
            "new_direction": "加仓" if d > 0.005 else ("减仓" if d < -0.005 else "不变"),
            "old_current_pct": o_today,
            "old_next_pct": o_next,
            "old_change_pp": f(o_d),
            "old_direction": o_dir,
        })

    def combo_payload(key):
        risk = R["risk_codes"][key]
        today, nxt = w_today[key], w_next[key]
        return {
            "name": R["names"][key],
            "risk_codes": risk,
            "bond_code": E.DEFENSIVE[0],
            "defensive_codes": E.DEFENSIVE,
            "current_holdings_pct": {c: f(today[c] * 100.0) for c in today.index},
            "next_target_pct": {c: f(nxt[c] * 100.0) for c in nxt.index},
            "changes_pp": {c: f((nxt[c] - today[c]) * 100.0) for c in nxt.index},
            "next_risk_total_pct": f(nxt[risk].sum() * 100.0),
            "current_risk_total_pct": f(today[risk].sum() * 100.0),
        }

    next_hint = (last_dt + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    payload = {
        "schema_version": "1.1",
        "strategy_version": params["strategy_version"],
        "params": params,
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
            "new": combo_payload("new"),
            "old": combo_payload("old"),
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

    # 每日存档
    with open(os.path.join(SITE_DIR, "history", f"{last_dt:%Y-%m-%d}.json"), "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    rows = []
    for it in instruments:
        if not it["in_new_combo"] and not it["in_old_combo"]:
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
            "海外组合_本交易日权重(%)": it["new_current_pct"],
            "海外组合_下一交易日权重(%)": it["new_next_pct"],
            "海外组合_变动(百分点)": it["new_change_pp"],
            "原股债金_本交易日权重(%)": it["old_current_pct"],
            "原股债金_下一交易日权重(%)": it["old_next_pct"],
            "原股债金_变动(百分点)": it["old_change_pp"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(SITE_DIR, "signal.csv"), index=False, encoding="utf-8-sig")

    print(f"[OK] 数据末日 {last_dt:%Y-%m-%d} → 信号日 {last_dt:%Y-%m-%d} → 下一交易日(预计) {next_hint}")
    print(f"     本交易日依据信号日 {prev_dt:%Y-%m-%d}")
    for key in ("new", "old"):
        print(f"     {R['names'][key]}")
        print(f"       明日目标：{payload['combos'][key]['next_target_pct']}")
    print(f"     数据质量：{payload['freshness']['data_quality']} / 更新结果：{payload['freshness']['update_result']}")
    print("     产出：site/signal.json, site/signal.csv, site/history/<date>.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
