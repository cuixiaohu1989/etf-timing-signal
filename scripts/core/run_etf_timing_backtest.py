import os
import json
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
# [云端副本] 仅将 matplotlib 改为可选导入，其余逻辑与本地研究脚本逐字一致。
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
except ImportError:
    plt = None

ETF_META = [
    ("510050.SH", "510050_SH_daily_hfq.csv", "上证50ETF", "A股宽基"),
    ("510300.SH", "510300_SH_daily_hfq.csv", "沪深300ETF", "A股宽基"),
    ("510500.SH", "510500_SH_daily_hfq.csv", "中证500ETF", "A股宽基"),
    ("512100.SH", "512100_SH_daily_hfq.csv", "中证1000ETF", "A股宽基"),
    ("159915.SZ", "159915_SZ_daily_hfq.csv", "创业板ETF", "A股宽基"),
    ("518880.SH", "518880_SH_daily_hfq.csv", "黄金ETF", "贵金属"),
    ("159934.SZ", "159934_SZ_daily_hfq.csv", "黄金ETF易方达", "贵金属"),
    ("159980.SZ", "159980_SZ_daily_hfq.csv", "有色ETF", "商品期货"),
    ("159981.SZ", "159981_SZ_daily_hfq.csv", "能源化工ETF", "商品期货"),
    ("511010.SH", "511010_SH_daily_hfq.csv", "5年国债ETF", "利率债"),
    ("511260.SH", "511260_SH_daily_hfq.csv", "十年国债ETF", "利率债"),
    ("511360.SH", "511360_SH_daily_hfq.csv", "短融ETF", "现金管理"),
]

EQ_CODES = ["510050.SH", "510300.SH", "510500.SH", "512100.SH"]
COMBO_RISK_CODES = ["518880.SH", "510500.SH", "510300.SH"]
BOND_CODE = "511010.SH"
COST_RATE = 0.0005
BUFFER_THRESH = 0.04
MARKET_TIMEZONE = ZoneInfo("Asia/Shanghai")


def infer_month_end_dates(index, as_of_date=None):
    idx = pd.DatetimeIndex(index)
    if idx.empty:
        return idx
    month_ends = idx.to_series().groupby(idx.to_period("M")).tail(1).index
    last_dt = idx[-1]
    reference = pd.Timestamp(as_of_date).normalize() if as_of_date else pd.Timestamp(datetime.now(MARKET_TIMEZONE).date())
    if last_dt.to_period("M") == reference.to_period("M"):
        business_month_end = (last_dt + pd.offsets.BMonthEnd(0)).normalize()
        if last_dt.normalize() != business_month_end:
            month_ends = month_ends[month_ends != last_dt]
    return pd.DatetimeIndex(month_ends)


def resolve_execution_context(prices, execution_date=None):
    execution_dt = pd.Timestamp(execution_date).normalize() if execution_date else pd.Timestamp(datetime.now(MARKET_TIMEZONE).date())
    prior_dates = prices.index[prices.index < execution_dt]
    if prior_dates.empty:
        raise RuntimeError(f"没有 {execution_dt:%Y-%m-%d} 之前的行情，无法计算 T+1 实际持仓")
    signal_dt = prior_dates[-1]
    expected_signal_dt = (execution_dt - pd.offsets.BDay(1)).normalize()
    if signal_dt != expected_signal_dt:
        raise RuntimeError(
            f"数据最新可用信号日为 {signal_dt:%Y-%m-%d}，但 {execution_dt:%Y-%m-%d} 实际持仓需要 "
            f"{expected_signal_dt:%Y-%m-%d} 的收盘信号；为避免滞后或盘中误判，停止生成正式持仓"
        )
    return execution_dt, signal_dt


def load_all_etf_data(data_dir):
    close_hfq = {}
    close_raw = {}
    adj_factor = {}
    latest_dates = {}
    for sym, fname, _, _ in ETF_META:
        fpath = os.path.join(data_dir, fname)
        df = pd.read_csv(fpath)
        required = {"datetime", "close", "close_raw", "adj_factor"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise RuntimeError(f"{sym} 数据缺少必要字段：{missing}")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
        if df["datetime"].duplicated().any():
            raise RuntimeError(f"{sym} 数据存在重复日期")
        df = df.sort_values("datetime").set_index("datetime")
        if df.empty:
            raise RuntimeError(f"{sym} 数据为空")
        close_hfq[sym] = df["close"].astype(float)
        close_raw[sym] = df["close_raw"].astype(float)
        adj_factor[sym] = df["adj_factor"].astype(float)
        latest_dates[sym] = df.index[-1]
    if len(set(latest_dates.values())) != 1:
        details = {s: d.strftime("%Y-%m-%d") for s, d in latest_dates.items()}
        raise RuntimeError(f"12只ETF最新日期不一致，停止以前向填充制造混合持仓：{details}")
    df_hfq = pd.DataFrame(close_hfq).sort_index().ffill()
    df_raw = pd.DataFrame(close_raw).sort_index().ffill()
    df_adj = pd.DataFrame(adj_factor).sort_index().ffill()
    return df_hfq, df_raw, df_adj


def compute_indicators(prices):
    rets = prices.pct_change()
    ma20 = prices.rolling(20, min_periods=20).mean()
    ma60 = prices.rolling(60, min_periods=60).mean()
    ma200 = prices.rolling(200, min_periods=200).mean()
    vol20 = rets.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252.0) * 100.0
    q80 = vol20.rolling(500, min_periods=120).quantile(0.80)
    hwm60 = prices.rolling(60, min_periods=20).max()
    hwm120 = prices.rolling(120, min_periods=20).max()
    vol60 = rets.rolling(60, min_periods=20).std(ddof=1) * np.sqrt(252.0) * 100.0
    return {
        "rets": rets,
        "ma20": ma20,
        "ma60": ma60,
        "ma200": ma200,
        "vol20": vol20,
        "q80": q80,
        "hwm60": hwm60,
        "hwm120": hwm120,
        "vol60": vol60,
    }


def generate_five_rule_signals(prices, ind, symbols, month_end_dates=None, as_of_date=None):
    if month_end_dates is None:
        month_end_dates = infer_month_end_dates(prices.index, as_of_date)
    me_set = set(pd.DatetimeIndex(month_end_dates))
    weights = pd.DataFrame(0.0, index=prices.index, columns=symbols)
    rules = pd.DataFrame("规则C0【破位清仓退守国债】", index=prices.index, columns=symbols)

    for sym in symbols:
        p = prices[sym]
        m20 = ind["ma20"][sym]
        m60 = ind["ma60"][sym]
        m200 = ind["ma200"][sym]
        v20 = ind["vol20"][sym]
        q80 = ind["q80"][sym]
        v20_lag10 = v20.shift(10)
        h60 = ind["hwm60"][sym]

        w_curr = 0.0
        r_curr = "规则C0【破位清仓退守国债】"
        w_list = []
        r_list = []

        for dt in prices.index:
            px = p.loc[dt]
            ma20_v = m20.loc[dt]
            ma60_v = m60.loc[dt]
            ma200_v = m200.loc[dt]
            vol_v = v20.loc[dt]
            q80_v = q80.loc[dt]
            vol_l10 = v20_lag10.loc[dt]
            hwm_v = h60.loc[dt]

            if np.isnan(ma200_v) or np.isnan(vol_v) or np.isnan(q80_v):
                w_list.append(0.0)
                r_list.append("样本预热期")
                continue

            is_me = dt in me_set
            zero_dd = px >= 0.995 * hwm_v
            mild_high_vol = (vol_v > q80_v) and (vol_v <= 1.15 * q80_v)

            if px < ma200_v:
                if is_me and (ma20_v > ma60_v) and (px >= ma60_v) and (vol_v <= q80_v) and (not np.isnan(vol_l10) and vol_v < vol_l10):
                    w_curr = 0.5
                    r_curr = "规则L1【高波回落快线抢跑】"
                elif w_curr == 0.5 and (ma20_v > ma60_v) and (px >= 0.985 * ma60_v) and (vol_v <= q80_v):
                    w_curr = 0.5
                    r_curr = "规则L1【高波回落快线抢跑】"
                else:
                    w_curr = 0.0
                    r_curr = "规则C0【破位清仓退守国债】"
            else:
                if (vol_v > q80_v) and (px < ma60_v):
                    w_curr = 0.0
                    r_curr = "规则C0【高波破MA60清仓】"
                elif px < ma60_v:
                    if w_curr > 0.0:
                        w_curr = 0.5
                        r_curr = "规则L1【跌破MA60减半防守】"
                    else:
                        w_curr = 0.0
                        r_curr = "规则C0【破位清仓退守国债】"
                elif vol_v > q80_v:
                    if w_curr > 0.0:
                        if zero_dd and mild_high_vol:
                            w_curr = 0.75
                            r_curr = "规则D1【创新高零回撤缓冲】"
                        else:
                            w_curr = 0.5
                            r_curr = "规则T0【双边高波踩刹车】"
                    else:
                        w_curr = 0.0
                        r_curr = "规则C0【高波禁入退守国债】"
                else:
                    if is_me and (ma20_v >= ma60_v):
                        w_curr = 1.0
                        r_curr = "规则A【趋势低波满仓】"
                    elif w_curr in (0.5, 0.75) and is_me and (ma20_v >= ma60_v):
                        w_curr = 1.0
                        r_curr = "规则A【趋势低波满仓】"

            w_list.append(w_curr)
            r_list.append(r_curr)

        weights[sym] = w_list
        rules[sym] = r_list

    return weights, rules


def build_multi_asset_portfolio(prices, ind, sig_weights):
    idx = prices.index
    cols = COMBO_RISK_CODES + [BOND_CODE]
    w_target = pd.DataFrame(0.0, index=idx, columns=cols)
    base_cap = {"518880.SH": 0.35, "510500.SH": 0.30, "510300.SH": 0.30}

    for dt in idx:
        v60 = ind["vol60"].loc[dt, COMBO_RISK_CODES]
        if v60.isna().any():
            w_target.loc[dt, BOND_CODE] = 1.0
            continue
        inv_v = (10.0 / v60.clip(lower=8.0)).clip(upper=1.2)
        raw_w = {}
        for c in COMBO_RISK_CODES:
            raw_w[c] = base_cap[c] * float(sig_weights.loc[dt, c]) * float(inv_v[c])
        tot_risk = sum(raw_w.values())
        if tot_risk > 0.85:
            scale = 0.85 / tot_risk
            for c in COMBO_RISK_CODES:
                raw_w[c] *= scale
            tot_risk = 0.85
        for c in COMBO_RISK_CODES:
            w_target.loc[dt, c] = raw_w[c]
        w_target.loc[dt, BOND_CODE] = max(0.0, 1.0 - tot_risk)

    w_exec = pd.DataFrame(0.0, index=idx, columns=cols)
    curr = pd.Series({c: 0.0 for c in cols})
    curr[BOND_CODE] = 1.0

    for dt in idx:
        des = w_target.loc[dt]
        risk_des = des[COMBO_RISK_CODES].sum()
        risk_curr = curr[COMBO_RISK_CODES].sum()
        half_l1 = 0.5 * (des - curr).abs().sum()
        if risk_des < risk_curr - 1e-6:
            curr = des.copy()
        elif half_l1 >= BUFFER_THRESH:
            curr = des.copy()
        w_exec.loc[dt] = curr

    return w_exec


def calc_perf_metrics(ret_series, turn_series=None, rf_annual=0.0):
    r = ret_series.dropna()
    n = len(r)
    if n == 0:
        return {}
    nav = (1.0 + r).cumprod()
    tot_ret = float(nav.iloc[-1] - 1.0)
    cagr = float(nav.iloc[-1] ** (252.0 / n) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(252.0))
    rf_daily = rf_annual / 252.0
    ex_r = r - rf_daily
    sharpe = float((ex_r.mean() / r.std(ddof=1)) * np.sqrt(252.0)) if r.std(ddof=1) > 0 else 0.0
    dd = nav / nav.cummax() - 1.0
    mdd = float(dd.min())
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    ann_turn = float(turn_series.reindex(r.index).fillna(0.0).sum() / (n / 252.0) * 0.5 * 100.0) if turn_series is not None else 0.0
    return {
        "累计收益率(%)": round(tot_ret * 100.0, 2),
        "年化收益率(%)": round(cagr * 100.0, 2),
        "年化波动率(%)": round(vol * 100.0, 2),
        "夏普比率": round(sharpe, 3),
        "最大回撤(%)": round(mdd * 100.0, 2),
        "卡玛比率": round(calmar, 2),
        "年化单边换手率(%)": round(ann_turn, 2),
        "样本交易日数": int(n),
    }


def extract_trade_ledger(prices, raw_prices, ind, sig_weights, sig_rules, start_dt="2023-01-01"):
    rows = []
    name_map = {s: n for s, _, n, _ in ETF_META}
    sub_idx = prices.loc[start_dt:].index

    for sym in COMBO_RISK_CODES:
        w_s = sig_weights[sym].loc[sub_idx]
        r_s = sig_rules[sym].loc[sub_idx]
        prev_w = 0.0
        entry_px = None

        for dt in sub_idx:
            new_w = float(w_s.loc[dt])
            if abs(new_w - prev_w) > 1e-6:
                px_hfq = float(prices.loc[dt, sym])
                px_raw = float(raw_prices.loc[dt, sym])
                ma60_v = float(ind["ma60"].loc[dt, sym])
                ma200_v = float(ind["ma200"].loc[dt, sym])
                vol20_v = float(ind["vol20"].loc[dt, sym])
                q80_v = float(ind["q80"].loc[dt, sym])
                rule_name = str(r_s.loc[dt])

                if prev_w == 0.0 and new_w > 0.0:
                    sig_type = "开仓进场"
                    entry_px = px_hfq
                    pnl_str = "建仓起点"
                elif new_w > prev_w:
                    sig_type = "顺势加仓"
                    pct = (px_hfq / entry_px - 1.0) * 100.0 if entry_px else 0.0
                    pnl_str = f"浮盈{pct:+.2f}%加仓"
                elif new_w == 0.0:
                    sig_type = "平仓清零"
                    pct = (px_hfq / entry_px - 1.0) * 100.0 if entry_px else 0.0
                    pnl_str = f"波段结账 {pct:+.2f}%"
                    entry_px = None
                else:
                    sig_type = "高波减仓"
                    pct = (px_hfq / entry_px - 1.0) * 100.0 if entry_px else 0.0
                    pnl_str = f"锁盈/控险 {pct:+.2f}%"

                rows.append(
                    {
                        "标的代码": sym,
                        "标的名称": name_map[sym],
                        "信号日期(T日收盘)": dt.strftime("%Y-%m-%d"),
                        "执行规则(T+1日生效)": sig_type,
                        "仓位变动": f"{int(prev_w*100)}% -> {int(new_w*100)}%",
                        "原仓位": prev_w,
                        "新仓位": new_w,
                        "不复权现价": round(px_raw, 3),
                        "后复权收盘价": round(px_hfq, 3),
                        "MA60均线": round(ma60_v, 3),
                        "MA200均线": round(ma200_v, 3),
                        "20日年化波动率(%)": round(vol20_v, 2),
                        "80%分位阈值(%)": round(q80_v, 2),
                        "波段累计盈亏": pnl_str,
                        "核心规则代号": rule_name,
                    }
                )
                prev_w = new_w

    return pd.DataFrame(rows)


def run_causality_audit(prices):
    audit_cols = COMBO_RISK_CODES + [BOND_CODE]
    p_sub = prices[audit_cols].loc["2020-01-01":].copy()
    month_end_dates = infer_month_end_dates(p_sub.index)
    ind_full = compute_indicators(p_sub)
    w_full, _ = generate_five_rule_signals(p_sub, ind_full, COMBO_RISK_CODES, month_end_dates=month_end_dates)
    port_full = build_multi_asset_portfolio(p_sub, ind_full, w_full)

    cutoff_dates = ["2024-06-28", "2025-06-30", "2026-03-31", "2026-10-15"]
    max_trunc_diff = 0.0
    max_perturb_diff = 0.0
    details = []

    for cut_str in cutoff_dates:
        cut_dt = pd.Timestamp(cut_str)
        prices_trunc = p_sub.loc[:cut_dt].copy()
        ind_trunc = compute_indicators(prices_trunc)
        w_trunc, _ = generate_five_rule_signals(prices_trunc, ind_trunc, COMBO_RISK_CODES, month_end_dates=month_end_dates)
        port_trunc = build_multi_asset_portfolio(prices_trunc, ind_trunc, w_trunc)
        diff_t = float((port_full.loc[:cut_dt] - port_trunc).abs().max().max())
        max_trunc_diff = max(max_trunc_diff, diff_t)

        prices_pert = p_sub.copy()
        after_mask = prices_pert.index > cut_dt
        rng = np.random.default_rng(20261106)
        noise = rng.uniform(0.5, 1.8, size=prices_pert.loc[after_mask].shape)
        prices_pert.loc[after_mask] = prices_pert.loc[after_mask] * noise
        ind_pert = compute_indicators(prices_pert)
        w_pert, _ = generate_five_rule_signals(prices_pert, ind_pert, COMBO_RISK_CODES, month_end_dates=month_end_dates)
        port_pert = build_multi_asset_portfolio(prices_pert, ind_pert, w_pert)
        diff_p = float((port_full.loc[:cut_dt] - port_pert.loc[:cut_dt]).abs().max().max())
        max_perturb_diff = max(max_perturb_diff, diff_p)

        details.append(
            {
                "cutoff_date": cut_str,
                "truncation_max_abs_diff": diff_t,
                "future_perturbation_max_abs_diff": diff_p,
                "status": "PASS" if (diff_t == 0.0 and diff_p == 0.0) else "FAIL",
            }
        )

    return {
        "audit_name": "ETF择时系统防未来函数与因果不变性检验 (Causality & Future-Perturbation Invariance Test)",
        "execution_lag_rule": "T日收盘计算信号，严格通过 shift(1) 于 T+1 日生效计入收益",
        "truncation_invariance_max_diff": max_trunc_diff,
        "future_perturbation_max_diff": max_perturb_diff,
        "all_checks_passed": bool(max_trunc_diff == 0.0 and max_perturb_diff == 0.0),
        "checkpoints": details,
    }


def main(execution_date=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, "..", "04-数据获得", "daily_hfq"))

    prices, raw_prices, adj_factors = load_all_etf_data(data_dir)
    execution_dt, signal_dt = resolve_execution_context(prices, execution_date)
    last_dt = prices.index[-1]
    ind = compute_indicators(prices)
    rets = ind["rets"].fillna(0.0)

    all_symbols = [s for s, _, _, _ in ETF_META]
    me_dates = infer_month_end_dates(prices.index, execution_dt)
    sig_w_all, sig_r_all = generate_five_rule_signals(prices, ind, all_symbols, month_end_dates=me_dates)

    lv_t2c1 = ((prices[EQ_CODES] > ind["ma200"][EQ_CODES]) & (ind["vol20"][EQ_CODES] <= ind["q80"][EQ_CODES])).astype(float)
    sig_t2c1 = pd.DataFrame(np.nan, index=prices.index, columns=EQ_CODES)
    sig_t2c1.loc[me_dates] = lv_t2c1.loc[me_dates]
    sig_t2c1_exec = sig_t2c1.ffill().fillna(0.0).shift(1).fillna(0.0)

    w_eq_bh = pd.DataFrame(0.25, index=prices.index, columns=EQ_CODES)
    r_eq_bh = (w_eq_bh * rets[EQ_CODES]).sum(axis=1)

    w_eq_cash = sig_t2c1_exec * 0.25
    turn_eq_cash = w_eq_cash.diff().abs().sum(axis=1).fillna(0.0)
    r_eq_cash = (w_eq_cash * rets[EQ_CODES]).sum(axis=1) - turn_eq_cash * COST_RATE

    w_eq_bond = w_eq_cash.copy()
    w_eq_bond[BOND_CODE] = 1.0 - w_eq_cash.sum(axis=1)
    turn_eq_bond = w_eq_bond.diff().abs().sum(axis=1).fillna(0.0)
    r_eq_bond = (w_eq_bond[EQ_CODES] * rets[EQ_CODES]).sum(axis=1) + w_eq_bond[BOND_CODE] * rets[BOND_CODE] - turn_eq_bond * COST_RATE

    single_results = {}
    for sym in ["518880.SH", "510500.SH", "510300.SH", "510050.SH", "512100.SH"]:
        w_s_exec = sig_w_all[sym].shift(1).fillna(0.0)
        w_b_exec = 1.0 - w_s_exec
        turn_s = w_s_exec.diff().abs().fillna(0.0) * 2.0
        r_s_timed_bond = w_s_exec * rets[sym] + w_b_exec * rets[BOND_CODE] - turn_s * COST_RATE
        r_s_timed_cash = w_s_exec * rets[sym] - w_s_exec.diff().abs().fillna(0.0) * COST_RATE
        single_results[sym] = {
            "bh": rets[sym],
            "timed_cash": r_s_timed_cash,
            "turn_cash": w_s_exec.diff().abs().fillna(0.0),
            "timed_bond": r_s_timed_bond,
            "turn_bond": turn_s,
            "w_exec": w_s_exec,
        }

    w_multi_raw = build_multi_asset_portfolio(prices, ind, sig_w_all[COMBO_RISK_CODES])
    w_multi_exec = w_multi_raw.shift(1).fillna(0.0)
    w_multi_exec.iloc[0] = [0.0, 0.0, 0.0, 1.0]
    turn_multi = w_multi_exec.diff().abs().sum(axis=1).fillna(0.0)
    r_multi = (w_multi_exec * rets[COMBO_RISK_CODES + [BOND_CODE]]).sum(axis=1) - turn_multi * COST_RATE

    last_date_str = last_dt.strftime("%Y-%m-%d")
    recent_label = f"近三年样本 (2023-01 至 {last_dt:%Y-%m})"
    ytd_label = f"{last_dt.year}年YTD压力窗 ({last_dt.year}-01 至 {last_dt:%Y-%m})"
    periods = [
        (recent_label, "2023-01-03", last_date_str),
        ("FOF标准三年窗 (2023-10 至 2026-09)", "2023-10-01", "2026-09-30"),
        (ytd_label, f"{last_dt.year}-01-01", last_date_str),
    ]

    summary_rows = []
    strat_list = [
        ("多资产组合", "股债金五规则论文择时组合 (518880+510500+510300+511010)", r_multi, turn_multi),
        ("四大宽基组合", "四宽基等权死拿基准 (50/300/500/1000 Buy&Hold)", r_eq_bh, None),
        ("四大宽基组合", "四宽基双门控择时-空仓持0息现金 (T2_C1->Cash)", r_eq_cash, turn_eq_cash),
        ("四大宽基组合", "四宽基双门控择时-空仓转511010国债ETF (T2_C1->Bond)", r_eq_bond, turn_eq_bond),
        ("单品种对照", "黄金ETF (518880.SH) 买入死拿基准", single_results["518880.SH"]["bh"], None),
        ("单品种对照", "黄金ETF (518880.SH) 五规则论文择时+转国债ETF", single_results["518880.SH"]["timed_bond"], single_results["518880.SH"]["turn_bond"]),
        ("单品种对照", "中证500ETF (510500.SH) 买入死拿基准", single_results["510500.SH"]["bh"], None),
        ("单品种对照", "中证500ETF (510500.SH) 五规则论文择时+转国债ETF", single_results["510500.SH"]["timed_bond"], single_results["510500.SH"]["turn_bond"]),
        ("单品种对照", "沪深300ETF (510300.SH) 买入死拿基准", single_results["510300.SH"]["bh"], None),
        ("单品种对照", "沪深300ETF (510300.SH) 五规则论文择时+转国债ETF", single_results["510300.SH"]["timed_bond"], single_results["510300.SH"]["turn_bond"]),
        ("单品种对照", "上证50ETF (510050.SH) 买入死拿基准", single_results["510050.SH"]["bh"], None),
        ("单品种对照", "上证50ETF (510050.SH) 五规则论文择时+转国债ETF", single_results["510050.SH"]["timed_bond"], single_results["510050.SH"]["turn_bond"]),
        ("单品种对照", "中证1000ETF (512100.SH) 买入死拿基准", single_results["512100.SH"]["bh"], None),
        ("单品种对照", "中证1000ETF (512100.SH) 五规则论文择时+转国债ETF", single_results["512100.SH"]["timed_bond"], single_results["512100.SH"]["turn_bond"]),
        ("防守底仓基准", "5年国债ETF (511010.SH) 买入持有基准", rets[BOND_CODE], None),
    ]

    for p_label, s_dt, e_dt in periods:
        for cat, s_name, r_ser, t_ser in strat_list:
            sub_r = r_ser.loc[s_dt:e_dt]
            sub_t = t_ser.loc[s_dt:e_dt] if t_ser is not None else None
            m = calc_perf_metrics(sub_r, sub_t)
            row = {"区间名称": p_label, "策略分类": cat, "策略名称": s_name}
            row.update(m)
            summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(base_dir, "backtest_summary_metrics.csv"), index=False, encoding="utf-8-sig")

    df_ledger = extract_trade_ledger(prices, raw_prices, ind, sig_w_all, sig_r_all, start_dt="2023-01-01")
    df_ledger.to_csv(os.path.join(base_dir, "trade_signals_ledger_2023_2026.csv"), index=False, encoding="utf-8-sig")

    signal_prices = prices.loc[:signal_dt]
    previous_signal_dt = prices.index[prices.index < signal_dt][-1]
    actual_weights = w_multi_raw.loc[signal_dt]
    previous_weights = w_multi_raw.loc[previous_signal_dt]
    audit_rows = []
    for sym, _, name, cat in ETF_META:
        px_hfq = float(prices.loc[signal_dt, sym])
        px_raw = float(raw_prices.loc[signal_dt, sym])
        adj_v = float(adj_factors.loc[signal_dt, sym])
        ma20_v = float(ind["ma20"].loc[signal_dt, sym])
        ma60_v = float(ind["ma60"].loc[signal_dt, sym])
        ma200_v = float(ind["ma200"].loc[signal_dt, sym])
        vol20_v = float(ind["vol20"].loc[signal_dt, sym])
        q80_v = float(ind["q80"].loc[signal_dt, sym])
        ret20 = float((signal_prices[sym].iloc[-1] / signal_prices[sym].iloc[-21] - 1.0) * 100.0)
        ret60 = float((signal_prices[sym].iloc[-1] / signal_prices[sym].iloc[-61] - 1.0) * 100.0)
        t2c1_v = 100 if (px_hfq > ma200_v and vol20_v <= q80_v) else 0
        w_5rule = float(sig_w_all.loc[signal_dt, sym])
        r_5rule = str(sig_r_all.loc[signal_dt, sym])
        combo_w = float(actual_weights[sym]) * 100.0 if sym in actual_weights.index else 0.0
        previous_combo_w = float(previous_weights[sym]) * 100.0 if sym in previous_weights.index else 0.0
        weight_change = combo_w - previous_combo_w
        if sym not in actual_weights.index:
            rebalance_direction = "组合外"
        elif weight_change > 0.005:
            rebalance_direction = "加仓"
        elif weight_change < -0.005:
            rebalance_direction = "减仓"
        else:
            rebalance_direction = "不变"

        audit_rows.append(
            {
                "数据截止日期": last_dt.strftime("%Y-%m-%d"),
                "信号日期": signal_dt.strftime("%Y-%m-%d"),
                "执行日期": execution_dt.strftime("%Y-%m-%d"),
                "标的代码": sym,
                "标的名称": name,
                "资产类别": cat,
                "后复权收盘价": round(px_hfq, 4),
                "不复权现价": round(px_raw, 4),
                "累计复权因子": round(adj_v, 6),
                "MA20": round(ma20_v, 4),
                "MA60": round(ma60_v, 4),
                "MA200": round(ma200_v, 4),
                "20日年化波动率(%)": round(vol20_v, 2),
                "500日80%分位阈值(%)": round(q80_v, 2),
                "近20日涨跌幅(%)": round(ret20, 2),
                "近60日涨跌幅(%)": round(ret60, 2),
                "双门控信号日状态(T2_C1_%)": t2c1_v,
                "单品种五规则信号仓位(%)": round(w_5rule * 100.0, 1),
                "信号日触发规则": r_5rule,
                "股债金组合今日实际权重(%)": round(combo_w, 2),
                "今日权重变动(百分点)": round(weight_change, 2),
                "今日调仓方向": rebalance_direction,
            }
        )

    df_audit = pd.DataFrame(audit_rows)
    audit_path = os.path.join(base_dir, f"current_holdings_audit_{execution_dt:%Y%m%d}.csv")
    df_audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    causality_res = run_causality_audit(prices)
    with open(os.path.join(base_dir, "causality_check_result.json"), "w", encoding="utf-8") as f:
        json.dump(causality_res, f, ensure_ascii=False, indent=2)

    sub_start = "2023-01-03"
    nav_multi = (1.0 + r_multi.loc[sub_start:]).cumprod()
    nav_eq_bh = (1.0 + r_eq_bh.loc[sub_start:]).cumprod()
    nav_eq_bond = (1.0 + r_eq_bond.loc[sub_start:]).cumprod()
    nav_gold_timed = (1.0 + single_results["518880.SH"]["timed_bond"].loc[sub_start:]).cumprod()
    nav_gold_bh = (1.0 + single_results["518880.SH"]["bh"].loc[sub_start:]).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3, 1.5]})
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.07, right=0.96, hspace=0.18)
    fig.suptitle(f"基于顶刊文献的 ETF 择时系统实证回测总览图 (2023.01 - {last_dt:%Y.%m} 真实日线)", fontsize=16, fontweight="bold", y=0.965)

    c_red = (0.86, 0.15, 0.15)
    c_blue = (0.11, 0.30, 0.85)
    c_teal = (0.06, 0.46, 0.43)
    c_amber = (0.85, 0.47, 0.02)
    c_gray = (0.45, 0.50, 0.58)

    axes[0].plot(nav_multi.index, nav_multi.values, color=c_red, lw=2.2, label="股债金五规则论文择时组合 (518880+510500+510300+511010)")
    axes[0].plot(nav_eq_bond.index, nav_eq_bond.values, color=c_blue, lw=1.8, label="四宽基双门控择时+转511010国债ETF (T2_C1->Bond)")
    axes[0].plot(nav_eq_bh.index, nav_eq_bh.values, color=c_gray, lw=1.4, ls="--", label="四宽基等权买入持有基准 (Buy & Hold)")
    axes[0].plot(nav_gold_timed.index, nav_gold_timed.values, color=c_teal, lw=1.6, label="黄金ETF (518880) 五规则论文择时+转国债ETF")
    axes[0].plot(nav_gold_bh.index, nav_gold_bh.values, color=c_amber, lw=1.2, ls=":", label="黄金ETF (518880) 买入持有基准")
    axes[0].set_ylabel("累计净值 (初始=1.0)", fontsize=11)
    axes[0].set_title("面板 A：多资产组合、四宽基择时与黄金单品种择时累计净值对比 (含单边5bp成本与T+1执行滞后)", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper left", ncol=2, fontsize=9.5)
    axes[0].grid(True, alpha=0.3)

    dd_multi = (nav_multi / nav_multi.cummax() - 1.0) * 100.0
    dd_eq_bh = (nav_eq_bh / nav_eq_bh.cummax() - 1.0) * 100.0
    dd_gold_bh = (nav_gold_bh / nav_gold_bh.cummax() - 1.0) * 100.0
    axes[1].fill_between(dd_eq_bh.index, dd_eq_bh.values, 0, color=c_gray, alpha=0.22, label="四宽基死拿回撤")
    axes[1].plot(dd_gold_bh.index, dd_gold_bh.values, color=c_amber, lw=1.2, ls=":", label=f"黄金ETF死拿最大回撤 ({dd_gold_bh.min():.2f}%)")
    axes[1].plot(dd_multi.index, dd_multi.values, color=c_red, lw=1.8, label="股债金论文择时组合动态回撤")
    axes[1].set_ylabel("动态回撤 (%)", fontsize=11)
    axes[1].set_title("面板 B：多资产择时、四宽基持有与黄金持有的动态回撤对比", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower left", ncol=3, fontsize=9.5)
    axes[1].grid(True, alpha=0.3)

    w_plot = w_multi_exec.loc[sub_start:] * 100.0
    axes[2].stackplot(
        w_plot.index,
        w_plot["518880.SH"],
        w_plot["510500.SH"],
        w_plot["510300.SH"],
        w_plot["511010.SH"],
        labels=["518880 黄金ETF", "510500 中证500ETF", "510300 沪深300ETF", "511010 国债ETF (防守收息底仓)"],
        colors=[c_amber, c_blue, c_teal, (0.82, 0.86, 0.92)],
        alpha=0.88,
    )
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("组合持仓权重 (%)", fontsize=11)
    axes[2].set_xlabel("交易日期", fontsize=11)
    holdings_desc = "，".join(f"{code.split('.')[0]} {actual_weights[code] * 100:.2f}%" for code in COMBO_RISK_CODES + [BOND_CODE])
    axes[2].set_title(f"面板 C：组合动态仓位（{execution_dt:%Y-%m-%d} 实际执行：{holdings_desc}）", fontsize=12, fontweight="bold")
    axes[2].legend(loc="upper right", ncol=4, fontsize=9.5)
    axes[2].grid(True, alpha=0.3)

    plt.savefig(os.path.join(base_dir, "student_backtest_result_chart.png"), dpi=220)
    plt.close(fig)

    print("Backtest completed successfully.")
    print(df_summary[df_summary["区间名称"] == recent_label].to_string(index=False))
    position_cols = ["信号日期", "执行日期", "标的代码", "标的名称", "股债金组合今日实际权重(%)", "今日权重变动(百分点)", "今日调仓方向", "信号日触发规则"]
    combo_audit = df_audit[df_audit["标的代码"].isin(COMBO_RISK_CODES + [BOND_CODE])][position_cols]
    print("Today actual holdings (T+1):")
    print(combo_audit.to_string(index=False))
    print(f"Holdings audit: {audit_path}")
    print("Causality check:", json.dumps(causality_res, ensure_ascii=False))
    if not causality_res["all_checks_passed"]:
        raise RuntimeError("因果检验未通过，请检查持仓结果")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-date", default=None, help="实际持仓执行日期，格式 YYYY-MM-DD；默认取上海当天日期")
    args = parser.parse_args()
    main(args.execution_date)
