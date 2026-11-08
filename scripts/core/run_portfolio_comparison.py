# -*- coding: utf-8 -*-
"""新建组合对比回测：标普500 + 纳指 + 黄金 + 国债  vs  原股债金组合。

设计原则
--------
1) 不修改 run_etf_timing_backtest.py 及任何原有产出文件，原组合结果零污染。
2) 通过 import 复用原脚本的 compute_indicators / generate_five_rule_signals /
   calc_perf_metrics / infer_month_end_dates / resolve_execution_context 等纯函数，
   保证信号体系、成本、T+1 执行口径与原组合完全一致。
3) 仅「把股票标的进行了替换」：
     原组合 518880 黄金 + 510500 中证500 + 510300 沪深300 + 511010 国债防守
     新组合 518880 黄金 + 513500 标普500 + 513100 纳指     + 511010 国债防守
   权重结构沿用 base_cap {黄金0.35, 股票各0.30}，0.85 风险上限、4% 换手缓冲、
   5bp 单边成本、五规则信号、T+1 执行全部保持不变。
4) 输出并列绩效、净值曲线、对比图、新组合 T+1 持仓审计、两组合因果检验。

产出文件（均在 05-实验代码/）
    portfolio_comparison_metrics.csv
    portfolio_comparison_nav.csv
    portfolio_comparison_chart.png
    new_combo_holdings_audit_YYYYMMDD.csv
    portfolio_comparison_summary.json
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
# [云端副本] 仅将 matplotlib 改为可选导入，其余逻辑与本地研究脚本逐字一致。
# CI 不生成图表，避免 ubuntu-latest 缺少中文字体导致的方框字与额外依赖。
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
except ImportError:
    plt = None

import run_etf_timing_backtest as bt

# 14 只标的：原 12 只 + 新增 2 只海外股票 ETF
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
    ("513500.SH", "513500_SH_daily_hfq.csv", "标普500ETF博时", "海外股票"),
    ("513100.SH", "513100_SH_daily_hfq.csv", "纳指ETF国泰", "海外股票"),
]

OLD_RISK = ["518880.SH", "510500.SH", "510300.SH"]
NEW_RISK = ["518880.SH", "513500.SH", "513100.SH"]
BOND_CODE = bt.BOND_CODE
OLD_BASE_CAP = {"518880.SH": 0.35, "510500.SH": 0.30, "510300.SH": 0.30}
NEW_BASE_CAP = {"518880.SH": 0.35, "513500.SH": 0.30, "513100.SH": 0.30}

OLD_NAME = "原股债金组合 (518880黄金+510500中证500+510300沪深300+511010国债)"
NEW_NAME = "新建海外组合 (518880黄金+513500标普500+513100纳指+511010国债)"


def load_prices(data_dir):
    """与原脚本 load_all_etf_data 同口径，多载入两只新标的。"""
    close_hfq, close_raw, adj_factor, latest = {}, {}, {}, {}
    for sym, fname, _, _ in ETF_META:
        df = pd.read_csv(os.path.join(data_dir, fname))
        required = {"datetime", "close", "close_raw", "adj_factor"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise RuntimeError(f"{sym} 缺少字段 {missing}")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
        if df["datetime"].duplicated().any():
            raise RuntimeError(f"{sym} 存在重复日期")
        df = df.sort_values("datetime").set_index("datetime")
        close_hfq[sym] = df["close"].astype(float)
        close_raw[sym] = df["close_raw"].astype(float)
        adj_factor[sym] = df["adj_factor"].astype(float)
        latest[sym] = df.index[-1]
    if len(set(latest.values())) != 1:
        raise RuntimeError(f"14只ETF最新日期不一致：{ {s: d.strftime('%Y-%m-%d') for s, d in latest.items()} }")
    prices = pd.DataFrame(close_hfq).sort_index().ffill()
    raw_prices = pd.DataFrame(close_raw).sort_index().ffill()
    adj_factors = pd.DataFrame(adj_factor).sort_index().ffill()
    return prices, raw_prices, adj_factors


def build_portfolio(prices, ind, sig_weights, risk_codes, base_cap):
    """参数化复刻原 build_multi_asset_portfolio，两个组合共用同一逻辑保证公平。"""
    idx = prices.index
    cols = risk_codes + [BOND_CODE]
    w_target = pd.DataFrame(0.0, index=idx, columns=cols)
    for dt in idx:
        v60 = ind["vol60"].loc[dt, risk_codes]
        if v60.isna().any():
            w_target.loc[dt, BOND_CODE] = 1.0
            continue
        inv_v = (10.0 / v60.clip(lower=8.0)).clip(upper=1.2)
        raw_w = {c: base_cap[c] * float(sig_weights.loc[dt, c]) * float(inv_v[c]) for c in risk_codes}
        tot_risk = sum(raw_w.values())
        if tot_risk > 0.85:
            scale = 0.85 / tot_risk
            raw_w = {c: w * scale for c, w in raw_w.items()}
            tot_risk = 0.85
        for c in risk_codes:
            w_target.loc[dt, c] = raw_w[c]
        w_target.loc[dt, BOND_CODE] = max(0.0, 1.0 - tot_risk)

    w_exec = pd.DataFrame(0.0, index=idx, columns=cols)
    curr = pd.Series({c: 0.0 for c in cols})
    curr[BOND_CODE] = 1.0
    for dt in idx:
        des = w_target.loc[dt]
        risk_des = des[risk_codes].sum()
        risk_curr = curr[risk_codes].sum()
        half_l1 = 0.5 * (des - curr).abs().sum()
        if risk_des < risk_curr - 1e-6:
            curr = des.copy()
        elif half_l1 >= bt.BUFFER_THRESH:
            curr = des.copy()
        w_exec.loc[dt] = curr
    return w_exec


def combo_returns(prices, ind, rets, sig_w_all, risk_codes, base_cap):
    w_raw = build_portfolio(prices, ind, sig_w_all[risk_codes], risk_codes, base_cap)
    w_exec = w_raw.shift(1).fillna(0.0)
    first = pd.Series(0.0, index=w_exec.columns)
    first[BOND_CODE] = 1.0
    w_exec.iloc[0] = first.values
    turn = w_exec.diff().abs().sum(axis=1).fillna(0.0)
    r = (w_exec * rets[risk_codes + [BOND_CODE]]).sum(axis=1) - turn * bt.COST_RATE
    return {"w_raw": w_raw, "w_exec": w_exec, "turn": turn, "ret": r}


def causality_audit(prices, risk_codes, base_cap, label):
    """参数化复刻原 run_causality_audit：截断不变性 + 未来扰动不变性。"""
    audit_cols = risk_codes + [BOND_CODE]
    p_sub = prices[audit_cols].loc["2020-01-01":].copy()
    me = bt.infer_month_end_dates(p_sub.index)
    ind_full = bt.compute_indicators(p_sub)
    w_full, _ = bt.generate_five_rule_signals(p_sub, ind_full, risk_codes, month_end_dates=me)
    port_full = build_portfolio(p_sub, ind_full, w_full, risk_codes, base_cap)

    cutoffs = ["2024-06-28", "2025-06-30", "2026-03-31", "2026-10-15"]
    max_t = max_p = 0.0
    details = []
    for cut in cutoffs:
        cut_dt = pd.Timestamp(cut)
        p_trunc = p_sub.loc[:cut_dt].copy()
        i_t = bt.compute_indicators(p_trunc)
        w_t, _ = bt.generate_five_rule_signals(p_trunc, i_t, risk_codes, month_end_dates=me)
        p_t = build_portfolio(p_trunc, i_t, w_t, risk_codes, base_cap)
        d_t = float((port_full.loc[:cut_dt] - p_t).abs().max().max())

        p_pert = p_sub.copy()
        mask = p_pert.index > cut_dt
        rng = np.random.default_rng(20261106)
        p_pert.loc[mask] = p_pert.loc[mask] * rng.uniform(0.5, 1.8, size=p_pert.loc[mask].shape)
        i_p = bt.compute_indicators(p_pert)
        w_p, _ = bt.generate_five_rule_signals(p_pert, i_p, risk_codes, month_end_dates=me)
        p_p = build_portfolio(p_pert, i_p, w_p, risk_codes, base_cap)
        d_p = float((port_full.loc[:cut_dt] - p_p.loc[:cut_dt]).abs().max().max())

        max_t, max_p = max(max_t, d_t), max(max_p, d_p)
        details.append({"cutoff_date": cut, "truncation_max_abs_diff": d_t,
                        "future_perturbation_max_abs_diff": d_p,
                        "status": "PASS" if (d_t == 0.0 and d_p == 0.0) else "FAIL"})
    return {"audit_name": f"{label} 防未来函数与因果不变性检验",
            "truncation_invariance_max_diff": max_t,
            "future_perturbation_max_diff": max_p,
            "all_checks_passed": bool(max_t == 0.0 and max_p == 0.0),
            "checkpoints": details}


def holdings_audit(prices, raw_prices, adj_factors, ind, sig_w_all, sig_r_all,
                   combo, execution_dt, signal_dt, last_dt):
    name_map = {s: n for s, _, n, _ in ETF_META}
    cat_map = {s: c for s, _, _, c in ETF_META}
    actual = combo["w_raw"].loc[signal_dt]
    prev_dt = prices.index[prices.index < signal_dt][-1]
    previous = combo["w_raw"].loc[prev_dt]
    rows = []
    for sym in NEW_RISK + [BOND_CODE]:
        w = float(actual[sym]) * 100.0
        dw = w - float(previous[sym]) * 100.0
        direction = "加仓" if dw > 0.005 else ("减仓" if dw < -0.005 else "不变")
        rows.append({
            "数据截止日期": last_dt.strftime("%Y-%m-%d"),
            "信号日期(T日收盘)": signal_dt.strftime("%Y-%m-%d"),
            "执行日期(T+1)": execution_dt.strftime("%Y-%m-%d"),
            "组合名称": NEW_NAME,
            "标的代码": sym, "标的名称": name_map[sym], "资产类别": cat_map[sym],
            "后复权收盘价": round(float(prices.loc[signal_dt, sym]), 4),
            "不复权现价": round(float(raw_prices.loc[signal_dt, sym]), 4),
            "累计复权因子": round(float(adj_factors.loc[signal_dt, sym]), 6),
            "MA20": round(float(ind["ma20"].loc[signal_dt, sym]), 4),
            "MA60": round(float(ind["ma60"].loc[signal_dt, sym]), 4),
            "MA200": round(float(ind["ma200"].loc[signal_dt, sym]), 4),
            "20日年化波动率(%)": round(float(ind["vol20"].loc[signal_dt, sym]), 2),
            "500日80%分位阈值(%)": round(float(ind["q80"].loc[signal_dt, sym]), 2),
            "单品种五规则信号仓位(%)": round(float(sig_w_all.loc[signal_dt, sym]) * 100.0, 1),
            "信号日触发规则": str(sig_r_all.loc[signal_dt, sym]),
            "新组合今日实际权重(%)": round(w, 2),
            "今日权重变动(百分点)": round(dw, 2),
            "今日调仓方向": direction,
        })
    return pd.DataFrame(rows)


def make_chart(base_dir, last_dt, execution_dt, old, new, rets, sub_start, out_name, window_label):
    nav_old = (1.0 + old["ret"].loc[sub_start:]).cumprod()
    nav_new = (1.0 + new["ret"].loc[sub_start:]).cumprod()
    bh_old = (1.0 + rets[OLD_RISK].mean(axis=1).loc[sub_start:]).cumprod()
    bh_new = (1.0 + rets[NEW_RISK].mean(axis=1).loc[sub_start:]).cumprod()

    c_red, c_blue = (0.86, 0.15, 0.15), (0.11, 0.30, 0.85)
    c_teal, c_amber, c_gray = (0.06, 0.46, 0.43), (0.85, 0.47, 0.02), (0.45, 0.50, 0.58)

    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1.2, 1.5, 1.2]})
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.07, right=0.96, hspace=0.20)
    fig.suptitle(f"组合对比：原「股债金」 vs 新建「标普500+纳指+黄金+国债」"
                 f"（{window_label} {sub_start[:7]} - {last_dt:%Y.%m}）",
                 fontsize=16, fontweight="bold", y=0.965)

    axes[0].plot(nav_old.index, nav_old.values, color=c_red, lw=2.3, label=OLD_NAME)
    axes[0].plot(nav_new.index, nav_new.values, color=c_blue, lw=2.3, label=NEW_NAME)
    axes[0].plot(bh_old.index, bh_old.values, color=c_gray, lw=1.3, ls="--",
                 label="原组合三资产等权买入持有 (518880/510500/510300)")
    axes[0].plot(bh_new.index, bh_new.values, color=c_amber, lw=1.3, ls=":",
                 label="新组合三资产等权买入持有 (518880/513500/513100)")
    axes[0].set_ylabel("累计净值 (初始=1.0)", fontsize=11)
    axes[0].set_title("面板 A：两组合累计净值对比（含单边5bp成本与T+1执行滞后，股票标的替换后其余规则完全一致）",
                      fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper left", ncol=2, fontsize=9.5)
    axes[0].grid(True, alpha=0.3)

    dd_old = (nav_old / nav_old.cummax() - 1.0) * 100.0
    dd_new = (nav_new / nav_new.cummax() - 1.0) * 100.0
    axes[1].fill_between(dd_old.index, dd_old.values, 0, color=c_red, alpha=0.18)
    axes[1].plot(dd_old.index, dd_old.values, color=c_red, lw=1.6,
                 label=f"原组合动态回撤 (最大 {dd_old.min():.2f}%)")
    axes[1].plot(dd_new.index, dd_new.values, color=c_blue, lw=1.6,
                 label=f"新组合动态回撤 (最大 {dd_new.min():.2f}%)")
    axes[1].set_ylabel("动态回撤 (%)", fontsize=11)
    axes[1].set_title("面板 B：两组合动态回撤对比", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower left", ncol=2, fontsize=9.5)
    axes[1].grid(True, alpha=0.3)

    w_new = new["w_exec"].loc[sub_start:] * 100.0
    axes[2].stackplot(w_new.index, w_new["518880.SH"], w_new["513500.SH"],
                      w_new["513100.SH"], w_new[BOND_CODE],
                      labels=["518880 黄金ETF", "513500 标普500ETF", "513100 纳指ETF", "511010 国债ETF (防守)"],
                      colors=[c_amber, c_blue, c_teal, (0.82, 0.86, 0.92)], alpha=0.88)
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("新组合权重 (%)", fontsize=11)
    axes[2].set_title("面板 C：新建组合动态仓位（空仓/减仓时退守 511010 国债ETF）", fontsize=12, fontweight="bold")
    axes[2].legend(loc="upper right", ncol=4, fontsize=9.5)
    axes[2].grid(True, alpha=0.3)

    risk_old = old["w_exec"].loc[sub_start:, OLD_RISK].sum(axis=1) * 100.0
    risk_new = new["w_exec"].loc[sub_start:, NEW_RISK].sum(axis=1) * 100.0
    axes[3].plot(risk_old.index, risk_old.values, color=c_red, lw=1.5, label="原组合风险资产总权重")
    axes[3].plot(risk_new.index, risk_new.values, color=c_blue, lw=1.5, label="新组合风险资产总权重")
    axes[3].set_ylim(0, 100)
    axes[3].set_ylabel("风险仓位 (%)", fontsize=11)
    axes[3].set_xlabel("交易日期", fontsize=11)
    axes[3].set_title("面板 D：两组合择时开仓强度对比（同一五规则信号，仅股票标的替换）",
                      fontsize=12, fontweight="bold")
    axes[3].legend(loc="upper right", ncol=2, fontsize=9.5)
    axes[3].grid(True, alpha=0.3)

    out = os.path.join(base_dir, out_name)
    plt.savefig(out, dpi=200)
    plt.close(fig)
    return out, nav_old, nav_new, dd_old, dd_new


def main(execution_date=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, "..", "04-数据获得", "daily_hfq"))
    prices, raw_prices, adj_factors = load_prices(data_dir)

    if execution_date is None:
        execution_date = prices.index[-1].strftime("%Y-%m-%d")
    execution_dt, signal_dt = bt.resolve_execution_context(prices, execution_date)
    last_dt = prices.index[-1]

    ind = bt.compute_indicators(prices)
    rets = ind["rets"].fillna(0.0)
    all_symbols = [s for s, _, _, _ in ETF_META]
    me_dates = bt.infer_month_end_dates(prices.index, execution_dt)
    sig_w_all, sig_r_all = bt.generate_five_rule_signals(prices, ind, all_symbols, month_end_dates=me_dates)

    old = combo_returns(prices, ind, rets, sig_w_all, OLD_RISK, OLD_BASE_CAP)
    new = combo_returns(prices, ind, rets, sig_w_all, NEW_RISK, NEW_BASE_CAP)

    # 校验：新脚本重算的原组合须与既有正式产出一致（防污染）
    ref_path = os.path.join(base_dir, "backtest_summary_metrics.csv")
    ref = pd.read_csv(ref_path)
    ref_row = ref[(ref["策略名称"].str.contains("股债金五规则论文择时组合", na=False)) &
                  (ref["区间名称"].str.contains("近三年样本", na=False))]
    ref_val = float(ref_row["累计收益率(%)"].iloc[0]) if len(ref_row) else None
    recent = ("近三年样本 (2023-01 至 %s)" % last_dt.strftime("%Y-%m"), "2023-01-03", last_dt.strftime("%Y-%m-%d"))
    old_recent = bt.calc_perf_metrics(old["ret"].loc[recent[1]:recent[2]], old["turn"].loc[recent[1]:recent[2]])
    print(f"[原组合一致性校验] 本脚本={old_recent['累计收益率(%)']}%  既有产出={ref_val}%")
    if ref_val is not None and abs(old_recent["累计收益率(%)"] - ref_val) > 0.02:
        raise RuntimeError("原组合结果被污染，停止生成对比产出")

    ytd_label = f"{last_dt.year}年YTD压力窗 ({last_dt.year}-01 至 {last_dt:%Y-%m})"
    ten_years_start = (last_dt - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
    ten_year_label = f"近十年样本 ({ten_years_start[:7]} 至 {last_dt:%Y-%m})"

    # 近十年起点有效性校验：所有风险标的的长周期指标须已预热完成，
    # 避免把「样本预热期」的 0 仓位计入对比区间，造成不公平对比。
    first_ten_dt = prices.index[prices.index >= pd.Timestamp(ten_years_start)][0]
    warmup = {}
    for sym in sorted(set(OLD_RISK + NEW_RISK)):
        warmup[sym] = all(not pd.isna(ind[k].loc[first_ten_dt, sym])
                          for k in ["ma20", "ma60", "ma200", "vol20", "q80", "vol60"])
    if not all(warmup.values()):
        bad = [s for s, ok in warmup.items() if not ok]
        raise RuntimeError(f"近十年起点 {first_ten_dt:%Y-%m-%d} 存在未预热指标：{bad}")
    print(f"[近十年区间校验] 起点={first_ten_dt:%Y-%m-%d}（近10年自然日起 {ten_years_start}），"
          f"{len(warmup)} 只风险标的 MA20/MA60/MA200/vol20/q80/vol60 全部已预热")

    periods = [
        (ten_year_label, ten_years_start, last_dt.strftime("%Y-%m-%d")),
        (recent[0], recent[1], recent[2]),
        ("共同全历史 (2014-01-15 起，新标的数据齐备)", "2014-01-15", last_dt.strftime("%Y-%m-%d")),
        ("FOF标准三年窗 (2023-10 至 2026-09)", "2023-10-01", "2026-09-30"),
        (ytd_label, f"{last_dt.year}-01-01", last_dt.strftime("%Y-%m-%d")),
    ]
    strategies = [
        ("原股债金组合", OLD_NAME, old),
        ("新建海外组合", NEW_NAME, new),
    ]
    rows = []
    for label, s_dt, e_dt in periods:
        for cat, sname, c in strategies:
            m = bt.calc_perf_metrics(c["ret"].loc[s_dt:e_dt], c["turn"].loc[s_dt:e_dt])
            rows.append({"区间名称": label, "策略分类": cat, "策略名称": sname, **m})
        for cat, sname, r in [
            ("基准", "原组合三资产等权买入持有 (518880/510500/510300)", rets[OLD_RISK].mean(axis=1)),
            ("基准", "新组合三资产等权买入持有 (518880/513500/513100)", rets[NEW_RISK].mean(axis=1)),
            ("基准", "5年国债ETF (511010.SH) 买入持有", rets[BOND_CODE]),
        ]:
            m = bt.calc_perf_metrics(r.loc[s_dt:e_dt])
            rows.append({"区间名称": label, "策略分类": cat, "策略名称": sname, **m})
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(os.path.join(base_dir, "portfolio_comparison_metrics.csv"),
                      index=False, encoding="utf-8-sig")

    sub_start = ten_years_start
    chart_path, nav_old, nav_new, dd_old, dd_new = make_chart(
        base_dir, last_dt, execution_dt, old, new, rets, sub_start,
        "portfolio_comparison_chart.png", "近十年")
    chart_3y_path, _, _, _, _ = make_chart(
        base_dir, last_dt, execution_dt, old, new, rets, "2023-01-03",
        "portfolio_comparison_chart_3y.png", "近三年")
    df_nav = pd.DataFrame({
        "date": nav_old.index,
        "原股债金组合净值": nav_old.values,
        "新建海外组合净值": nav_new.values,
        "原组合动态回撤(%)": dd_old.values,
        "新组合动态回撤(%)": dd_new.values,
        "原组合风险仓位(%)": old["w_exec"].loc[sub_start:, OLD_RISK].sum(axis=1).values * 100.0,
        "新组合风险仓位(%)": new["w_exec"].loc[sub_start:, NEW_RISK].sum(axis=1).values * 100.0,
    })
    df_nav.to_csv(os.path.join(base_dir, "portfolio_comparison_nav.csv"),
                  index=False, encoding="utf-8-sig")

    df_hold = holdings_audit(prices, raw_prices, adj_factors, ind, sig_w_all, sig_r_all,
                             new, execution_dt, signal_dt, last_dt)
    hold_path = os.path.join(base_dir, f"new_combo_holdings_audit_{execution_dt:%Y%m%d}.csv")
    df_hold.to_csv(hold_path, index=False, encoding="utf-8-sig")

    caus_old = causality_audit(prices, OLD_RISK, OLD_BASE_CAP, "原股债金组合")
    caus_new = causality_audit(prices, NEW_RISK, NEW_BASE_CAP, "新建海外组合")
    summary = {
        "execution_date": execution_dt.strftime("%Y-%m-%d"),
        "signal_date": signal_dt.strftime("%Y-%m-%d"),
        "data_end_date": last_dt.strftime("%Y-%m-%d"),
        "old_combo": {"name": OLD_NAME, "risk_codes": OLD_RISK, "bond": BOND_CODE, "base_cap": OLD_BASE_CAP},
        "new_combo": {"name": NEW_NAME, "risk_codes": NEW_RISK, "bond": BOND_CODE, "base_cap": NEW_BASE_CAP},
        "shared_rules": {"cost_rate": bt.COST_RATE, "buffer_thresh": bt.BUFFER_THRESH,
                         "risk_cap": 0.85, "execution": "T日收盘信号，T+1执行",
                         "signal": "五规则(MA20/MA60/MA200/vol20/q80/hwm60) + 月末门控"},
        "comparison_windows": {
            "primary": ten_year_label,
            "ten_years_start": ten_years_start,
            "ten_years_first_trade_date": first_ten_dt.strftime("%Y-%m-%d"),
            "warmup_all_passed": all(warmup.values()),
            "warmup_detail": warmup,
            "note": "近十年为主对比区间；另保留近三年、共同全历史、FOF三年窗、YTD 四个窗口。",
            "chart_files": ["portfolio_comparison_chart.png (近十年全览)",
                            "portfolio_comparison_chart_3y.png (近三年细节)"],
        },
        "data_notes": [
            "513500.SH 2022-03-29 份额折算1:2；513100.SH 2022-01-13 份额折算1:5；两者历史无分红",
            "后复权价来自腾讯 fqkline hfq 分段拼接，与新浪 quotes_service 前复权逐日核验价差为0",
            "跨境ETF场内价格含溢价，与基金净值存在差异；成交额为估算、换手率置0，均不参与回测",
            "共同全历史区间起点取 2014-01-15（513500 上市首日），此前新标的无数据",
        ],
        "old_combo_recheck": {"script_recent_total_return_pct": old_recent["累计收益率(%)"],
                              "existing_output_pct": ref_val, "consistent": ref_val is None or abs(old_recent["累计收益率(%)"] - ref_val) <= 0.02},
        "causality": {"old": caus_old, "new": caus_new},
        "head_to_head": {},
    }
    for label, s_dt, e_dt in periods:
        mo = bt.calc_perf_metrics(old["ret"].loc[s_dt:e_dt], old["turn"].loc[s_dt:e_dt])
        mn = bt.calc_perf_metrics(new["ret"].loc[s_dt:e_dt], new["turn"].loc[s_dt:e_dt])
        summary["head_to_head"][label] = {"old": mo, "new": mn,
                                          "return_diff_pp": round(mn["累计收益率(%)"] - mo["累计收益率(%)"], 2),
                                          "mdd_diff_pp": round(mn["最大回撤(%)"] - mo["最大回撤(%)"], 2),
                                          "sharpe_diff": round(mn["夏普比率"] - mo["夏普比率"], 3)}
    with open(os.path.join(base_dir, "portfolio_comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== 两组合并列绩效（全部区间，近十年为首位主区间） =====")
    show = df_metrics[df_metrics["策略分类"].isin(["原股债金组合", "新建海外组合"])]
    print(show[["区间名称", "策略分类", "累计收益率(%)", "年化收益率(%)", "夏普比率",
                "最大回撤(%)", "卡玛比率", "年化单边换手率(%)", "样本交易日数"]].to_string(index=False))
    print("\n===== 新组合今日实际持仓 (T+1) =====")
    print(df_hold[["标的代码", "标的名称", "新组合今日实际权重(%)", "今日权重变动(百分点)", "今日调仓方向", "信号日触发规则"]].to_string(index=False))
    print(f"\n因果检验 原组合={caus_old['all_checks_passed']} 新组合={caus_new['all_checks_passed']}")
    print(f"产出: {chart_path} (近十年全览)")
    print(f"      {chart_3y_path} (近三年细节)")
    print(f"      {hold_path}")
    if not (caus_old["all_checks_passed"] and caus_new["all_checks_passed"]):
        raise RuntimeError("因果检验未通过")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execution-date", default=None, help="实际持仓执行日期 YYYY-MM-DD；默认取数据最后一天")
    args = ap.parse_args()
    main(args.execution_date)
