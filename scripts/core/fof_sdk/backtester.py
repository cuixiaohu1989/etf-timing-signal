from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .signals import compute_five_rule_signals


def build_course1_true_a0(prices: pd.DataFrame) -> pd.DataFrame:
    risk_codes = ["518880.SH", "510500.SH", "510300.SH"]
    bond_code = "511010.SH"
    cols = risk_codes + [bond_code]
    idx = pd.DatetimeIndex(prices.index)
    month_ends = idx.to_series().groupby(idx.to_period("M")).tail(1).index
    if not idx.empty:
        last_dt = idx[-1]
        bme = (last_dt + pd.offsets.BMonthEnd(0)).normalize()
        if last_dt.normalize() < bme:
            month_ends = month_ends[month_ends != last_dt]

    sig_w, _, _ = compute_five_rule_signals(prices, pd.DatetimeIndex(month_ends), tickers=risk_codes)
    rets = prices[cols].pct_change(fill_method=None)
    vol60 = rets[risk_codes].rolling(60, min_periods=20).std(ddof=1) * np.sqrt(252.0) * 100.0

    base_cap = np.array([0.35, 0.30, 0.30], dtype=float)
    v60_arr = vol60.values
    sig_arr = sig_w[risk_codes].values
    n_steps = len(idx)

    w_out = np.zeros((n_steps, 4), dtype=float)
    curr = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    for i in range(n_steps):
        v_row = v60_arr[i]
        if np.isnan(v_row).any():
            des = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        else:
            inv_v = np.clip(10.0 / np.maximum(v_row, 8.0), 0.0, 1.2)
            raw_r = base_cap * sig_arr[i] * inv_v
            tot_r = float(raw_r.sum())
            if tot_r > 0.85:
                raw_r = raw_r * (0.85 / tot_r)
                tot_r = 0.85
            des = np.array([raw_r[0], raw_r[1], raw_r[2], max(0.0, 1.0 - tot_r)], dtype=float)

        risk_des = float(des[:3].sum())
        risk_curr = float(curr[:3].sum())
        half_l1 = 0.5 * float(np.abs(des - curr).sum())
        if risk_des < risk_curr - 1e-6:
            curr = des.copy()
        elif half_l1 >= 0.04:
            curr = des.copy()
        w_out[i] = curr

    return pd.DataFrame(w_out, index=idx, columns=cols)


def calc_perf_metrics(
    ret_s: pd.Series,
    turn_s: Optional[pd.Series] = None,
    ann_factor: float = 252.0,
) -> dict:
    r = ret_s.dropna()
    n = len(r)
    if n == 0:
        return {}
    n_yr = n / ann_factor
    cum = float((1.0 + r).prod() - 1.0) * 100.0
    cagr = float((1.0 + r).prod() ** (1.0 / n_yr) - 1.0) * 100.0
    vol = float(r.std(ddof=1) * np.sqrt(ann_factor)) * 100.0 if n >= 2 else 0.0
    mean_ann = float(r.mean() * ann_factor) * 100.0
    sharpe = mean_ann / vol if vol > 1e-9 else 0.0
    neg = r.clip(upper=0.0)
    dvol = float(np.sqrt(np.mean(neg ** 2) * ann_factor)) * 100.0
    sortino = mean_ann / dvol if dvol > 1e-9 else 0.0
    nav = (1.0 + r).cumprod()
    mdd = float((nav / nav.cummax() - 1.0).min()) * 100.0
    calmar = cagr / abs(mdd) if abs(mdd) > 1e-9 else 0.0
    if turn_s is not None:
        t_aligned = turn_s.reindex(r.index).fillna(0.0)
        one_way_turn = float(t_aligned.mean() * ann_factor * 0.5) * 100.0
    else:
        one_way_turn = 0.0
    return {
        "累计收益率(%)": round(cum, 2),
        "年化收益率(%)": round(cagr, 2),
        "年化波动率(%)": round(vol, 2),
        "夏普比率": round(sharpe, 3),
        "索提诺比率": round(sortino, 3),
        "最大回撤(%)": round(mdd, 2),
        "卡玛比率": round(calmar, 2),
        "年化单边换手率(%)": round(one_way_turn, 2),
        "样本交易日数": int(n),
    }


def evaluate_t1_portfolio(
    w_raw: pd.DataFrame,
    rets_clean: pd.DataFrame,
    defensive_anchor: Optional[str] = None,
    cost_one_way: float = 0.0005,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    cols = list(w_raw.columns)
    w_exec = w_raw.shift(1).copy()
    w_exec.iloc[0] = 0.0
    anchor = defensive_anchor if (defensive_anchor and defensive_anchor in cols) else cols[-1]
    w_exec.iloc[0, cols.index(anchor)] = 1.0

    turn_l1 = w_exec.diff().abs().sum(axis=1).fillna(0.0)
    gross_ret = (w_exec * rets_clean[cols].reindex(w_exec.index).fillna(0.0)).sum(axis=1)
    net_ret = gross_ret - turn_l1 * cost_one_way
    return w_exec, net_ret, turn_l1


def evaluate_execution_modes(
    w_raw: pd.DataFrame,
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    defensive_cols: List[str],
    cost_rate: float = 0.0005,
) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    cols = list(w_raw.columns)
    c_sub = close_df[cols]
    o_sub = open_df[cols].copy()
    for bcol in defensive_cols:
        if bcol in o_sub.columns:
            prev_c = c_sub[bcol].shift(1)
            o_sub[bcol] = o_sub[bcol].clip(lower=prev_c * 0.996, upper=prev_c * 1.004)

    rets_cc = c_sub.pct_change(fill_method=None).fillna(0.0)
    r_ov = (o_sub / c_sub.shift(1) - 1.0).fillna(0.0)
    r_id = (c_sub / o_sub - 1.0).fillna(0.0)

    anchor = defensive_cols[0] if defensive_cols[0] in cols else cols[-1]
    w_t1 = w_raw.shift(1).fillna(0.0)
    w_t1.iloc[0] = 0.0
    w_t1.iloc[0, cols.index(anchor)] = 1.0

    w_t2 = w_t1.shift(1).fillna(0.0)
    w_t2.iloc[0] = 0.0
    w_t2.iloc[0, cols.index(anchor)] = 1.0

    turn_s1 = w_t1.diff().abs().sum(axis=1).fillna(0.0)
    turn_op = (w_t1 - w_t2).abs().sum(axis=1)
    turn_s3 = w_t2.diff().abs().sum(axis=1).fillna(0.0)

    ret_s1 = (w_t1 * rets_cc).sum(axis=1) - turn_s1 * cost_rate
    ret_s2 = (1.0 + (w_t2 * r_ov).sum(axis=1)) * (1.0 + (w_t1 * r_id).sum(axis=1)) - 1.0 - turn_op * cost_rate
    ret_s3 = (w_t2 * rets_cc).sum(axis=1) - turn_s3 * cost_rate

    return {
        "SOP-1: T日14:55快照→14:57尾盘竞价成交": (ret_s1, turn_s1),
        "SOP-2: T日盘后出单→T+1日09:31限价开盘成交": (ret_s2, turn_op),
        "SOP-3: T日盘后出单→T+1日14:57延后1整天成交": (ret_s3, turn_s3),
    }


def simulate_100share_lot_ledger(
    w_raw: pd.DataFrame,
    prices_hfq: pd.DataFrame,
    close_raw_df: pd.DataFrame,
    initial_capital_rmb: float,
    primary_bond_col: str = "511010.SH",
    start_dt: str = "2023-01-03",
    end_dt: str = "2026-11-16",
    comm_rate: float = 0.0001,
    slip_rate: float = 0.0004,
    cash_repo_rate: float = 0.015,
) -> dict:
    sub_idx = prices_hfq.loc[start_dt:end_dt].index
    cols = list(w_raw.columns)
    w_exec = w_raw.shift(1).reindex(sub_idx).ffill().fillna(0.0)
    hfq_rets = prices_hfq[cols].pct_change(fill_method=None).fillna(0.0).reindex(sub_idx)

    holdings_val = np.zeros(len(cols), dtype=float)
    init_dt = prices_hfq.index[max(0, prices_hfq.index.get_loc(sub_idx[0]) - 1)]
    init_w = w_exec.loc[sub_idx[0]].values
    init_lots_px = close_raw_df[cols].ffill().loc[init_dt].values * 100.0
    for j in range(len(cols)):
        lc = init_lots_px[j]
        if not np.isnan(lc) and lc > 0 and (initial_capital_rmb * 0.9992 * init_w[j]) >= lc:
            holdings_val[j] = np.floor((initial_capital_rmb * 0.9992 * init_w[j]) / lc) * lc
    b5_j0 = cols.index(primary_bond_col) if primary_bond_col in cols else -1
    if b5_j0 >= 0 and init_lots_px[b5_j0] > 0:
        rem0 = initial_capital_rmb * 0.9992 - holdings_val.sum()
        if rem0 >= init_lots_px[b5_j0]:
            holdings_val[b5_j0] += np.floor(rem0 / init_lots_px[b5_j0]) * init_lots_px[b5_j0]
    cash = float(initial_capital_rmb - holdings_val.sum())
    prev_target = init_w.copy()

    nav_series = []
    cash_ratio_list = []
    for dt in sub_idx:
        w_t = w_exec.loc[dt].values
        tot_eq_pre = float(holdings_val.sum() + cash)
        if np.abs(w_t - prev_target).sum() > 1e-8:
            prev_dt = prices_hfq.index[max(0, prices_hfq.index.get_loc(dt) - 1)]
            lot_costs = close_raw_df[cols].ffill().loc[prev_dt].values * 100.0
            investable = tot_eq_pre * 0.9992
            desired_vals = investable * w_t
            new_vals = np.zeros(len(cols), dtype=float)
            for j in range(len(cols)):
                lc = lot_costs[j]
                if not np.isnan(lc) and lc > 0 and desired_vals[j] >= lc:
                    new_vals[j] = np.floor(desired_vals[j] / lc) * lc
            if b5_j0 >= 0 and not np.isnan(lot_costs[b5_j0]) and lot_costs[b5_j0] > 0:
                leftover = investable - new_vals.sum()
                if leftover >= lot_costs[b5_j0]:
                    new_vals[b5_j0] += np.floor(leftover / lot_costs[b5_j0]) * lot_costs[b5_j0]
            total_fee = 0.0
            for j in range(len(cols)):
                trade_rmb = abs(new_vals[j] - holdings_val[j])
                if trade_rmb > 1.0:
                    total_fee += max(5.0, trade_rmb * comm_rate) + trade_rmb * slip_rate
            cash = tot_eq_pre - new_vals.sum() - total_fee
            holdings_val = new_vals
            prev_target = w_t.copy()

        holdings_val = holdings_val * (1.0 + hfq_rets.loc[dt].values)
        cash *= (1.0 + cash_repo_rate / 252.0)
        end_eq = float(holdings_val.sum() + cash)
        nav_series.append(end_eq / initial_capital_rmb)
        cash_ratio_list.append(cash / end_eq * 100.0)

    nav_s = pd.Series(nav_series, index=sub_idx)
    ret_s = nav_s.pct_change().fillna(nav_s.iloc[0] - 1.0)
    m = calc_perf_metrics(ret_s)
    return {
        "账户初始资金(万元)": int(round(initial_capital_rmb / 1e4, 0)),
        "实盘整手年化(%)": m["年化收益率(%)"],
        "实盘整手夏普": m["夏普比率"],
        "实盘整手最大回撤(%)": m["最大回撤(%)"],
        "日均零头现金占比(%)": round(float(np.mean(cash_ratio_list)), 2),
    }
