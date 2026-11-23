import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


BUILTIN_CN_MONTHEND_HOLIDAYS = [
    "2012-04-30", "2013-04-29", "2013-04-30", "2014-01-31",
    "2017-01-27", "2017-01-30", "2017-01-31", "2018-04-30", "2018-12-31",
    "2020-01-24", "2020-01-27", "2020-01-28", "2020-01-29", "2020-01-30", "2020-01-31",
    "2022-01-31", "2023-09-29",
    "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2027-02-08", "2028-01-26", "2028-01-27", "2028-01-28", "2028-01-31",
]
DEFAULT_MAX_VERIFIED_YEAR = 2028


def load_holiday_config(holiday_json: Optional[Path] = None) -> Tuple[set, int]:
    dates = list(BUILTIN_CN_MONTHEND_HOLIDAYS)
    max_yr = DEFAULT_MAX_VERIFIED_YEAR
    if holiday_json is not None and Path(holiday_json).exists():
        with open(holiday_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        dates.extend(cfg.get("cn_monthend_holidays", []))
        max_yr = max(max_yr, int(cfg.get("max_verified_year", max_yr)))
    return set(pd.to_datetime(dates)), max_yr


def infer_cn_month_end_dates(
    index: pd.DatetimeIndex,
    as_of_date=None,
    holiday_json: Optional[Path] = None,
) -> pd.DatetimeIndex:
    hol_set, max_yr = load_holiday_config(holiday_json)
    idx = pd.DatetimeIndex(index).sort_values()
    if as_of_date is not None:
        idx = idx[idx <= pd.Timestamp(as_of_date)]
    if len(idx) == 0:
        return idx
    last_dt = idx[-1]
    if last_dt.year > max_yr:
        raise RuntimeError(
            f"CALENDAR_OUT_OF_RANGE_ERROR: 当前数据日期 {last_dt.strftime('%Y-%m-%d')} 超出节假日表已核验年份上限 ({max_yr})！"
            f"请先在 config/cn_holidays_2012_2028.json 中补充当年春节/国庆月末休市日并调高 max_verified_year。"
        )
    s = pd.Series(idx, index=idx)
    me = s.groupby(idx.to_period("M")).tail(1)
    cal_end = last_dt + pd.offsets.MonthEnd(0)
    rem_b = pd.date_range(last_dt + pd.Timedelta(days=1), cal_end, freq="B")
    open_rem = [d for d in rem_b if d not in hol_set]
    if len(open_rem) > 0:
        me = me.iloc[:-1]
    return pd.DatetimeIndex(me.values)


def compute_five_rule_signals(
    prices: pd.DataFrame,
    month_end_dates: pd.DatetimeIndex,
    tickers: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    cols = list(tickers) if tickers is not None else list(prices.columns)
    px = prices[cols]
    rets = px.pct_change(fill_method=None)

    ma20 = px.rolling(20, min_periods=20).mean()
    ma60 = px.rolling(60, min_periods=60).mean()
    ma200 = px.rolling(200, min_periods=200).mean()
    vol20 = rets.rolling(20, min_periods=20).std() * np.sqrt(252.0) * 100.0
    vol60 = rets.rolling(60, min_periods=30).std() * np.sqrt(252.0) * 100.0
    q80 = vol20.rolling(500, min_periods=120).quantile(0.80)
    vol20_lag10 = vol20.shift(10)
    hwm60 = px.rolling(60, min_periods=20).max()

    me_set = set(pd.DatetimeIndex(month_end_dates).intersection(px.index))
    sig_df = pd.DataFrame(0.0, index=px.index, columns=cols)
    lbl_df = pd.DataFrame("空仓观望(0%)", index=px.index, columns=cols)

    for sym in cols:
        w_curr = 0.0
        cur_lbl = "预热期空仓(0%)"
        w_list = []
        l_list = []
        p_s = px[sym].values
        m20_s = ma20[sym].values
        m60_s = ma60[sym].values
        m200_s = ma200[sym].values
        v20_s = vol20[sym].values
        q80_s = q80[sym].values
        vl10_s = vol20_lag10[sym].values
        h60_s = hwm60[sym].values

        for i, dt in enumerate(px.index):
            if np.isnan(m200_s[i]) or np.isnan(v20_s[i]) or np.isnan(q80_s[i]):
                w_list.append(0.0)
                l_list.append("预热期空仓(0%)")
                continue

            is_me = dt in me_set
            zero_dd = p_s[i] >= 0.995 * h60_s[i]
            mild_hv = (v20_s[i] > q80_s[i]) and (v20_s[i] <= 1.15 * q80_s[i])

            if p_s[i] < m200_s[i]:
                if (
                    is_me
                    and (m20_s[i] > m60_s[i])
                    and (p_s[i] >= m60_s[i])
                    and (v20_s[i] <= q80_s[i])
                    and (not np.isnan(vl10_s[i]) and v20_s[i] < vl10_s[i])
                ):
                    w_curr = 0.5
                    cur_lbl = "规则5:熊市末期左侧试多(50%)"
                elif w_curr == 0.5 and (m20_s[i] > m60_s[i]) and (p_s[i] >= 0.985 * m60_s[i]) and (v20_s[i] <= q80_s[i]):
                    w_curr = 0.5
                    cur_lbl = "规则5:左侧试多持有(50%)"
                else:
                    w_curr = 0.0
                    cur_lbl = "规则4:年线下方空仓避险(0%)"
            else:
                if (v20_s[i] > q80_s[i]) and (p_s[i] < m60_s[i]):
                    w_curr = 0.0
                    cur_lbl = "规则4:破60日线+高波清仓(0%)"
                elif p_s[i] < m60_s[i]:
                    if w_curr > 0.0:
                        w_curr = 0.5
                        cur_lbl = "规则3:破60日线低波减半(50%)"
                    else:
                        w_curr = 0.0
                        cur_lbl = "空仓等待月末突破(0%)"
                elif v20_s[i] > q80_s[i]:
                    if w_curr > 0.0:
                        if zero_dd and mild_hv:
                            w_curr = 0.75
                            cur_lbl = "规则2:强势新高轻度高波(75%)"
                        else:
                            w_curr = 0.5
                            cur_lbl = "规则2:高波风控减半(50%)"
                    else:
                        w_curr = 0.0
                        cur_lbl = "高波期暂缓建仓(0%)"
                else:
                    if is_me and (m20_s[i] >= m60_s[i]):
                        w_curr = 1.0
                        cur_lbl = "规则1:月末多头满信号(100%)"
                    elif w_curr == 1.0:
                        cur_lbl = "规则1:多头趋势持有(100%)"

            w_list.append(w_curr)
            l_list.append(cur_lbl)

        sig_df[sym] = w_list
        lbl_df[sym] = l_list

    indicators = {"rets": rets, "ma20": ma20, "ma60": ma60, "ma200": ma200, "vol20": vol20, "vol60": vol60, "q80": q80}
    return sig_df, lbl_df, indicators
