from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


DATE_COL_CANDIDATES = ["datetime", "date", "trade_date", "日期", "交易日期", "time", "时间"]
HFQ_CLOSE_CANDIDATES = ["close", "close_hfq", "hfq_close", "后复权收盘价", "后复权价", "复权收盘价", "收盘价", "收盘"]
RAW_CLOSE_CANDIDATES = ["close_raw", "raw_close", "spot_close", "不复权收盘价", "实盘收盘价", "除权收盘价", "最新价"]
OPEN_CANDIDATES = ["open", "open_hfq", "开盘价", "开盘", "后复权开盘价"]
HIGH_CANDIDATES = ["high", "high_hfq", "最高价", "最高", "后复权最高价"]
LOW_CANDIDATES = ["low", "low_hfq", "最低价", "最低", "后复权最低价"]
TURNOVER_CANDIDATES = ["turnover", "amount", "成交额", "成交金额", "total_turnover"]
PCT_CHG_CANDIDATES = ["pct_chg", "pct_change", "涨跌幅", "涨跌幅(%)"]
ADJ_FACTOR_CANDIDATES = ["adj_factor", "hfq_factor", "复权因子"]


def read_csv_auto_encoding(csv_path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def to_clean_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.strip().lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def resolve_etf_csv_path(data_dir: Path, ticker: str) -> Path:
    code6 = ticker.split(".")[0]
    exch = ticker.split(".")[1].upper() if "." in ticker else ""
    patterns = [
        f"{code6}_{exch}_daily_hfq.csv",
        f"{code6}_{exch}.csv",
        f"{ticker}.csv",
        f"{code6}.csv",
        f"{exch.lower()}{code6}.csv",
        f"{exch}{code6}.csv",
    ]
    for pat in patterns:
        cand = data_dir / pat
        if cand.exists():
            return cand
    matches = sorted(data_dir.glob(f"*{code6}*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"未在数据目录 {data_dir} 中找到标的 {ticker} 的 CSV 行情文件（已尝试 {code6}_*.csv / {ticker}.csv）。"
    )


def load_single_etf_csv(csv_path: Path, turnover_unit: str = "yuan") -> pd.DataFrame:
    df = read_csv_auto_encoding(csv_path)
    cols = list(df.columns)

    dt_col = _pick_col(cols, DATE_COL_CANDIDATES) or cols[0]
    raw_dt = df[dt_col].astype(str).str.strip()
    if raw_dt.str.match(r"^\d{8}$").all():
        dt_idx = pd.to_datetime(raw_dt, format="%Y%m%d")
    else:
        dt_idx = pd.to_datetime(raw_dt)

    df = df.assign(_dt=dt_idx).sort_values("_dt").drop_duplicates("_dt").set_index("_dt")

    c_col = _pick_col(cols, HFQ_CLOSE_CANDIDATES)
    if c_col is None:
        raise ValueError(f"文件 {csv_path.name} 缺少收盘价字段（支持 close / 后复权收盘价 / 收盘价 等）。")

    close_s = to_clean_numeric(df[c_col])
    raw_c_col = _pick_col(cols, RAW_CLOSE_CANDIDATES)
    adj_col = _pick_col(cols, ADJ_FACTOR_CANDIDATES)

    if raw_c_col is not None:
        close_raw_s = to_clean_numeric(df[raw_c_col])
    elif adj_col is not None:
        adj_s = to_clean_numeric(df[adj_col])
        close_raw_s = close_s.copy()
        close_s = close_s * adj_s
    else:
        close_raw_s = close_s.copy()

    o_col = _pick_col(cols, OPEN_CANDIDATES)
    h_col = _pick_col(cols, HIGH_CANDIDATES)
    l_col = _pick_col(cols, LOW_CANDIDATES)
    t_col = _pick_col(cols, TURNOVER_CANDIDATES)
    p_col = _pick_col(cols, PCT_CHG_CANDIDATES)

    open_s = to_clean_numeric(df[o_col]) if o_col else close_s.copy()
    high_s = to_clean_numeric(df[h_col]) if h_col else close_s.copy()
    low_s = to_clean_numeric(df[l_col]) if l_col else close_s.copy()

    if t_col is not None:
        turn_s = to_clean_numeric(df[t_col])
        mult = 1000.0 if turnover_unit == "qian_yuan" else (10000.0 if turnover_unit == "wan_yuan" else 1.0)
        turn_s = turn_s * mult
    else:
        turn_s = pd.Series(np.nan, index=df.index, dtype=float)

    if p_col is not None:
        pct_s = to_clean_numeric(df[p_col])
    else:
        pct_s = close_s.pct_change(fill_method=None) * 100.0

    return pd.DataFrame({
        "open": open_s,
        "high": high_s,
        "low": low_s,
        "close": close_s,
        "close_raw": close_raw_s,
        "turnover": turn_s,
        "pct_chg": pct_s,
    }, index=df.index)


def load_domestic_etf_panel(
    data_dir: Path,
    tickers: List[str],
    as_of_date: Optional[str] = None,
    turnover_unit: str = "yuan",
) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    per_tk = {}
    last_dates = {}
    for tk in tickers:
        fp = resolve_etf_csv_path(data_dir, tk)
        sub = load_single_etf_csv(fp, turnover_unit=turnover_unit)
        if as_of_date is not None:
            sub = sub.loc[:pd.Timestamp(as_of_date)]
        if len(sub) == 0:
            raise ValueError(f"标的 {tk} 在截止日期 {as_of_date} 前无有效行情记录。")
        per_tk[tk] = sub
        last_dates[tk] = sub.index[-1]

    panel = {}
    for field in ["close", "close_raw", "open", "high", "low", "turnover", "pct_chg"]:
        df_f = pd.DataFrame({tk: per_tk[tk][field] for tk in tickers}).sort_index()
        if field in ("close", "close_raw", "open", "high", "low"):
            df_f = df_f.ffill()
        panel[field] = df_f
    panel["last_dates"] = pd.Series(last_dates)
    return panel


def validate_live_panel_hygiene(
    panel: Dict[str, pd.DataFrame],
    required_tickers: List[str],
    allow_stale: bool = False,
    min_warmup_bars: int = 252,
) -> Tuple[pd.Timestamp, List[str]]:
    warnings_list = []
    last_dts = panel["last_dates"][required_tickers]
    max_dt = last_dts.max()
    lagging = last_dts[last_dts < max_dt]
    if len(lagging) > 0:
        msg = (
            f"STALE_DATA_ERROR: 最新交易日不一致！最新日期为 {max_dt.strftime('%Y-%m-%d')}，"
            f"但以下标的停留在旧日期：{ {k: v.strftime('%Y-%m-%d') for k, v in lagging.items()} }。"
            f"请先补齐当日行情后再生成实盘指令（如确系全天停牌，可加 --allow-stale 参数放行）。"
        )
        if not allow_stale:
            raise RuntimeError(msg)
        warnings_list.append(msg)

    close_df = panel["close"][required_tickers]
    if len(close_df) < min_warmup_bars:
        raise RuntimeError(
            f"INSUFFICIENT_WARMUP_ERROR: 当前历史总交易日数仅 {len(close_df)} 天，少于五规则 MA200 与分位数所需的 {min_warmup_bars} 天预热期！"
        )

    latest_rets = close_df.pct_change(fill_method=None).iloc[-1].abs()
    abnormal = latest_rets[latest_rets > 0.22]
    if len(abnormal) > 0:
        warnings_list.append(
            f"EXTREME_RETURN_WARNING: 当日发现单日复权涨跌幅超 22% 的标的 {abnormal.round(4).to_dict()}，请核实是否发生份额折算未复权！"
        )

    return max_dt, warnings_list
