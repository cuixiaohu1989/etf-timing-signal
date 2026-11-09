# -*- coding: utf-8 -*-
"""云端每日增量取数：14 只 ETF 后复权日线，幂等、可补跑、原子写回。

与本地 04-数据获得/update_daily_incremental.py 的差异（云端必需）：
  1. 标的池补齐到 14 只（含 513500 / 513100 两只跨境 ETF）
  2. 用新浪返回的真实交易日列表推断交易日历，从本地末日逐日补到最新已收盘交易日，
     不用 BDay，因此春节/国庆等长假后不会算错，也不会在非交易日误报失败
  3. 跨境 ETF 跳过成交量/换手率/成交额口径核验（这些字段不参与策略计算）
  4. 支持 QUOTE_STRICT=0 单源降级（预检发现腾讯不通时），并在 status 中标注 data_quality
  5. 每次运行写 site/update_status.json，供页面做新鲜度与数据质量判断

严格约束（保持与本地一致）：
  - 只接纳已收盘（>=15:00）且双源价格一致的完整日线
  - 新浪前一交易日收盘必须与本地 close_raw 一致，否则判定除权事件并中止该标的
  - 14 只全部核验通过后才原子写回，任一失败整批不更新
"""
import os
import sys
import json
import time
import shutil
import hashlib
import tempfile
import argparse
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT, "data", "daily_hfq")
SUMMARY_PATH = os.path.join(ROOT, "data", "etf_data_summary.csv")
STATUS_PATH = os.path.join(ROOT, "site", "update_status.json")
SHANGHAI = ZoneInfo("Asia/Shanghai")

# code, exch, name, category, cross_border
ETF_POOL = [
    ("510050", "sh", "上证50ETF", "宽基股票", False),
    ("510300", "sh", "沪深300ETF", "宽基股票", False),
    ("510500", "sh", "中证500ETF", "宽基股票", False),
    ("512100", "sh", "中证1000ETF", "宽基股票", False),
    ("159915", "sz", "创业板ETF", "宽基股票", False),
    ("518880", "sh", "黄金ETF", "贵金属", False),
    ("159934", "sz", "黄金ETF易方达", "贵金属", False),
    ("159980", "sz", "有色ETF", "商品期货ETF", False),
    ("159981", "sz", "能源化工ETF", "商品期货ETF", False),
    ("511010", "sh", "国债ETF", "利率债", False),
    ("511260", "sh", "十年国债ETF", "利率债", False),
    ("511360", "sh", "短融ETF", "货币短债", False),
    ("513500", "sh", "标普500ETF博时", "海外股票", True),
    ("513100", "sh", "纳指ETF国泰", "海外股票", True),
]

SINA_URL = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            "CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen=30")
TX_QT_URL = "https://qt.gtimg.cn/q={sym}"
TX_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param={sym},day,,,30"
CSV_COLS = ["datetime", "open", "high", "low", "close", "volume", "turnover",
            "amplitude", "pct_chg", "chg", "turnover_rate", "close_raw",
            "close_qfq", "adj_factor"]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://gu.qq.com/",
})


def now_sh():
    return datetime.now(SHANGHAI)


def csv_path(code, exch):
    return os.path.join(DATA_DIR, f"{code}_{exch.upper()}_daily_hfq.csv")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url, retries=3, timeout=15, gbk=False):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            if gbk:
                r.encoding = "gbk"
            return r
        except Exception as exc:
            last = exc
            time.sleep(0.8 * (i + 1))
    raise last


def close_enough(a, b, tol=0.002):
    return abs(float(a) - float(b)) <= tol


def fetch_sina(sym):
    arr = json.loads(http_get(SINA_URL.format(sym=sym), timeout=15).text.strip())
    if not isinstance(arr, list) or len(arr) < 3:
        raise RuntimeError(f"{sym} 新浪日线数据不足")
    return {str(r["day"]): r for r in arr}, [str(r["day"]) for r in arr]


def fetch_tencent_qt(sym):
    p = http_get(TX_QT_URL.format(sym=sym), gbk=True, timeout=12).text.split("~")
    if len(p) < 45:
        raise RuntimeError(f"{sym} 腾讯 qt 字段不足（{len(p)}）")
    return p


def fetch_tencent_calendar(sym="sh510300"):
    """腾讯不复权 K 线作为交易日历的备用来源。"""
    j = http_get(TX_KLINE_URL.format(sym=sym), timeout=15).json()
    d = next(iter((j.get("data") or {}).values()), {})
    return [row[0] for row in (d.get("day") or [])]


def norm_day(d):
    """腾讯 K 线偶尔返回 YYYYMMDD，统一成 YYYY-MM-DD。"""
    s = str(d).strip()[:10]
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def get_trade_days():
    """返回 (真实交易日列表, 日历来源, 源错误列表)。

    2026-11-09 事故修复：原来「新浪成功就直接返回」，而新浪 quotes_service
    在盘中不生成当日日线，单源日历会把「还没收盘」误判成「已是最新」。
    现在同时取新浪与腾讯两套日历取并集，任一源先更新都能驱动补数。
    """
    errors, all_days, sources = [], [], []
    try:
        _, days = fetch_sina("sh510300")
        if days:
            all_days.extend(days)
            sources.append("sina_quotes_service")
    except Exception as exc:
        errors.append(f"sina: {type(exc).__name__}: {exc}")
    try:
        days = fetch_tencent_calendar()
        if days:
            all_days.extend(days)
            sources.append("tencent_kline")
    except Exception as exc:
        errors.append(f"tencent_kline: {type(exc).__name__}: {exc}")
    if not all_days:
        raise RuntimeError("无法获取交易日历；" + " | ".join(errors))
    days = sorted({norm_day(d) for d in all_days if norm_day(d)})
    return days, "+".join(sources), errors


def market_state_today(today_str):
    """用腾讯实时行情时间戳区分「已收盘 / 盘中 / 休市」，返回四态之一。

    休市日行情时间戳会停在上一个交易日，据此可把法定节假日与
    「行情日历滞后」区分开，避免节假日误报。行情源不可用时返回 None。
    """
    try:
        qt = fetch_tencent_qt("sh510300")
    except Exception:
        return None
    ts = str(qt[30])
    if not ts.startswith(today_str.replace("-", "")):
        return "holiday"
    return "closed" if (len(ts) < 12 or ts[8:12] >= "1500") else "open"


def build_row(code, day, prev, local_prev, target_date, qt, strict, cross_border):
    raw_open = float(day["open"])
    raw_high = float(day["high"])
    raw_low = float(day["low"])
    raw_close = float(day["close"])
    volume_shares = float(day["volume"])

    if not (raw_low <= min(raw_open, raw_close) and max(raw_open, raw_close) <= raw_high):
        raise RuntimeError(f"{code} OHLC 关系不合理")

    prev_close_raw = float(prev["close"])
    if not close_enough(prev_close_raw, float(local_prev["close_raw"])):
        raise RuntimeError(
            f"{code} 新浪前收 {prev_close_raw} 与本地前收 {local_prev['close_raw']} 不一致，"
            "疑似除权/份额折算，需人工核验复权因子"
        )

    factor = float(local_prev["adj_factor"])
    prev_close_hfq = float(local_prev["close"])
    hfq_open = round(raw_open * factor, 3)
    hfq_high = round(raw_high * factor, 3)
    hfq_low = round(raw_low * factor, 3)
    hfq_close = round(raw_close * factor, 3)

    volume_lots = int(round(volume_shares / 100.0))
    turnover, turnover_rate = 0.0, 0.0
    if strict and qt is not None and not cross_border:
        if not close_enough(volume_lots * 100.0, volume_shares, tol=100.0):
            raise RuntimeError(f"{code} 成交量换算异常")
        if not close_enough(volume_lots, float(qt[36]), tol=1.0):
            raise RuntimeError(f"{code} 成交量与腾讯不一致：{volume_lots} vs {qt[36]}")
        turnover = float(qt[37]) * 10000.0
        turnover_rate = float(qt[38])

    amplitude = round((raw_high - raw_low) / prev_close_raw * 100.0, 2)
    pct_chg = round((hfq_close / prev_close_hfq - 1.0) * 100.0, 2)
    chg = round(hfq_close - prev_close_hfq, 3)
    return {
        "datetime": target_date,
        "open": hfq_open, "high": hfq_high, "low": hfq_low, "close": hfq_close,
        "volume": volume_lots, "turnover": turnover, "amplitude": amplitude,
        "pct_chg": pct_chg, "chg": chg, "turnover_rate": turnover_rate,
        "close_raw": raw_close, "close_qfq": raw_close,
        "adj_factor": round(hfq_close / raw_close, 6),
    }


def verify_one(code, exch, cross_border, target_date, prev_date, local_prev, strict, is_latest):
    sym = exch + code
    sina_map, _ = fetch_sina(sym)
    if target_date not in sina_map:
        raise RuntimeError(f"{code} 新浪未覆盖 {target_date}（可能停牌）")
    if prev_date not in sina_map:
        raise RuntimeError(f"{code} 新浪未覆盖前一交易日 {prev_date}")
    day, prev = sina_map[target_date], sina_map[prev_date]

    qt = None
    if strict and is_latest and not cross_border:
        try:
            qt = fetch_tencent_qt(sym)
        except Exception as exc:
            raise RuntimeError(f"{code} 腾讯交叉核验不可用：{type(exc).__name__}: {exc}")

        ts = str(qt[30])
        if not ts.startswith(target_date.replace("-", "")):
            raise RuntimeError(f"{code} 腾讯行情时间戳不是 {target_date}：{ts}")
        if len(ts) >= 12 and ts[8:12] < "1500":
            raise RuntimeError(f"{code} 腾讯行情尚未收盘：{ts}")
        for label, sina_v, tx_v in [
            ("close", day["close"], qt[3]), ("open", day["open"], qt[5]),
            ("high", day["high"], qt[33]), ("low", day["low"], qt[34]),
        ]:
            if not close_enough(sina_v, tx_v):
                raise RuntimeError(f"{code} {label} 跨源不一致：新浪 {sina_v} vs 腾讯 {tx_v}")
        if not close_enough(float(prev["close"]), float(qt[4]), tol=0.002):
            raise RuntimeError(f"{code} 腾讯昨收 {qt[4]} 与新浪前收 {prev['close']} 不一致，疑似除权")

    return build_row(code, day, prev, local_prev, target_date, qt, strict, cross_border)


def load_local_frames():
    frames, lasts = {}, {}
    for code, exch, _, _, _ in ETF_POOL:
        p = csv_path(code, exch)
        df = pd.read_csv(p)
        miss = set(CSV_COLS) - set(df.columns)
        if miss:
            raise RuntimeError(f"{code} CSV 缺字段 {sorted(miss)}")
        frames[code] = df
        lasts[code] = str(df.iloc[-1]["datetime"])
    if len(set(lasts.values())) != 1:
        raise RuntimeError(f"14 只 ETF 本地末日不一致，停止更新：{lasts}")
    return frames, next(iter(lasts.values()))


def atomic_write(frames, rows_by_code, target_date, source_note):
    """先把全部新 CSV 与摘要写进临时目录，全部成功后再一次性替换。"""
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        staged = {}
        for code, exch, _, _, _ in ETF_POOL:
            df = frames[code]
            new_df = pd.concat([df, pd.DataFrame([rows_by_code[code]])], ignore_index=True)
            dates = pd.to_datetime(new_df["datetime"])
            if not dates.is_monotonic_increasing or dates.duplicated().any():
                raise RuntimeError(f"{code} 追加后日期未严格递增或存在重复")
            p = os.path.join(tmp, os.path.basename(csv_path(code, exch)))
            new_df.to_csv(p, index=False, encoding="utf-8-sig")
            staged[csv_path(code, exch)] = (p, new_df)

        summary = pd.read_csv(SUMMARY_PATH)
        for code, exch, _, _, _ in ETF_POOL:
            symbol = f"{code}.{exch.upper()}"
            mask = summary["symbol"] == symbol
            if not mask.any():
                raise RuntimeError(f"摘要中缺少 {symbol}")
            row = rows_by_code[code]
            staged_csv, new_df = staged[csv_path(code, exch)]
            summary.loc[mask, "end_date"] = target_date
            summary.loc[mask, "rows"] = len(new_df)
            summary.loc[mask, "latest_close_hfq"] = row["close"]
            summary.loc[mask, "latest_close_raw"] = row["close_raw"]
            summary.loc[mask, "latest_adj_factor"] = row["adj_factor"]
            summary.loc[mask, "sha256"] = sha256_file(staged_csv)
            summary.loc[mask, "source"] = source_note
        sp = os.path.join(tmp, "etf_data_summary.csv")
        summary.to_csv(sp, index=False, encoding="utf-8-sig")

        for dst, (src, _) in staged.items():
            shutil.copy2(src, dst)
        shutil.copy2(sp, SUMMARY_PATH)
        return {code: new_df for code, (_, new_df) in
                ((c, staged[csv_path(c, e)]) for c, e, _, _, _ in ETF_POOL)}


def write_status(status):
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", default=os.environ.get("QUOTE_STRICT", "1"),
                    help="1=双源严格核验，0=新浪单源降级")
    ap.add_argument("--max-days", type=int, default=5, help="单次最多补齐交易日数")
    args = ap.parse_args()
    strict = str(args.strict) not in ("0", "false", "False", "no")

    status = {
        "last_attempt_at": now_sh().strftime("%Y-%m-%d %H:%M:%S"),
        "strict_mode": strict,
        "data_quality": "dual_source" if strict else "single_source",
        "updated_dates": [],
        "errors": [],
    }
    try:
        frames, local_last = load_local_frames()
        trade_days, cal_source, cal_errors = get_trade_days()
        today = now_sh()
        today_str = today.strftime("%Y-%m-%d")
        is_weekday = today.weekday() < 5
        # 只接纳已收盘的交易日：当日 15:30 之前不取当日数据
        cutoff_ok = today.time() >= dtime(15, 30)
        market_state = market_state_today(today_str) if is_weekday else "holiday"

        status["calendar_source"] = cal_source
        status["calendar_errors"] = cal_errors
        status["cutoff_ok"] = cutoff_ok
        status["market_state"] = market_state

        # 两路日线日历都滞后时的最后兜底：腾讯实时行情已给出「今日收盘」时间戳，
        # 就把今天补进候选交易日，再交给逐日实查确认新浪是否已有当日日线。
        if cutoff_ok and market_state == "closed" and today_str not in trade_days:
            trade_days.append(today_str)
            trade_days.sort()
            status["calendar_patched"] = today_str

        status["recent_trade_days"] = trade_days[-12:]
        status["local_last_before"] = local_last

        pending = [d for d in trade_days
                   if d > local_last and d <= today_str and (d != today_str or cutoff_ok)]
        pending = pending[:args.max_days]

        # 逐日实查：日历里有该日、但新浪基准日线缺失（停牌 / 数据源滞后）时跳过该日，
        # 不让整批 14 只因为一个不存在的日期全部失败。
        ref_map, _ = fetch_sina("sh510300")
        confirmed = []
        for d in pending:
            if d in ref_map:
                confirmed.append(d)
            else:
                status.setdefault("skipped_dates", []).append(d)
        pending = confirmed
        status["pending_dates"] = pending

        # 「上次取数成功」只在真的取到数、或确认无新交易日时才刷新。
        # 盘中等待 / 静默缺数都不能冒充成功，否则页面会误显示「今天已更新」。
        prev_success = None
        if os.path.exists(STATUS_PATH):
            try:
                with open(STATUS_PATH, encoding="utf-8") as fp:
                    prev_success = json.load(fp).get("last_success_at")
            except Exception:
                pass
        status["last_success_at"] = prev_success

        if not pending:
            cal_last = trade_days[-1] if trade_days else local_last
            status["expected_last_trade_day"] = cal_last
            status["data_end_date"] = local_last
            if not cutoff_ok and is_weekday and market_state != "holiday":
                status["result"] = "pre_close"
                write_status(status)
                print(f"[WAIT] 尚未收盘（{today:%H:%M}），今日日线未生成，等待收盘后定时任务补数")
                return 0
            if cal_last > local_last or today_str in status.get("skipped_dates", []):
                status["result"] = "stale"
                status["errors"].append(
                    f"行情日历最新交易日 {cal_last} 晚于本地数据末日 {local_last}，"
                    "但本次没有生成可补的交易日，疑似行情源滞后或定时任务漏跑")
                write_status(status)
                print(f"[STALE] {status['errors'][-1]}", file=sys.stderr)
                return 1
            status["result"] = "up_to_date"
            status["last_success_at"] = status["last_attempt_at"]
            write_status(status)
            print(f"[SKIP] 本地末日 {local_last} 已是最新已收盘交易日，无需更新（日历来源 {cal_source}）")
            return 0

        print(f"[INFO] 本地末日 {local_last}，待补 {len(pending)} 个交易日：{pending}（严格模式={strict}）")
        for idx, target_date in enumerate(pending):
            is_latest = idx == len(pending) - 1
            prev_date = str(frames["510050"].iloc[-1]["datetime"])
            rows = {}
            for code, exch, name, _, cross_border in ETF_POOL:
                rows[code] = verify_one(
                    code, exch, cross_border, target_date, prev_date,
                    frames[code].iloc[-1], strict, is_latest,
                )
                print(f"  [OK] {code} {name:<14} {target_date} 不复权 {rows[code]['close_raw']} / 后复权 {rows[code]['close']}")
                time.sleep(0.25)
            frames = atomic_write(
                frames, rows, target_date,
                f"gh_actions | {target_date}: sina_quotes_service"
                + ("+tencent_qt_crosscheck" if strict else "_single_source")
                + ", adj_factor_carried",
            )
            status["updated_dates"].append(target_date)
            print(f"[DONE] {target_date} 14 只全部核验通过并原子写回")

        status["result"] = "updated"
        status["data_end_date"] = status["updated_dates"][-1]
        status["last_success_at"] = status["last_attempt_at"]
        write_status(status)
        print(json.dumps({k: status[k] for k in ["result", "data_end_date", "updated_dates", "data_quality"]},
                        ensure_ascii=False, indent=2))
        return 0

    except Exception as exc:
        # 失败必须让 Actions 标红；同时把错误写进 status，页面就能显示「数据源异常」
        status["result"] = "failed"
        status["errors"].append(f"{type(exc).__name__}: {exc}")
        write_status(status)
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
