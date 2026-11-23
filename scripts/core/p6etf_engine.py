# -*- coding: utf-8 -*-
"""P6ETF 6ETF 升级引擎：为网站两个组合提供目标权重与信号明细。

架构（源自 P6.2 课程包 fof_sdk，完整 6ETF 策略）：
  - 信号层：fof_sdk.compute_five_rule_signals（五规则状态机，月末门控）
  - 分配层：fof_sdk.build_p62_golive_allocator（克隆广度 / 下行半协方差+ERC /
            TIPP 回撤垫 / 三阶固收护盾 / 6.5% 漂移缓冲），完整开启
  - 两个组合仅风险袖套不同，防守底仓统一为三阶固收：

  old 原股债金升级版  = 518880 黄金 + 510500 中证500 + 510300 沪深300 + 511010/511260/511360 三债
  new 海外升级版      = 518880 黄金 + 513500 标普500 + 513100 纳指   + 511010/511260/511360 三债

不启用 ADV20 流动性锁（不传 daily_turnover_rmb / aum_rmb），适配个人中小资金。
权重和恒为 1.0，供 build_signal.py 做 T+1 切分与自检。
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fof_sdk import (
    load_domestic_etf_panel,
    infer_cn_month_end_dates,
    compute_five_rule_signals,
    build_p62_golive_allocator,
)

DEFENSIVE = ["511010.SH", "511260.SH", "511360.SH"]

# 6ETF 两袖套：黄金单袖 + 股票双袖（个股 cap 0.38，袖 cap 0.48/0.60，与 P6.2 课程 6etf 对齐）
SLEEVES_OLD = {
    "Gold": {"tickers": ["518880.SH"], "cap": 0.48, "indiv_cap": 0.48, "satellite_monthly": False},
    "Equity": {"tickers": ["510500.SH", "510300.SH"], "cap": 0.60, "indiv_cap": 0.38, "satellite_monthly": False},
}
SLEEVES_NEW = {
    "Gold": {"tickers": ["518880.SH"], "cap": 0.48, "indiv_cap": 0.48, "satellite_monthly": False},
    "Equity": {"tickers": ["513500.SH", "513100.SH"], "cap": 0.60, "indiv_cap": 0.38, "satellite_monthly": False},
}

COMBO_SLEEVES = {"old": SLEEVES_OLD, "new": SLEEVES_NEW}
COMBO_RISK = {
    "old": ["518880.SH", "510500.SH", "510300.SH"],
    "new": ["518880.SH", "513500.SH", "513100.SH"],
}
COMBO_NAME = {
    "old": "原股债金升级版 6ETF (518880黄金+510500中证500+510300沪深300+511010/511260/511360三阶固收)",
    "new": "海外升级版 6ETF (518880黄金+513500标普500+513100纳指+511010/511260/511360三阶固收)",
}

UNIVERSE = sorted(set(
    ["518880.SH", "510500.SH", "510300.SH", "513500.SH", "513100.SH"] + DEFENSIVE
))


def build_p6etf_panel(data_dir) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (后复权收盘 panel, 不复权收盘 panel)。"""
    panel = load_domestic_etf_panel(Path(data_dir), UNIVERSE)
    return panel["close"], panel["close_raw"]


def run_combo(
    prices: pd.DataFrame,
    sig: pd.DataFrame,
    me_dates: pd.DatetimeIndex,
    combo_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """跑单个组合，返回 (w_raw 权重表, sig 信号表子集, indicators)。"""
    sleeves = COMBO_SLEEVES[combo_key]
    risk = COMBO_RISK[combo_key]
    cols = risk + DEFENSIVE
    sub_px = prices[cols]
    sub_sig = sig[cols]
    w = build_p62_golive_allocator(
        prices=sub_px,
        sig_weights=sub_sig,
        sleeve_config=sleeves,
        defensive_cols=DEFENSIVE,
        month_end_dates=me_dates,
    )
    return w, sub_sig, {}


def run_all(data_dir) -> dict:
    """主入口：一次加载数据，产出两个组合的权重与信号。

    返回 dict:
      prices, prices_raw, me_dates, sig, lbl, indicators
      weights: {'old': w_df, 'new': w_df}
      risk_codes: {'old': [...], 'new': [...]}
      names: {'old': ..., 'new': ...}
    """
    prices, prices_raw = build_p6etf_panel(data_dir)
    me_dates = infer_cn_month_end_dates(prices.index)
    sig, lbl, indicators = compute_five_rule_signals(prices, me_dates, tickers=UNIVERSE)

    weights = {}
    for key in ("old", "new"):
        w, _, _ = run_combo(prices, sig, me_dates, key)
        total = float(w.sum(axis=1).iloc[-1])
        if abs(total - 1.0) > 1e-6:
            raise RuntimeError(f"{COMBO_NAME[key]} 权重合计 {total:.6f} != 1.0")
        weights[key] = w

    return {
        "prices": prices,
        "prices_raw": prices_raw,
        "me_dates": me_dates,
        "sig": sig,
        "lbl": lbl,
        "indicators": indicators,
        "weights": weights,
        "risk_codes": COMBO_RISK,
        "names": COMBO_NAME,
        "defensive": DEFENSIVE,
        "universe": UNIVERSE,
    }
