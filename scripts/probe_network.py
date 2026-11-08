# -*- coding: utf-8 -*-
"""GitHub Actions 境外出口 -> 国内行情接口 连通性预检。

这一步是全云端方案的生死线，必须先于正式 workflow 跑通。
只做只读探测，不写任何数据 CSV，即使失败也不影响仓库内容。

用法：
    python scripts/probe_network.py
    # CI 中会自动把结果写进 $GITHUB_STEP_SUMMARY
"""
import os
import sys
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

SHANGHAI = ZoneInfo("Asia/Shanghai")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://gu.qq.com/",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

SINA = ("{scheme}://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        "CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen=10")
TX_QT = "https://qt.gtimg.cn/q={sym}"
TX_FQ = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
         "param={sym},day,2026-01-01,2026-12-31,20,qfq")
TX_KLINE = "https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param={sym},day,,,20"
EM_HIST = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?"
           "secid={market}.{code}&fields1=f1,f2,f3,f4,f5,f6"
           "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
           "&klt=101&fqt=2&beg=0&end=20500101&lmt=20")
EM_QT = ("https://push2.eastmoney.com/api/qt/stock/get?"
         "secid={market}.{code}&fields=f43,f44,f45,f46,f47,f48,f60,f86")

SYMBOLS = ["sh510300", "sh513500"]


def sniff(name, url, parser, timeout=12):
    """返回 (ok, 详情文本, 耗时秒, 字节数)。任何异常都只记录不抛出。"""
    t0 = time.perf_counter()
    try:
        r = SESSION.get(url, timeout=timeout, headers=HEADERS)
        elapsed = round(time.perf_counter() - t0, 2)
        body = r.content
        detail = f"HTTP {r.status_code}, {len(body)} bytes"
        if r.status_code != 200:
            return False, detail, elapsed, len(body)
        try:
            parsed = parser(r)
            detail += f" | {parsed}"
            return True, detail, elapsed, len(body)
        except Exception as exc:
            return False, detail + f" | 解析失败 {type(exc).__name__}: {exc}", elapsed, len(body)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:160]}", round(time.perf_counter() - t0, 2), 0


def p_json_list(r):
    arr = json.loads(r.text.strip())
    if not isinstance(arr, list) or not arr:
        return "0 条"
    return f"{len(arr)} 条, 最新 {arr[-1].get('day')} 收 {arr[-1].get('close')}"


def p_qt(r):
    r.encoding = "gbk"
    p = r.text.split("~")
    if len(p) < 31:
        return f"字段仅 {len(p)} 个"
    return f"名称 {p[1]}, 现价 {p[3]}, 昨收 {p[4]}, 时间 {p[30]}"


def p_fq(r):
    j = r.json()
    sym = SYMBOLS[0] if SYMBOLS[0] in j.get("data", {}) else next(iter(j.get("data", {})), "")
    d = j.get("data", {}).get(sym, {})
    key = "qfqday" if "qfqday" in d else "day"
    arr = d.get(key, []) or []
    return f"key={key}, {len(arr)} 条" + (f", 尾={arr[-1][:3]}" if arr else "")


def p_kline(r):
    j = r.json()
    d = next(iter(j.get("data", {}).values()), {})
    arr = d.get("day", []) or []
    return f"{len(arr)} 条" + (f", 尾={arr[-1][:3]}" if arr else "")


def p_em(r):
    j = r.json()
    d = j.get("data") or {}
    kl = d.get("klines") or []
    if kl:
        return f"{len(kl)} 条, 尾={kl[-1][:40]}"
    return f"现价={d.get('f43')} 时间戳={d.get('f86')}"


def build_cases():
    cases = []
    # 网络基线与出口归属
    cases.append(("基线 / GitHub API", "https://api.github.com", lambda r: f"status={r.json().get('current_user_url', 'ok')[:40]}"))
    cases.append(("基线 / 出口 IP", "https://ipinfo.io/json", lambda r: f"ip={r.json().get('ip')}, region={r.json().get('region')}, org={str(r.json().get('org'))[:40]}"))
    for sym in SYMBOLS:
        code, market = sym[2:], ("1" if sym.startswith("sh") else "0")
        cases += [
            (f"{sym} 新浪 quotes_service HTTPS", SINA.format(scheme="https", sym=sym), p_json_list),
            (f"{sym} 新浪 quotes_service HTTP", SINA.format(scheme="http", sym=sym), p_json_list),
            (f"{sym} 腾讯 qt.gtimg.cn 实时", TX_QT.format(sym=sym), p_qt),
            (f"{sym} 腾讯 fqkline qfq", TX_FQ.format(sym=sym), p_fq),
            (f"{sym} 腾讯 kline 不复权", TX_KLINE.format(sym=sym), p_kline),
            (f"{sym} 东财 push2his 历史", EM_HIST.format(market=market, code=code), p_em),
            (f"{sym} 东财 push2 实时", EM_QT.format(market=market, code=code), p_em),
        ]
    return cases


def main():
    now = datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"===== GitHub Actions 出口连通性预检 @ {now} =====")
    results = []
    for name, url, parser in build_cases():
        ok, detail, elapsed, size = sniff(name, url, parser)
        results.append({"name": name, "ok": ok, "detail": detail, "elapsed_s": elapsed, "bytes": size})
        print(f"[{'OK  ' if ok else 'FAIL'}] {name:<42} {elapsed:>5}s  {detail}")

    ok_names = [r["name"] for r in results if r["ok"]]
    groups = {
        "sina": any(r["ok"] and "新浪" in r["name"] for r in results),
        "tencent_qt": any(r["ok"] and "qt.gtimg" in r["name"] for r in results),
        "tencent_fq": any(r["ok"] and ("腾讯 fqkline" in r["name"] or "腾讯 kline" in r["name"]) for r in results),
        "eastmoney": any(r["ok"] and "东财" in r["name"] for r in results),
        "baseline": any(r["ok"] and "基线" in r["name"] for r in results),
    }
    print("\n===== 分组结论 =====")
    for k, v in groups.items():
        print(f"  {k:<14} {'可用' if v else '不可用'}")

    if not groups["baseline"]:
        verdict = "ABORT: 连 GitHub API / ipinfo 都不通，属 Actions 环境整体异常，与国内接口无关。"
    elif not any([groups["sina"], groups["tencent_qt"], groups["tencent_fq"], groups["eastmoney"]]):
        verdict = ("BLOCKED: 境外出口无法访问任何国内行情接口。全云端直连方案不可行。"
                   "备选：① 国际数据源覆盖度不足，A股 ETF 基本不可用；"
                   "② 国内云函数/CF Worker 反向代理中转；③ Gitee / 腾讯云 CI 等国内托管 CI；"
                   "④ 退回本地计算 + 推 JSON（但这会重新依赖本地开机）。")
    elif groups["sina"] and (groups["tencent_qt"] or groups["tencent_fq"]):
        verdict = ("PASS: 新浪与腾讯双源均可用，fetch_and_update.py 可按默认严格模式上线。"
                   "建议把每日 workflow 的 QUOTE_STRICT 保持为 1。")
    elif groups["eastmoney"]:
        verdict = ("PARTIAL: 仅东财可用。需把 fetch_and_update.py 的数据源切到东财 fqt=2 后复权口径，"
                   "并经本地逐日比对复权因子后再上线，勿直接启用。")
    else:
        verdict = ("PARTIAL: 至少一个源可用但不足以做双源交叉核验。"
                   "可设 QUOTE_STRICT=0 走单源模式，signal.json 会标注 data_quality=single_source，"
                   "页面会显示黄色提示；建议同时开启人工复核。")

    print(f"\n===== 总判定 =====\n{verdict}")

    payload = {
        "checked_at": now,
        "groups": groups,
        "verdict": verdict,
        "results": results,
        "ok_count": len(ok_names),
        "total": len(results),
    }
    # CI 中落在 scripts/probe_result.json 供 artifact 归档；
    # 本地对照测试可用 PROBE_OUT 指到临时路径，避免把本机结果带进仓库。
    out = os.environ.get("PROBE_OUT") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "probe_result.json")
    with open(os.path.abspath(out), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        lines = [
            "## 境外出口连通性预检",
            "",
            f"时间：{now}　通过：{len(ok_names)}/{len(results)}",
            "",
            "| 分组 | 结果 |",
            "| --- | --- |",
        ]
        lines += [f"| {k} | {'可用' if v else '不可用'} |" for k, v in groups.items()]
        lines += ["", "| 探测项 | 结果 | 耗时 | 详情 |", "| --- | --- | --- | --- |"]
        lines += [f"| {r['name']} | {'OK' if r['ok'] else 'FAIL'} | {r['elapsed_s']}s | {r['detail'][:110]} |" for r in results]
        lines += ["", f"**总判定**：{verdict}", ""]
        with open(summary, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # 预检是诊断工具，无论结论如何都正常退出，避免 CI 显示刺眼的红叉
    return 0


if __name__ == "__main__":
    sys.exit(main())
