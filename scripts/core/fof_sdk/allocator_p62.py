from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def proj_capped_simplex(v, target_sum: float = 1.0, low=0.0, high=1.0) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0 or target_sum <= 1e-14:
        return np.zeros(n, dtype=float)
    low_arr = np.broadcast_to(np.asarray(low, dtype=float), (n,))
    high_arr = np.broadcast_to(np.asarray(high, dtype=float), (n,))
    if target_sum <= float(low_arr.sum()) + 1e-14:
        return low_arr.copy()
    if target_sum >= float(high_arr.sum()) - 1e-14:
        return high_arr.copy()
    lo_mu = float(np.min(v - high_arr)) - 1.0
    hi_mu = float(np.max(v - low_arr)) + 1.0
    for _ in range(65):
        mid = 0.5 * (lo_mu + hi_mu)
        w = np.clip(v - mid, low_arr, high_arr)
        s = float(w.sum())
        if abs(s - target_sum) < 1e-13:
            return w
        if s > target_sum:
            lo_mu = mid
        else:
            hi_mu = mid
    return np.clip(v - 0.5 * (lo_mu + hi_mu), low_arr, high_arr)


def shrunk_downside_semicov(
    neg_rets: np.ndarray,
    tot_rets: np.ndarray,
    ann_factor: float = 252.0,
    shrink_mult: float = 1.0,
    downside_mode: bool = True,
) -> np.ndarray:
    T, N = neg_rets.shape
    if N == 1:
        if downside_mode:
            v = (0.70 * (2.0 * np.mean(neg_rets[:, 0] ** 2)) + 0.30 * np.var(tot_rets[:, 0], ddof=1)) * ann_factor
        else:
            v = np.var(tot_rets[:, 0], ddof=1) * ann_factor
        return np.array([[max(float(v), 1e-8)]])
    if downside_mode:
        semi = (2.0 * (neg_rets.T @ neg_rets) / float(T)) * ann_factor
        tcov = np.cov(tot_rets, rowvar=False, ddof=1) * ann_factor
        S = 0.70 * semi + 0.30 * tcov
    else:
        S = np.cov(tot_rets, rowvar=False, ddof=1) * ann_factor
    diag = np.maximum(np.diag(S), 1e-8)
    if not downside_mode:
        np.fill_diagonal(S, diag)
        return 0.85 * S + 0.15 * np.diag(diag)
    std = np.sqrt(diag)
    outer_std = np.outer(std, std)
    corr = np.clip(S / outer_std, -0.95, 0.999)
    np.fill_diagonal(corr, 1.0)
    off_mask = ~np.eye(N, dtype=bool)
    rho_bar = float(np.mean(corr[off_mask])) if np.any(off_mask) else 0.0
    F_corr = np.full((N, N), max(rho_bar, 0.0))
    np.fill_diagonal(F_corr, 1.0)
    F = F_corr * outer_std
    base_delta = (N + 2.0) / (N + 2.0 + T / 2.5)
    delta = float(np.clip(shrink_mult * base_delta, 0.15, 0.65))
    return (1.0 - delta) * S + delta * F


def solve_erc_weights(cov: np.ndarray, budgets: Optional[np.ndarray] = None, n_iter: int = 35) -> np.ndarray:
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0], dtype=float)
    b = np.full(n, 1.0 / n, dtype=float) if budgets is None else (np.asarray(budgets, dtype=float) / np.sum(budgets))
    diag = np.maximum(np.diag(cov), 1e-8)
    x = b / np.sqrt(diag)
    x = x / x.sum()
    for _ in range(n_iter):
        cx = cov @ x
        for i in range(n):
            aii = diag[i]
            ci_excl = cx[i] - aii * x[i]
            x_new = (-ci_excl + np.sqrt(max(ci_excl * ci_excl + 4.0 * aii * b[i], 1e-16))) / (2.0 * aii)
            diff = x_new - x[i]
            if abs(diff) > 1e-14:
                x[i] = x_new
                cx += cov[:, i] * diff
    return x / x.sum()


def build_p62_golive_allocator(
    prices: pd.DataFrame,
    sig_weights: pd.DataFrame,
    sleeve_config: Dict[str, dict],
    defensive_cols: List[str],
    month_end_dates: pd.DatetimeIndex,
    daily_turnover_rmb: Optional[pd.DataFrame] = None,
    aum_rmb: Optional[float] = None,
    ann_factor: float = 252.0,
    vol_ref: float = 0.10,
    port_semi_cap: float = 0.052,
    max_total_risk: float = 0.85,
    cov_lookback: Optional[int] = None,
    shrink_mult: float = 1.0,
    drift_buf: float = 0.065,
    dd_soft_daily: float = 0.016,
    dd_slope_daily: float = 10.0,
    dd_min_mult: float = 0.52,
    use_clone_breadth: bool = True,
    use_downside_semicov: bool = True,
    use_tipp_cushion: bool = True,
    use_3tier_defense: bool = True,
    use_local_buffer: bool = True,
) -> pd.DataFrame:
    idx = prices.index
    n_steps = len(idx)
    me_set = set(pd.DatetimeIndex(month_end_dates).intersection(idx))

    is_daily = ann_factor > 50.0
    w_fast = 20 if is_daily else 3
    w_slow = 60 if is_daily else 6
    w_long = 200 if is_daily else 10
    w_anchor = 252 if is_daily else 24
    cov_win = cov_lookback if cov_lookback is not None else (60 if is_daily else 12)

    rets = prices.pct_change(fill_method=None)
    rets_clean = rets.fillna(0.0)

    vol_fast = rets.rolling(w_fast, min_periods=max(3, w_fast // 2)).std() * np.sqrt(ann_factor)
    vol_slow = rets.rolling(w_slow, min_periods=max(4, w_slow // 2)).std() * np.sqrt(ann_factor)
    neg_sq = rets.clip(upper=0.0) ** 2
    pos_sq = rets.clip(lower=0.0) ** 2
    dvol_fast = np.sqrt(2.0 * neg_sq.rolling(w_fast, min_periods=max(3, w_fast // 2)).mean()) * np.sqrt(ann_factor)
    dvol_slow = np.sqrt(2.0 * neg_sq.rolling(w_slow, min_periods=max(4, w_slow // 2)).mean()) * np.sqrt(ann_factor)
    uvol_slow = np.sqrt(2.0 * pos_sq.rolling(w_slow, min_periods=max(4, w_slow // 2)).mean()) * np.sqrt(ann_factor)
    dvol_long = np.sqrt(2.0 * neg_sq.rolling(w_anchor, min_periods=max(6, w_slow)).mean()) * np.sqrt(ann_factor)

    q80_fast = vol_fast.rolling(w_anchor, min_periods=max(6, w_slow)).quantile(0.80)
    ma_fast = prices.rolling(w_fast, min_periods=max(3, w_fast // 2)).mean()
    ma_slow = prices.rolling(w_slow, min_periods=max(4, w_slow // 2)).mean()
    ma_long = prices.rolling(w_long, min_periods=max(6, w_long // 2)).mean()
    ret_fast = prices / prices.shift(w_fast) - 1.0
    ret_slow = prices / prices.shift(w_slow) - 1.0

    sleeve_names = [s for s, cfg in sleeve_config.items() if any(t in prices.columns for t in cfg["tickers"])]
    risk_cols = []
    ticker_to_sleeve = {}
    indiv_caps_list = []
    sat_monthly_flags = []
    for s_name in sleeve_names:
        cfg = sleeve_config[s_name]
        ic = float(cfg.get("indiv_cap", 0.38))
        sm_flag = bool(cfg.get("satellite_monthly", False))
        for tk in cfg["tickers"]:
            if tk in prices.columns and tk not in ticker_to_sleeve:
                risk_cols.append(tk)
                ticker_to_sleeve[tk] = s_name
                indiv_caps_list.append(ic)
                sat_monthly_flags.append(sm_flag)

    base_indiv_caps = np.array(indiv_caps_list, dtype=float)
    sat_flags = np.array(sat_monthly_flags, dtype=bool)
    sleeve_idx_map = {s: [k for k, tk in enumerate(risk_cols) if ticker_to_sleeve[tk] == s] for s in sleeve_names}
    sleeve_caps = np.array([
        float(sleeve_config[s].get("cap", 0.48 if len(sleeve_idx_map[s]) == 1 else 0.60))
        for s in sleeve_names
    ], dtype=float)

    if daily_turnover_rmb is not None and aum_rmb is not None and aum_rmb > 0:
        adv20_arr = daily_turnover_rmb[risk_cols].rolling(20, min_periods=5).median().values
    else:
        adv20_arr = None

    def_cols = [c for c in defensive_cols if c in prices.columns]
    all_cols = risk_cols + def_cols
    n_risk = len(risk_cols)
    n_def = len(def_cols)

    def_mat = np.zeros((n_steps, n_def), dtype=float)
    def_mat[:, 0] = 1.0
    if use_3tier_defense and n_def >= 2:
        b_mid = def_cols[0]
        b_long = def_cols[1]
        c_shld = def_cols[2] if n_def >= 3 else None
        long_share = 0.0
        cash_share = 0.0
        for i, dt in enumerate(idx):
            px_l = prices[b_long].iloc[i]
            ms_l = ma_slow[b_long].iloc[i]
            ml_l = ma_long[b_long].iloc[i]
            dv_l = dvol_fast[b_long].iloc[i]
            q8_l = q80_fast[b_long].iloc[i]
            has_l = pd.notna(px_l) and pd.notna(ms_l) and pd.notna(ml_l)

            px_m = prices[b_mid].iloc[i]
            mf_m = ma_fast[b_mid].iloc[i]
            ms_m = ma_slow[b_mid].iloc[i]
            ml_m = ma_long[b_mid].iloc[i]
            mid_weak = pd.notna(ms_m) and (px_m < ms_m) and (px_m < mf_m if is_daily else (pd.notna(ml_m) and px_m < ml_m))
            has_c = (c_shld is not None) and pd.notna(prices[c_shld].iloc[i]) and (i > w_fast and pd.notna(prices[c_shld].iloc[i - w_fast]))
            urgent_bond_break = is_daily and has_l and (long_share > 0.30) and (px_l < 0.994 * ms_l) and (px_l < ml_l)

            if (not is_daily) or (dt in me_set) or urgent_bond_break:
                if has_l:
                    rho_l = float(dv_l / max(q8_l, 0.015)) if pd.notna(q8_l) and q8_l > 0 else 0.80
                    if px_l >= 0.998 * ms_l and ms_l >= 0.999 * ml_l and rho_l <= 0.95:
                        des_l, des_c = 0.72, 0.0
                    elif long_share >= 0.45 and px_l >= 0.995 * ms_l and px_l > ml_l and rho_l <= 1.05:
                        des_l, des_c = long_share, 0.0
                    elif px_l > ml_l and (not mid_weak):
                        des_l, des_c = 0.30, 0.0
                    elif has_c and mid_weak:
                        des_l, des_c = 0.0, 0.75
                    else:
                        des_l, des_c = 0.0, 0.0
                elif has_c and mid_weak:
                    des_l, des_c = 0.0, 0.75
                else:
                    des_l, des_c = 0.0, 0.0

                if abs(des_l - long_share) >= 0.22 or abs(des_c - cash_share) >= 0.22:
                    long_share = (0.25 * long_share + 0.75 * des_l) if des_c == 0.0 else 0.0
                    cash_share = des_c

            def_mat[i, 0] = max(0.0, 1.0 - long_share - cash_share)
            def_mat[i, 1] = long_share
            if n_def >= 3:
                def_mat[i, 2] = cash_share

    px_arr = prices[risk_cols].values
    ma_f_arr = ma_fast[risk_cols].values
    vf_arr = vol_fast[risk_cols].values
    vs_arr = vol_slow[risk_cols].values
    dvf_arr = dvol_fast[risk_cols].values
    dvs_arr = dvol_slow[risk_cols].values
    uvs_arr = uvol_slow[risk_cols].values
    dvl_arr = dvol_long[risk_cols].values
    q80_arr = q80_fast[risk_cols].values
    rf_arr = ret_fast[risk_cols].values
    rs_arr = ret_slow[risk_cols].values
    sig_arr = sig_weights[risk_cols].fillna(0.0).values

    neg_r_mat = rets_clean[risk_cols].clip(upper=0.0).values
    tot_r_mat = rets_clean[risk_cols].values
    all_r_mat = rets_clean[all_cols].values

    w_out = np.zeros((n_steps, len(all_cols)), dtype=float)
    curr = np.zeros(len(all_cols), dtype=float)
    curr[n_risk] = 1.0
    prev_sig = np.zeros(n_risk, dtype=float)
    sat_gate_state = np.where(sat_flags, 0.0, 1.0)
    vol_floor = 0.80 * vol_ref

    port_nav = 1.0
    nav_hist = np.ones(n_steps, dtype=float)
    dd_win = 60 if is_daily else 12
    dd_soft = dd_soft_daily if is_daily else 0.055
    dd_slope = dd_slope_daily if is_daily else 6.0

    def project_two_stage(r_in: np.ndarray, indiv_c: np.ndarray, max_r: float) -> np.ndarray:
        s_sums = np.array([float(r_in[sleeve_idx_map[s]].sum()) for s in sleeve_names], dtype=float)
        eff_s_caps = np.array([
            min(sleeve_caps[s_idx], float(indiv_c[sleeve_idx_map[s_name]].sum()))
            for s_idx, s_name in enumerate(sleeve_names)
        ], dtype=float)
        s_clipped = np.minimum(s_sums, eff_s_caps)
        if s_clipped.sum() > max_r:
            s_targets = proj_capped_simplex(s_clipped, target_sum=max_r, low=0.0, high=eff_s_caps)
        else:
            s_targets = s_clipped
        out_r = np.zeros(n_risk, dtype=float)
        for s_idx, s_name in enumerate(sleeve_names):
            ks = sleeve_idx_map[s_name]
            t_sum = s_targets[s_idx]
            if t_sum > 1e-12 and len(ks) > 0 and r_in[ks].sum() > 1e-12:
                out_r[ks] = proj_capped_simplex(r_in[ks] * (t_sum / r_in[ks].sum()), target_sum=t_sum, low=0.0, high=indiv_c[ks])
        return out_r

    for i, dt in enumerate(idx):
        if i > 0:
            port_nav *= (1.0 + float(np.dot(curr, all_r_mat[i])))
        nav_hist[i] = port_nav
        if use_tipp_cushion:
            rolling_peak = float(np.max(nav_hist[max(0, i - dd_win): i + 1]))
            curr_dd = max(0.0, 1.0 - port_nav / rolling_peak)
            dd_mult = float(np.clip(1.0 - dd_slope * max(0.0, curr_dd - dd_soft), dd_min_mult, 1.0))
        else:
            dd_mult = 1.0

        valid_mask = (~np.isnan(px_arr[i])) & (~np.isnan(dvs_arr[i])) & (dvs_arr[i] > 1e-6)
        if not np.any(valid_mask):
            w_out[i] = curr
            continue

        if adv20_arr is not None:
            adv_row = adv20_arr[i]
            indiv_caps = np.where(
                (~np.isnan(adv_row)) & (adv_row > 0) & (adv_row < 2.0e8),
                np.minimum(base_indiv_caps, np.maximum(0.0, 0.04 * adv_row / aum_rmb)),
                base_indiv_caps,
            )
        else:
            indiv_caps = base_indiv_caps

        is_me = (not is_daily) or (dt in me_set)
        raw_sig_i = np.where(valid_mask, sig_arr[i], 0.0)

        if is_me:
            for k in range(n_risk):
                if sat_flags[k]:
                    if (
                        valid_mask[k]
                        and raw_sig_i[k] >= 0.99
                        and (not np.isnan(rs_arr[i, k]))
                        and rs_arr[i, k] > 0.05
                        and (np.isnan(q80_arr[i, k]) or dvf_arr[i, k] <= 0.80 * q80_arr[i, k])
                    ):
                        sat_gate_state[k] = 1.0
                    else:
                        sat_gate_state[k] = 0.0

        eff_sig = raw_sig_i * sat_gate_state
        sig_diff = eff_sig - prev_sig

        if use_downside_semicov:
            eff_dv = np.where(
                valid_mask,
                np.maximum(vs_arr[i], 0.55 * dvs_arr[i] + 0.45 * np.nan_to_num(dvf_arr[i], nan=dvs_arr[i])),
                1.0,
            )
            eff_dv = np.maximum(eff_dv, vol_floor)
            long_dv = np.where(np.isnan(dvl_arr[i]) | (dvl_arr[i] <= 1e-5), eff_dv, np.maximum(dvl_arr[i], vol_floor))
            omega = np.where(
                valid_mask,
                np.clip(np.nan_to_num(uvs_arr[i], nan=dvs_arr[i]) / np.maximum(dvs_arr[i], 0.60 * vol_ref), 0.85, 1.45),
                1.0,
            )
            calm_mask = np.isnan(q80_arr[i]) | (vf_arr[i] <= q80_arr[i])
            omega_bonus = np.where(calm_mask & (omega > 1.0), omega ** 0.30, 1.0)
            inv_v = np.clip(vol_ref / eff_dv, 0.0, 1.24) * omega_bonus
            blowoff_cut = False
            blowoff_mult = np.ones(n_risk, dtype=float)
            if is_daily:
                for k in range(n_risk):
                    if valid_mask[k] and (not np.isnan(q80_arr[i, k])):
                        if (
                            (not np.isnan(rf_arr[i, k]) and rf_arr[i, k] > 0.18 and vf_arr[i, k] > 1.15 * q80_arr[i, k])
                            or (not np.isnan(rs_arr[i, k]) and rs_arr[i, k] > 0.26 and vf_arr[i, k] > 0.95 * q80_arr[i, k])
                        ):
                            blowoff_mult[k] = 0.45
                            blowoff_cut = True
        else:
            eff_dv = np.maximum(np.where(valid_mask, np.nan_to_num(vs_arr[i], nan=1.0), 1.0), vol_floor)
            long_dv = eff_dv.copy()
            omega = np.ones(n_risk, dtype=float)
            omega_bonus = np.ones(n_risk, dtype=float)
            inv_v = np.clip(vol_ref / eff_dv, 0.0, 1.24)
            blowoff_cut = False
            blowoff_mult = np.ones(n_risk, dtype=float)

        n_sleeves = len(sleeve_names)
        prior_raw = np.zeros(n_sleeves, dtype=float)
        neff_per_sleeve = np.zeros(n_sleeves, dtype=float)
        valid_per_sleeve = []
        for s_idx, s_name in enumerate(sleeve_names):
            ks_v = [k for k in sleeve_idx_map[s_name] if valid_mask[k] and indiv_caps[k] > 1e-4]
            valid_per_sleeve.append(ks_v)
            m_s = len(ks_v)
            if m_s > 0:
                if is_daily:
                    if use_clone_breadth:
                        if m_s >= 2 and i >= cov_win:
                            sub_tot_s = tot_r_mat[i - cov_win + 1 : i + 1, :][:, ks_v]
                            if np.all(np.std(sub_tot_s, axis=0, ddof=1) > 1e-8):
                                cmat = np.corrcoef(sub_tot_s, rowvar=False)
                                rho_plus = float(np.clip(np.nanmean(cmat[~np.eye(m_s, dtype=bool)]), 0.0, 0.999))
                            else:
                                rho_plus = 0.80
                            n_indep = float(m_s / (1.0 + (m_s - 1.0) * rho_plus))
                        else:
                            n_indep = 1.0
                        n_eff_s = min(n_indep * 1.15 if m_s >= 2 else 1.0, sleeve_caps[s_idx] / 0.34, 2.0)
                        neff_per_sleeve[s_idx] = n_eff_s
                        s_ldv = float(np.mean(long_dv[ks_v]))
                        prior_raw[s_idx] = (n_eff_s ** 0.75) * ((vol_ref / s_ldv) ** 0.35)
                    else:
                        n_eff_s = float(m_s)
                        neff_per_sleeve[s_idx] = n_eff_s
                        prior_raw[s_idx] = n_eff_s
                else:
                    n_eff_s = min(float(m_s), sleeve_caps[s_idx] / 0.14, 2.2) if use_clone_breadth else float(m_s)
                    neff_per_sleeve[s_idx] = n_eff_s
                    s_ldv = float(np.mean(long_dv[ks_v]))
                    prior_raw[s_idx] = (n_eff_s ** 0.60) * ((vol_ref / s_ldv) ** 0.35)

        if prior_raw.sum() > 0:
            n_valid_sleeves = sum(1 for kv in valid_per_sleeve if len(kv) > 0)
            sum_neff = float(neff_per_sleeve.sum())
            sleeve_div_cap = 0.56 if n_valid_sleeves == 1 else (0.90 if n_valid_sleeves == 2 else 1.00)
            if use_clone_breadth:
                tot_prior_cap = min(sleeve_div_cap, 0.36 * sum_neff, float(sleeve_caps.sum()))
            else:
                tot_prior_cap = float(sleeve_caps.sum())
            base_budgets = np.minimum(sleeve_caps, (prior_raw / prior_raw.sum()) * tot_prior_cap)
        else:
            base_budgets = np.zeros(n_sleeves, dtype=float)

        raw_sleeve_w = np.zeros(n_sleeves, dtype=float)
        intra_raw = np.zeros(n_risk, dtype=float)
        for s_idx, s_name in enumerate(sleeve_names):
            ks_valid = valid_per_sleeve[s_idx]
            ks_act = [k for k in ks_valid if eff_sig[k] > 1e-6]
            if len(ks_act) == 0:
                continue
            if len(ks_act) >= 2 and i >= cov_win:
                cov_s = shrunk_downside_semicov(
                    neg_r_mat[i - cov_win + 1 : i + 1, :][:, ks_act],
                    tot_r_mat[i - cov_win + 1 : i + 1, :][:, ks_act],
                    ann_factor=ann_factor,
                    shrink_mult=shrink_mult,
                    downside_mode=use_downside_semicov,
                )
                rb_in = np.array([eff_sig[k] * blowoff_mult[k] * (omega[k] ** 0.50) for k in ks_act], dtype=float)
                rel_share = solve_erc_weights(cov_s, budgets=rb_in)
            else:
                sc_s = np.array([eff_sig[k] * inv_v[k] * blowoff_mult[k] for k in ks_act], dtype=float)
                rel_share = sc_s / sc_s.sum()

            act_ratio = len(ks_act) / float(len(ks_valid))
            breadth_factor = (0.38 + 0.62 * (act_ratio ** 0.75)) if len(ks_valid) >= 2 else 1.0
            avg_slot_inv = float(np.sum(rel_share * inv_v[ks_act] * eff_sig[ks_act] * blowoff_mult[ks_act]))
            w_s = min(sleeve_caps[s_idx], base_budgets[s_idx] * avg_slot_inv * breadth_factor)
            raw_sleeve_w[s_idx] = w_s
            for loc_i, k in enumerate(ks_act):
                intra_raw[k] = w_s * rel_share[loc_i]

        tot_s = float(raw_sleeve_w.sum())
        if 0.05 < tot_s < 0.60 and n_sleeves >= 2:
            inactive_cnt = sum(1 for s_idx in range(n_sleeves) if len(valid_per_sleeve[s_idx]) > 0 and raw_sleeve_w[s_idx] < 0.06)
            if inactive_cnt >= 1:
                for s_idx, s_name in enumerate(sleeve_names):
                    if raw_sleeve_w[s_idx] <= 0.05:
                        continue
                    ks_strong = [k for k in valid_per_sleeve[s_idx] if eff_sig[k] >= 0.75]
                    if len(ks_strong) >= max(1, int(np.ceil(0.5 * len(valid_per_sleeve[s_idx])))):
                        avg_eff_dv = float(np.mean(eff_dv[ks_strong]))
                        dyn_cap = min(sleeve_caps[s_idx], 0.46 * (vol_ref / max(avg_eff_dv, 0.92 * vol_ref)) * float(np.mean(omega_bonus[ks_strong])))
                        if raw_sleeve_w[s_idx] < dyn_cap:
                            boost = min(dyn_cap - raw_sleeve_w[s_idx], 0.26 * (0.60 - tot_s))
                            scale_up = (raw_sleeve_w[s_idx] + boost) / raw_sleeve_w[s_idx]
                            raw_sleeve_w[s_idx] += boost
                            for k in sleeve_idx_map[s_name]:
                                intra_raw[k] *= scale_up

        risk_vec = intra_raw * dd_mult
        eff_semi_cap = port_semi_cap * dd_mult
        eff_max_risk = max_total_risk * dd_mult

        act = risk_vec > 1e-5
        if i >= cov_win and np.sum(act) >= 2 and risk_vec.sum() > 0.08:
            act_idx = np.where(act)[0]
            cov_shrunk = shrunk_downside_semicov(
                neg_r_mat[i - cov_win + 1 : i + 1, :][:, act_idx],
                tot_r_mat[i - cov_win + 1 : i + 1, :][:, act_idx],
                ann_factor=ann_factor,
                shrink_mult=shrink_mult,
                downside_mode=use_downside_semicov,
            )
            w_a = risk_vec[act_idx]
            p_var = float(w_a @ cov_shrunk @ w_a)
            if p_var > 1e-10:
                p_vol = np.sqrt(p_var)
                mrc = (cov_shrunk @ w_a) / p_vol
                indiv_v = np.sqrt(np.maximum(np.diag(cov_shrunk), 1e-8))
                inv_corr_tilt = np.clip(mrc / indiv_v, 0.25, 1.0) ** (-0.30)
                w_tilted = w_a * (inv_corr_tilt / np.average(inv_corr_tilt, weights=w_a))
                div_ratio = float((w_a @ indiv_v) / p_vol)
                scale_div = float(np.clip(div_ratio ** 0.25, 0.98, 1.05)) if (p_vol < 0.85 * eff_semi_cap and dd_mult > 0.95) else 1.0
                w_tilted = w_tilted * scale_div
                new_pvol = float(np.sqrt(max(w_tilted @ cov_shrunk @ w_tilted, 1e-10)))
                if new_pvol > eff_semi_cap:
                    w_tilted *= (eff_semi_cap / new_pvol)
                risk_vec[act_idx] = w_tilted

        final_risk = project_two_stage(risk_vec, indiv_caps, eff_max_risk)
        tot_risk = float(final_risk.sum())
        rem_def = max(0.0, 1.0 - tot_risk)
        def_s = def_mat[i]

        if not use_local_buffer:
            curr = np.concatenate([final_risk, rem_def * def_s])
        elif not is_daily:
            if 0.5 * float(np.abs(final_risk - curr[:n_risk]).sum()) >= 0.020 or np.any(np.abs(eff_sig - prev_sig) > 0.05):
                curr = np.concatenate([final_risk, rem_def * def_s])
            else:
                rem_keep = max(0.0, 1.0 - float(curr[:n_risk].sum()))
                curr = np.concatenate([curr[:n_risk], rem_keep * def_s])
        else:
            stop_to_zero = bool(np.any((prev_sig > 0.01) & (eff_sig <= 1e-6)))
            urgent_cut = (
                bool(np.any((sig_diff <= -0.25) & ((dvf_arr[i] > 0.80 * np.nan_to_num(q80_arr[i], nan=99.0)) | (px_arr[i] < ma_f_arr[i]))))
                or (dd_mult < 0.85 and tot_risk < float(curr[:n_risk].sum()) - 0.03)
            )
            calm_entry = bool(np.all((sig_diff <= 1e-6) | np.isnan(q80_arr[i]) | (vf_arr[i] <= 0.88 * q80_arr[i])))
            risk_curr = float(curr[:n_risk].sum())
            risk_half_l1 = 0.5 * float(np.abs(final_risk - curr[:n_risk]).sum())

            def sweep_def(r_v: np.ndarray, force_align: bool = False) -> np.ndarray:
                r_v_clean = project_two_stage(r_v, indiv_caps, eff_max_risk)
                rem = max(0.0, 1.0 - float(r_v_clean.sum()))
                if n_def == 1:
                    return np.concatenate([r_v_clean, [rem]])
                b10_sh = def_s[1]
                cash_sh = def_s[2] if n_def >= 3 else 0.0
                curr_cash = curr[n_risk + 2] if n_def >= 3 else 0.0
                if force_align or cash_sh > 0.01 or curr_cash > 0.01:
                    return np.concatenate([r_v_clean, rem * def_s])
                w10_keep = min(curr[n_risk + 1], rem)
                if abs(w10_keep - rem * b10_sh) >= 0.22:
                    return np.concatenate([r_v_clean, rem * def_s])
                return np.concatenate([r_v_clean, [max(0.0, rem - w10_keep), w10_keep] + ([0.0] if n_def >= 3 else [])])

            changed_sleeves = [
                s_name for s_name in sleeve_names
                if np.any(np.abs(sig_diff[sleeve_idx_map[s_name]]) >= 0.20)
            ]

            if stop_to_zero or blowoff_cut or urgent_cut or (tot_risk < risk_curr - 0.050):
                curr = sweep_def(final_risk, force_align=False)
            elif len(changed_sleeves) > 0 and risk_half_l1 >= 0.025:
                lam = 0.90 if (calm_entry or np.any(sig_diff < -0.20)) else 0.65
                cand_r = curr[:n_risk].copy()
                for s_name in sleeve_names:
                    ks = sleeve_idx_map[s_name]
                    if (s_name in changed_sleeves) or (float(np.abs(final_risk[ks] - curr[ks]).sum()) >= 0.045):
                        cand_r[ks] = (1.0 - lam) * curr[ks] + lam * final_risk[ks]
                curr = sweep_def(cand_r, force_align=False)
            elif risk_half_l1 >= drift_buf:
                curr = sweep_def(0.45 * curr[:n_risk] + 0.55 * final_risk, force_align=False)
            elif is_me and n_def >= 2:
                if abs(curr[n_risk + 1] - rem_def * def_s[1]) >= 0.22 or (n_def >= 3 and abs(curr[n_risk + 2] - rem_def * def_s[2]) >= 0.20):
                    curr = sweep_def(curr[:n_risk], force_align=True)

        w_out[i] = curr
        prev_sig = eff_sig.copy()

    return pd.DataFrame(w_out, index=idx, columns=all_cols)
