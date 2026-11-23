from .data_loader import load_domestic_etf_panel, validate_live_panel_hygiene
from .signals import infer_cn_month_end_dates, compute_five_rule_signals
from .allocator_p62 import build_p62_golive_allocator, proj_capped_simplex, shrunk_downside_semicov, solve_erc_weights
from .backtester import (
    evaluate_t1_portfolio,
    evaluate_execution_modes,
    simulate_100share_lot_ledger,
    calc_perf_metrics,
    build_course1_true_a0,
)

__version__ = "6.2.0"
