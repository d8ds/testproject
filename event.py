"""
Event Study Analysis Template
==============================
适用信号格式: polars DataFrame with columns [qid, date, value, data_date]
- qid       : 股票/资产标识符
- date      : 信号所属的交易日（持仓日）
- value     : 信号值（如情绪分数、文本特征等）
- data_date : 信号实际披露/可知日期（事件时间，用于避免 look-ahead bias）

主要功能:
1. 构建事件窗口面板（Event Panel）
2. Short-term tilt 分析（事件后 1-5 日）
3. Long-term tilt 分析（事件后 1-60 日）
4. CAR（累计超额收益）按分位数分组展示
5. 统计显著性检验（t-test / bootstrap）
6. 可视化输出
"""

import polars as pl
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from bisect import bisect_right
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 0. 配置区（按需修改）
# ============================================================

class EventStudyConfig:
    # 事件窗口
    PRE_WINDOW  = 20    # 事件前交易日数
    POST_SHORT  = 5     # short-term 截止日
    POST_LONG   = 60    # long-term 截止日

    # 分组
    N_QUANTILES = 5     # 分位数组数（5=五分位）
    LONG_LEG    = "Q5"  # 多头分组
    SHORT_LEG   = "Q1"  # 空头分组

    # 收益计算方式: "raw" | "market_adj" | "factor_adj"
    RETURN_TYPE = "market_adj"

    # 最小事件间隔（同一 qid 两次事件之间最少相隔天数，-1 表示不过滤）
    MIN_EVENT_GAP = 30

    # Bootstrap 显著性检验次数
    N_BOOTSTRAP = 500

    # 图表输出路径（None 则只展示不保存）
    PLOT_SAVE_PATH = "event_study_results.png"


# ============================================================
# 1. 工具函数
# ============================================================

def build_trading_calendar(returns: pl.DataFrame, date_col: str = "date") -> list:
    """从收益率 DataFrame 提取交易日历（排序后的唯一日期列表）"""
    return sorted(returns[date_col].unique().to_list())


def get_nth_trading_day(trading_days: list, anchor_date, offset: int) -> Optional[object]:
    """
    获取 anchor_date 之后（含当日）第 offset 个交易日。
    offset=0 → anchor_date 本身或其后第一个交易日
    offset=1 → anchor_date 之后第一个交易日（常用于 t=0）
    """
    idx = bisect_right(trading_days, anchor_date)  # anchor_date 后的第一个位置
    target = idx + offset - 1
    if 0 <= target < len(trading_days):
        return trading_days[target]
    return None


def filter_min_gap(signal_df: pl.DataFrame, min_gap_days: int) -> pl.DataFrame:
    """
    过滤同一 qid 过于密集的事件，保留每个 qid 间隔 >= min_gap_days 的事件。
    """
    if min_gap_days <= 0:
        return signal_df

    result = []
    df_pd = signal_df.sort(["qid", "data_date"]).to_pandas()
    for qid, grp in df_pd.groupby("qid"):
        grp = grp.sort_values("data_date").reset_index(drop=True)
        keep = [True]
        last_kept = grp.loc[0, "data_date"]
        for i in range(1, len(grp)):
            gap = (grp.loc[i, "data_date"] - last_kept).days
            if gap >= min_gap_days:
                keep.append(True)
                last_kept = grp.loc[i, "data_date"]
            else:
                keep.append(False)
        result.append(grp[keep])

    return pl.from_pandas(pd.concat(result).reset_index(drop=True))


# ============================================================
# 2. 构建事件面板
# ============================================================

def build_event_panel(
    signal_df: pl.DataFrame,
    returns_df: pl.DataFrame,
    market_ret_df: Optional[pl.DataFrame],
    config: EventStudyConfig,
) -> pl.DataFrame:
    """
    构建以 data_date 为事件零点的面板数据。

    参数:
        signal_df    : [qid, date, value, data_date]
        returns_df   : [qid, date, ret]
        market_ret_df: [date, mkt_ret]（RETURN_TYPE="market_adj" 时需要）
        config       : EventStudyConfig

    返回:
        event_panel  : [qid, data_date, event_day, date, ret, ar, signal]
    """
    trading_days = build_trading_calendar(returns_df)
    pre, post = config.PRE_WINDOW, config.POST_LONG

    # 过滤过密事件
    signal_filtered = filter_min_gap(signal_df, config.MIN_EVENT_GAP)
    print(f"[Info] Events after gap filter: {len(signal_filtered)}")

    # 转成 pandas 方便行级操作
    sig_pd = signal_filtered.to_pandas()
    ret_pd = returns_df.to_pandas().set_index(["qid", "date"])["ret"]

    if market_ret_df is not None:
        mkt_pd = market_ret_df.to_pandas().set_index("date")["mkt_ret"]
    else:
        mkt_pd = None

    records = []
    for _, row in sig_pd.iterrows():
        qid       = row["qid"]
        data_date = row["data_date"]
        signal    = row["value"]

        # t=0 → data_date 之后第一个交易日（避免 look-ahead bias）
        anchor_idx = bisect_right(trading_days, data_date)

        for t in range(-pre, post + 1):
            target_idx = anchor_idx + t
            if target_idx < 0 or target_idx >= len(trading_days):
                continue
            trade_date = trading_days[target_idx]

            # 原始收益
            try:
                ret = ret_pd.loc[(qid, trade_date)]
            except KeyError:
                ret = np.nan

            # 超额收益
            if config.RETURN_TYPE == "market_adj" and mkt_pd is not None:
                try:
                    ar = ret - mkt_pd.loc[trade_date]
                except KeyError:
                    ar = ret
            else:
                ar = ret  # raw 或未提供市场收益时直接用原始收益

            records.append({
                "qid":       qid,
                "data_date": data_date,
                "event_day": t,
                "date":      trade_date,
                "ret":       ret,
                "ar":        ar,
                "signal":    signal,
            })

    panel = pl.DataFrame(records)
    print(f"[Info] Event panel built: {len(panel)} rows, "
          f"{panel['data_date'].n_unique()} unique events")
    return panel


# ============================================================
# 3. 分位数分组
# ============================================================

def assign_quantiles(
    event_panel: pl.DataFrame,
    config: EventStudyConfig,
) -> pl.DataFrame:
    """
    在每个 data_date（事件截面）内对信号值做分位数分组，
    避免跨时间比较信号值绝对大小带来的偏差。
    """
    n = config.N_QUANTILES
    labels = [f"Q{i+1}" for i in range(n)]

    # 用 pandas 做截面分组（polars qcut over group 有限制）
    df_pd = event_panel.to_pandas()

    def cut_group(grp):
        try:
            grp["quantile"] = pd.qcut(
                grp["signal"],
                q=n,
                labels=labels,
                duplicates="drop"
            ).astype(str)
        except Exception:
            grp["quantile"] = "Q_NA"
        return grp

    df_pd = df_pd.groupby("data_date", group_keys=False).apply(cut_group)
    return pl.from_pandas(df_pd)


# ============================================================
# 4. CAR 计算
# ============================================================

def compute_car(
    event_panel: pl.DataFrame,
    return_col: str = "ar",
) -> pl.DataFrame:
    """
    计算每个分位组在每个 event_day 的平均收益 & 累计异常收益（CAR）。
    """
    avg = (
        event_panel
        .group_by(["quantile", "event_day"])
        .agg([
            pl.col(return_col).mean().alias("avg_ret"),
            pl.col(return_col).std().alias("std_ret"),
            pl.col(return_col).count().alias("n"),
        ])
        .sort(["quantile", "event_day"])
    )

    # CAR = cum_sum(avg_ret) within each quantile
    car = avg.with_columns(
        pl.col("avg_ret").cum_sum().over("quantile").alias("CAR")
    )

    # 标准误（用于置信区间带）
    car = car.with_columns(
        (pl.col("std_ret") / pl.col("n").sqrt()).alias("se")
    )

    return car


# ============================================================
# 5. Short/Long-term Tilt 汇总
# ============================================================

def compute_tilt_summary(
    event_panel: pl.DataFrame,
    config: EventStudyConfig,
    return_col: str = "ar",
) -> dict:
    """
    分别计算 short-term 和 long-term 窗口内的累计收益，
    并做 long-short spread 的 t 检验。

    返回 dict 包含：
        short_by_quantile, long_by_quantile,
        short_ls_ttest, long_ls_ttest,
        short_ls_bootstrap, long_ls_bootstrap
    """
    results = {}

    for label, (t_start, t_end) in [
        ("short", (1, config.POST_SHORT)),
        ("long",  (1, config.POST_LONG)),
    ]:
        window = event_panel.filter(
            pl.col("event_day").is_between(t_start, t_end)
        )

        # 每个事件的累计收益
        ev_cum = (
            window
            .group_by(["qid", "data_date", "quantile"])
            .agg(pl.col(return_col).sum().alias("cum_ret"))
        )

        # 各分位组均值
        by_q = (
            ev_cum
            .group_by("quantile")
            .agg([
                pl.col("cum_ret").mean().alias("mean_ret"),
                pl.col("cum_ret").std().alias("std_ret"),
                pl.col("cum_ret").count().alias("n"),
            ])
            .sort("quantile")
        )
        results[f"{label}_by_quantile"] = by_q

        # Long-short spread
        long_leg  = ev_cum.filter(pl.col("quantile") == config.LONG_LEG)["cum_ret"]
        short_leg = ev_cum.filter(pl.col("quantile") == config.SHORT_LEG)["cum_ret"]

        if len(long_leg) > 1 and len(short_leg) > 1:
            tstat, pval = stats.ttest_ind(long_leg.to_numpy(), short_leg.to_numpy())
            results[f"{label}_ls_ttest"] = {
                "mean_spread": long_leg.mean() - short_leg.mean(),
                "t_stat": tstat,
                "p_value": pval,
            }
            # Bootstrap
            boot_spreads = []
            l_arr, s_arr = long_leg.to_numpy(), short_leg.to_numpy()
            rng = np.random.default_rng(42)
            for _ in range(config.N_BOOTSTRAP):
                l_sample = rng.choice(l_arr, size=len(l_arr), replace=True)
                s_sample = rng.choice(s_arr, size=len(s_arr), replace=True)
                boot_spreads.append(l_sample.mean() - s_sample.mean())
            boot_spreads = np.array(boot_spreads)
            results[f"{label}_ls_bootstrap"] = {
                "mean_spread": boot_spreads.mean(),
                "ci_5":  np.percentile(boot_spreads, 5),
                "ci_95": np.percentile(boot_spreads, 95),
            }

    return results


# ============================================================
# 6. 可视化
# ============================================================

def plot_event_study(
    car: pl.DataFrame,
    tilt_summary: dict,
    config: EventStudyConfig,
):
    """
    三图布局：
    1. 所有分位组的 CAR（含 Q5-Q1 spread）
    2. Short-term vs Long-term 各分位组均值柱状图
    3. Bootstrap 分布图（L/S spread）
    """
    car_pd = car.to_pandas()
    quantiles = sorted(car_pd["quantile"].unique())
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(quantiles)))
    color_map = dict(zip(quantiles, colors))

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 图1：完整 CAR 曲线 ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])

    for q, color in zip(quantiles, colors):
        sub = car_pd[car_pd["quantile"] == q].sort_values("event_day")
        ax1.plot(sub["event_day"], sub["CAR"] * 100,
                 label=q, color=color, lw=2)
        # 95% CI 带（± 1.96 SE，累计近似）
        ci = sub["se"].cumsum() * 1.96 * 100
        ax1.fill_between(sub["event_day"],
                         sub["CAR"] * 100 - ci,
                         sub["CAR"] * 100 + ci,
                         alpha=0.08, color=color)

    # Long-short spread
    q5 = car_pd[car_pd["quantile"] == config.LONG_LEG].set_index("event_day")["CAR"]
    q1 = car_pd[car_pd["quantile"] == config.SHORT_LEG].set_index("event_day")["CAR"]
    common_days = q5.index.intersection(q1.index).sort_values()
    spread = (q5.loc[common_days] - q1.loc[common_days]) * 100
    ax1.plot(common_days, spread, "k--", lw=2.5, label=f"{config.LONG_LEG}-{config.SHORT_LEG} Spread")

    ax1.axvline(0, color="black", linestyle=":", lw=1.5, label="Event (t=0)")
    ax1.axvspan(1, config.POST_SHORT, alpha=0.06, color="blue", label=f"Short [{1},{config.POST_SHORT}]")
    ax1.axvspan(config.POST_SHORT + 1, config.POST_LONG, alpha=0.04,
                color="green", label=f"Long [{config.POST_SHORT+1},{config.POST_LONG}]")
    ax1.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax1.set_title("Cumulative Abnormal Return (CAR) by Signal Quantile", fontsize=13)
    ax1.set_xlabel("Event Day (t=0: first trading day after disclosure)")
    ax1.set_ylabel("CAR (%)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ── 图2：CAR 放大 pre-window ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    pre_window_pd = car_pd[car_pd["event_day"].between(-config.PRE_WINDOW, 0)]
    for q, color in zip(quantiles, colors):
        sub = pre_window_pd[pre_window_pd["quantile"] == q].sort_values("event_day")
        ax2.plot(sub["event_day"], sub["CAR"] * 100, color=color, lw=1.8, label=q)
    ax2.axvline(0, color="black", linestyle=":", lw=1.5)
    ax2.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax2.set_title("Pre-Event CAR (Drift Check)", fontsize=11)
    ax2.set_xlabel("Event Day")
    ax2.set_ylabel("CAR (%)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── 图3：Short-term 柱状图 ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if "short_by_quantile" in tilt_summary:
        sq = tilt_summary["short_by_quantile"].to_pandas().sort_values("quantile")
        bars = ax3.bar(
            sq["quantile"],
            sq["mean_ret"] * 100,
            color=[color_map.get(q, "gray") for q in sq["quantile"]],
            edgecolor="black", linewidth=0.5
        )
        ax3.errorbar(
            sq["quantile"],
            sq["mean_ret"] * 100,
            yerr=sq["std_ret"] / np.sqrt(sq["n"]) * 1.96 * 100,
            fmt="none", color="black", capsize=4
        )
        ax3.axhline(0, color="gray", lw=0.8, linestyle="--")
        ax3.set_title(f"Short-term CAR [1, {config.POST_SHORT}d] by Quantile", fontsize=10)
        ax3.set_ylabel("Mean CAR (%)")
        if "short_ls_ttest" in tilt_summary:
            t = tilt_summary["short_ls_ttest"]
            ax3.set_xlabel(
                f"L/S Spread: {t['mean_spread']*100:.2f}%  "
                f"(t={t['t_stat']:.2f}, p={t['p_value']:.3f})"
            )
        ax3.grid(True, alpha=0.3, axis="y")

    # ── 图4：Long-term 柱状图 ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if "long_by_quantile" in tilt_summary:
        lq = tilt_summary["long_by_quantile"].to_pandas().sort_values("quantile")
        ax4.bar(
            lq["quantile"],
            lq["mean_ret"] * 100,
            color=[color_map.get(q, "gray") for q in lq["quantile"]],
            edgecolor="black", linewidth=0.5
        )
        ax4.errorbar(
            lq["quantile"],
            lq["mean_ret"] * 100,
            yerr=lq["std_ret"] / np.sqrt(lq["n"]) * 1.96 * 100,
            fmt="none", color="black", capsize=4
        )
        ax4.axhline(0, color="gray", lw=0.8, linestyle="--")
        ax4.set_title(f"Long-term CAR [1, {config.POST_LONG}d] by Quantile", fontsize=10)
        ax4.set_ylabel("Mean CAR (%)")
        if "long_ls_ttest" in tilt_summary:
            t = tilt_summary["long_ls_ttest"]
            ax4.set_xlabel(
                f"L/S Spread: {t['mean_spread']*100:.2f}%  "
                f"(t={t['t_stat']:.2f}, p={t['p_value']:.3f})"
            )
        ax4.grid(True, alpha=0.3, axis="y")

    # ── 图5：Bootstrap 分布 ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    for label, color, ls in [("short", "steelblue", "-"), ("long", "forestgreen", "--")]:
        key = f"{label}_ls_bootstrap"
        if key in tilt_summary:
            b = tilt_summary[key]
            # 重新 bootstrap（这里只画 CI 区间示意）
            mu   = b["mean_spread"] * 100
            ci_l = b["ci_5"] * 100
            ci_u = b["ci_95"] * 100
            ax5.barh(
                label,
                mu,
                xerr=[[mu - ci_l], [ci_u - mu]],
                color=color, alpha=0.7, height=0.4,
                ecolor="black", capsize=6,
                label=f"{label}: {mu:.2f}% [{ci_l:.2f}%, {ci_u:.2f}%]"
            )
    ax5.axvline(0, color="black", lw=1, linestyle="--")
    ax5.set_title("Bootstrap L/S Spread (90% CI)", fontsize=10)
    ax5.set_xlabel("Mean Spread (%)")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Event Study Analysis", fontsize=15, y=1.01, fontweight="bold")

    if config.PLOT_SAVE_PATH:
        plt.savefig(config.PLOT_SAVE_PATH, dpi=150, bbox_inches="tight")
        print(f"[Info] Plot saved to: {config.PLOT_SAVE_PATH}")
    plt.show()


# ============================================================
# 7. 主流程
# ============================================================

def run_event_study(
    signal_df: pl.DataFrame,
    returns_df: pl.DataFrame,
    market_ret_df: Optional[pl.DataFrame] = None,
    config: Optional[EventStudyConfig] = None,
) -> dict:
    """
    主入口。

    参数:
        signal_df     : [qid, date, value, data_date]
        returns_df    : [qid, date, ret]
        market_ret_df : [date, mkt_ret]（可选，RETURN_TYPE="market_adj" 时使用）
        config        : EventStudyConfig（默认使用类内默认值）

    返回:
        dict 包含 event_panel, car, tilt_summary
    """
    if config is None:
        config = EventStudyConfig()

    print("=" * 60)
    print("Step 1/4: Building event panel...")
    event_panel = build_event_panel(signal_df, returns_df, market_ret_df, config)

    print("Step 2/4: Assigning quantiles...")
    event_panel = assign_quantiles(event_panel, config)

    print("Step 3/4: Computing CAR...")
    car = compute_car(event_panel, return_col="ar")

    print("Step 4/4: Computing tilt summary & significance tests...")
    tilt_summary = compute_tilt_summary(event_panel, config, return_col="ar")

    # 打印汇总
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for period in ["short", "long"]:
        key = f"{period}_ls_ttest"
        if key in tilt_summary:
            t = tilt_summary[key]
            sig = "***" if t["p_value"] < 0.01 else ("**" if t["p_value"] < 0.05 else
                  ("*" if t["p_value"] < 0.10 else ""))
            window = f"[1, {config.POST_SHORT}d]" if period == "short" else f"[1, {config.POST_LONG}d]"
            print(f"  {period.upper()}-TERM {window}: "
                  f"Spread={t['mean_spread']*100:.3f}%  "
                  f"t={t['t_stat']:.2f}  p={t['p_value']:.4f} {sig}")
    print("=" * 60)

    plot_event_study(car, tilt_summary, config)

    return {
        "event_panel":  event_panel,
        "car":          car,
        "tilt_summary": tilt_summary,
    }


# ============================================================
# 8. 快速示例（生成模拟数据测试）
# ============================================================

if __name__ == "__main__":
    import datetime

    np.random.seed(42)

    # ── 生成交易日历（2020-01-01 到 2023-12-31）──
    all_days = pd.bdate_range("2020-01-01", "2023-12-31").tolist()
    all_days = [d.date() for d in all_days]

    # ── 模拟 returns：500 只股票，所有交易日 ──
    n_stocks = 200
    qids = [f"Q{str(i).zfill(4)}" for i in range(n_stocks)]
    ret_records = []
    for d in all_days:
        for qid in qids:
            ret_records.append({
                "qid":  qid,
                "date": d,
                "ret":  np.random.normal(0.0005, 0.02),
            })
    returns_df = pl.DataFrame(ret_records).with_columns(
        pl.col("date").cast(pl.Date)
    )

    # ── 模拟市场收益 ──
    mkt_records = [{"date": d, "mkt_ret": np.random.normal(0.0004, 0.01)} for d in all_days]
    market_ret_df = pl.DataFrame(mkt_records).with_columns(
        pl.col("date").cast(pl.Date)
    )

    # ── 模拟信号（每只股票约每月一个事件，信号值正相关于未来 5 日收益）──
    sig_records = []
    for qid in qids:
        # 随机抽取约 30 个事件日
        event_days = sorted(np.random.choice(range(20, len(all_days) - 70),
                                              size=30, replace=False))
        for idx in event_days:
            data_date = all_days[idx]
            # 信号值：加入部分前瞻性（模拟信号有预测力）
            future_ret = sum(
                ret_records[i]["ret"]
                for i in range(len(ret_records))
                if ret_records[i]["qid"] == qid
                   and ret_records[i]["date"] in all_days[idx+1:idx+6]
            ) if False else np.random.normal(0, 1)  # 简化：纯噪声信号作为 baseline
            sig_records.append({
                "qid":       qid,
                "date":      data_date,
                "value":     float(future_ret + np.random.normal(0, 0.5)),
                "data_date": data_date,
            })

    signal_df = pl.DataFrame(sig_records).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("data_date").cast(pl.Date),
    )

    print(f"Signal shape: {signal_df.shape}")
    print(signal_df.head(5))

    # ── 运行 event study ──
    config = EventStudyConfig()
    config.POST_LONG = 40       # 缩短方便测试
    config.MIN_EVENT_GAP = 15
    config.N_BOOTSTRAP = 200

    results = run_event_study(
        signal_df=signal_df,
        returns_df=returns_df,
        market_ret_df=market_ret_df,
        config=config,
    )
