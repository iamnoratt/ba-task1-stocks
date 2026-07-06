from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Microsoft YaHei",
    "Heiti TC",
    "Arial Unicode MS",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


TRADING_DAYS = 252

STOCK_CONFIG = {
    "china_jushi": {
        "label": "中国巨石（600176）",
        "symbol": "600176",
        "file_name": "china-jushi_600176.csv",
    },
    "zhongji_xuchuang": {
        "label": "中际旭创（300308）",
        "symbol": "300308",
        "file_name": "zhongji-xuchuang_300308.csv",
    },
}


@dataclass(frozen=True)
class StrategyParams:
    short_window: int
    long_window: int


def load_price_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    numeric_columns = ["open", "close", "high", "low", "volume", "change", "pct_change", "amplitude_pct"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def apply_dual_ma_strategy(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")

    result = df.copy()
    result["ma_short"] = result["close"].rolling(short_window).mean()
    result["ma_long"] = result["close"].rolling(long_window).mean()

    valid_ma = result["ma_short"].notna() & result["ma_long"].notna()
    result["signal"] = np.where(valid_ma & (result["ma_short"] > result["ma_long"]), 1.0, 0.0)
    result["signal_change"] = result["signal"].diff().fillna(0.0)
    result["buy_signal"] = result["signal_change"] > 0
    result["sell_signal"] = result["signal_change"] < 0

    # Shift position by one day to avoid using same-day close information for execution.
    result["position"] = result["signal"].shift(1).fillna(0.0)
    result["daily_return"] = result["close"].pct_change().fillna(0.0)
    result["strategy_return"] = result["position"] * result["daily_return"]
    result["benchmark_wealth"] = (1.0 + result["daily_return"]).cumprod()
    result["strategy_wealth"] = (1.0 + result["strategy_return"]).cumprod()
    result["benchmark_cumulative_return"] = result["benchmark_wealth"] - 1.0
    result["strategy_cumulative_return"] = result["strategy_wealth"] - 1.0
    result["rolling_drawdown"] = result["strategy_wealth"] / result["strategy_wealth"].cummax() - 1.0
    return result


def compute_max_drawdown(wealth: pd.Series) -> float:
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def compute_sharpe_ratio(strategy_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = strategy_returns - risk_free_rate / TRADING_DAYS
    volatility = excess.std()
    if volatility == 0 or np.isnan(volatility):
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / volatility)


def compute_annualized_return(wealth: pd.Series) -> float:
    if len(wealth) <= 1:
        return 0.0
    total_return = float(wealth.iloc[-1])
    years = (len(wealth) - 1) / TRADING_DAYS
    if years <= 0 or total_return <= 0:
        return 0.0
    return total_return ** (1 / years) - 1.0


def summarize_strategy(strategy_df: pd.DataFrame, stock_label: str, params: StrategyParams) -> dict[str, object]:
    strategy_wealth = strategy_df["strategy_wealth"]
    benchmark_wealth = strategy_df["benchmark_wealth"]

    return {
        "stock": stock_label,
        "short_window": params.short_window,
        "long_window": params.long_window,
        "sample_start": strategy_df["date"].iloc[0].strftime("%Y-%m-%d"),
        "sample_end": strategy_df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "buy_count": int(strategy_df["buy_signal"].sum()),
        "sell_count": int(strategy_df["sell_signal"].sum()),
        "benchmark_cumulative_return": float(benchmark_wealth.iloc[-1] - 1.0),
        "strategy_cumulative_return": float(strategy_wealth.iloc[-1] - 1.0),
        "max_drawdown": compute_max_drawdown(strategy_wealth),
        "sharpe_ratio": compute_sharpe_ratio(strategy_df["strategy_return"]),
        "annualized_return": compute_annualized_return(strategy_wealth),
    }


def save_signal_figure(
    strategy_df: pd.DataFrame,
    stock_label: str,
    params: StrategyParams,
    output_base: Path,
) -> None:
    fig, (ax_price, ax_signal) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_price.plot(strategy_df["date"], strategy_df["close"], label="收盘价", color="#1f2937", linewidth=1.8)
    ax_price.plot(
        strategy_df["date"],
        strategy_df["ma_short"],
        label=f"短均线 MA{params.short_window}",
        color="#ef4444",
        linewidth=1.5,
    )
    ax_price.plot(
        strategy_df["date"],
        strategy_df["ma_long"],
        label=f"长均线 MA{params.long_window}",
        color="#2563eb",
        linewidth=1.5,
    )

    buys = strategy_df[strategy_df["buy_signal"]]
    sells = strategy_df[strategy_df["sell_signal"]]
    ax_price.scatter(buys["date"], buys["close"], marker="^", s=80, color="#16a34a", label="买入信号（金叉）", zorder=5)
    ax_price.scatter(sells["date"], sells["close"], marker="v", s=80, color="#b91c1c", label="卖出信号（死叉）", zorder=5)

    ax_price.set_title(f"图1 {stock_label} 双均线策略交易信号图（MA{params.short_window}/MA{params.long_window}）", fontweight="bold")
    ax_price.set_ylabel("价格")
    ax_price.grid(alpha=0.25)
    ax_price.legend(loc="upper left", ncol=2)

    ax_signal.step(strategy_df["date"], strategy_df["position"], where="post", color="#7c3aed", linewidth=1.6)
    ax_signal.fill_between(strategy_df["date"], strategy_df["position"], step="post", alpha=0.18, color="#7c3aed")
    ax_signal.set_ylabel("仓位")
    ax_signal.set_ylim(-0.05, 1.05)
    ax_signal.set_yticks([0, 1])
    ax_signal.set_yticklabels(["空仓", "持仓"])
    ax_signal.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def save_equity_figure(
    strategy_df: pd.DataFrame,
    stock_label: str,
    params: StrategyParams,
    output_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.plot(strategy_df["date"], strategy_df["benchmark_wealth"], label="买入并持有", color="#94a3b8", linewidth=1.8)
    ax.plot(strategy_df["date"], strategy_df["strategy_wealth"], label="双均线策略", color="#16a34a", linewidth=2.0)
    ax.set_title(f"图2 {stock_label} 策略净值与基准净值对比（MA{params.short_window}/MA{params.long_window}）", fontweight="bold")
    ax.set_ylabel("财富净值")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(metrics_df: pd.DataFrame, output_base: Path) -> None:
    labels = [
        f"{row.stock}\nMA{int(row.short_window)}/{int(row.long_window)}"
        for row in metrics_df.itertuples(index=False)
    ]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 11), sharex=True)
    colors = ["#16a34a" if "中际" in label else "#2563eb" for label in labels]

    axes[0].bar(x, metrics_df["strategy_cumulative_return"], color=colors)
    axes[0].set_title("图3 双均线策略参数组合绩效对比", fontweight="bold")
    axes[0].set_ylabel("累计回报")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x, metrics_df["sharpe_ratio"], color=colors)
    axes[1].set_ylabel("夏普比率")
    axes[1].grid(axis="y", alpha=0.2)

    axes[2].bar(x, metrics_df["max_drawdown"], color=colors)
    axes[2].set_ylabel("最大回撤")
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=0)

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
