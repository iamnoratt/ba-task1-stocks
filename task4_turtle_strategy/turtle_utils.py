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
class TurtleParams:
    entry_window: int
    exit_window: int
    atr_window: int
    stop_atr_multiplier: float


def load_price_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    numeric_columns = ["open", "close", "high", "low", "volume", "change", "pct_change", "amplitude_pct"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def compute_atr(df: pd.DataFrame, atr_window: int) -> pd.Series:
    true_range = compute_true_range(df)
    return true_range.rolling(atr_window).mean()


def apply_turtle_strategy(
    df: pd.DataFrame,
    entry_window: int,
    exit_window: int,
    atr_window: int,
    stop_atr_multiplier: float,
) -> pd.DataFrame:
    if exit_window >= entry_window:
        raise ValueError("exit_window should be smaller than entry_window for a classic turtle setup")

    result = df.copy()
    result["true_range"] = compute_true_range(result)
    result["atr"] = result["true_range"].rolling(atr_window).mean()
    result["entry_high"] = result["high"].rolling(entry_window).max().shift(1)
    result["entry_low"] = result["low"].rolling(entry_window).min().shift(1)
    result["exit_low"] = result["low"].rolling(exit_window).min().shift(1)
    result["position"] = 0.0
    result["buy_signal"] = False
    result["sell_signal"] = False
    result["sell_reason"] = ""
    result["stop_line"] = np.nan

    in_position = False
    stop_line = np.nan

    for idx, row in result.iterrows():
        ready = (
            pd.notna(row["entry_high"])
            and pd.notna(row["exit_low"])
            and pd.notna(row["atr"])
        )
        if not ready:
            result.at[idx, "position"] = 1.0 if in_position else 0.0
            result.at[idx, "stop_line"] = stop_line
            continue

        if not in_position:
            if row["close"] > row["entry_high"]:
                in_position = True
                stop_line = row["close"] - stop_atr_multiplier * row["atr"]
                result.at[idx, "buy_signal"] = True
        else:
            trailing_stop = row["close"] - stop_atr_multiplier * row["atr"]
            stop_line = max(stop_line, trailing_stop) if pd.notna(stop_line) else trailing_stop

            exit_by_channel = row["close"] < row["exit_low"]
            exit_by_stop = row["close"] < stop_line

            if exit_by_channel or exit_by_stop:
                result.at[idx, "sell_signal"] = True
                if exit_by_channel and exit_by_stop:
                    result.at[idx, "sell_reason"] = "通道+止损"
                elif exit_by_channel:
                    result.at[idx, "sell_reason"] = "通道跌破"
                else:
                    result.at[idx, "sell_reason"] = "ATR止损"
                in_position = False
                stop_line = np.nan

        result.at[idx, "position"] = 1.0 if in_position else 0.0
        result.at[idx, "stop_line"] = stop_line

    result["exec_position"] = result["position"].shift(1).fillna(0.0)
    result["daily_return"] = result["close"].pct_change().fillna(0.0)
    result["strategy_return"] = result["exec_position"] * result["daily_return"]
    result["benchmark_wealth"] = (1.0 + result["daily_return"]).cumprod()
    result["strategy_wealth"] = (1.0 + result["strategy_return"]).cumprod()
    result["benchmark_cumulative_return"] = result["benchmark_wealth"] - 1.0
    result["strategy_cumulative_return"] = result["strategy_wealth"] - 1.0
    result["rolling_drawdown"] = result["strategy_wealth"] / result["strategy_wealth"].cummax() - 1.0
    return result


def compute_max_drawdown(wealth: pd.Series) -> float:
    return float((wealth / wealth.cummax() - 1.0).min())


def compute_sharpe_ratio(strategy_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = strategy_returns - risk_free_rate / TRADING_DAYS
    volatility = excess.std()
    if volatility == 0 or np.isnan(volatility):
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / volatility)


def compute_annualized_return(wealth: pd.Series) -> float:
    if len(wealth) <= 1:
        return 0.0
    total_wealth = float(wealth.iloc[-1])
    years = (len(wealth) - 1) / TRADING_DAYS
    if years <= 0 or total_wealth <= 0:
        return 0.0
    return total_wealth ** (1 / years) - 1.0


def summarize_strategy(strategy_df: pd.DataFrame, stock_label: str, params: TurtleParams) -> dict[str, object]:
    strategy_wealth = strategy_df["strategy_wealth"]
    benchmark_wealth = strategy_df["benchmark_wealth"]
    sell_reason_counts = strategy_df.loc[strategy_df["sell_signal"], "sell_reason"].value_counts()

    return {
        "stock": stock_label,
        "entry_window": params.entry_window,
        "exit_window": params.exit_window,
        "atr_window": params.atr_window,
        "stop_atr_multiplier": params.stop_atr_multiplier,
        "sample_start": strategy_df["date"].iloc[0].strftime("%Y-%m-%d"),
        "sample_end": strategy_df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "buy_count": int(strategy_df["buy_signal"].sum()),
        "sell_count": int(strategy_df["sell_signal"].sum()),
        "stop_exit_count": int(sell_reason_counts.get("ATR止损", 0) + sell_reason_counts.get("通道+止损", 0)),
        "channel_exit_count": int(sell_reason_counts.get("通道跌破", 0) + sell_reason_counts.get("通道+止损", 0)),
        "benchmark_cumulative_return": float(benchmark_wealth.iloc[-1] - 1.0),
        "strategy_cumulative_return": float(strategy_wealth.iloc[-1] - 1.0),
        "max_drawdown": compute_max_drawdown(strategy_wealth),
        "sharpe_ratio": compute_sharpe_ratio(strategy_df["strategy_return"]),
        "annualized_return": compute_annualized_return(strategy_wealth),
        "latest_atr": float(strategy_df["atr"].dropna().iloc[-1]),
    }


def save_signal_figure(
    strategy_df: pd.DataFrame,
    stock_label: str,
    params: TurtleParams,
    output_base: Path,
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.5, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2, 0.8]},
    )
    ax_price, ax_atr, ax_position = axes

    ax_price.plot(strategy_df["date"], strategy_df["close"], label="收盘价", color="#111827", linewidth=1.8)
    ax_price.plot(strategy_df["date"], strategy_df["entry_high"], label=f"{params.entry_window}日上轨", color="#2563eb", linewidth=1.5)
    ax_price.plot(strategy_df["date"], strategy_df["exit_low"], label=f"{params.exit_window}日下轨", color="#f97316", linewidth=1.5)
    ax_price.plot(strategy_df["date"], strategy_df["stop_line"], label=f"{params.stop_atr_multiplier}ATR止损线", color="#dc2626", linewidth=1.2, linestyle="--")

    buys = strategy_df[strategy_df["buy_signal"]]
    sells = strategy_df[strategy_df["sell_signal"]]
    ax_price.scatter(buys["date"], buys["close"], marker="^", s=85, color="#16a34a", label="买入信号", zorder=6)
    ax_price.scatter(sells["date"], sells["close"], marker="v", s=85, color="#b91c1c", label="卖出信号", zorder=6)
    ax_price.set_title(
        f"图1 {stock_label} 海龟策略信号图（{params.entry_window}/{params.exit_window}/ATR{params.atr_window}）",
        fontweight="bold",
    )
    ax_price.set_ylabel("价格")
    ax_price.grid(alpha=0.25)
    ax_price.legend(loc="upper left", ncol=3)

    ax_atr.plot(strategy_df["date"], strategy_df["atr"], color="#7c3aed", linewidth=1.6)
    ax_atr.set_title("ATR 波动指标", fontsize=11)
    ax_atr.set_ylabel("ATR")
    ax_atr.grid(alpha=0.25)

    ax_position.step(strategy_df["date"], strategy_df["position"], where="post", color="#0f766e", linewidth=1.6)
    ax_position.fill_between(strategy_df["date"], strategy_df["position"], step="post", alpha=0.18, color="#0f766e")
    ax_position.set_ylim(-0.05, 1.05)
    ax_position.set_yticks([0, 1])
    ax_position.set_yticklabels(["空仓", "持仓"])
    ax_position.set_ylabel("仓位")
    ax_position.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def save_equity_figure(
    strategy_df: pd.DataFrame,
    stock_label: str,
    params: TurtleParams,
    output_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.plot(strategy_df["date"], strategy_df["benchmark_wealth"], label="买入并持有", color="#94a3b8", linewidth=1.8)
    ax.plot(strategy_df["date"], strategy_df["strategy_wealth"], label="海龟策略", color="#16a34a", linewidth=2.0)
    ax.set_title(
        f"图2 {stock_label} 海龟策略净值与基准净值对比（{params.entry_window}/{params.exit_window}/ATR{params.atr_window}）",
        fontweight="bold",
    )
    ax.set_ylabel("财富净值")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(metrics_df: pd.DataFrame, output_base: Path) -> None:
    labels = [
        f"{row.stock}\n{int(row.entry_window)}/{int(row.exit_window)}"
        for row in metrics_df.itertuples(index=False)
    ]
    x = np.arange(len(labels))
    colors = ["#16a34a" if "中际" in label else "#2563eb" for label in labels]

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 11), sharex=True)
    axes[0].bar(x, metrics_df["strategy_cumulative_return"], color=colors)
    axes[0].set_title("图3 海龟策略参数组合绩效对比", fontweight="bold")
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
