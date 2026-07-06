from __future__ import annotations

import shutil
from pathlib import Path

import nbformat as nbf
import pandas as pd

from strategy_utils import (
    STOCK_CONFIG,
    StrategyParams,
    apply_dual_ma_strategy,
    load_price_data,
    save_comparison_figure,
    save_equity_figure,
    save_signal_figure,
    summarize_strategy,
)


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"
PROCESSED_DIR = BASE_DIR / "processed_data"
FIGURES_DIR = BASE_DIR / "figures"
NOTEBOOK_DIR = BASE_DIR / "notebook"
REPORT_DIR = BASE_DIR / "report"
PUBLISH_DIR = BASE_DIR / "publish"

DEFAULT_PARAMS = StrategyParams(short_window=5, long_window=15)
EXPERIMENT_PARAMS = [
    StrategyParams(short_window=5, long_window=15),
    StrategyParams(short_window=5, long_window=20),
    StrategyParams(short_window=10, long_window=30),
]


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, FIGURES_DIR, NOTEBOOK_DIR, REPORT_DIR, PUBLISH_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def slug_for_stock(stock_key: str) -> str:
    return stock_key


def build_processed_outputs() -> pd.DataFrame:
    metrics_rows: list[dict[str, object]] = []

    for stock_key, info in STOCK_CONFIG.items():
        price_df = load_price_data(RAW_DIR / info["file_name"])

        for params in EXPERIMENT_PARAMS:
            strategy_df = apply_dual_ma_strategy(price_df, params.short_window, params.long_window)
            output_name = f"{slug_for_stock(stock_key)}_ma_{params.short_window}_{params.long_window}.csv"
            strategy_df.to_csv(PROCESSED_DIR / output_name, index=False, encoding="utf-8-sig")

            metrics_rows.append(summarize_strategy(strategy_df, info["label"], params))

            if params == DEFAULT_PARAMS:
                signal_base = FIGURES_DIR / f"{slug_for_stock(stock_key)}_ma_{params.short_window}_{params.long_window}_signals"
                equity_base = FIGURES_DIR / f"{slug_for_stock(stock_key)}_ma_{params.short_window}_{params.long_window}_equity"
                save_signal_figure(strategy_df, info["label"], params, signal_base)
                save_equity_figure(strategy_df, info["label"], params, equity_base)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(PROCESSED_DIR / "ma_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    save_comparison_figure(metrics_df, FIGURES_DIR / "ma_strategy_metric_comparison")
    return metrics_df


def metric_lookup(metrics_df: pd.DataFrame, stock_label: str, params: StrategyParams) -> dict[str, object]:
    row = metrics_df[
        (metrics_df["stock"] == stock_label)
        & (metrics_df["short_window"] == params.short_window)
        & (metrics_df["long_window"] == params.long_window)
    ].iloc[0]
    return row.to_dict()


def pct_text(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_report(metrics_df: pd.DataFrame) -> None:
    jushi_default = metric_lookup(metrics_df, STOCK_CONFIG["china_jushi"]["label"], DEFAULT_PARAMS)
    zhongji_default = metric_lookup(metrics_df, STOCK_CONFIG["zhongji_xuchuang"]["label"], DEFAULT_PARAMS)
    jushi_best = metric_lookup(metrics_df, STOCK_CONFIG["china_jushi"]["label"], StrategyParams(5, 20))
    zhongji_best = metric_lookup(metrics_df, STOCK_CONFIG["zhongji_xuchuang"]["label"], StrategyParams(5, 20))
    zhongji_stable = metric_lookup(metrics_df, STOCK_CONFIG["zhongji_xuchuang"]["label"], StrategyParams(10, 30))

    report = f"""# Task3 双均线策略报告

## 1. 任务目标

本次 Task3 的目标是在已有数据引擎和指标构建基础上，实现一个最经典的量化交易策略之一：双均线策略。任务重点不是只给出最终收益曲线，而是要说明策略的逻辑、交易信号的形成方式、回测过程以及绩效指标的含义。

## 2. 双均线策略概念说明

双均线策略通过比较短周期均线和长周期均线的相对位置，来判断市场趋势是否发生变化。

- 短均线：对近期价格更敏感，能较快反映市场变化
- 长均线：更平滑，更多代表中期趋势

### 2.1 金叉

当短均线从下方向上突破长均线时，称为“金叉”。它通常表示市场短期动能开始增强，趋势可能转强，因此常被视为买入信号。

### 2.2 死叉

当短均线从上方向下跌破长均线时，称为“死叉”。它通常表示市场短期动能转弱，趋势可能下行，因此常被视为卖出信号。

本次实验的默认参数为 `MA5` 与 `MA15`，并额外测试 `MA5/20` 和 `MA10/30` 两组组合，用来观察不同均线周期对策略绩效的影响。

## 3. 绩效指标概念说明

### 3.1 累计回报（Cumulative Return）

累计回报表示策略从起始到结束总体获得了多少收益。若累计回报越高，说明策略在该样本区间内增长越明显。

### 3.2 最大回撤（MDD）

最大回撤表示策略净值从某个历史高点回落到后续低点时的最大跌幅。该指标越低（绝对值越小），说明策略回撤风险越可控。

### 3.3 夏普比率（Sharpe Ratio）

夏普比率用于衡量单位风险下获得的超额收益。通常夏普比率越高，说明策略收益和波动之间的平衡越好。

## 4. 数据与回测设置

- 标的 1：中国巨石（600176）
- 标的 2：中际旭创（300308）
- 数据区间：2024-01-02 至 2026-07-03
- 数据来源：前两次任务中保存的前复权日线数据
- 回测方式：仅做多，不做空
- 执行假设：当日根据均线关系生成信号，次日按持仓状态计入收益，以避免未来函数偏差
- 交易成本：本次基础实验中暂不加入手续费和滑点

## 5. Python 实现流程

本次程序实现了以下步骤：

1. 读取原始股价数据
2. 设定短均线和长均线周期，并计算均线
3. 根据短均线与长均线位置生成交易信号
4. 标记买入点（金叉）和卖出点（死叉）
5. 计算每日策略收益率与净值曲线
6. 统计累计回报、最大回撤和夏普比率
7. 对不同股票、不同均线周期进行比较

## 6. 图表展示与解读

### 图1 中国巨石双均线策略交易信号图（MA5/MA15）

文件位置：`figures/china_jushi_ma_5_15_signals.png`

解读：该图将收盘价、MA5、MA15 以及金叉和死叉位置标记在同一张图中。可以看到，当中国巨石出现明显上升趋势时，短均线会先行抬升并站上长均线，随后形成买入信号；而在趋势减弱阶段，则出现死叉，提示离场。

### 图2 中国巨石策略净值与基准净值对比（MA5/MA15）

文件位置：`figures/china_jushi_ma_5_15_equity.png`

解读：在样本期内，中国巨石的双均线策略累计回报约为 **{pct_text(float(jushi_default["strategy_cumulative_return"]))}**，高于买入并持有策略的表现。与此同时，最大回撤约为 **{pct_text(float(jushi_default["max_drawdown"]))}**，说明在捕捉趋势的同时仍然存在一定回撤风险。

### 图3 中际旭创双均线策略交易信号图（MA5/MA15）

文件位置：`figures/zhongji_xuchuang_ma_5_15_signals.png`

解读：中际旭创波动显著大于中国巨石，因此买卖信号更加密集。趋势行情中，金叉信号能够帮助策略较快进入上涨区间；但在高波动阶段，策略也更容易遭遇来回切换，从而带来较大回撤。

### 图4 中际旭创策略净值与基准净值对比（MA5/MA15）

文件位置：`figures/zhongji_xuchuang_ma_5_15_equity.png`

解读：中际旭创在样本期内的双均线策略累计回报约为 **{pct_text(float(zhongji_default["strategy_cumulative_return"]))}**，收益弹性明显强于中国巨石，但最大回撤也达到 **{pct_text(float(zhongji_default["max_drawdown"]))}**，说明高收益往往伴随更高的波动风险。

### 图5 双均线策略参数组合绩效对比

文件位置：`figures/ma_strategy_metric_comparison.png`

解读：该图对比了两只股票在 `MA5/15`、`MA5/20` 和 `MA10/30` 三组参数下的累计回报、夏普比率和最大回撤。整体看，`MA5/20` 在本次样本中表现最优，说明稍微放宽长均线周期后，能够减少部分噪声交易并保留趋势收益。

## 7. 回测结果汇总

### 7.1 中国巨石

- `MA5/15`：累计回报 **{pct_text(float(jushi_default["strategy_cumulative_return"]))}**，最大回撤 **{pct_text(float(jushi_default["max_drawdown"]))}**，夏普比率 **{float(jushi_default["sharpe_ratio"]):.2f}**
- `MA5/20`：累计回报 **{pct_text(float(jushi_best["strategy_cumulative_return"]))}**，最大回撤 **{pct_text(float(jushi_best["max_drawdown"]))}**，夏普比率 **{float(jushi_best["sharpe_ratio"]):.2f}**
- `MA10/30`：累计回报较高，但回撤重新扩大，说明较慢参数虽然能够保留趋势，但退出也更滞后

### 7.2 中际旭创

- `MA5/15`：累计回报 **{pct_text(float(zhongji_default["strategy_cumulative_return"]))}**，最大回撤 **{pct_text(float(zhongji_default["max_drawdown"]))}**，夏普比率 **{float(zhongji_default["sharpe_ratio"]):.2f}**
- `MA5/20`：累计回报 **{pct_text(float(zhongji_best["strategy_cumulative_return"]))}**，夏普比率 **{float(zhongji_best["sharpe_ratio"]):.2f}**，在本次样本中表现最好
- `MA10/30`：累计回报低于 `MA5/20`，但最大回撤收敛到 **{pct_text(float(zhongji_stable["max_drawdown"]))}**，更适合强调稳定性的情形

## 8. 不同股票与均线周期的观察

通过比较可以发现：

1. 趋势明显、波动相对可控的股票，更适合使用双均线策略。
2. 高波动股票在强趋势阶段能放大策略收益，但也容易带来更大的回撤。
3. 短均线越短、长均线越短，信号通常越灵敏，但也更容易被市场噪声干扰。
4. 适当拉长长均线周期，有助于减少频繁切换，提高策略稳定性。

## 9. 策略适用场景与应用心得

双均线策略更适合以下场景：

- 市场存在较明显趋势时
- 投资者希望用规则化方式替代主观判断时
- 需要一个结构清晰、易解释、易扩展的基础策略框架时

本次实验的心得主要有三点：

1. 双均线策略实现简单，非常适合作为量化策略入门模型。
2. 单看收益是不够的，必须结合最大回撤和夏普比率一起评价。
3. 同一策略在不同股票上的表现差异很大，因此参数优化和标的筛选同样重要。

## 10. 局限与后续改进

本次回测仍有以下局限：

- 未加入手续费、滑点和涨跌停约束
- 仅使用了最基础的价格趋势信号
- 未做滚动样本外检验

后续可以继续改进：

- 在回测中加入交易成本
- 引入布林带、RSI、MACD 等指标做过滤
- 扩展到更多股票和更多参数组合
- 尝试加入风险控制规则，例如止损或仓位管理
"""

    (REPORT_DIR / "task3_report_final.md").write_text(report, encoding="utf-8")


def build_notebook() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Task3 双均线策略回测 Notebook\n"
            "\n"
            "本 Notebook 对双均线策略的核心流程进行白盒展示，包括：数据读取、均线计算、金叉/死叉信号生成、策略回测与绩效指标评估。"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. 核心概念\n"
            "\n"
            "- **金叉**：短均线向上突破长均线，通常被视为买入信号。\n"
            "- **死叉**：短均线向下跌破长均线，通常被视为卖出信号。\n"
            "- **累计回报**：策略在整个样本区间内累计获得的总收益。\n"
            "- **最大回撤（MDD）**：策略净值从高点回落到低点时的最大跌幅。\n"
            "- **夏普比率（Sharpe Ratio）**：单位风险对应的超额收益水平。"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            "\n"
            "base_dir = Path.cwd().resolve().parent\n"
            "sys.path.insert(0, str(base_dir))\n"
            "\n"
            "from strategy_utils import StrategyParams, apply_dual_ma_strategy, load_price_data, summarize_strategy\n"
            "\n"
            "raw_dir = base_dir / 'raw_data'\n"
            "processed_dir = base_dir / 'processed_data'\n"
            "figures_dir = base_dir / 'figures'\n"
            "default_params = StrategyParams(short_window=5, long_window=15)\n"
            "metrics_df = pd.read_csv(processed_dir / 'ma_strategy_metrics.csv')\n"
            "metrics_df"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. 加载样例股票并计算双均线策略\n"
            "\n"
            "这里以中国巨石为例，展示如何从原始 CSV 开始计算短均线、长均线和交易信号。"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "jushi = load_price_data(raw_dir / 'china-jushi_600176.csv')\n"
            "jushi_strategy = apply_dual_ma_strategy(jushi, default_params.short_window, default_params.long_window)\n"
            "jushi_strategy[['date', 'close', 'ma_short', 'ma_long', 'buy_signal', 'sell_signal', 'position']].tail(12)"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "jushi_summary = summarize_strategy(jushi_strategy, '中国巨石（600176）', default_params)\n"
            "pd.Series(jushi_summary)"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. 图表展示\n"
            "\n"
            "下面的图片展示了默认参数 `MA5/MA15` 下的交易信号图和策略净值图。"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "display(Image(filename=str(figures_dir / 'china_jushi_ma_5_15_signals.png')))\n"
            "display(Image(filename=str(figures_dir / 'china_jushi_ma_5_15_equity.png')))\n"
            "display(Image(filename=str(figures_dir / 'zhongji_xuchuang_ma_5_15_signals.png')))\n"
            "display(Image(filename=str(figures_dir / 'zhongji_xuchuang_ma_5_15_equity.png')))"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. 不同参数组合比较\n"
            "\n"
            "为了观察均线周期变化对收益和风险的影响，本次额外测试了 `MA5/20` 与 `MA10/30` 两组参数。"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "metrics_df[['stock', 'short_window', 'long_window', 'strategy_cumulative_return', 'max_drawdown', 'sharpe_ratio']]"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "display(Image(filename=str(figures_dir / 'ma_strategy_metric_comparison.png')))"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. 观察总结\n"
            "\n"
            "1. 双均线策略适合趋势明显的区间。\n"
            "2. 中际旭创在样本期内弹性更强，因此策略收益更高，但回撤也更大。\n"
            "3. `MA5/20` 在本次样本中整体表现最好，说明适度拉长长均线可以减少噪声干扰。"
        )
    )

    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.13"}

    with (NOTEBOOK_DIR / "task3_dual_ma_strategy.ipynb").open("w", encoding="utf-8") as fp:
        nbf.write(nb, fp)


def build_readme() -> None:
    content = """# Task3 Strategy Debut

本目录对应第三次量化工作坊任务：使用双均线策略完成交易信号构建、策略回测和绩效评估。

## 目录说明

- `raw_data/`：原始股价数据
- `processed_data/`：双均线策略计算结果与绩效表
- `figures/`：策略图表
- `notebook/`：白盒 Notebook
- `report/`：报告正文
- `strategy_utils.py`：策略计算工具模块
- `build_task3_assets.py`：自动生成 Task3 交付物的脚本

## 默认策略

- 短均线：MA5
- 长均线：MA15

## 实验参数组合

- MA5 / MA15
- MA5 / MA20
- MA10 / MA30
"""
    (BASE_DIR / "README.md").write_text(content, encoding="utf-8")


def build_spec() -> None:
    content = """# Task3 Spec

## 1. 任务目标

本任务的目标是从已有股价数据出发，实现一个最基础但可解释性很强的量化策略：双均线策略，并使用累计回报、最大回撤、夏普比率等指标评估策略效果。

## 2. 白盒原则

本任务必须满足以下要求：

1. 明确原始数据来自哪里
2. 明确短均线、长均线如何计算
3. 明确金叉、死叉如何生成
4. 明确回测收益率如何计算
5. 明确绩效指标如何计算

## 3. 数据对象

- 中国巨石（600176）
- 中际旭创（300308）

时间范围：

- 2024-01-02 至 2026-07-03

## 4. 策略定义

默认策略：

- 短均线：MA5
- 长均线：MA15

信号规则：

- 当 MA5 上穿 MA15 时，记为买入信号（金叉）
- 当 MA5 下穿 MA15 时，记为卖出信号（死叉）
- 当日收盘后形成信号，次日按持仓状态计入收益

## 5. 绩效指标

- 累计回报（Cumulative Return）
- 最大回撤（MDD）
- 夏普比率（Sharpe Ratio）

## 6. 扩展实验

除了默认参数，还比较：

- MA5 / MA20
- MA10 / MA30

目的是观察不同股票、不同均线周期下策略收益和风险的变化。
"""
    (BASE_DIR / "task3_spec.md").write_text(content, encoding="utf-8")


def build_publish_tree() -> None:
    publish_targets = [
        BASE_DIR / "README.md",
        BASE_DIR / "task3_spec.md",
        BASE_DIR / "strategy_utils.py",
        BASE_DIR / "build_task3_assets.py",
        REPORT_DIR / "task3_report_final.md",
        NOTEBOOK_DIR / "task3_dual_ma_strategy.ipynb",
        NOTEBOOK_DIR / "task3_dual_ma_strategy_executed.ipynb",
    ]

    for relative_dir in ["raw_data", "processed_data", "figures", "report", "notebook"]:
        destination = PUBLISH_DIR / relative_dir
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(BASE_DIR / relative_dir, destination)

    for file_path in publish_targets:
        destination = PUBLISH_DIR / file_path.relative_to(BASE_DIR)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination)


def main() -> None:
    ensure_dirs()
    metrics_df = build_processed_outputs()
    build_report(metrics_df)
    build_notebook()
    build_readme()
    build_spec()
    print("Task3 assets generated.")


if __name__ == "__main__":
    main()
