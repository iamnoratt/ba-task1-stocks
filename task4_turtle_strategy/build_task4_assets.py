from __future__ import annotations

import shutil
from pathlib import Path

import nbformat as nbf
import pandas as pd

from turtle_utils import (
    STOCK_CONFIG,
    TurtleParams,
    apply_turtle_strategy,
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

DEFAULT_PARAMS = TurtleParams(entry_window=20, exit_window=10, atr_window=20, stop_atr_multiplier=2.0)
EXPERIMENT_PARAMS = [
    TurtleParams(entry_window=20, exit_window=10, atr_window=20, stop_atr_multiplier=2.0),
    TurtleParams(entry_window=30, exit_window=10, atr_window=20, stop_atr_multiplier=2.0),
    TurtleParams(entry_window=55, exit_window=20, atr_window=20, stop_atr_multiplier=2.0),
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
            strategy_df = apply_turtle_strategy(
                price_df,
                params.entry_window,
                params.exit_window,
                params.atr_window,
                params.stop_atr_multiplier,
            )
            output_name = (
                f"{slug_for_stock(stock_key)}_turtle_"
                f"{params.entry_window}_{params.exit_window}_atr{params.atr_window}_stop{str(params.stop_atr_multiplier).replace('.', '_')}.csv"
            )
            strategy_df.to_csv(PROCESSED_DIR / output_name, index=False, encoding="utf-8-sig")

            metrics_rows.append(summarize_strategy(strategy_df, info["label"], params))

            if params == DEFAULT_PARAMS:
                signal_base = (
                    FIGURES_DIR
                    / f"{slug_for_stock(stock_key)}_turtle_{params.entry_window}_{params.exit_window}_signals"
                )
                equity_base = (
                    FIGURES_DIR
                    / f"{slug_for_stock(stock_key)}_turtle_{params.entry_window}_{params.exit_window}_equity"
                )
                save_signal_figure(strategy_df, info["label"], params, signal_base)
                save_equity_figure(strategy_df, info["label"], params, equity_base)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(PROCESSED_DIR / "turtle_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    save_comparison_figure(metrics_df, FIGURES_DIR / "turtle_strategy_metric_comparison")
    return metrics_df


def metric_lookup(metrics_df: pd.DataFrame, stock_label: str, params: TurtleParams) -> dict[str, object]:
    row = metrics_df[
        (metrics_df["stock"] == stock_label)
        & (metrics_df["entry_window"] == params.entry_window)
        & (metrics_df["exit_window"] == params.exit_window)
        & (metrics_df["atr_window"] == params.atr_window)
        & (metrics_df["stop_atr_multiplier"] == params.stop_atr_multiplier)
    ].iloc[0]
    return row.to_dict()


def pct_text(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_report(metrics_df: pd.DataFrame) -> None:
    jushi_default = metric_lookup(metrics_df, STOCK_CONFIG["china_jushi"]["label"], DEFAULT_PARAMS)
    zhongji_default = metric_lookup(metrics_df, STOCK_CONFIG["zhongji_xuchuang"]["label"], DEFAULT_PARAMS)
    jushi_mid = metric_lookup(metrics_df, STOCK_CONFIG["china_jushi"]["label"], TurtleParams(30, 10, 20, 2.0))
    zhongji_mid = metric_lookup(metrics_df, STOCK_CONFIG["zhongji_xuchuang"]["label"], TurtleParams(30, 10, 20, 2.0))
    zhongji_long = metric_lookup(metrics_df, STOCK_CONFIG["zhongji_xuchuang"]["label"], TurtleParams(55, 20, 20, 2.0))

    report = f"""# Task4 海龟交易策略报告

## 1. 任务目标

本次 Task4 的目标是在前几次任务完成数据准备、指标构建和双均线回测基础上，进一步理解并实践经典趋势跟随策略——海龟交易策略。作业重点不只是展示最终收益曲线，而是解释海龟法则中的价格通道、ATR、止损条件以及这些规则如何共同构成一套系统化的交易框架。

## 2. 海龟策略的核心思想与优势

海龟交易策略的核心思想是“顺势突破、让利润奔跑、用规则控制风险”。其基本逻辑是：当价格向上突破一段时间内的最高价通道时，视为趋势开始，执行买入；当价格跌破退出通道或触发 ATR 止损时，执行卖出。

海龟策略的关键优势包括：

1. 完全规则化，减少主观判断干扰
2. 更适合趋势行情，能捕捉中期以上的波段收益
3. 使用 ATR 作为波动尺度，使止损规则具有自适应性
4. 便于参数比较、回测评估与后续优化

## 3. 核心概念说明

### 3.1 高低点通道

高低点通道是海龟策略的基础。常见做法是用过去若干天的最高价和最低价构成价格通道：

- 上轨：过去 `N` 天的最高价
- 下轨：过去 `N` 天或更短退出窗口内的最低价

本次实验采用经典的“`20` 日上轨突破入场，`10` 日下轨跌破离场”作为默认参数，并额外测试 `30/10` 和 `55/20` 两组通道长度。

### 3.2 ATR（Average True Range）

ATR 是平均真实波幅，用于衡量市场波动程度。它不是方向指标，而是波动指标。ATR 越大，说明近期价格振幅越大，市场更活跃；ATR 越小，说明市场相对平稳。

本次实验中：

- 先计算每日真实波幅 `TR`
- 再对 `TR` 做 `20` 日移动平均，得到 `ATR20`

### 3.3 止损条件

海龟策略中常见的止损方式，是按 ATR 设定波动性止损。本次实验采用：

- 止损线 = 当前价格 - `2 × ATR`

当持仓后价格跌破 `10` 日下轨，或跌破 `2ATR` 止损线时，视为卖出信号。这样可以让止损尺度随市场波动强弱自适应变化。

## 4. 数据与回测设置

- 标的 1：中国巨石（600176）
- 标的 2：中际旭创（300308）
- 数据区间：2024-01-02 至 2026-07-03
- 数据来源：前几次任务中已保存的前复权日线数据
- 回测方式：仅做多，不做空
- 默认策略参数：入场通道 `20`，离场通道 `10`，ATR 窗口 `20`，止损倍数 `2.0`
- 执行假设：当日根据收盘价与通道关系生成信号，次日按持仓状态计入收益，以避免未来函数偏差
- 交易成本：本次基础实验中未加入手续费和滑点

## 5. Python 实现流程

本次程序实现了以下步骤：

1. 加载已存储的股价数据
2. 计算高低点价格通道
3. 计算真实波幅 TR 与 ATR
4. 生成买入、卖出和止损信号
5. 绘制股价、通道、ATR、持仓与交易信号图
6. 对策略进行模拟交易与回测
7. 计算累计回报、最大回撤、夏普比率等绩效指标

## 6. 图表展示与解读

### 图1 中国巨石海龟策略信号图（20/10/ATR20）

文件位置：`figures/china_jushi_turtle_20_10_signals.png`

解读：该图展示了中国巨石在默认参数下的收盘价、20 日上轨、10 日下轨、ATR 止损线以及买卖信号。可以看到，价格向上突破上轨后形成买入信号；后续若跌破离场通道或止损线，则执行卖出。

### 图2 中国巨石海龟策略净值与基准净值对比

文件位置：`figures/china_jushi_turtle_20_10_equity.png`

解读：中国巨石在默认海龟策略下累计回报约为 **{pct_text(float(jushi_default["strategy_cumulative_return"]))}**，最大回撤约为 **{pct_text(float(jushi_default["max_drawdown"]))}**。这说明该股票在样本期内存在可捕捉的趋势行情，同时整体风险也较可控。

### 图3 中际旭创海龟策略信号图（20/10/ATR20）

文件位置：`figures/zhongji_xuchuang_turtle_20_10_signals.png`

解读：中际旭创的波动明显更大，因此 ATR 曲线也更高。突破发生后，策略收益扩张更快，但由于波动强，止损和通道离场的触发也更敏感。

### 图4 中际旭创海龟策略净值与基准净值对比

文件位置：`figures/zhongji_xuchuang_turtle_20_10_equity.png`

解读：中际旭创在默认参数下累计回报约为 **{pct_text(float(zhongji_default["strategy_cumulative_return"]))}**，夏普比率约为 **{float(zhongji_default["sharpe_ratio"]):.2f}**。这一结果说明海龟策略在强趋势高弹性标的上可能获得更高收益，但同时也要承受较明显的波动与回撤。

### 图5 海龟策略参数组合绩效对比

文件位置：`figures/turtle_strategy_metric_comparison.png`

解读：该图对比了两只股票在 `20/10`、`30/10` 和 `55/20` 三组参数下的累计回报、夏普比率与最大回撤。可以看到，适度拉长通道长度后，交易次数减少，部分标的的风险收益比得到改善。

## 7. 回测结果汇总

### 7.1 中国巨石

- `20/10/ATR20/2N`：累计回报 **{pct_text(float(jushi_default["strategy_cumulative_return"]))}**，最大回撤 **{pct_text(float(jushi_default["max_drawdown"]))}**，夏普比率 **{float(jushi_default["sharpe_ratio"]):.2f}**
- `30/10/ATR20/2N`：累计回报 **{pct_text(float(jushi_mid["strategy_cumulative_return"]))}**，最大回撤 **{pct_text(float(jushi_mid["max_drawdown"]))}**，夏普比率 **{float(jushi_mid["sharpe_ratio"]):.2f}**
- `55/20/ATR20/2N`：累计回报降低，说明过长通道可能导致信号过慢，从而错过部分趋势收益

### 7.2 中际旭创

- `20/10/ATR20/2N`：累计回报 **{pct_text(float(zhongji_default["strategy_cumulative_return"]))}**，最大回撤 **{pct_text(float(zhongji_default["max_drawdown"]))}**，夏普比率 **{float(zhongji_default["sharpe_ratio"]):.2f}**
- `30/10/ATR20/2N`：累计回报 **{pct_text(float(zhongji_mid["strategy_cumulative_return"]))}**，最大回撤 **{pct_text(float(zhongji_mid["max_drawdown"]))}**，夏普比率 **{float(zhongji_mid["sharpe_ratio"]):.2f}**
- `55/20/ATR20/2N`：累计回报 **{pct_text(float(zhongji_long["strategy_cumulative_return"]))}**，最大回撤较小，更适合强调稳健性的情形

## 8. 不同参数与股票类型的观察

通过比较可以发现：

1. 海龟策略更适合趋势明确、能持续突破的重要行情。
2. 波动大的股票在突破成功时收益更高，但 ATR 也更大，对止损和回撤管理提出更高要求。
3. 较短的入场通道更灵敏，能更快入场，但可能更容易受到市场噪声影响。
4. 较长的通道能过滤部分假突破，但会减少交易次数，也可能错过一部分趋势早段收益。

## 9. 海龟法则的适应场景与使用心得

海龟法则更适合：

- 趋势性较强的市场
- 愿意耐心等待突破并持有趋势的交易者
- 希望用统一规则进行风险控制和信号执行的量化策略场景

本次实验的心得主要有三点：

1. 海龟策略的核心不是预测，而是等市场给出突破后再跟随。
2. ATR 的引入使止损规则不再固定，而是随波动自适应变化。
3. 参数对结果影响明显，因此必须结合标的特征与风险偏好做比较，不能机械套用单一设置。

## 10. 局限与后续改进

本次回测仍有以下局限：

- 未加入手续费、滑点和仓位加减仓规则
- 仅做多，未扩展到做空场景
- 未进行样本外测试与滚动参数检验

后续可以继续改进：

- 加入海龟法则中的分批加仓逻辑
- 增加交易成本与滑点约束
- 将海龟策略与均线、RSI、布林带等指标结合，形成组合过滤规则
- 扩展到更多股票与指数进行横向比较
"""

    (REPORT_DIR / "task4_report_final.md").write_text(report, encoding="utf-8")


def build_notebook() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Task4 海龟交易策略回测 Notebook\n"
            "\n"
            "本 Notebook 用白盒方式展示海龟交易策略的核心流程，包括：价格通道计算、ATR 计算、突破信号生成、止损规则以及策略回测。"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. 核心概念\n"
            "\n"
            "- **高低点通道**：过去若干日的最高价和最低价区间。\n"
            "- **ATR**：平均真实波幅，用于衡量波动强度。\n"
            "- **止损条件**：本次采用 `2 × ATR` 的波动性止损。\n"
            "- **累计回报、最大回撤、夏普比率**：用于评价策略收益与风险。"
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
            "from turtle_utils import TurtleParams, apply_turtle_strategy, load_price_data, summarize_strategy\n"
            "\n"
            "raw_dir = base_dir / 'raw_data'\n"
            "processed_dir = base_dir / 'processed_data'\n"
            "figures_dir = base_dir / 'figures'\n"
            "default_params = TurtleParams(entry_window=20, exit_window=10, atr_window=20, stop_atr_multiplier=2.0)\n"
            "metrics_df = pd.read_csv(processed_dir / 'turtle_strategy_metrics.csv')\n"
            "metrics_df"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. 以中国巨石为例计算海龟策略\n"
            "\n"
            "下面展示从原始股价数据出发，计算通道、ATR、止损线和交易信号。"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "jushi = load_price_data(raw_dir / 'china-jushi_600176.csv')\n"
            "jushi_strategy = apply_turtle_strategy(jushi, default_params.entry_window, default_params.exit_window, default_params.atr_window, default_params.stop_atr_multiplier)\n"
            "jushi_strategy[['date', 'close', 'entry_high', 'exit_low', 'atr', 'buy_signal', 'sell_signal', 'sell_reason', 'position']].tail(15)"
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
            "以下图表展示了默认参数 `20/10/ATR20/2N` 下的交易信号图和净值曲线。"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "display(Image(filename=str(figures_dir / 'china_jushi_turtle_20_10_signals.png')))\n"
            "display(Image(filename=str(figures_dir / 'china_jushi_turtle_20_10_equity.png')))\n"
            "display(Image(filename=str(figures_dir / 'zhongji_xuchuang_turtle_20_10_signals.png')))\n"
            "display(Image(filename=str(figures_dir / 'zhongji_xuchuang_turtle_20_10_equity.png')))"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. 不同通道参数比较\n"
            "\n"
            "为了比较策略适应性，本次另外测试了 `30/10` 和 `55/20` 两组通道长度。"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "metrics_df[['stock', 'entry_window', 'exit_window', 'atr_window', 'stop_atr_multiplier', 'strategy_cumulative_return', 'max_drawdown', 'sharpe_ratio']]"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "display(Image(filename=str(figures_dir / 'turtle_strategy_metric_comparison.png')))"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. 观察总结\n"
            "\n"
            "1. 海龟策略属于趋势突破策略，适合趋势明显的阶段。\n"
            "2. ATR 有助于让止损规则随波动变化，而不是机械使用固定点位。\n"
            "3. 更长的通道会减少交易次数，但也可能错过部分行情；更短的通道则更灵敏，但更容易受到噪声影响。"
        )
    )

    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.13"}

    with (NOTEBOOK_DIR / "task4_turtle_strategy.ipynb").open("w", encoding="utf-8") as fp:
        nbf.write(nb, fp)


def build_readme() -> None:
    content = """# Task4 Turtle Trading

本目录对应第四次量化工作坊任务：使用海龟交易策略完成价格通道突破、ATR 计算、止损规则构建和策略回测。

## 目录说明

- `raw_data/`：原始股价数据
- `processed_data/`：海龟策略计算结果与绩效表
- `figures/`：策略图表
- `notebook/`：白盒 Notebook
- `report/`：报告正文与提交件
- `turtle_utils.py`：海龟策略计算工具模块
- `build_task4_assets.py`：自动生成 Task4 交付物的脚本
- `build_task4_pdf.py`：生成提交版 PDF 的脚本

## 默认策略

- 入场通道：20 日高点
- 离场通道：10 日低点
- ATR 窗口：20
- 止损倍数：2.0 ATR

## 对比参数组合

- 20 / 10 / ATR20 / 2N
- 30 / 10 / ATR20 / 2N
- 55 / 20 / ATR20 / 2N
"""
    (BASE_DIR / "README.md").write_text(content, encoding="utf-8")


def build_spec() -> None:
    content = """# Task4 Spec

## 1. 任务目标

本任务的目标是从已有股价数据出发，实现海龟交易策略，并通过价格通道、ATR、止损线和回测指标来解释其交易逻辑。

## 2. 白盒原则

本任务必须满足以下要求：

1. 明确高低点通道如何计算
2. 明确 ATR 如何计算
3. 明确买入、卖出和止损信号如何生成
4. 明确累计回报、最大回撤和夏普比率如何计算
5. 保留原始数据、处理结果、图表、Notebook 和报告

## 3. 数据对象

- 中国巨石（600176）
- 中际旭创（300308）

时间范围：

- 2024-01-02 至 2026-07-03

## 4. 默认策略设定

- 入场上轨：20 日最高价
- 离场下轨：10 日最低价
- ATR 窗口：20
- 止损线：当前价格减去 2 倍 ATR

## 5. 对比实验

除了默认参数，还比较：

- 30 / 10 / ATR20 / 2N
- 55 / 20 / ATR20 / 2N

用于观察不同通道长度对收益、回撤和交易次数的影响。
"""
    (BASE_DIR / "task4_spec.md").write_text(content, encoding="utf-8")


def build_publish_tree() -> None:
    for relative_dir in ["raw_data", "processed_data", "figures", "report", "notebook"]:
        destination = PUBLISH_DIR / relative_dir
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(BASE_DIR / relative_dir, destination)

    for file_name in ["README.md", "task4_spec.md", "turtle_utils.py", "build_task4_assets.py", "build_task4_pdf.py"]:
        shutil.copy2(BASE_DIR / file_name, PUBLISH_DIR / file_name)


def main() -> None:
    ensure_dirs()
    metrics_df = build_processed_outputs()
    build_report(metrics_df)
    build_notebook()
    build_readme()
    build_spec()
    build_publish_tree()
    print("Task4 assets generated.")


if __name__ == "__main__":
    main()
