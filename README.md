# Task3 Strategy Debut

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
