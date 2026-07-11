# Task4 Turtle Trading

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
