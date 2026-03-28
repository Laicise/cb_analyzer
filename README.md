# 可转债量化估值分析系统 (CB Analyzer)

一个基于量化评估的可转债分析工程，支持相似度匹配、价格预测、机器学习优化。

## 功能特性

- ✅ 可转债实时行情数据获取
- ✅ 可转债基本信息（含转股价值、溢价率）
- ✅ 正股行业分类（申万行业）和主营业务
- ✅ 量化评估预测算法（参考表3-7逻辑）
- ✅ 预测结果可信度评级
- ✅ 历史数据学习优化
- ✅ 预测结果存储到数据库

## 工程结构

```
cb_analyzer/
├── config.py                   # 配置文件
├── main.py                     # 主程序入口 (CLI)
├── requirements.txt            # 依赖包
├── db/
│   └── models.py              # 数据库模型
├── scripts/
│   ├── fetch_cov_data.py      # 获取可转债数据
│   ├── fetch_stock_info.py    # 获取正股信息
│   ├── calculate_yield.py     # 计算到期收益率
│   ├── fetch_history.py       # 获取单只债券历史
│   ├── batch_fetch_history.py # 批量获取历史数据
│   └── save_prediction.py    # 保存预测结果
├── analysis/
│   └── similarity.py          # 相似度匹配和预测算法
└── data/
    └── cb_data.db             # SQLite数据库
```

## 快速开始

### 1. 安装依赖

```bash
cd ~/Documents/openclaw/workspace_tony/code/cb_analyzer
pip install -r requirements.txt
```

### 2. 初始化数据库

```bash
python3 main.py init
```

### 3. 更新数据

```bash
python3 main.py update
```

### 4. 预测新债价格

```bash
python3 main.py predict --code 110074
```

### 5. 查看统计

```bash
python3 main.py stats
```

## 命令行选项

| 命令 | 说明 |
|------|------|
| `init` | 初始化数据库 |
| `update` | 更新所有数据 |
| `show` | 显示可转债列表 |
| `predict --code <代码>` | 预测新债开盘价 |
| `stats` | 显示统计信息 |

## 预测算法

### 量化评估方法（参考表3-7）

考虑以下因素并量化评分：
1. **转股价值差异** - 每差1元价格调整1元
2. **票息差异** - 6年总票息差异
3. **上市时间差异** - 每月约影响0.2元
4. **信用评级差异** - 每级差0.5元
5. **行业差异** - 行业相同不加减价

### 可信度评级

| 等级 | 相似度要求 | 说明 |
|------|------------|------|
| A | ≥70% | 高可信度，市场存在高度相似的可转债 |
| B | ≥50% | 中等可信度，存在较相似的可转债 |
| C | ≥30% | 较低可信度，相似可转债较少 |
| D | <30% | 可信度不足，建议观望 |

### 机器学习优化

- 从历史预测误差中学习
- 自动调整预测参数
- 积累足够数据后自动启用

## 数据库表说明

### bond_info - 可转债信息
| 字段 | 说明 |
|------|------|
| bond_code | 债券代码 |
| bond_name | 债券简称 |
| stock_code | 正股代码 |
| conversion_value | 转股价值 |
| premium_rate | 转股溢价率 |
| credit_rating | 信用评级 |
| predicted_price | 预测开盘价 |
| first_open | 首日开盘价 |
| error_rate | 预测误差率 |

### prediction_record - 预测记录
| 字段 | 说明 |
|------|------|
| bond_code | 债券代码 |
| predicted_price | 预测价格 |
| confidence_level | 可信度等级 |
| actual_price | 实际价格 |
| error_rate | 误差率 |
| status | 状态(pending/confirmed) |

### stock_info - 正股信息
| 字段 | 说明 |
|------|------|
| stock_code | 股票代码 |
| industry_sw_l1 | 申万一级行业 |
| business | 主营业务 |

## 后续计划

1. 完善历史数据获取（剩余700+只）
2. 引入机器学习模型
3. 添加定时任务自动更新
4. 增加Web界面

## 注意事项

- 部分历史数据获取可能失败（API限制）
- 预测结果仅供参考，不构成投资建议
- 建议定期更新数据以保持准确性