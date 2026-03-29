# 可转债量化估值分析系统 (CB Analyzer) v4.0

一个基于量化评估 + 机器学习的可转债首日开盘价预测系统。

## 核心特性

- ✅ 可转债实时行情数据获取（AKShare）
- ✅ 正股基本面数据（PE、PB、市值、ROE等）
- ✅ 申万行业分类 + 行业热度评分
- ✅ 相似度匹配预测算法
- ✅ 集成机器学习模型（线性回归 + 随机森林 + GBDT）
- ✅ 历史回测验证
- ✅ 预测结果存储到数据库

## 工程结构

```
cb_analyzer/
├── config.py                     # 配置文件
├── main.py                       # 主程序入口 (CLI)
├── db/
│   └── models.py                # 数据库模型（v2扩展）
├── scripts/
│   ├── fetch_cov_data.py        # 获取可转债数据
│   ├── fetch_stock_info.py      # 获取正股行业/主营业务
│   ├── fetch_stock_fundamentals.py  # 获取正股基本面（PE/PB/ROE等）
│   ├── calculate_yield.py       # 计算到期收益率
│   ├── batch_fetch_history.py   # 批量获取历史首日数据
│   └── continue_fetch.py        # 增量补全历史数据
├── analysis/
│   ├── similarity.py            # 相似度匹配算法
│   ├── fundamental_features.py  # 正股基本面特征工程
│   ├── ml_model.py              # 基础ML模型
│   └── ml_model_v2.py           # 集成ML模型（LR+RF+GBDT）
└── data/
    └── cb_data.db               # SQLite数据库
```

## 快速开始

```bash
cd ~/Documents/openclaw/workspace_tony/code/cb_analyzer

# 初始化数据库
python3 main.py init

# 更新所有数据（转债+正股+基本面）
python3 main.py update

# 预测新债价格
python3 main.py predict --code 110074

# 训练机器学习模型
python3 main.py train

# 历史回测
python3 main.py backtest

# 查看统计
python3 main.py stats
```

## 预测算法详解

### 1. 相似度匹配

基于5个维度计算相似度：
- 行业（20%权重）
- 主营业务（10%权重）
- 信用评级（15%权重）
- 转股价值（30%权重）
- 转股溢价率（25%权重）

### 2. 机器学习模型（集成学习）

**特征工程（19维特征）：**

| 类别 | 特征 | 说明 |
|------|------|------|
| 债券 | conversion_value | 转股价值 |
| 债券 | premium_rate | 转股溢价率 |
| 债券 | issue_size | 发行规模 |
| 债券 | coupon_rate | 票面利率 |
| 债券 | years_to_expiry | 剩余年限 |
| 债券 | rating_score | 信用评级分数 |
| 正股 | stock_pe | 市盈率 |
| 正股 | stock_pb | 市净率 |
| 正股 | stock_market_cap | 正股总市值 |
| 正股 | stock_listing_days | 上市天数 |
| 正股 | stock_roe | 净资产收益率 |
| 财务 | revenue_growth | 营收增长率 |
| 财务 | profit_growth | 净利润增长率 |
| 财务 | gross_margin | 毛利率 |
| 财务 | debt_ratio | 资产负债率 |
| 市场 | market_score | 大盘点位分数 |
| 行业 | industry_heat | 行业热度 |

**模型：**
- 线性回归（带L2正则化）
- 随机森林（10棵树，深度5）
- 梯度提升树（GBDT，20棵树）

**集成策略：** 根据各模型在验证集上的MAE自动加权平均

### 3. 综合预测

最终预测 = (相似度匹配 + ML预测) / 2

## 数据库表说明

### bond_info - 可转债信息
包含：债券代码、名称、正股、转股价、转股价值、溢价率、评级、发行规模、首日开盘价/收盘价/最高价/最低价、预测价格、误差率等。

### stock_info - 正股信息
包含：股票代码、行业（申万三级）、PE、PB、ROE、营收增长、净利润增长、毛利率、资产负债率、总市值、流通市值、主营业务等。

### prediction_record - 预测记录
包含：债券代码、预测日期、预测价格（综合/相似度/ML）、可信度等级、参考债券、实际价格、误差率等。

### model_meta - 模型元数据
包含：模型名称、版本、训练日期、训练样本数、验证MAE、验证R²、特征名称、模型权重等。

## 后续计划

1. 增加更多基本面特征（现金流、存货周转等）
2. 加入时序特征（正股近期涨跌）
3. 深度学习模型（LSTM/Transformer）
4. Web界面可视化
5. 定时任务自动更新

## 注意事项

- 部分数据源可能有限流，建议分批获取
- 预测结果仅供参考，不构成投资建议
- 建议积累至少100条有首日数据的样本后再使用ML模型