# 可转债价格预测工具

基于机器学习的可转债首日开盘价预测系统。

## 功能特点

- 🤖 ML模型v5预测（线性回归 + K近邻 + 梯度提升集成）
- 📊 24维特征工程（正股基本面、市场情绪、溢价率等）
- 🎯 预测精度MAE=7.55元（回测样本356条）
- 💾 模型持久化（启动即用，无需重训练）

## 安装运行

### Linux桌面版

```bash
cd cb_analyzer

# 安装依赖
pip3 install -r requirements.txt

# 启动GUI
python3 desktop_app.py

# 或使用启动器
chmod +x run.sh
./run.sh
```

### 命令行预测

```bash
# 预测指定债券
python3 main.py predict --code 110074 --method ml

# 训练模型
python3 main.py train

# 回测评估
python3 main.py backtest
```

## 快速开始

1. 输入6位债券代码（如110074、113701）
2. 点击"开始预测"
3. 查看预测价格和可信区间

### 快捷代码示例

| 代码 | 名称 |
|-----|-----|
| 110074 | 精达转债 |
| 118067 | 上26转债 |
| 113701 | 祥和转债 |
| 127113 | 长高转债 |

## 项目结构

```
cb_analyzer/
├── desktop_app.py      # Linux桌面GUI应用
├── run.sh              # Linux启动器
├── main.py             # 命令行主程序
├── requirements.txt    # Python依赖
├── db/
│   └── models.py       # 数据库模型
├── analysis/
│   ├── ml_model_v5.py  # ML模型v5
│   ├── fundamental_features.py  # 特征工程
│   └── model_persistence.py     # 模型持久化
├── scripts/
│   ├── fetch_real_pe_pb.py     # 获取PE/PB数据
│   └── fetch_stock_fundamentals.py
├── models/
│   ├── ensemble_model.pkl      # 训练好的模型
│   └── model_config.json       # 模型配置
└── data/
    └── cb_analyzer.db          # SQLite数据库
```

## 模型性能

| 指标 | 数值 |
|------|------|
| 验证集MAE | 7.55元 |
| 验证集R² | 0.248 |
| 回测样本 | 356条 |
| 误差<8元占比 | 76.2% |

## 数据来源

- 可转债数据：[AKShare](https://akshare.akfamily.xyz/)
- PE/PB数据：东方财富实时行情
- 训练样本：360条历史可转债

## 技术栈

- Python 3.10+
- tkinter（GUI）
- NumPy（数值计算）
- SQLAlchemy（数据库）
- AKShare（金融数据）

## License

MIT