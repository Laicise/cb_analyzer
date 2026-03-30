# 可转债分析器 - 设计文档

## 1. 核心功能设计

### 1.1 价格预测

**输入**: 6位可转债代码（如 `110074`）

**处理流程**:
1. 查询债券基本信息（正股、评级、转股价值等）
2. 提取17维特征向量
3. 并行运行两种预测算法
4. 综合给出预测价格和置信区间

**输出**:
```json
{
    "bond_code": "110074",
    "predicted_price": 128.5,
    "confidence_interval": [120, 135],
    "method": "ml",
    "similar_bonds": [
        {"code": "113009", "name": "精达转债", "similarity": 0.92}
    ]
}
```

### 1.2 特征工程设计

**17维特征向量**:

| 序号 | 特征名 | 来源 | 说明 |
|------|--------|------|------|
| 1 | conversion_value | 债券 | 转股价值 |
| 2 | premium_rate | 债券 | 转股溢价率 |
| 3 | issue_size | 债券 | 发行规模 |
| 4 | credit_rating | 债券 | 信用评级(数值化) |
| 5 | pe_ttm | 正股 | 市盈率TTM |
| 6 | pb | 正股 | 市净率 |
| 7 | roe | 正股 | 净资产收益率 |
| 8 | market_cap | 正股 | 总市值 |
| 9 | revenue_growth | 正股 | 营收增长率 |
| 10 | profit_growth | 正股 | 净利润增长率 |
| 11 | industry_code | 正股 | 行业编码 |
| 12 | same_day_cb_return | 市场 | 同日转债涨幅 |
| 13 | cb_market_sentiment | 市场 | 转债市场情绪 |
| 14 | stock_volatility | 正股 | 正股波动率 |
| 15 | bond_volatility | 债券 | 债券波动率 |
| 16 | conversion_ratio | 债券 | 转股比例 |
| 17 | days_to_listed | 债券 | 距上市天数 |

### 1.3 算法设计

#### 1.3.1 相似度匹配算法

```python
def calculate_similarity(bond_a, bond_b):
    """基于加权特征的余弦相似度"""
    weights = {
        'conversion_value': 0.25,
        'premium_rate': 0.20,
        'industry': 0.15,
        'pe_ttm': 0.15,
        'pb': 0.10,
        'issue_size': 0.10,
        'credit_rating': 0.05
    }
    # 标准化 + 加权余弦相似度
    return weighted_cosine_similarity(features_a, features_b, weights)
```

#### 1.3.2 ML集成模型

```python
class EnsembleModelV5:
    """三模型集成"""
    def __init__(self):
        self.lr = LinearRegression()
        self.knn = KNeighborsRegressor(n_neighbors=5)
        self.gbdt = GradientBoostingRegressor(n_estimators=100)
    
    def predict(self, X):
        # 训练时用加权平均，预测时用三个模型投票
        pred_lr = self.lr.predict(X)
        pred_knn = self.knn.predict(X)
        pred_gbdt = self.gbdt.predict(X)
        
        # 加权集成
        return 0.3 * pred_lr + 0.3 * pred_knn + 0.4 * pred_gbdt
```

### 1.4 数据库设计

```sql
-- 可转债基本信息表
CREATE TABLE bond_info (
    id INTEGER PRIMARY KEY,
    bond_code VARCHAR(10) UNIQUE,
    bond_name VARCHAR(50),
    stock_code VARCHAR(10),
    stock_name VARCHAR(50),
    conversion_value FLOAT,
    premium_rate FLOAT,
    first_open FLOAT,  -- 目标变量
    credit_rating VARCHAR(10),
    -- ... 其他字段
);

-- 正股信息表（含基本面）
CREATE TABLE stock_info (
    id INTEGER PRIMARY KEY,
    stock_code VARCHAR(10) UNIQUE,
    pe_ttm FLOAT,
    pb FLOAT,
    roe FLOAT,
    market_cap FLOAT,
    industry_sw_l1 VARCHAR(50),
    -- ... 其他基本面字段
);

-- 每日行情表
CREATE TABLE bond_daily (
    id INTEGER PRIMARY KEY,
    bond_code VARCHAR(10),
    trade_date DATETIME,
    close_price FLOAT,
    volume INTEGER,
    -- ... 
);

-- 预测记录表
CREATE TABLE prediction_record (
    id INTEGER PRIMARY KEY,
    bond_code VARCHAR(10),
    predicted_price FLOAT,
    actual_price FLOAT,
    method VARCHAR(20),
    created_at DATETIME
);
```

### 1.5 GUI设计

```python
class CBAnalyzerGUI:
    """主窗口布局"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("可转债价格预测器")
        self.root.geometry("600x500")
        
        # 输入区
        self.input_frame = Frame()
        self.code_entry = Entry()  # 债券代码输入
        self.predict_btn = Button(text="开始预测")
        
        # 结果区
        self.result_frame = Frame()
        self.price_label = Label()  # 预测价格
        self.confidence_label = Label()  # 置信区间
        self.similar_list = Listbox()  # 相似债券列表
        
        # 状态栏
        self.status_bar = Label()
```

## 2. 配置管理

```python
# config.py
DB_PATH = "data/cb_analyzer.db"
DB_URL = f"sqlite:///{DB_PATH}"

RATING_MAP = {
    'AAA': 5, 'AA+': 4, 'AA': 3, 'AA-': 2, 'A+': 1
}

# 模型配置
MODEL_CONFIG = {
    'n_neighbors': 5,
    'gbdt_n_estimators': 100,
    'ensemble_weights': [0.3, 0.3, 0.4]
}
```

## 3. 错误处理

| 错误类型 | 处理方式 |
|----------|----------|
| 网络请求失败 | 重试3次，间隔2秒，超时则跳过 |
| 数据缺失 | 使用默认值或同类均值填充 |
| 预测异常 | 返回"数据不足，无法预测" |
| 数据库错误 | 记录日志，提示用户检查 |

## 4. 扩展性设计

### 4.1 算法扩展

```python
# 预留扩展接口
class Predictor(ABC):
    @abstractmethod
    def predict(self, bond_code) -> PredictionResult:
        pass

class SimilarityPredictor(Predictor):
    pass

class MLPredictor(Predictor):
    pass

# 轻松添加新算法
class XGBoostPredictor(Predictor):
    pass
```

### 4.2 数据源扩展

```python
# 预留数据源接口
class DataSource(ABC):
    @abstractmethod
    def fetch_bonds(self) -> List[BondInfo]:
        pass

class AkshareSource(DataSource):
    pass

# 轻松添加新数据源
class EastMoneySource(DataSource):
    pass
```
