# 可转债分析器 - 测试文档

## 1. 测试策略

### 1.1 测试金字塔

```
           ┌─────────┐
           │  E2E   │  ← 少量，端到端验证
           ├─────────┤
           │集成测试│  ← 中量，模块交互
           ├─────────┤
           │单元测试│  ← 大量，核心逻辑
           └─────────┘
```

### 1.2 测试覆盖范围

| 模块 | 测试类型 | 优先级 |
|------|----------|--------|
| 数据获取 | 集成测试 | 高 |
| 特征工程 | 单元测试 | 高 |
| ML模型 | 单元测试 + 回测 | 高 |
| 相似度算法 | 单元测试 | 高 |
| CLI命令 | 集成测试 | 中 |
| GUI交互 | E2E测试 | 中 |

## 2. 单元测试

### 2.1 特征工程测试

```python
# test_fundamental_features.py
import pytest
from analysis.fundamental_features import extract_features

def test_extract_features_complete():
    """测试完整特征提取"""
    bond = get_mock_bond()
    features = extract_features(bond)
    
    assert len(features) == 17
    assert features['conversion_value'] > 0
    assert 0 <= features['credit_rating'] <= 5

def test_extract_features_missing_data():
    """测试数据缺失时的处理"""
    bond = get_mock_bond_missing_data()
    features = extract_features(bond)
    
    # 应使用默认值填充
    assert features['pe_ttm'] == 0 or features['pe_ttm'] is not None
```

### 2.2 相似度算法测试

```python
# test_similarity.py
import pytest
from analysis.similarity import calculate_similarity

def test_similarity_identical():
    """相同债券相似度应为1"""
    bond = get_mock_bond()
    sim = calculate_similarity(bond, bond)
    
    assert sim == 1.0

def test_similarity_bounds():
    """相似度应在0-1之间"""
    bonds = get_mock_bonds(10)
    
    for i, a in enumerate(bonds):
        for b in bonds[i+1:]:
            sim = calculate_similarity(a, b)
            assert 0 <= sim <= 1
```

### 2.3 ML模型测试

```python
# test_ml_model.py
import pytest
from analysis.ml_model_v5 import EnsembleModelV5
import numpy as np

def test_model_training():
    """测试模型训练"""
    X_train, y_train = get_training_data(100)
    model = EnsembleModelV5()
    model.fit(X_train, y_train)
    
    assert model.lr is not None
    assert model.knn is not None
    assert model.gbdt is not None

def test_model_prediction():
    """测试模型预测"""
    model = get_trained_model()
    X = np.array([[100, 10, 5, ...]])  # 17维特征
    
    pred = model.predict(X)
    
    assert isinstance(pred, (float, np.floating))
    assert pred > 0
```

## 3. 集成测试

### 3.1 数据获取测试

```python
# test_data_fetch.py
import pytest
from scripts.fetch_cov_data import update_all

def test_fetch_cov_data():
    """测试可转债数据获取"""
    count_before = get_bond_count()
    update_all()
    count_after = get_bond_count()
    
    assert count_after > count_before
    assert count_after >= 500  # 至少有500只转债
```

### 3.2 CLI命令测试

```python
# test_cli.py
import pytest
from click.testing import CliRunner
from main import cli

def test_cli_init():
    """测试初始化命令"""
    runner = CliRunner()
    result = runner.invoke(cli, ['init'])
    
    assert result.exit_code == 0
    assert '初始化完成' in result.output

def test_cli_predict():
    """测试预测命令"""
    runner = CliRunner()
    result = runner.invoke(cli, ['predict', '--code', '110074'])
    
    assert result.exit_code == 0
    assert '预测' in result.output
```

## 4. 回测测试

### 4.1 回测框架

```python
# test_backtest.py
def backtest_model(model, start_date, end_date):
    """历史回测"""
    results = []
    
    for date in date_range(start_date, end_date):
        bonds = get_bonds_listed_before(date)
        
        for bond in bonds:
            X = extract_features(bond)
            predicted = model.predict(X)
            actual = bond.first_open
            
            results.append({
                'bond_code': bond.bond_code,
                'predicted': predicted,
                'actual': actual,
                'error': abs(predicted - actual),
                'error_rate': abs(predicted - actual) / actual
            })
    
    return analyze_results(results)
```

### 4.2 回测指标

| 指标 | 计算方式 | 合格标准 |
|------|----------|----------|
| MAE | mean(abs(error)) | < 10元 |
| RMSE | sqrt(mean(error²)) | < 12元 |
| R² | 1 - ss_res/ss_tot | > 0.2 |
| 误差<8元占比 | count(error<8)/total | > 70% |
| 误差<5元占比 | count(error<5)/total | > 50% |

### 4.3 回测报告

```markdown
## 回测结果 (2020-01 至 2024-12)

### v6模型滚动回测

| 指标 | 数值 | 状态 |
|------|------|------|
| 样本数 | 330 | ✅ |
| 平均误差 | 9.36% | ✅ |
| 误差中位数 | 5.70% | ✅ |
| 优秀(≤5%) | 44% | ✅ |
| 良好(5-10%) | 30% | ✅ |
| 一般(10-15%) | 12% | ✅ |
| 较差(>15%) | 14% | ⚠️ |

### 训练集验证指标

| 指标 | 数值 |
|------|------|
| Stacking MAE | 5.24元 |
| R² | 0.3145 |
| 预测区间覆盖率(P20-P80) | 47.2% |
| 平均区间宽度 | 12.9元 |

### 错误分布
- 误差 ≤ 5%: 69.1% (回测)
- 误差 5-10%: 91.0% (回测)
- 误差 > 15%: 14% (回测, 多为市场时机异常)

### 特征重要性Top 5
1. years_to_expiry (到期时间)
2. issue_size (发行规模)
3. stock_market_cap (正股市值)
4. stock_listing_days (正股上市天数)
5. batch_count (同批次数量)
```

## 5. GUI测试

### 5.1 GUI交互测试

```python
# test_gui.py
import pytest
from desktop_app import CBAnalyzerGUI

def test_gui_input():
    """测试输入框"""
    gui = CBAnalyzerGUI()
    
    gui.code_entry.insert(0, '110074')
    assert gui.code_entry.get() == '110074'

def test_gui_predict_button():
    """测试预测按钮"""
    gui = CBAnalyzerGUI()
    
    gui.code_entry.insert(0, '110074')
    gui.predict_btn.invoke()
    
    # 验证结果
    assert gui.price_label.cget('text') != ''
```

### 5.2 边界情况测试

```python
def test_gui_invalid_code():
    """测试无效代码"""
    gui = CBAnalyzerGUI()
    
    gui.code_entry.insert(0, '000000')
    gui.predict_btn.invoke()
    
    assert '未找到' in gui.status_bar.cget('text')

def test_gui_empty_input():
    """测试空输入"""
    gui = CBAnalyzerGUI()
    
    gui.predict_btn.invoke()
    
    assert '请输入' in gui.status_bar.cget('text')
```

## 6. 性能测试

### 6.1 性能基准

| 操作 | 目标时间 | 警告阈值 |
|------|----------|----------|
| 单个预测 | < 2秒 | > 5秒 |
| 模型训练(100样本) | < 10秒 | > 30秒 |
| 数据更新 | < 60秒 | > 120秒 |
| GUI启动 | < 3秒 | > 10秒 |

### 6.2 性能测试代码

```python
# test_performance.py
import time
import pytest

def test_predict_performance():
    """测试预测性能"""
    model = get_trained_model()
    
    start = time.time()
    for _ in range(10):
        predict('110074')
    elapsed = time.time() - start
    
    assert elapsed < 20  # 10次预测 < 20秒
```

## 7. 运行测试

### 7.1 运行所有测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行特定模块
pytest tests/test_ml_model.py -v

# 运行带覆盖率
pytest tests/ --cov=analysis --cov-report=html

# 运行回测
python main.py backtest
```

### 7.2 持续集成

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v
      - name: Run backtest
        run: python main.py backtest
```
