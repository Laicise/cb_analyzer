"""
机器学习预测模型（轻量版）
基于统计学习实现，无需额外依赖
"""
import pandas as pd
import numpy as np
from db.models import get_session, BondInfo
from config import RATING_MAP
import warnings
warnings.filterwarnings('ignore')

# 特征名称
FEATURE_NAMES = [
    'conversion_value',
    'premium_rate',
    'issue_size',
    'coupon_rate',
    'years_to_expiry',
    'rating_score',
]


def prepare_features(bond):
    """准备特征"""
    features = {}
    
    features['conversion_value'] = bond.conversion_value or 100
    features['premium_rate'] = bond.premium_rate or 20
    features['issue_size'] = bond.issue_size or 10
    features['coupon_rate'] = bond.coupon_rate or 1.5
    
    if bond.expiry_date:
        from datetime import datetime
        days = (bond.expiry_date - datetime.now()).days
        features['years_to_expiry'] = max(0, days / 365)
    else:
        features['years_to_expiry'] = 5
    
    rating = bond.credit_rating or 'AA'
    features['rating_score'] = RATING_MAP.get(rating, 3)
    
    return features


class LinearRegressionModel:
    """线性回归模型（纯Python实现）"""
    
    def __init__(self):
        self.weights = None
        self.bias = None
        self.scaler_mean = None
        self.scaler_std = None
    
    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape
        
        # 标准化
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.scaler_mean) / self.scaler_std
        
        # 简单的梯度下降
        self.weights = np.zeros(n_features)
        self.bias = 0
        learning_rate = 0.01
        n_iterations = 1000
        
        for _ in range(n_iterations):
            y_pred = np.dot(X_norm, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X_norm.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
    
    def predict(self, X):
        """预测"""
        X_norm = (X - self.scaler_mean) / self.scaler_std
        return np.dot(X_norm, self.weights) + self.bias


class LearningModel:
    """基于统计学习的模型"""
    
    def __init__(self):
        self.model = None
        self.feature_importance = {}
    
    def fit(self, X, y, feature_names):
        """训练模型"""
        n_features = X.shape[1]
        
        # 计算每个特征与目标的相关性
        correlations = []
        for i in range(n_features):
            if np.std(X[:, i]) > 0:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append(abs(corr))
            else:
                correlations.append(0)
        
        # 归一化相关性作为特征重要性
        total = sum(correlations) + 1e-8
        self.feature_importance = {
            feature_names[i]: correlations[i] / total 
            for i in range(n_features)
        }
        
        # 训练线性回归模型
        self.model = LinearRegressionModel()
        self.model.fit(X, y)
        
        # 调整权重
        self.adjusted_weights = self.model.weights * np.array(correlations)
    
    def predict(self, X):
        """预测"""
        if self.model is None:
            return None
        
        X_norm = (X - self.model.scaler_mean) / (self.model.scaler_std + 1e-8)
        return np.dot(X_norm, self.adjusted_weights) + self.model.bias


def train_model():
    """训练模型"""
    print("="*50)
    print("训练机器学习模型...")
    print("="*50)
    
    session = get_session()
    
    bonds = session.query(BondInfo).filter(
        BondInfo.first_open != None,
        BondInfo.conversion_value != None
    ).all()
    
    X = []
    y = []
    bond_codes = []
    
    for bond in bonds:
        features = prepare_features(bond)
        feature_values = [features[name] for name in FEATURE_NAMES]
        
        if all(isinstance(v, (int, float)) and not np.isnan(v) for v in feature_values):
            X.append(feature_values)
            y.append(bond.first_open)
            bond_codes.append(bond.bond_code)
    
    session.close()
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) < 20:
        print(f"数据不足，需要至少20条数据，当前只有{len(X)}条")
        return None
    
    print(f"训练数据: {len(X)} 条")
    
    # 训练模型
    model = LearningModel()
    model.fit(X, y, FEATURE_NAMES)
    
    # 评估
    y_pred = model.predict(X)
    
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    print(f"\n训练集评估:")
    print(f"  MAE: {mae:.2f}元")
    print(f"  RMSE: {rmse:.2f}元")
    print(f"  R²: {r2:.4f}")
    
    print(f"\n特征重要性:")
    for name, imp in sorted(model.feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.4f}")
    
    # 测试集评估（20%）
    test_size = int(len(X) * 0.2)
    X_test = X[:test_size]
    y_test = y[:test_size]
    
    y_test_pred = model.predict(X_test)
    test_mae = np.mean(np.abs(y_test - y_test_pred))
    test_r2 = 1 - np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    print(f"\n测试集评估:")
    print(f"  MAE: {test_mae:.2f}元")
    print(f"  R²: {test_r2:.4f}")
    
    return {
        'model': model,
        'mae': mae,
        'test_mae': test_mae,
        'r2': r2,
        'test_r2': test_r2
    }


def predict_with_ml(bond_code):
    """使用机器学习模型预测"""
    session = get_session()
    
    global _ml_model
    if '_ml_model' not in globals() or _ml_model is None:
        _ml_model = train_model()
    
    if _ml_model is None:
        session.close()
        return None
    
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    if not bond:
        session.close()
        return None
    
    features = prepare_features(bond)
    X = np.array([[features[name] for name in FEATURE_NAMES]])
    
    predicted_price = _ml_model['model'].predict(X)[0]
    
    session.close()
    
    return round(max(80, min(180, predicted_price)), 2)  # 限制范围


_ml_model = None


if __name__ == '__main__':
    result = train_model()
    if result:
        print(f"\n模型训练完成! 测试集MAE: {result['test_mae']:.2f}元")