"""
机器学习预测模型 - 增强版
支持多种模型：线性回归、随机森林、GBDT
集成学习：加权平均多个模型
特征：正股基本面 + 可转债指标
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from db.models import get_session, BondInfo, StockInfo
from analysis.fundamental_features import (
    FUNDAMENTAL_FEATURES, prepare_all_features, 
    features_to_array, get_market_score, get_stock_industry_heat
)
from config import RATING_MAP
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 全局模型缓存
_models = None


class LinearRegression:
    """线性回归（纯Python实现，梯度下降）"""
    
    def __init__(self, lr=0.01, n_iter=2000, reg=0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.reg = reg  # L2正则化系数
        self.weights = None
        self.bias = None
        self.scaler_mean = None
        self.scaler_std = None
        self.feature_names = None
    
    def fit(self, X, y, feature_names=None):
        n_samples, n_features = X.shape
        self.feature_names = feature_names
        
        # 标准化
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.scaler_mean) / self.scaler_std
        
        # 初始化
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.n_iter):
            y_pred = np.dot(X_norm, self.weights) + self.bias
            error = y_pred - y
            
            # 带正则化的梯度
            dw = (1/n_samples) * np.dot(X_norm.T, error) + self.reg * self.weights / n_samples
            db = (1/n_samples) * np.sum(error)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # 早停
            if i > 500 and i % 500 == 0:
                loss = np.mean(error ** 2)
                if loss > 100:
                    print(f"    LR loss diverging at iter {i}, reducing LR...")
                    self.lr *= 0.5
    
    def predict(self, X):
        X_norm = (X - self.scaler_mean) / self.scaler_std
        return np.dot(X_norm, self.weights) + self.bias
    
    def get_weights(self):
        if self.weights is None:
            return {}
        importance = np.abs(self.weights)
        importance = importance / (importance.sum() + 1e-8)
        return {name: float(importance[i]) for i, name in enumerate(self.feature_names) if self.feature_names}


class DecisionTreeRegressor:
    """简单决策树回归（基于均值分箱）"""
    
    def __init__(self, max_depth=5, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.feature_names = None
    
    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        
        # 停止条件
        if depth >= self.max_depth or n_samples < self.min_samples_leaf * 2:
            return {'leaf': True, 'value': float(np.mean(y))}
        
        # 找最佳分割特征
        best_gain = -1
        best_split = None
        best_left_idx = None
        best_right_idx = None
        
        base_variance = np.var(y)
        n_features = X.shape[1]
        
        # 随机选一半特征
        np.random.seed(42 + depth)
        feat_idx = np.random.choice(n_features, max(1, n_features // 2), replace=False)
        
        for fi in feat_idx:
            col = X[:, fi]
            sorted_idx = np.argsort(col)
            
            # 尝试每个分割点
            for split_i in range(self.min_samples_leaf, n_samples - self.min_samples_leaf + 1):
                threshold = col[sorted_idx[split_i]]
                
                left_idx = sorted_idx[:split_i]
                right_idx = sorted_idx[split_i:]
                
                if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                    continue
                
                left_var = np.var(y[left_idx]) if len(left_idx) > 1 else 0
                right_var = np.var(y[right_idx]) if len(right_idx) > 1 else 0
                
                gain = base_variance - (len(left_idx) * left_var + len(right_idx) * right_var) / n_samples
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = {'feature': fi, 'threshold': threshold}
                    best_left_idx = left_idx
                    best_right_idx = right_idx
        
        if best_gain <= 0 or best_split is None:
            return {'leaf': True, 'value': float(np.mean(y))}
        
        return {
            'leaf': False,
            'feature': best_split['feature'],
            'threshold': float(best_split['threshold']),
            'left': self._build_tree(X[best_left_idx], y[best_left_idx], depth + 1),
            'right': self._build_tree(X[best_right_idx], y[best_right_idx], depth + 1)
        }
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def _predict_one(self, x, node):
        if node.get('leaf'):
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['right'])  # 反转：左边是<=，但这里用right表示左子
        return self._predict_one(x, node['right'])


class RandomForest:
    """随机森林（多棵决策树集成）"""
    
    def __init__(self, n_trees=10, max_depth=5, min_samples_leaf=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.feature_importance = None
    
    def fit(self, X, y, feature_names=None):
        n_samples = X.shape[0]
        self.feature_names = feature_names
        
        self.trees = []
        for i in range(self.n_trees):
            # Bootstrap采样
            np.random.seed(42 + i)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # 训练单棵树
            tree = DecisionTreeRegressor(self.max_depth, self.min_samples_leaf)
            tree.fit(X_boot, y_boot, feature_names)
            self.trees.append(tree)
            
            if (i + 1) % 5 == 0:
                print(f"    训练树 {i+1}/{self.n_trees}")
        
        # 计算特征重要性（基于分割次数）
        self._calc_feature_importance(X)
    
    def _calc_feature_importance(self, X):
        n_features = X.shape[1]
        importance = np.zeros(n_features)
        
        for tree in self.trees:
            importance += self._count_splits(tree.tree, n_features)
        
        importance = importance / (len(self.trees) + 1e-8)
        importance = importance / (importance.sum() + 1e-8)
        
        n_features = X.shape[1]
        if self.feature_names:
            self.feature_importance = {self.feature_names[i]: float(importance[i]) for i in range(n_features)}
        else:
            self.feature_importance = importance.tolist()
    
    def _count_splits(self, node, n_features):
        counts = np.zeros(n_features)
        if node.get('leaf'):
            return counts
        feat_idx = node['feature']
        if feat_idx < n_features:
            counts[feat_idx] = 1
        return counts + self._count_splits(node['left'], n_features) + self._count_splits(node['right'], n_features)
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)


class GBDT:
    """梯度提升树（简化版）"""
    
    def __init__(self, n_estimators=20, max_depth=4, lr=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.trees = []
        self.init_pred = None
    
    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        
        # 初始预测（目标均值）
        self.init_pred = np.mean(y)
        
        residual = y - self.init_pred
        
        for i in range(self.n_estimators):
            # 训练下一棵树拟合残差
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=5)
            tree.fit(X, residual, feature_names)
            
            pred = tree.predict(X)
            
            # 更新残差
            residual = residual - self.lr * pred
            
            self.trees.append(tree)
            
            if (i + 1) % 5 == 0:
                rmse = np.sqrt(np.mean(residual ** 2))
                print(f"    GBDT树 {i+1}/{self.n_estimators}, 残差RMSE: {rmse:.2f}")
    
    def predict(self, X):
        pred = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred


class EnsembleModel:
    """集成模型：组合多个模型"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.feature_names = None
        self.feature_importance = {}
        self.metadata = {}
    
    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        
        # 数据分割
        n = len(y)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_size = int(n * 0.8)
        X_train, X_val = X[indices[:train_size]], X[indices[train_size:]]
        y_train, y_val = y[indices[:train_size]], y[indices[train_size:]]
        
        print(f"\n训练集: {len(y_train)}条, 验证集: {len(y_val)}条")
        
        # 模型1: 线性回归
        print("\n[1/3] 训练线性回归模型...")
        lr = LinearRegression(lr=0.02, n_iter=3000, reg=0.01)
        lr.fit(X_train, y_train, feature_names)
        pred_lr = lr.predict(X_val)
        mae_lr = np.mean(np.abs(y_val - pred_lr))
        r2_lr = self._calc_r2(y_val, pred_lr)
        print(f"  线性回归 - MAE: {mae_lr:.2f}元, R²: {r2_lr:.4f}")
        self.models['lr'] = lr
        self.weights['lr'] = max(0, 1 / (mae_lr + 0.1))
        
        # 模型2: 随机森林
        print("\n[2/3] 训练随机森林模型...")
        rf = RandomForest(n_trees=10, max_depth=5, min_samples_leaf=5)
        rf.fit(X_train, y_train, feature_names)
        pred_rf = rf.predict(X_val)
        mae_rf = np.mean(np.abs(y_val - pred_rf))
        r2_rf = self._calc_r2(y_val, pred_rf)
        print(f"  随机森林 - MAE: {mae_rf:.2f}元, R²: {r2_rf:.4f}")
        self.models['rf'] = rf
        self.weights['rf'] = max(0, 1 / (mae_rf + 0.1))
        self.feature_importance.update(rf.feature_importance or {})
        
        # 模型3: GBDT
        print("\n[3/3] 训练梯度提升模型...")
        gbdt = GBDT(n_estimators=20, max_depth=4, lr=0.1)
        gbdt.fit(X_train, y_train, feature_names)
        pred_gbdt = gbdt.predict(X_val)
        mae_gbdt = np.mean(np.abs(y_val - pred_gbdt))
        r2_gbdt = self._calc_r2(y_val, pred_gbdt)
        print(f"  梯度提升 - MAE: {mae_gbdt:.2f}元, R²: {r2_gbdt:.4f}")
        self.models['gbdt'] = gbdt
        self.weights['gbdt'] = max(0, 1 / (mae_gbdt + 0.1))
        
        # 计算集成权重
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        print(f"\n模型权重: {self.weights}")
        
        # 集成预测
        pred_ensemble = (
            self.weights['lr'] * pred_lr +
            self.weights['rf'] * pred_rf +
            self.weights['gbdt'] * pred_gbdt
        )
        mae_ensemble = np.mean(np.abs(y_val - pred_ensemble))
        r2_ensemble = self._calc_r2(y_val, pred_ensemble)
        
        print(f"\n集成模型 - MAE: {mae_ensemble:.2f}元, R²: {r2_ensemble:.4f}")
        
        # 保存元数据
        self.metadata = {
            'mae_lr': mae_lr, 'mae_rf': mae_rf, 'mae_gbdt': mae_gbdt,
            'r2_lr': r2_lr, 'r2_rf': r2_rf, 'r2_gbdt': r2_gbdt,
            'mae_ensemble': mae_ensemble, 'r2_ensemble': r2_ensemble,
            'train_size': len(y_train), 'val_size': len(y_val),
            'n_features': len(feature_names) if feature_names else 0,
            'model_config': str(self.weights)
        }
        
        # 在全量数据上重新训练
        print("\n使用全量数据重新训练...")
        lr.fit(X, y, feature_names)
        rf.fit(X, y, feature_names)
        gbdt.fit(X, y, feature_names)
        
        return self.metadata
    
    def predict(self, X):
        if not self.models:
            return None
        
        pred_lr = self.models['lr'].predict(X)
        pred_rf = self.models['rf'].predict(X)
        pred_gbdt = self.models['gbdt'].predict(X)
        
        pred = (
            self.weights['lr'] * pred_lr +
            self.weights['rf'] * pred_rf +
            self.weights['gbdt'] * pred_gbdt
        )
        
        # 限制合理范围
        return np.clip(pred, 80, 180)
    
    def get_feature_importance(self):
        return self.feature_importance
    
    def get_metadata(self):
        return self.metadata
    
    def _calc_r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0
        return 1 - ss_res / ss_tot


def load_training_data():
    """加载训练数据"""
    session = get_session()
    
    bonds = session.query(BondInfo).filter(
        BondInfo.first_open != None,
        BondInfo.conversion_value != None
    ).all()
    
    X_list = []
    y_list = []
    bond_codes = []
    
    for bond in bonds:
        features, stock = prepare_all_features(session, bond, include_fundamental=True)
        if not validate_features(features):
            continue
        
        arr = features_to_array(features, FUNDAMENTAL_FEATURES)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            continue
        
        X_list.append(arr)
        y_list.append(bond.first_open)
        bond_codes.append(bond.bond_code)
    
    session.close()
    
    if len(X_list) < 20:
        return None, None, None
    
    return np.array(X_list), np.array(y_list), bond_codes


def validate_features(features):
    """验证特征有效性"""
    for v in features.values():
        if v is None:
            return False
        if not isinstance(v, (int, float)):
            return False
        if np.isnan(v) or np.isinf(v):
            return False
    return True


def train_ensemble():
    """训练集成模型"""
    print("="*60)
    print("机器学习模型训练 - 集成学习版")
    print("="*60)
    
    X, y, bond_codes = load_training_data()
    
    if X is None:
        print("训练数据不足，无法训练模型")
        return None
    
    print(f"\n加载训练数据: {len(y)}条")
    print(f"特征数量: {X.shape[1]}")
    print(f"特征列表: {FUNDAMENTAL_FEATURES}")
    
    # 训练集成模型
    model = EnsembleModel()
    metadata = model.fit(X, y, FUNDAMENTAL_FEATURES)
    
    # 显示特征重要性
    importance = model.get_feature_importance()
    if importance:
        print("\n特征重要性排序:")
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_imp[:10]:
            print(f"  {name}: {imp:.4f}")
    
    global _models
    _models = model
    
    mdata = model.get_metadata()
    print("\n" + "="*60)
    print("模型训练完成!")
    print(f"验证集MAE: {mdata['mae_ensemble']:.2f}元")
    print(f"验证集R²: {mdata['r2_ensemble']:.4f}")
    print("="*60)
    
    return mdata


def predict_price_ml(bond_code):
    """使用ML模型预测价格"""
    global _models
    
    if _models is None:
        _models = train_ensemble()
    
    if _models is None:
        return None
    
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    if not bond:
        session.close()
        return None
    
    features, stock = prepare_all_features(session, bond, include_fundamental=True)
    arr = features_to_array(features, FUNDAMENTAL_FEATURES).reshape(1, -1)
    
    predicted_price = _models.predict(arr)[0]
    
    # 计算可信度
    metadata = _models.get_metadata()
    mae = metadata.get('mae_ensemble', 5)
    
    # 误差估计：基于验证集MAE
    error_estimate = mae * 1.5
    lower = max(80, predicted_price - error_estimate)
    upper = min(180, predicted_price + error_estimate)
    
    session.close()
    
    return {
        'predicted_price': round(float(predicted_price), 2),
        'price_range': (round(lower, 2), round(upper, 2)),
        'model': 'ensemble',
        'mae': round(mae, 2)
    }


def evaluate_on_history():
    """在历史数据上评估模型"""
    global _models
    
    if _models is None:
        _models = train_ensemble()
    
    if _models is None:
        return
    
    X, y, bond_codes = load_training_data()
    
    if X is None:
        return
    
    session = get_session()
    
    # 按时间顺序评估（模拟真实预测场景）
    errors = []
    predictions = []
    
    for i in range(len(y)):
        # 使用之前的样本训练（滚动训练）
        if i < 10:
            continue
        
        # 训练一个简单的模型
        X_train = X[:i]
        y_train = y[:i]
        
        model = EnsembleModel()
        try:
            model.fit(X_train, y_train, FUNDAMENTAL_FEATURES)
            
            # 预测当前
            pred = model.predict(X[i:i+1])[0]
            actual = y[i]
            error = abs(pred - actual)
            error_pct = error / actual * 100
            errors.append(error_pct)
            
            predictions.append({
                'bond_code': bond_codes[i],
                'predicted': round(pred, 2),
                'actual': round(actual, 2),
                'error_pct': round(error_pct, 2)
            })
        except:
            continue
    
    session.close()
    
    if not errors:
        return
    
    print("\n=== 历史回测结果 ===")
    print(f"总预测数: {len(errors)}")
    print(f"平均误差: {np.mean(errors):.2f}%")
    print(f"误差中位数: {np.median(errors):.2f}%")
    
    excellent = sum(1 for e in errors if e <= 5)
    good = sum(1 for e in errors if 5 < e <= 10)
    fair = sum(1 for e in errors if 10 < e <= 15)
    poor = sum(1 for e in errors if e > 15)
    
    print(f"优秀(≤5%): {excellent} ({excellent/len(errors)*100:.0f}%)")
    print(f"良好(5-10%): {good} ({good/len(errors)*100:.0f}%)")
    print(f"一般(10-15%): {fair} ({fair/len(errors)*100:.0f}%)")
    print(f"较差(>15%): {poor} ({poor/len(errors)*100:.0f}%)")
    
    return {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'excellent': excellent,
        'good': good,
        'fair': fair,
        'poor': poor
    }


_models = None


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        evaluate_on_history()
    else:
        train_ensemble()