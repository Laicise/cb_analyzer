"""
ML模型 v3 - 增强优化版
优化: 
1. 更好的特征归一化
2. 时序/动量特征
3. 分层预测 + 分位数回归
4. Stacking集成
5. 异常值检测
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from db.models import get_session, BondInfo, StockInfo
from analysis.fundamental_features import FUNDAMENTAL_FEATURES, prepare_all_features, features_to_array
from config import RATING_MAP
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

_models_v3 = None


def normalize_features(X_train, X_test=None):
    """Robust归一化：去除异常值后标准化"""
    X_train = X_train.astype(float)
    
    # 分位数裁剪（去除1%和99%分位数以外的异常值）
    for j in range(X_train.shape[1]):
        lo = np.nanpercentile(X_train[:, j], 1)
        hi = np.nanpercentile(X_train[:, j], 99)
        X_train[:, j] = np.clip(X_train[:, j], lo, hi)
    
    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0) + 1e-8
    X_norm = (X_train - mean) / std
    
    if X_test is not None:
        X_test = X_test.astype(float)
        for j in range(X_test.shape[1]):
            X_test[:, j] = np.clip(X_test[:, j], 
                np.nanpercentile(X_train[:, j], 1), 
                np.nanpercentile(X_train[:, j], 99))
        X_test_norm = (X_test - mean) / std
        return X_norm, X_test_norm, mean, std
    
    return X_norm, mean, std


class RidgeRegression:
    """岭回归（带早停）"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
    
    def fit(self, X, y):
        n, f = X.shape
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0) + 1e-8
        X_norm = (X - self.mean) / self.std
        
        # 闭式解：w = (X'X + αI)^-1 X'y
        XtX = X_norm.T @ X_norm
        XtX += self.alpha * np.eye(f)
        Xty = X_norm.T @ y
        try:
            weights = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        
        self.weights = weights
        self.bias = np.mean(y)
        return self
    
    def predict(self, X_raw=None, X_norm=None):
        if self.weights is None:
            return None
        if X_norm is not None:
            return X_norm @ self.weights + self.bias
        X_norm = (X_raw - self.mean) / self.std
        return X_norm @ self.weights + self.bias


class KNNRegressor:
    """K近邻回归"""
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X.astype(float)
        self.y_train = y.astype(float)
        return self
    
    def predict(self, X):
        X = X.astype(float)
        preds = []
        for x in X:
            dists = np.sum((self.X_train - x) ** 2, axis=1)
            idx = np.argsort(dists)[:self.k]
            preds.append(np.mean(self.y_train[idx]))
        return np.array(preds)


class HuberRegressor:
    """Huber回归（抗异常值）"""
    def __init__(self, epsilon=1.35, lr=0.01, n_iter=500):
        self.epsilon = epsilon
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n, f = X.shape
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0) + 1e-8
        X_norm = (X - mean) / std
        
        self.weights = np.zeros(f)
        self.bias = np.median(y)
        
        for i in range(self.n_iter):
            pred = X_norm @ self.weights + self.bias
            error = y - pred
            abs_e = np.abs(error)
            
            # Huber权重
            mask = abs_e <= self.epsilon
            grad = np.where(mask, error, self.epsilon * np.sign(error))
            
            dw = X_norm.T @ grad / n
            db = np.mean(grad)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db * 0.1
        
        self.mean = mean
        self.std = std
        return self
    
    def predict(self, X):
        X_norm = (X - self.mean) / self.std
        return X_norm @ self.weights + self.bias


class QuantileRegressor:
    """分位数回归（预测区间）"""
    def __init__(self, quantile=0.5, lr=0.05, n_iter=300):
        self.quantile = quantile
        self.lr = lr
        self.n_iter = n_iter
    
    def fit(self, X, y):
        n, f = X.shape
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0) + 1e-8
        X_norm = (X - mean) / std
        
        self.weights = np.zeros(f)
        self.bias = np.quantile(y, self.quantile)
        self.mean = mean
        self.std = std
        
        for _ in range(self.n_iter):
            pred = X_norm @ self.weights + self.bias
            error = y - pred
            
            # 分位数梯度
            grad = np.where(error > 0, self.quantile, self.quantile - 1)
            self.weights += self.lr * (X_norm.T @ grad / n)
        
        return self
    
    def predict(self, X):
        X_norm = (X - self.mean) / self.std
        return X_norm @ self.weights + self.bias


class StackingEnsemble:
    """Stacking集成模型"""
    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.feature_names = None
        self.metadata = {}
    
    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        n = len(y)
        
        # 数据分割：80%训练，20%验证
        np.random.seed(42)
        idx = np.random.permutation(n)
        train_idx = idx[:int(n * 0.8)]
        val_idx = idx[int(n * 0.8):]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"\n训练集: {len(y_train)}条, 验证集: {len(y_val)}条")
        
        # 归一化
        X_train_norm, X_val_norm, mean, std = normalize_features(X_train, X_val)
        
        # 基础模型1: 岭回归
        print("\n[1/5] 岭回归...")
        lr = RidgeRegression(alpha=0.5)
        lr.fit(X_train_norm, y_train)
        pred_lr = np.clip(lr.predict(X_val_norm), 90, 140)
        mae_lr = np.mean(np.abs(y_val - pred_lr))
        print(f"  MAE: {mae_lr:.2f}元, R²: {self._r2(y_val, pred_lr):.4f}")
        self.base_models['lr'] = lr
        
        # 基础模型2: Huber回归
        print("\n[2/5] Huber回归...")
        huber = HuberRegressor(epsilon=1.5, lr=0.02, n_iter=400)
        huber.fit(X_train_norm, y_train)
        pred_huber = np.clip(huber.predict(X_val_norm), 90, 140)
        mae_huber = np.mean(np.abs(y_val - pred_huber))
        print(f"  MAE: {mae_huber:.2f}元, R²: {self._r2(y_val, pred_huber):.4f}")
        self.base_models['huber'] = huber
        
        # 基础模型3: KNN
        print("\n[3/5] K近邻回归...")
        knn = KNNRegressor(k=7)
        knn.fit(X_train, y_train)
        pred_knn = np.clip(knn.predict(X_val), 90, 140)
        mae_knn = np.mean(np.abs(y_val - pred_knn))
        print(f"  MAE: {mae_knn:.2f}元, R²: {self._r2(y_val, pred_knn):.4f}")
        self.base_models['knn'] = knn
        
        # 基础模型4: 分位数回归(下界)
        print("\n[4/5] 分位数回归(下界/上界)...")
        q_low = QuantileRegressor(quantile=0.25, lr=0.03, n_iter=300)
        q_low.fit(X_train_norm, y_train)
        pred_q_low = np.clip(q_low.predict(X_val_norm), 90, 140)
        mae_q_low = np.mean(np.abs(y_val - pred_q_low))
        print(f"  下界MAE: {mae_q_low:.2f}元")
        self.base_models['q_low'] = q_low
        
        q_high = QuantileRegressor(quantile=0.75, lr=0.03, n_iter=300)
        q_high.fit(X_train_norm, y_train)
        pred_q_high = np.clip(q_high.predict(X_val_norm), 90, 140)
        mae_q_high = np.mean(np.abs(y_val - pred_q_high))
        print(f"  上界MAE: {mae_q_high:.2f}元")
        self.base_models['q_high'] = q_high
        
        # Stacking: 用基础模型预测作为元特征训练元模型
        print("\n[5/5] Stacking元模型...")
        meta_features = np.column_stack([pred_lr, pred_huber, pred_knn, pred_q_low, pred_q_high])
        
        # 简单加权平均作为元模型
        errors = np.array([mae_lr, mae_huber, mae_knn, mae_q_low, mae_q_high])
        weights = 1 / (errors + 1)
        weights = weights / weights.sum()
        
        # 评估集成
        pred_ensemble = np.zeros_like(y_val)
        for i, name in enumerate(['lr', 'huber', 'knn', 'q_low', 'q_high']):
            model = self.base_models[name]
            if name == 'knn':
                pred = np.clip(model.predict(X_val), 90, 140)
            else:
                if hasattr(model, 'mean'):
                    pred = np.clip(model.predict(X_val), 90, 140)
            if name == 'lr':
                pred_ensemble += weights[0] * pred_lr
            elif name == 'huber':
                pred_ensemble += weights[1] * pred_huber
            elif name == 'knn':
                pred_ensemble += weights[2] * pred_knn
            elif name == 'q_low':
                pred_ensemble += weights[3] * pred_q_low
            elif name == 'q_high':
                pred_ensemble += weights[4] * pred_q_high
        
        mae_ensemble = np.mean(np.abs(y_val - pred_ensemble))
        r2_ensemble = self._r2(y_val, pred_ensemble)
        
        print(f"\n模型权重: LR={weights[0]:.3f}, Huber={weights[1]:.3f}, KNN={weights[2]:.3f}, Q25={weights[3]:.3f}, Q75={weights[4]:.3f}")
        print(f"集成模型: MAE={mae_ensemble:.2f}元, R²={r2_ensemble:.4f}")
        
        self.metadata = {
            'mae_lr': mae_lr, 'mae_huber': mae_huber, 'mae_knn': mae_knn,
            'mae_q_low': mae_q_low, 'mae_q_high': mae_q_high,
            'mae_ensemble': mae_ensemble, 'r2_ensemble': r2_ensemble,
            'ensemble_weights': weights.tolist(),
            'train_size': len(y_train), 'val_size': len(y_val)
        }
        
        # 全量数据重训练
        print("\n使用全量数据重新训练...")
        self._X_norm_full, self._mean_full, self._std_full = normalize_features(X)
        
        for name, model in self.base_models.items():
            if name == 'knn':
                model.fit(X, y)
            else:
                model.fit(self._X_norm_full, y)
        
        return self.metadata
    
    def predict(self, X):
        preds = []
        for name, model in self.base_models.items():
            if name == 'knn':
                p = np.clip(model.predict(X.astype(float)), 90, 140)
            else:
                X_norm = (X - model.mean) / model.std
                p = np.clip(model.predict(X_norm), 90, 140)
            preds.append(p)
        
        weights = np.array(self.metadata.get('ensemble_weights', [0.2]*5))
        ensemble_pred = sum(w * p for w, p in zip(weights, preds))
        
        # 预测区间
        lower = preds[3]  # q_low
        upper = preds[4]    # q_high
        
        return ensemble_pred, lower, upper
    
    def _r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0
        return 1 - ss_res / ss_tot
    
    def get_metadata(self):
        return self.metadata


def load_training_data():
    session = get_session()
    bonds = session.query(BondInfo).filter(
        BondInfo.first_open != None,
        BondInfo.conversion_value != None
    ).order_by(BondInfo.listing_date).all()
    
    X_list, y_list, bond_codes = [], [], []
    
    for bond in bonds:
        features, stock = prepare_all_features(session, bond, include_fundamental=True)
        
        # 跳过有异常值的数据
        if bond.first_open < 90 or bond.first_open > 160:
            continue
        
        arr = features_to_array(features, FUNDAMENTAL_FEATURES)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            continue
        
        X_list.append(arr)
        y_list.append(bond.first_open)
        bond_codes.append(bond.bond_code)
    
    session.close()
    
    if len(X_list) < 30:
        return None, None, None
    
    return np.array(X_list), np.array(y_list), bond_codes


def train_v3():
    """训练v3模型"""
    print("=" * 60)
    print("机器学习模型训练 - v3 Stacking集成版")
    print("=" * 60)
    
    X, y, bond_codes = load_training_data()
    if X is None:
        print("训练数据不足")
        return None
    
    print(f"\n加载数据: {len(y)}条, 特征数: {X.shape[1]}")
    
    # 特征重要性分析（用相关性）
    print("\n特征与目标变量的相关性:")
    for i, name in enumerate(FUNDAMENTAL_FEATURES):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        if not np.isnan(corr):
            print(f"  {name}: {corr:.3f}")
    
    model = StackingEnsemble()
    metadata = model.fit(X, y, FUNDAMENTAL_FEATURES)
    
    global _models_v3
    _models_v3 = model
    
    print("\n" + "=" * 60)
    print("模型训练完成!")
    print(f"验证集MAE: {metadata['mae_ensemble']:.2f}元")
    print(f"验证集R²: {metadata['r2_ensemble']:.4f}")
    print("=" * 60)
    
    return metadata


def predict_price_ml_v3(bond_code):
    """使用v3模型预测"""
    global _models_v3
    
    if _models_v3 is None:
        result = train_v3()
        if result is None:
            return None
    
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    if not bond:
        session.close()
        return None
    
    features, stock = prepare_all_features(session, bond)
    arr = features_to_array(features, FUNDAMENTAL_FEATURES).reshape(1, -1)
    session.close()
    
    pred, lower, upper = _models_v3.predict(arr)
    mae = _models_v3.metadata.get('mae_ensemble', 5)
    
    return {
        'predicted_price': round(float(pred[0]), 2),
        'price_range': (round(float(lower[0]), 2), round(float(upper[0]), 2)),
        'model': 'stacking_v3',
        'mae': round(mae, 2)
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        from analysis.ml_model_v2 import evaluate_on_history
        evaluate_on_history()
    else:
        train_v3()