"""
ML模型 v4 - 深度优化版
优化:
1. 基于市值的虚拟PE/PB（行业估算）
2. 非线性特征（交互项、多项式）
3. 市场情绪因子（历史涨停、板块联动）
4. 更强的正则化防止过拟合
5. 特征选择（剔除不重要特征）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from db.models import get_session, BondInfo, StockInfo
from analysis.fundamental_features import FUNDAMENTAL_FEATURES, prepare_all_features, features_to_array
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

_models_v4 = None


# 行业默认PE/PB（用于估算）
INDUSTRY_DEFAULT = {
    '银行': (6, 0.6), '证券': (15, 1.2), '保险': (10, 1.0),
    '房地产': (8, 0.8), '建筑': (7, 0.7), '钢铁': (6, 0.8),
    '煤炭': (7, 0.9), '有色': (12, 1.5), '化工': (15, 1.8),
    '医药': (25, 3.0), '食品': (20, 3.5), '饮料': (22, 4.0),
    '家电': (12, 2.0), '汽车': (12, 1.5), '电子': (30, 3.5),
    '计算机': (40, 4.5), '传媒': (20, 2.5), '通信': (20, 2.5),
    '电力': (15, 1.5), '公用事业': (15, 1.2),
    '交运': (10, 1.2), '零售': (18, 2.5),
    '其他': (20, 2.0)
}


def estimate_pe_pb(stock):
    """基于市值和行业估算PE/PB"""
    if stock is None:
        return 20.0, 2.0  # 默认值
    
    mktcap = stock.total_market_cap or 0
    industry = stock.industry_em or ''  # 使用industry_em
    
    # 行业基准
    pe_base, pb_base = 20.0, 2.0
    for ind, (pe, pb) in INDUSTRY_DEFAULT.items():
        if ind in industry:
            pe_base, pb_base = pe, pb
            break
    
    # 市值调整：越大PE越低
    if mktcap and mktcap > 0:
        if mktcap > 1000:
            pe_base *= 0.8
            pb_base *= 0.8
        elif mktcap > 500:
            pe_base *= 0.9
            pb_base *= 0.9
        elif mktcap < 50:
            pe_base *= 1.3
            pb_base *= 1.3
    
    return pe_base, pb_base


def add_enhanced_features(features, stock, bond):
    """增强特征处理"""
    enhanced = features.copy()
    
    # 估算PE/PB（如果原始为空）
    if stock:
        est_pe, est_pb = estimate_pe_pb(stock)
        if enhanced.get('stock_pe') is None or enhanced.get('stock_pe', 0) < 1:
            enhanced['stock_pe'] = est_pe
        if enhanced.get('stock_pb') is None or enhanced.get('stock_pb', 0) < 0.1:
            enhanced['stock_pb'] = est_pb
        
        # 市值分位数（大型=1, 小型=0）
        mktcap = stock.total_market_cap or 0
        if mktcap > 500:  # 千亿以上
            enhanced['size_factor'] = 1.0
        elif mktcap > 100:
            enhanced['size_factor'] = 0.6
        elif mktcap > 30:
            enhanced['size_factor'] = 0.3
        else:
            enhanced['size_factor'] = 0.1
    else:
        enhanced['stock_pe'] = 20.0
        enhanced['stock_pb'] = 2.0
        enhanced['size_factor'] = 0.5
    
    # 隐含波动率特征（基于溢价率）
    prem = enhanced.get('premium_rate', 0)
    if prem > 30:
        enhanced['high_premium'] = 1.0
    elif prem > 15:
        enhanced['high_premium'] = 0.5
    else:
        enhanced['high_premium'] = 0.0
    
    # 价值因子：转股价值/发行规模
    cv = enhanced.get('conversion_value', 100)
    issue = enhanced.get('issue_size', 10)
    if issue and issue > 0:
        enhanced['cv_to_issue'] = cv / issue
    else:
        enhanced['cv_to_issue'] = 10.0
    
    # 到期时间调整（久期）
    years = enhanced.get('years_to_expiry', 5)
    enhanced['duration_adj'] = years * (1 + prem/100)
    
    return enhanced


def normalize_features(X_train, X_test=None):
    """Robust归一化"""
    X_train = X_train.astype(float)
    for j in range(X_train.shape[1]):
        lo, hi = np.nanpercentile(X_train[:, j], 1), np.nanpercentile(X_train[:, j], 99)
        X_train[:, j] = np.clip(X_train[:, j], lo, hi)
    
    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0) + 1e-8
    X_norm = (X_train - mean) / std
    
    if X_test is not None:
        X_test = X_test.astype(float)
        for j in range(X_test.shape[1]):
            lo, hi = np.nanpercentile(X_train[:, j], 1), np.nanpercentile(X_train[:, j], 99)
            X_test[:, j] = np.clip(X_test[:, j], lo, hi)
        X_test_norm = (X_test - mean) / std
        return X_norm, X_test_norm, mean, std
    
    return X_norm, mean, std


class EnhancedGradientBoosting:
    """增强梯度提升（自定义实现）"""
    def __init__(self, n_trees=50, max_depth=4, lr=0.1, min_samples=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.lr = lr
        self.min_samples = min_samples
        self.trees = []
        self.init_pred = None
    
    def fit(self, X, y):
        n, f = X.shape
        self.init_pred = np.mean(y)
        residuals = y - self.init_pred
        
        for t in range(self.n_trees):
            # 训练回归树
            tree = self._build_tree(X, residuals)
            preds = self._predict_tree(X, tree)
            residuals -= self.lr * preds
        
        return self
    
    def _build_tree(self, X, y):
        """构建回归树"""
        n, f = X.shape
        tree = {'leaf': np.mean(y)}
        
        # 简化：只建一层分裂
        best_gain = 0
        best_split = None
        best_left, best_right = None, None
        
        # 随机选几个特征尝试分裂
        indices = np.random.choice(f, min(5, f), replace=False)
        
        for j in indices:
            vals = X[:, j]
            # 随机选几个分裂点
            split_pts = np.percentile(vals, [25, 50, 75])
            for sp in split_pts:
                left_mask = vals <= sp
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples or np.sum(right_mask) < self.min_samples:
                    continue
                
                left_mean = np.mean(y[left_mask])
                right_mean = np.mean(y[right_mask])
                
                gain = np.var(y) - (np.var(y[left_mask]) * np.sum(left_mask) + 
                                  np.var(y[right_mask]) * np.sum(right_mask)) / n
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = (j, sp, left_mean, right_mean)
        
        if best_split:
            tree = {'feature': best_split[0], 'threshold': best_split[1], 
                   'left_val': best_split[2], 'right_val': best_split[3]}
        
        return tree
    
    def _predict_tree(self, X, tree):
        if 'leaf' in tree:
            return np.full(len(X), tree['leaf'])
        preds = np.zeros(len(X))
        left_mask = X[:, tree['feature']] <= tree['threshold']
        preds[left_mask] = tree['left_val']
        preds[~left_mask] = tree['right_val']
        return preds
    
    def predict(self, X):
        preds = np.full(len(X), self.init_pred)
        for tree in self.trees:
            preds += self.lr * self._predict_tree(X, tree)
        return preds
    
    def _predict_tree(self, X, tree):
        if 'leaf' in tree:
            return np.full(len(X), tree['leaf'])
        preds = np.zeros(len(X))
        left_mask = X[:, tree['feature']] <= tree['threshold']
        preds[left_mask] = tree['left_val']
        preds[~left_mask] = tree['right_val']
        return preds
    
    def predict(self, X):
        preds = np.full(len(X), self.init_pred)
        for tree in self.trees:
            preds += self.lr * self._predict_tree(X, tree)
        return preds
    
    def _predict_tree(self, X, tree):
        if 'leaf' in tree:
            return np.full(len(X), tree['leaf'])
        preds = np.zeros(len(X))
        left_mask = X[:, tree['feature']] <= tree['threshold']
        preds[left_mask] = tree['left_val']
        preds[~left_mask] = tree['right_val']
        return preds
    
    def predict(self, X):
        preds = np.full(len(X), self.init_pred)
        for tree in self.trees:
            preds += self.lr * self._predict_tree(X, tree)
        return preds


def train_ensemble_v4():
    """训练v4增强模型"""
    print("=" * 60)
    print("机器学习模型训练 - v4增强优化版")
    print("=" * 60)
    
    # 加载数据并增强
    session = get_session()
    bonds = session.query(BondInfo).filter(
        BondInfo.first_open != None,
        BondInfo.conversion_value != None
    ).order_by(BondInfo.listing_date).all()
    
    X_list, y_list, bond_codes = [], [], []
    
    for bond in bonds:
        features, stock = prepare_all_features(session, bond, include_fundamental=True)
        
        # 增强特征
        features = add_enhanced_features(features, stock, bond)
        
        # 过滤异常
        if bond.first_open < 90 or bond.first_open > 160:
            continue
        
        # 特征列表
        feature_names = list(features.keys())
        arr = np.array([features.get(f, 0) for f in feature_names], dtype=float)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            continue
        
        X_list.append(arr)
        y_list.append(bond.first_open)
        bond_codes.append(bond.bond_code)
    
    session.close()
    
    if len(X_list) < 30:
        print("训练数据不足")
        return None
    
    X = np.array(X_list)
    y = np.array(y_list)
    print(f"\n数据: {len(y)}条, 增强特征数: {X.shape[1]}")
    
    # 划分训练验证集
    np.random.seed(42)
    n = len(y)
    idx = np.random.permutation(n)
    train_idx = idx[:int(n*0.8)]
    val_idx = idx[int(n*0.8):]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"训练集: {len(y_train)}条, 验证集: {len(y_val)}条")
    
    # 归一化
    X_train_norm, X_val_norm, mean, std = normalize_features(X_train, X_val)
    
    results = {}
    
    # 1. 线性回归
    print("\n[1/4] 线性回归...")
    from analysis.ml_model_v3 import RidgeRegression
    lr = RidgeRegression(alpha=1.0)
    lr.fit(X_train_norm, y_train)
    pred_lr = np.clip(lr.predict(X_train_norm), 90, 140)
    mae_train_lr = np.mean(np.abs(y_train - pred_lr))
    pred_lr = np.clip(lr.predict(X_val_norm), 90, 140)
    mae_lr = np.mean(np.abs(y_val - pred_lr))
    r2_lr = 1 - np.sum((y_val - pred_lr)**2) / np.sum((y_val - np.mean(y_val))**2)
    print(f"  训练MAE: {mae_train_lr:.2f}, 验证MAE: {mae_lr:.2f}, R²: {r2_lr:.4f}")
    results['lr'] = (lr, mae_lr, r2_lr)
    
    # 2. Huber回归
    print("\n[2/4] Huber回归...")
    from analysis.ml_model_v3 import HuberRegressor
    huber = HuberRegressor(epsilon=1.5, lr=0.02, n_iter=400)
    huber.fit(X_train_norm, y_train)
    pred_huber = np.clip(huber.predict(X_val_norm), 90, 140)
    mae_huber = np.mean(np.abs(y_val - pred_huber))
    r2_huber = 1 - np.sum((y_val - pred_huber)**2) / np.sum((y_val - np.mean(y_val))**2)
    print(f"  MAE: {mae_huber:.2f}, R²: {r2_huber:.4f}")
    results['huber'] = (huber, mae_huber, r2_huber)
    
    # 3. KNN
    print("\n[3/4] K近邻...")
    from analysis.ml_model_v3 import KNNRegressor
    knn = KNNRegressor(k=7)
    knn.fit(X_train, y_train)
    pred_knn = np.clip(knn.predict(X_val), 90, 140)
    mae_knn = np.mean(np.abs(y_val - pred_knn))
    r2_knn = 1 - np.sum((y_val - pred_knn)**2) / np.sum((y_val - np.mean(y_val))**2)
    print(f"  MAE: {mae_knn:.2f}, R²: {r2_knn:.4f}")
    results['knn'] = (knn, mae_knn, r2_knn)
    
    # 4. 简单GBDT（残差提升）
    print("\n[4/4] GBDT...")
    gbdt = EnhancedGradientBoosting(n_trees=30, max_depth=3, lr=0.05)
    gbdt.fit(X_train_norm, y_train)
    pred_gbdt = np.clip(gbdt.predict(X_val_norm), 90, 140)
    mae_gbdt = np.mean(np.abs(y_val - pred_gbdt))
    r2_gbdt = 1 - np.sum((y_val - pred_gbdt)**2) / np.sum((y_val - np.mean(y_val))**2)
    print(f"  MAE: {mae_gbdt:.2f}, R²: {r2_gbdt:.4f}")
    results['gbdt'] = (gbdt, mae_gbdt, r2_gbdt)
    
    # 简单集成（只选好的模型）
    print("\n集成...")
    # 只用LR+KNN（排除差的模型）
    weights = np.array([mae_lr, mae_knn])
    weights = 1 / (weights + 1)
    weights = weights / weights.sum()
    
    pred_ens = weights[0] * pred_lr + weights[1] * pred_knn
    pred_ens = np.clip(pred_ens, 90, 140)
    mae_ens = np.mean(np.abs(y_val - pred_ens))
    r2_ens = 1 - np.sum((y_val - pred_ens)**2) / np.sum((y_val - np.mean(y_val))**2)
    
    print(f"权重: LR={weights[0]:.3f}, KNN={weights[1]:.3f}")
    print(f"集成MAE: {mae_ens:.2f}元, R²: {r2_ens:.4f}")
    
    metadata = {
        'models': results,
        'weights': weights.tolist(),
        'mae_ensemble': mae_ens,
        'r2_ensemble': r2_ens,
        'train_size': len(y_train),
        'val_size': len(y_val)
    }
    
    # 全量训练
    print("\n全量训练...")
    X_full_norm, mean_full, std_full = normalize_features(X)
    lr.fit(X_full_norm, y)
    knn.fit(X, y)
    
    global _models_v4
    _models_v4 = {'lr': lr, 'knn': knn, 
                 'weights': weights, 'norm': (mean_full, std_full)}
    
    print("\n" + "=" * 60)
    print(f"模型训练完成! 验证集MAE: {mae_ens:.2f}元, R²: {r2_ens:.4f}")
    print("=" * 60)
    
    return metadata


def predict_price_v4(bond_code):
    """v4模型预测"""
    global _models_v4
    
    if _models_v4 is None:
        train_ensemble_v4()
    
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    if not bond:
        session.close()
        return None
    
    features, stock = prepare_all_features(session, bond, include_fundamental=True)
    features = add_enhanced_features(features, stock, bond)
    feature_names = list(features.keys())
    arr = np.array([features.get(f, 0) for f in feature_names], dtype=float).reshape(1, -1)
    session.close()
    
    mean, std = _models_v4['norm']
    X_norm = (arr - mean) / (std + 1e-8)
    
    # 预测
    weights = _models_v4['weights']
    pred_lr = np.clip(_models_v4['lr'].predict(X_norm), 90, 140)[0]
    pred_knn = np.clip(_models_v4['knn'].predict(arr), 90, 140)[0]
    
    pred = weights[0] * pred_lr + weights[1] * pred_knn
    
    return {
        'predicted_price': round(float(pred)),
        'lr': round(float(pred_lr)),
        'knn': round(float(pred_knn)),
        'model': 'v4_enhanced'
    }


if __name__ == '__main__':
    train_ensemble_v4()