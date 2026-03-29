"""
ML模型 v5 - 最终优化版
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from db.models import get_session, BondInfo, StockInfo
from analysis.fundamental_features import prepare_all_features
from analysis.model_persistence import save_v5_model as save_model, load_v5_model as load_model, model_exists, get_model_age_days
import warnings, json
warnings.filterwarnings('ignore')

_models_v5 = None

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
    if stock is None:
        return 20.0, 2.0, False
    if stock.pe and stock.pe > 1:
        return stock.pe, stock.pb or 2.0, True
    if stock.pb and stock.pb > 0.1:
        return stock.pe or 20.0, stock.pb, True
    industry = stock.industry_em or ''
    mktcap = stock.total_market_cap or 0
    pe_base, pb_base = 20.0, 2.0
    for ind, (pe, pb) in INDUSTRY_DEFAULT.items():
        if ind in industry:
            pe_base, pb_base = pe, pb
            break
    if mktcap > 1000:
        pe_base *= 0.8; pb_base *= 0.8
    elif mktcap > 500:
        pe_base *= 0.9; pb_base *= 0.9
    elif mktcap < 50:
        pe_base *= 1.3; pb_base *= 1.3
    return pe_base, pb_base, False


def add_enhanced_features(features, stock, bond):
    enhanced = features.copy()
    pe, pb, is_real = estimate_pe_pb(stock)
    enhanced['stock_pe'] = pe
    enhanced['stock_pb'] = pb
    enhanced['pe_real'] = 1.0 if is_real else 0.0
    mktcap = stock.total_market_cap if stock else 0
    if mktcap and mktcap > 500:
        enhanced['size_factor'] = 1.0
    elif mktcap and mktcap > 100:
        enhanced['size_factor'] = 0.7
    elif mktcap and mktcap > 30:
        enhanced['size_factor'] = 0.4
    else:
        enhanced['size_factor'] = 0.1
    prem = enhanced.get('premium_rate', 0)
    enhanced['high_premium'] = 1.0 if prem > 30 else (0.5 if prem > 15 else 0.0)
    cv = enhanced.get('conversion_value', 100)
    issue = enhanced.get('issue_size', 10)
    enhanced['cv_to_issue'] = cv / issue if issue > 0 else 10.0
    years = enhanced.get('years_to_expiry', 5)
    enhanced['duration_adj'] = years * (1 + prem/100)
    return enhanced


def normalize(X_train, X_test=None):
    X_train = X_train.astype(float)
    for j in range(X_train.shape[1]):
        lo, hi = np.nanpercentile(X_train[:, j], 1), np.nanpercentile(X_train[:, j], 99)
        X_train[:, j] = np.clip(X_train[:, j], lo, hi)
    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0) + 1e-8
    Xn = (X_train - mean) / std
    if X_test is not None:
        X_test = X_test.astype(float)
        for j in range(X_test.shape[1]):
            lo, hi = np.nanpercentile(X_train[:, j], 1), np.nanpercentile(X_train[:, j], 99)
            X_test[:, j] = np.clip(X_test[:, j], lo, hi)
        Xn_test = (X_test - mean) / std
        return Xn, Xn_test, mean, std
    return Xn, mean, std


class LinearRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        n, f = X.shape
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0) + 1e-8
        Xn = (X - self.mean) / self.std
        XtX = Xn.T @ Xn + self.alpha * np.eye(f)
        Xty = Xn.T @ y
        try:
            w = np.linalg.solve(XtX, Xty)
        except:
            w = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        self.weights = w
        self.bias = np.mean(y)
        return self

    def predict(self, X):
        Xn = (X - self.mean) / self.std
        return Xn @ self.weights + self.bias


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X.astype(float)
        self.y = y.astype(float)
        return self

    def predict(self, X):
        X = X.astype(float)
        preds = []
        for x in X:
            d = np.sum((self.X - x) ** 2, axis=1)
            idx = np.argsort(d)[:self.k]
            preds.append(np.mean(self.y[idx]))
        return np.array(preds)


class GradientBoosting:
    def __init__(self, n_trees=30, max_depth=3, lr=0.08):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.lr = lr
        self.trees = []
        self.init_pred = None

    def fit(self, X, y):
        n, f = X.shape
        self.init_pred = np.mean(y)
        residuals = y - self.init_pred
        for t in range(self.n_trees):
            tree = self._build_tree(X, residuals)
            self.trees.append(tree)
            preds = self._predict_tree(X, tree)
            residuals -= self.lr * preds
        return self

    def _build_tree(self, X, y):
        n, f = X.shape
        best_gain = 0
        best = None
        indices = np.random.choice(f, min(6, f), replace=False)
        for j in indices:
            vals = X[:, j]
            for thresh in np.percentile(vals, [30, 50, 70]):
                left = vals <= thresh
                right = ~left
                if np.sum(left) < 3 or np.sum(right) < 3:
                    continue
                gain = np.var(y) - (np.var(y[left]) * np.sum(left) + np.var(y[right]) * np.sum(right)) / n
                if gain > best_gain:
                    best_gain = gain
                    best = (j, thresh, np.mean(y[left]), np.mean(y[right]))
        if best:
            return {'feature': best[0], 'thresh': best[1], 'left_val': best[2], 'right_val': best[3]}
        return {'leaf': np.mean(y)}

    def _predict_tree(self, X, tree):
        if 'leaf' in tree:
            return np.full(len(X), tree['leaf'])
        preds = np.zeros(len(X))
        left = X[:, tree['feature']] <= tree['thresh']
        preds[left] = tree['left_val']
        preds[~left] = tree['right_val']
        return preds

    def predict(self, X):
        preds = np.full(len(X), self.init_pred)
        for tree in self.trees:
            preds += self.lr * self._predict_tree(X, tree)
        return preds


def load_training_data():
    session = get_session()
    bonds = session.query(BondInfo).filter(
        BondInfo.first_open != None,
        BondInfo.conversion_value != None
    ).order_by(BondInfo.listing_date).all()
    X_list, y_list = [], []
    for bond in bonds:
        if bond.first_open < 90 or bond.first_open > 160:
            continue
        features, stock = prepare_all_features(session, bond, include_fundamental=True)
        features = add_enhanced_features(features, stock, bond)
        feature_names = sorted(features.keys())
        arr = np.array([features.get(f, 0) for f in feature_names], dtype=float)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            continue
        X_list.append(arr)
        y_list.append(bond.first_open)
    session.close()
    return np.array(X_list), np.array(y_list), feature_names


def train_ensemble_v5(force_retrain=False):
    global _models_v5

    print("=" * 60)
    print("机器学习模型训练 - v5最终优化版")
    print("=" * 60)

    # 尝试加载已有模型
    if not force_retrain and model_exists():
        age = get_model_age_days()
        if age is not None and age < 1:
            models, metadata = load_model()
            if models is not None:
                print(f"加载已有模型（{age:.1f}小时前训练）")
                _models_v5 = models
                return metadata

    X, y, feature_names = load_training_data()
    if len(X) < 30:
        print("训练数据不足")
        return None

    print(f"\n数据: {len(y)}条, 特征数: {X.shape[1]}")

    np.random.seed(42)
    n = len(y)
    idx = np.random.permutation(n)
    train_idx = idx[:int(n*0.8)]
    val_idx = idx[int(n*0.8):]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"训练集: {len(y_train)}条, 验证集: {len(y_val)}条")
    X_train_n, X_val_n, mean, std = normalize(X_train, X_val)

    # 训练
    print("\n[1/3] 线性回归...")
    lr = LinearRegression(alpha=1.0)
    lr.fit(X_train_n, y_train)
    pred_lr = np.clip(lr.predict(X_val_n), 90, 140)
    mae_lr = np.mean(np.abs(y_val - pred_lr))
    r2_lr = 1 - np.sum((y_val - pred_lr)**2) / np.sum((y_val - np.mean(y_val))**2)
    print(f"  MAE: {mae_lr:.2f}元, R2: {r2_lr:.4f}")

    print("\n[2/3] K近邻...")
    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    pred_knn = np.clip(knn.predict(X_val), 90, 140)
    mae_knn = np.mean(np.abs(y_val - pred_knn))
    r2_knn = 1 - np.sum((y_val - pred_knn)**2) / np.sum((y_val - np.mean(y_val))**2)
    print(f"  MAE: {mae_knn:.2f}元, R2: {r2_knn:.4f}")

    print("\n[3/3] 梯度提升...")
    gb = GradientBoosting(n_trees=50, max_depth=3, lr=0.08)
    gb.fit(X_train_n, y_train)
    pred_gb = np.clip(gb.predict(X_val_n), 90, 140)
    mae_gb = np.mean(np.abs(y_val - pred_gb))
    r2_gb = 1 - np.sum((y_val - pred_gb)**2) / np.sum((y_val - np.mean(y_val))**2)
    print(f"  MAE: {mae_gb:.2f}元, R2: {r2_gb:.4f}")

    # 集成
    errors = np.array([mae_lr, mae_knn, mae_gb])
    min_err = np.min(errors)
    good_mask = errors < min_err * 1.5
    good_errors = errors[good_mask]
    weights_all = 1 / (good_errors + 1)
    weights_all = weights_all / weights_all.sum()
    weights = np.zeros(3)
    j = 0
    for i in range(3):
        if good_mask[i]:
            weights[i] = weights_all[j]
            j += 1

    pred_ens = weights[0] * pred_lr + weights[1] * pred_knn + weights[2] * pred_gb
    pred_ens = np.clip(pred_ens, 90, 140)
    mae_ens = np.mean(np.abs(y_val - pred_ens))
    r2_ens = 1 - np.sum((y_val - pred_ens)**2) / np.sum((y_val - np.mean(y_val))**2)

    print(f"\n权重: LR={weights[0]:.3f}, KNN={weights[1]:.3f}, GB={weights[2]:.3f}")
    print(f"集成MAE: {mae_ens:.2f}元, R2: {r2_ens:.4f}")

    metadata = {
        'mae_ensemble': mae_ens, 'r2_ensemble': r2_ens,
        'ensemble_weights': weights.tolist(),
        'train_size': len(y_train), 'val_size': len(y_val),
        'n_features': X.shape[1],
        'feature_names': feature_names
    }

    # 全量训练并保存
    print("\n全量训练...")
    X_n, mean_full, std_full = normalize(X)
    lr.fit(X_n, y)
    knn.fit(X, y)
    gb.fit(X_n, y)

    models_to_save = {
        'lr': lr, 'knn': knn, 'gb': gb,
        'weights': weights, 'norm': (mean_full, std_full)
    }

    save_model(models_to_save, metadata)

    _models_v5 = models_to_save

    print("\n" + "=" * 60)
    print(f"模型训练完成! 验证集MAE: {mae_ens:.2f}元, R2: {r2_ens:.4f}")
    print("=" * 60)

    return metadata


def predict_price_v5(bond_code):
    global _models_v5
    _cached_meta = None

    # 尝试加载已保存的模型
    if _models_v5 is None:
        models_cache, meta = load_model()
        if models_cache is not None:
            _models_v5 = models_cache
            _cached_meta = meta
        else:
            train_ensemble_v5()
            models_cache, _cached_meta = load_model()

    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    if not bond:
        session.close()
        return None

    features, stock = prepare_all_features(session, bond, include_fundamental=True)
    features = add_enhanced_features(features, stock, bond)

    # 从配置获取字段顺序
    _, meta = load_model()
    if _cached_meta is None:
        _cached_meta = meta
    feature_names = _cached_meta.get('feature_names', sorted(features.keys()))

    arr = np.array([features.get(f, 0) for f in feature_names], dtype=float).reshape(1, -1)
    session.close()

    mean, std = _models_v5['norm']
    Xn = (arr - mean) / (std + 1e-8)

    pred_lr = np.clip(_models_v5['lr'].predict(Xn), 90, 140)[0]
    pred_knn = np.clip(_models_v5['knn'].predict(arr), 90, 140)[0]
    pred_gb = np.clip(_models_v5['gb'].predict(Xn), 90, 140)[0]

    weights = _models_v5['weights']
    pred = weights[0] * pred_lr + weights[1] * pred_knn + weights[2] * pred_gb

    mae = _cached_meta.get('mae_ensemble', 8.0)
    pe_real = stock and stock.pe and stock.pe > 1

    return {
        'predicted_price': round(float(pred), 1),
        'lr': round(float(pred_lr), 1),
        'knn': round(float(pred_knn), 1),
        'gb': round(float(pred_gb), 1),
        'model': 'v5_ensemble',
        'mae': round(float(mae), 2),
        'pe_source': 'real' if pe_real else 'estimated'
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'force':
        train_ensemble_v5(force_retrain=True)
    else:
        train_ensemble_v5()