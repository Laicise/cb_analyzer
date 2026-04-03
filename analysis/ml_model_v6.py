"""
ML模型 v6 - 优化版
==================
改进：
1. LightGBM原生分位数回归替代自定义GBDT
2. Ridge元学习器替代简单加权平均
3. 特征重要性分析
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from db.models import get_session, BondInfo, StockInfo, BondDaily
from datetime import datetime, timedelta
from config import RATING_MAP, ML_CONFIG
import warnings
import json
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# ==================== 市场情绪因子 ====================

def get_market_sentiment(date):
    """获取市场情绪因子"""
    if date is None:
        return {'market_score': 0.5, 'sentiment': 'neutral'}

    # 直接返回默认值，避免网络请求卡住
    return {'market_score': 0.5, 'sentiment': 'neutral', 'change_pct': 0}


def get_batch_info(session, listing_date):
    """获取同批次新债信息"""
    if listing_date is None:
        return {'batch_count': 1, 'batch_avg_premium': 20, 'batch_heat': 0.5}

    # 查询前后7天内的其他新债
    start = listing_date - timedelta(days=7)
    end = listing_date + timedelta(days=7)

    batch_bonds = session.query(BondInfo).filter(
        BondInfo.listing_date >= start,
        BondInfo.listing_date <= end,
        BondInfo.bond_code != None
    ).all()

    count = len(batch_bonds)
    if count > 1:
        premiums = [b.premium_rate for b in batch_bonds if b.premium_rate]
        avg_premium = sum(premiums) / len(premiums) if premiums else 20
        heat = min(1.0, count / 5)  # 最多5只，heat=1
    else:
        avg_premium = 20
        heat = 0.3

    return {'batch_count': count, 'batch_avg_premium': avg_premium, 'batch_heat': heat}


def get_subscription_heat(session, bond_code):
    """获取申购热度

    注意：真实申购数据需要从东方财富API获取，
    当前返回默认值0.5以保证模型稳定性
    """
    return 0.5


# ==================== 特征工程 ====================

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
    """估算PE/PB"""
    if stock is None:
        return 20.0, 2.0, False
    if stock.pe and stock.pe > 1:
        return stock.pe, stock.pb or 2.0, True
    if stock.pb and stock.pb > 0.1:
        return stock.pe or 20.0, stock.pb, True
    industry = stock.industry_sw_l1 or stock.industry_em or ''
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
    elif mktcap and mktcap < 50:
        pe_base *= 1.3; pb_base *= 1.3
    return pe_base, pb_base, False


def prepare_v6_features(session, bond, include_market=True):
    """准备v6版本的所有特征"""
    features = {}

    # === 债券基础特征 ===
    features['conversion_value'] = bond.conversion_value or 100
    features['premium_rate'] = bond.premium_rate or 20
    features['issue_size'] = bond.issue_size or 10
    features['coupon_rate'] = bond.coupon_rate or 1.5

    # 剩余年限
    if bond.expiry_date:
        days = (bond.expiry_date - datetime.now()).days
        features['years_to_expiry'] = max(0, days / 365)
    else:
        features['years_to_expiry'] = 5

    # 评级分数
    rating = bond.credit_rating or 'AA'
    features['rating_score'] = RATING_MAP.get(rating, 3)

    # 规模分层
    issue_size = features['issue_size']
    if issue_size < 5:
        features['size_tier'] = 0  # 小盘
    elif issue_size < 10:
        features['size_tier'] = 1  # 中小盘
    elif issue_size < 20:
        features['size_tier'] = 2  # 中大盘
    else:
        features['size_tier'] = 3  # 大盘

    # === 正股基本面特征 ===
    stock = None
    if bond.stock_code:
        stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()

    pe, pb, is_real = estimate_pe_pb(stock)
    features['stock_pe'] = pe
    features['stock_pb'] = pb
    features['pe_real'] = 1.0 if is_real else 0.0

    mktcap = stock.total_market_cap if stock and stock.total_market_cap else 100
    features['stock_market_cap'] = mktcap

    # 市值分层
    if mktcap > 1000:
        features['mcap_tier'] = 3
    elif mktcap > 300:
        features['mcap_tier'] = 2
    elif mktcap > 100:
        features['mcap_tier'] = 1
    else:
        features['mcap_tier'] = 0

    features['stock_roe'] = stock.roe if stock and stock.roe else 8
    features['stock_listing_days'] = stock.listing_days if stock and stock.listing_days else 1000

    # 行业热度
    industry = stock.industry_sw_l1 if stock and stock.industry_sw_l1 else '其他'
    INDUSTRY_HEAT = {
        '医药生物': 1.05, '电子': 1.08, '计算机': 1.06,
        '电力设备': 1.04, '汽车': 1.02, '机械设备': 1.00,
        '基础化工': 0.98, '有色金属': 1.00, '食品饮料': 0.95,
        '银行': 0.88, '非银金融': 0.90, '房地产': 0.82,
    }
    features['industry_heat'] = INDUSTRY_HEAT.get(industry, 1.0)

    # === 市场情绪特征 ===
    if include_market:
        listing_date = bond.listing_date or datetime.now()

        # 大盘情绪
        market_info = get_market_sentiment(listing_date)
        features['market_score'] = market_info.get('market_score', 0.5)
        features['market_change'] = market_info.get('change_pct', 0)

        # 同批次信息
        batch_info = get_batch_info(session, listing_date)
        features['batch_count'] = batch_info['batch_count']
        features['batch_avg_premium'] = batch_info['batch_avg_premium']
        features['batch_heat'] = batch_info['batch_heat']

        # 申购热度
        features['subscription_heat'] = get_subscription_heat(session, bond.bond_code)
    else:
        features['market_score'] = 0.5
        features['market_change'] = 0
        features['batch_count'] = 1
        features['batch_avg_premium'] = 20
        features['batch_heat'] = 0.5
        features['subscription_heat'] = 0.5

    # === 衍生特征 ===
    cv = features['conversion_value']
    prem = features['premium_rate']
    features['cv_to_prem'] = cv / (prem + 1) if prem > 0 else cv
    features['size_adj_premium'] = prem * (1 + features['size_tier'] * 0.1)
    features['mcap_adj_pe'] = features['stock_pe'] * (1 + features['mcap_tier'] * 0.05)

    return features, stock


# ==================== 基础模型 ====================

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
    """自定义GBDT用于点预测"""
    def __init__(self, n_trees=50, max_depth=3, lr=0.08):
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


# ==================== LightGBM分位数回归 ====================

def train_lgb_quantile(X_train, y_train, quantile, params=None):
    """训练LightGBM分位数模型"""
    if params is None:
        params = {
            'objective': 'quantile',
            'alpha': quantile,
            'metric': 'quantile',
            'n_estimators': 100,
            'max_depth': 5,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True
        }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


def train_lgb_default(X_train, y_train, params=None):
    """训练LightGBM标准回归模型"""
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'n_estimators': 100,
            'max_depth': 5,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'force_col_wise': True
        }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


# ==================== 工具函数 ====================

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


def load_training_data_v6():
    """加载v6训练数据"""
    session = get_session()
    bonds = session.query(BondInfo).filter(
        BondInfo.first_open != None,
        BondInfo.conversion_value != None
    ).order_by(BondInfo.listing_date).all()

    X_list, y_list = [], []
    for bond in bonds:
        # 过滤异常值
        if bond.first_open < ML_CONFIG['price_min'] or bond.first_open > ML_CONFIG['price_max']:
            continue
        features, stock = prepare_v6_features(session, bond, include_market=True)
        feature_names = sorted(features.keys())
        arr = np.array([features.get(f, 0) for f in feature_names], dtype=float)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            continue
        X_list.append(arr)
        y_list.append(bond.first_open)

    session.close()
    return np.array(X_list), np.array(y_list), feature_names


def save_v6_model(models, metadata, path='models/ensemble_model_v6.pkl'):
    """保存v6模型"""
    Path('models').mkdir(exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'models': models, 'metadata': metadata}, f)
    print(f"模型已保存到 {path}")


def load_v6_model(path='models/ensemble_model_v6.pkl'):
    """加载v6模型"""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            return data['models'], data['metadata']
    except:
        return None, None


def model_exists(path='models/ensemble_model_v6.pkl'):
    return Path(path).exists()


def get_model_age_days(path='models/ensemble_model_v6.pkl'):
    try:
        mtime = Path(path).stat().st_mtime
        import time
        age_hours = (time.time() - mtime) / 3600
        return age_hours
    except:
        return None


# ==================== 特征重要性 ====================

def compute_feature_importance(X, y, feature_names, model=None):
    """计算特征重要性（使用LightGBM）"""
    if not HAS_LIGHTGBM:
        return None

    # 训练一个简单的LightGBM模型来获取特征重要性
    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        verbose=-1,
        random_state=42
    )
    model.fit(X, y)

    importance = model.feature_importances_
    importance_dict = dict(zip(feature_names, importance))
    return importance_dict


def print_feature_importance(importance_dict, top_n=10):
    """打印特征重要性"""
    if importance_dict is None:
        print("特征重要性分析不可用（需要LightGBM）")
        return

    sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    print("\n特征重要性 (Top {}):".format(top_n))
    print("-" * 40)
    for i, (name, imp) in enumerate(sorted_imp[:top_n], 1):
        bar = "█" * int(imp * 100)
        print(f"  {i:2d}. {name:<25} {imp:.4f} {bar}")


# ==================== 训练入口 ====================

def train_ensemble_v6(force_retrain=False):
    """训练v6集成模型 - 优化版

    改进：
    1. LightGBM原生分位数回归替代自定义GBDT
    2. Ridge元学习器替代简单加权平均
    3. 特征重要性分析
    """

    print("=" * 60)
    print("机器学习模型训练 - v6 (优化版)")
    print("=" * 60)
    print(f"LightGBM可用: {HAS_LIGHTGBM}")
    print("改进: LightGBM分位数回归 + Ridge元学习器 + 特征重要性")

    # 尝试加载已有模型
    if not force_retrain and model_exists():
        age = get_model_age_days()
        if age is not None and age < 1:
            models, metadata = load_v6_model()
            if models is not None:
                print(f"加载已有模型（{age:.1f}小时前训练）")
                return metadata

    X, y, feature_names = load_training_data_v6()
    if len(X) < 30:
        print("训练数据不足")
        return None

    print(f"\n数据: {len(y)}条, 特征数: {X.shape[1]}")
    print(f"特征: {feature_names}")

    np.random.seed(42)
    n = len(y)
    idx = np.random.permutation(n)
    train_idx = idx[:int(n*0.8)]
    val_idx = idx[int(n*0.8):]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"\n训练集: {len(y_train)}条, 验证集: {len(y_val)}条")
    X_train_n, X_val_n, mean, std = normalize(X_train, X_val)

    results = {}

    # 1. 线性回归
    print("\n[1/5] 线性回归...")
    lr = LinearRegression(alpha=1.0)
    lr.fit(X_train_n, y_train)
    pred_lr = np.clip(lr.predict(X_val_n), ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])
    mae_lr = np.mean(np.abs(y_val - pred_lr))
    r2_lr = 1 - np.sum((y_val - pred_lr)**2) / np.sum((y_val - np.mean(y_val))**2)
    print(f"  MAE: {mae_lr:.2f}元, R2: {r2_lr:.4f}")
    results['lr'] = {'mae': mae_lr, 'r2': r2_lr, 'pred': pred_lr}

    # 2. K近邻
    print("\n[2/5] K近邻...")
    knn = KNN(k=5)
    knn.fit(X_train_n, y_train)
    pred_knn = np.clip(knn.predict(X_val_n), ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])
    mae_knn = np.mean(np.abs(y_val - pred_knn))
    r2_knn = 1 - np.sum((y_val - pred_knn)**2) / np.sum((y_val - np.mean(y_val))**2)
    print(f"  MAE: {mae_knn:.2f}元, R2: {r2_knn:.4f}")
    results['knn'] = {'mae': mae_knn, 'r2': r2_knn, 'pred': pred_knn}

    # 3. 自定义GBDT
    print("\n[3/5] 梯度提升...")
    gb = GradientBoosting(n_trees=ML_CONFIG['gb_n_trees'], max_depth=ML_CONFIG['gb_max_depth'], lr=ML_CONFIG['gb_lr'])
    gb.fit(X_train_n, y_train)
    pred_gb = np.clip(gb.predict(X_val_n), ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])
    mae_gb = np.mean(np.abs(y_val - pred_gb))
    r2_gb = 1 - np.sum((y_val - pred_gb)**2) / np.sum((y_val - np.mean(y_val))**2)
    print(f"  MAE: {mae_gb:.2f}元, R2: {r2_gb:.4f}")
    results['gb'] = {'mae': mae_gb, 'r2': r2_knn, 'pred': pred_gb}

    # 4. LightGBM分位数回归 (P20/P50/P80)
    print("\n[4/5] LightGBM分位数回归...")

    if HAS_LIGHTGBM:
        # 使用LightGBM进行分位数回归
        lgb_q25 = train_lgb_quantile(X_train_n, y_train, quantile=0.20)
        pred_q25 = np.clip(lgb_q25.predict(X_val_n), ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])

        lgb_q50 = train_lgb_quantile(X_train_n, y_train, quantile=0.50)
        pred_q50 = np.clip(lgb_q50.predict(X_val_n), ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])
        mae_q50 = np.mean(np.abs(y_val - pred_q50))
        r2_q50 = 1 - np.sum((y_val - pred_q50)**2) / np.sum((y_val - np.mean(y_val))**2)
        print(f"  Q50 MAE: {mae_q50:.2f}元, R2: {r2_q50:.4f}")
        results['q50'] = {'mae': mae_q50, 'r2': r2_q50, 'pred': pred_q50}

        lgb_q75 = train_lgb_quantile(X_train_n, y_train, quantile=0.80)
        pred_q75 = np.clip(lgb_q75.predict(X_val_n), ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])
    else:
        # 降级：使用自定义QuantileGBDT
        from QuantileGBDT_backup import QuantileGBDT
        q25 = QuantileGBDT(n_trees=50, max_depth=3, lr=0.08, quantile=0.25)
        q25.fit(X_train_n, y_train)
        pred_q25 = np.clip(q25.predict(X_val_n), ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])

        q50 = QuantileGBDT(n_trees=50, max_depth=3, lr=0.08, quantile=0.50)
        q50.fit(X_train_n, y_train)
        pred_q50 = np.clip(q50.predict(X_val_n), ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])
        mae_q50 = np.mean(np.abs(y_val - pred_q50))
        r2_q50 = 1 - np.sum((y_val - pred_q50)**2) / np.sum((y_val - np.mean(y_val))**2)
        print(f"  Q50 MAE: {mae_q50:.2f}元, R2: {r2_q50:.4f}")
        results['q50'] = {'mae': mae_q50, 'r2': r2_q50, 'pred': pred_q50}

        q75 = QuantileGBDT(n_trees=50, max_depth=3, lr=0.08, quantile=0.75)
        q75.fit(X_train_n, y_train)
        pred_q75 = np.clip(q75.predict(X_val_n), ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])

    # 5. Ridge元学习器Stacking
    print("\n[5/5] Ridge元学习器Stacking...")

    # 构建元特征：各模型验证集预测 (pred_lr等都是val set的预测)
    meta_features = np.column_stack([pred_lr, pred_knn, pred_gb, pred_q50])

    # Ridge回归作为元学习器
    from sklearn.linear_model import Ridge
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features, y_val)  # 用验证集预测和真实值训练

    # 在验证集上评估Stacking
    pred_stack = np.clip(meta_model.predict(meta_features), ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])
    mae_stack = np.mean(np.abs(y_val - pred_stack))
    r2_stack = 1 - np.sum((y_val - pred_stack)**2) / np.sum((y_val - np.mean(y_val))**2)

    print(f"\nRidge元学习器系数: {meta_model.coef_}")
    print(f"Stacking MAE: {mae_stack:.2f}元, R2: {r2_stack:.4f}")
    results['stack'] = {'mae': mae_stack, 'r2': r2_stack}

    # 计算预测区间
    interval_width = pred_q75 - pred_q25
    if np.any(interval_width < 0):
        mask = interval_width < 0
        temp = pred_q25[mask].copy()
        pred_q25[mask] = pred_q75[mask]
        pred_q75[mask] = temp
        interval_width = pred_q75 - pred_q25
    coverage = np.mean((y_val >= pred_q25) & (y_val <= pred_q75))
    print(f"\n预测区间覆盖率: {coverage*100:.1f}% (P20-P80)")
    print(f"平均区间宽度: {np.mean(interval_width):.1f}元")

    # 特征重要性分析
    print("\n[特征重要性分析]")
    importance_dict = compute_feature_importance(X_train_n, y_train, feature_names)
    print_feature_importance(importance_dict, top_n=10)

    metadata = {
        'mae_stack': mae_stack,
        'r2_stack': r2_stack,
        'meta_coef': meta_model.coef_.tolist(),
        'meta_intercept': float(meta_model.intercept_),
        'train_size': len(y_train),
        'val_size': len(y_val),
        'n_features': X.shape[1],
        'feature_names': feature_names,
        'coverage_p20_p80': coverage,
        'avg_interval_width': float(np.mean(interval_width)),
        'use_lightgbm': HAS_LIGHTGBM,
        'phase': 'optimized',
        'improvements': ['lgb_quantile', 'ridge_stacking', 'feature_importance']
    }

    # 全量训练
    print("\n全量训练...")
    X_n, mean_full, std_full = normalize(X)
    lr.fit(X_n, y)
    knn.fit(X_n, y)
    gb.fit(X_n, y)

    if HAS_LIGHTGBM:
        lgb_q25_full = train_lgb_quantile(X_n, y, quantile=0.20)
        lgb_q50_full = train_lgb_quantile(X_n, y, quantile=0.50)
        lgb_q75_full = train_lgb_quantile(X_n, y, quantile=0.80)
        lgb_models = {'q25': lgb_q25_full, 'q50': lgb_q50_full, 'q75': lgb_q75_full}
    else:
        lgb_models = None

    # 全量数据上重新训练元学习器
    meta_X_full = np.column_stack([
        lr.predict(X_n),
        knn.predict(X_n),
        gb.predict(X_n),
        lgb_q50_full.predict(X_n) if HAS_LIGHTGBM else q50.predict(X_n)
    ])
    meta_model_full = Ridge(alpha=1.0)
    meta_model_full.fit(meta_X_full, y)

    models_to_save = {
        'lr': lr, 'knn': knn, 'gb': gb,
        'lgb_models': lgb_models,
        'q25_model': q25 if not HAS_LIGHTGBM else None,
        'q50_model': q50 if not HAS_LIGHTGBM else None,
        'q75_model': q75 if not HAS_LIGHTGBM else None,
        'meta_model': meta_model_full,
        'norm': (mean_full, std_full)
    }

    save_v6_model(models_to_save, metadata)

    print("\n" + "=" * 60)
    print(f"模型训练完成!")
    print(f"  Stacking MAE: {mae_stack:.2f}元")
    print(f"  R²: {r2_stack:.4f}")
    print(f"  预测区间覆盖率: {coverage*100:.1f}%")
    print("=" * 60)

    return metadata


# ==================== 预测入口 ====================

_models_v6 = None
_cached_meta_v6 = None


def predict_price_v6(bond_code):
    """v6预测 - 优化版"""
    global _models_v6, _cached_meta_v6

    if _models_v6 is None:
        models_cache, meta = load_v6_model()
        if models_cache is not None:
            _models_v6 = models_cache
            _cached_meta_v6 = meta
        else:
            train_ensemble_v6()
            models_cache, _cached_meta_v6 = load_v6_model()

    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    if not bond:
        session.close()
        return None

    features, stock = prepare_v6_features(session, bond, include_market=True)
    feature_names = sorted(features.keys())

    arr = np.array([features.get(f, 0) for f in feature_names], dtype=float).reshape(1, -1)
    session.close()

    mean, std = _models_v6['norm']
    Xn = (arr - mean) / (std + 1e-8)

    # 各模型预测
    pred_lr = np.clip(_models_v6['lr'].predict(Xn), ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])[0]
    pred_knn = np.clip(_models_v6['knn'].predict(Xn), ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])[0]
    pred_gb = np.clip(_models_v6['gb'].predict(Xn), ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])[0]

    use_lgb = _cached_meta_v6.get('use_lightgbm', False) if _cached_meta_v6 else False

    if use_lgb and _models_v6.get('lgb_models'):
        lgb_models = _models_v6['lgb_models']
        pred_q50 = np.clip(lgb_models['q50'].predict(Xn)[0], ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])
        pred_q25 = np.clip(lgb_models['q25'].predict(Xn)[0], ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])
        pred_q75 = np.clip(lgb_models['q75'].predict(Xn)[0], ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])
    else:
        pred_q50 = np.clip(_models_v6['q50_model'].predict(Xn)[0], ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])
        pred_q25 = np.clip(_models_v6['q25_model'].predict(Xn)[0], ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])
        pred_q75 = np.clip(_models_v6['q75_model'].predict(Xn)[0], ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])

    # Ridge元学习器集成
    meta_model = _models_v6['meta_model']
    meta_features = np.array([[pred_lr, pred_knn, pred_gb, pred_q50]])
    pred = np.clip(meta_model.predict(meta_features)[0], ML_CONFIG['pred_min'], ML_CONFIG['pred_max'])

    mae = _cached_meta_v6.get('mae_stack', 8.0) if _cached_meta_v6 else 8.0

    # 市场信息
    market_info = get_market_sentiment(bond.listing_date)

    return {
        'predicted_price': round(float(pred), 1),
        'lr': round(float(pred_lr), 1),
        'knn': round(float(pred_knn), 1),
        'gb': round(float(pred_gb), 1),
        'q50': round(float(pred_q50), 1),
        # 预测区间
        'p20': round(float(pred_q25), 1),
        'p50': round(float(pred_q50), 1),
        'p80': round(float(pred_q75), 1),
        'confidence_interval': [round(float(pred_q25), 1), round(float(pred_q75), 1)],
        'interval_width': round(float(pred_q75 - pred_q25), 1),
        'model': 'v6_optimized',
        'mae': round(float(mae), 2),
        'market_sentiment': market_info.get('sentiment', 'neutral'),
        'use_lightgbm': use_lgb
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'force':
        train_ensemble_v6(force_retrain=True)
    else:
        train_ensemble_v6()
