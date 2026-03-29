"""
正股基本面特征工程
将正股基本面数据转换为机器学习可用的特征
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from db.models import get_session, BondInfo, StockInfo
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# 可转债分析的核心基本面特征
FUNDAMENTAL_FEATURES = [
    # 债券自身特征
    'conversion_value',      # 转股价值
    'premium_rate',         # 转股溢价率
    'issue_size',           # 发行规模(亿元)
    'coupon_rate',          # 票面利率
    'years_to_expiry',      # 剩余年限
    'rating_score',         # 信用评级分
    
    # 正股基本面特征
    'stock_pe',             # 正股市盈率
    'stock_pb',             # 正股市净率
    'stock_market_cap',     # 正股总市值(亿元)
    'stock_listing_days',   # 正股上市天数
    'stock_roe',            # 正股净资产收益率
    
    # 正股财务指标
    'revenue_growth',       # 营收增长率
    'profit_growth',        # 净利润增长率
    'gross_margin',         # 毛利率
    'debt_ratio',           # 资产负债率
    
    # 市场情绪特征
    'market_score',         # 大盘点位分数
    'industry_heat',        # 行业热度
    'cb_market_avg',        # 可转债市场平均价格
    'cb_count',             # 市场可转债数量
]


# 申万一级行业映射到热度分数（根据市场统计调整）
INDUSTRY_HEAT_SCORE = {
    '医药生物': 1.05,
    '电子': 1.08,
    '计算机': 1.06,
    '电力设备': 1.04,
    '汽车': 1.02,
    '机械设备': 1.00,
    '基础化工': 0.98,
    '有色金属': 1.00,
    '食品饮料': 0.95,
    '银行': 0.88,
    '非银金融': 0.90,
    '房地产': 0.82,
    '建筑装饰': 0.85,
    '交通运输': 0.90,
    '传媒': 1.00,
    '通信': 1.03,
    '国防军工': 1.05,
    '家用电器': 0.95,
    '轻工制造': 0.92,
    '纺织服装': 0.88,
    '商贸零售': 0.90,
    '社会服务': 0.92,
    '农林牧渔': 0.85,
    '环保': 0.95,
    '建筑材料': 0.88,
    '钢铁': 0.85,
    '煤炭': 0.88,
    '石油石化': 0.90,
    '公用事业': 0.90,
    '综合': 0.95,
}


def get_stock_industry_heat(industry):
    """获取行业热度分数"""
    return INDUSTRY_HEAT_SCORE.get(industry, 1.0)


def get_market_score(date):
    """获取大盘点位分数"""
    if date is None:
        return 0.5
    try:
        import akshare as ak
        df = ak.index_zh_a_hist(symbol='000001', period='monthly', 
                                  start_date='20200101', end_date='20251231')
        date_str = date.strftime('%Y-%m') if isinstance(date, datetime) else str(date)[:7]
        df['月份'] = df['日期'].astype(str).str[:7]
        if date_str in df['月份'].values:
            idx = float(df[df['月份'] == date_str]['收盘'].values[0])
            # 归一化到0-1区间（以3000点为中心）
            score = max(0, min(1, (idx - 3000) / 1500 + 0.5))
            return score
    except:
        pass
    return 0.5


def get_cb_market_avg(session, date):
    """获取当时可转债市场平均价格"""
    if date is None:
        return 115.0
    
    # 查询前后30天内上市的转债的平均首日价格
    from datetime import timedelta
    start = date - timedelta(days=60)
    end = date + timedelta(days=30)
    
    bonds = session.query(BondInfo).filter(
        BondInfo.listing_date >= start,
        BondInfo.listing_date <= end,
        BondInfo.first_open != None
    ).all()
    
    if bonds:
        avg = sum(b.first_open for b in bonds) / len(bonds)
        return avg
    
    return 115.0


def get_cb_market_count(session, date):
    """获取当时市场可转债数量"""
    if date is None:
        return 500
    
    from datetime import timedelta
    end = date + timedelta(days=30)
    
    count = session.query(BondInfo).filter(
        BondInfo.listing_date <= end
    ).count()
    
    return count


def prepare_all_features(session, bond, include_fundamental=True):
    """准备所有特征"""
    features = {}
    
    # === 债券自身特征 ===
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
    from config import RATING_MAP
    rating = bond.credit_rating or 'AA'
    features['rating_score'] = RATING_MAP.get(rating, 3)
    
    # === 正股基本面特征 ===
    stock = None
    if bond.stock_code:
        stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
    
    if stock and include_fundamental:
        features['stock_pe'] = stock.pe if stock.pe and stock.pe > 0 else 30
        features['stock_pb'] = stock.pb if stock.pb and stock.pb > 0 else 3
        features['stock_market_cap'] = stock.total_market_cap or 100
        features['stock_listing_days'] = stock.listing_days if stock.listing_days else 1000
        features['stock_roe'] = stock.roe if stock.roe else 8
        
        # 行业热度
        industry = stock.industry_sw_l1 or '其他'
        features['industry_heat'] = get_stock_industry_heat(industry)
    else:
        features['stock_pe'] = 30
        features['stock_pb'] = 3
        features['stock_market_cap'] = 100
        features['stock_listing_days'] = 1000
        features['stock_roe'] = 8
        features['industry_heat'] = 1.0
    
    # === 市场情绪特征 ===
    listing_date = bond.listing_date or datetime.now()
    features['market_score'] = get_market_score(listing_date)
    features['cb_market_avg'] = get_cb_market_avg(session, listing_date)
    features['cb_count'] = get_cb_market_count(session, listing_date)
    
    # === 财务指标（暂用默认值） ===
    features['revenue_growth'] = 5
    features['profit_growth'] = 3
    features['gross_margin'] = 25
    features['debt_ratio'] = 50
    
    return features, stock


def features_to_array(features, feature_names):
    """将特征字典转换为numpy数组"""
    values = []
    for name in feature_names:
        val = features.get(name, 0)
        if val is None:
            val = 0
        values.append(float(val))
    return np.array(values)


def get_feature_names():
    """获取所有特征名称"""
    return FUNDAMENTAL_FEATURES


def get_feature_names_with_fundamental():
    """获取包含基本面的完整特征名称"""
    return FUNDAMENTAL_FEATURES.copy()


def validate_features(features):
    """验证特征是否有效"""
    for k, v in features.items():
        if v is None:
            return False
        if not isinstance(v, (int, float)):
            return False
        if np.isnan(v) or np.isinf(v):
            return False
    return True


if __name__ == '__main__':
    # 测试特征工程
    session = get_session()
    
    bonds = session.query(BondInfo).filter(
        BondInfo.first_open != None,
        BondInfo.conversion_value != None
    ).limit(5).all()
    
    for bond in bonds:
        features, stock = prepare_all_features(session, bond)
        arr = features_to_array(features, FUNDAMENTAL_FEATURES)
        print(f"{bond.bond_code} {bond.bond_name}:")
        print(f"  转股价值={features['conversion_value']:.1f}, 溢价率={features['premium_rate']:.1f}%, PE={features['stock_pe']}")
        print(f"  特征向量: {arr[:6]}")
        print()
    
    session.close()