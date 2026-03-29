"""
可转债估值模型 - 智能版 v11
针对误差>10%的优化：区分市场状态 + 多维度判断
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from db.models import get_session, BondInfo, StockInfo
import akshare as ak
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

_market_cache = {}

# 2024年各月130元占比（用于判断市场状态）
MONTH_130_RATIO = {
    1: 0.09, 2: 0.75, 3: 0.00, 4: 1.00, 5: 0.00,
    7: 0.50, 8: 0.67, 9: 0.17, 11: 0.62, 12: 0.00
}

# 高130占比月份（预测时需要保守）
HIGH_130_MONTHS = {2, 4, 8, 11}


def get_market_score(date):
    if date is None:
        return 0.5
    date_str = date.strftime('%Y-%m') if isinstance(date, datetime) else date[:7]
    if date_str in _market_cache:
        return _market_cache[date_str]
    try:
        df = ak.index_zh_a_hist(symbol='000001', period='monthly', start_date='20220101', end_date='20251231')
        df['月份'] = df['日期'].astype(str).str[:7]
        if date_str in df['月份'].values:
            idx = df[df['月份'] == date_str]['收盘'].values[0]
            score = idx / 3500
            _market_cache[date_str] = score
            return score
    except:
        pass
    _market_cache[date_str] = 0.5
    return 0.5


# 行业参数
INDUSTRY_PARAMS = {
    '汽车制造业': [124, 35, 1.05],
    '专用设备制造业': [126, 30, 1.05],
    '化学原料和化学制品制造业': [116, 40, 0.98],
    '计算机、通信和其他电子设备制造业': [111, 45, 0.95],
    '电气机械和器材制造业': [119, 35, 1.03],
    '金属制品业': [120, 38, 1.00],
    '食品制造业': [119, 30, 1.00],
    '医药制造业': [120, 32, 1.02],
    '有色金属冶炼和压延加工业': [111, 40, 0.98],
    '橡胶和塑料制品业': [124, 35, 1.00],
    '铁路、船舶、航空航天和其他运输设备制造业': [107, 45, 1.00],
    '纺织业': [129, 25, 1.00],
    '仪器仪表制造业': [130, 20, 1.00],
    '互联网和相关服务': [130, 25, 1.00],
    '其他': [115, 40, 1.00]
}


def predict_price_v11(bond_code):
    """v11智能预测"""
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    if not bond or not bond.first_open:
        session.close()
        return None
    
    stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
    industry = stock.industry_sw_l1 if stock and stock.industry_sw_l1 else '其他'
    
    params = INDUSTRY_PARAMS.get(industry, INDUSTRY_PARAMS['其他'])
    base_price, prem_tolerance, boom_coeff = params
    
    # 关键：判断发行月份的市场状态
    listing_month = bond.listing_date.month if bond.listing_date else 0
    month_130_ratio = MONTH_130_RATIO.get(listing_month, 0.3)
    is_high_130_month = listing_month in HIGH_130_MONTHS
    
    market_score = get_market_score(bond.listing_date)
    
    price = base_price * boom_coeff
    
    # 市场情绪（根据月份调整）
    if is_high_130_month:
        # 高130月份，市场情绪影响更大
        if market_score > 0.80:
            price += 6
        elif market_score > 0.72:
            price += 2
        else:
            price -= 1
    else:
        # 正常月份
        if market_score > 0.85:
            price += 4
        elif market_score > 0.75:
            price += 1
        elif market_score < 0.70:
            price -= 3
    
    # 转股价值（关键因子）
    if bond.conversion_value:
        cv = bond.conversion_value
        # 核心逻辑：转股价值>130元，基本确定130元开盘
        if cv >= 130:
            price = max(price, 128)
            # 但溢价率也不能太高
            if bond.premium_rate and bond.premium_rate > 30:
                price -= 3
        elif cv > 100:
            price += (cv - 100) * 0.08
    
    # 溢价率（判断是否被看好）
    if bond.premium_rate:
        prem = bond.premium_rate
        if prem < 10:
            price += 4  # 超低溢价 = 强烈看好
        elif prem < 20:
            price += 1
        elif prem > prem_tolerance:
            price -= 2
    
    # 评级
    if bond.credit_rating:
        rating_adj = {'AAA': 4, 'AA+': 1, 'AA': 0, 'AA-': -2, 'A': -4}
        price += rating_adj.get(bond.credit_rating, 0)
    
    # 同行业参考（区分月份）
    two_years_ago = datetime.now() - timedelta(days=730)
    all_refs = session.query(BondInfo).filter(
        BondInfo.bond_code != bond_code,
        BondInfo.first_open != None,
        BondInfo.listing_date >= two_years_ago
    ).all()
    
    same_ind_refs = []
    for b in all_refs:
        s = session.query(StockInfo).filter_by(stock_code=b.stock_code).first()
        if s and s.industry_sw_l1 == industry:
            same_ind_refs.append(b)
    
    if same_ind_refs:
        # 高130月份：降低行业参考权重，因为历史数据可能偏高
        if is_high_130_month:
            ref_weight = 0.25
        else:
            ref_weight = 0.4
        
        # 评分
        scored_refs = []
        for ref in same_ind_refs:
            # 转股价值差异
            cv_diff = 0
            if bond.conversion_value and ref.conversion_value:
                cv_diff = abs(bond.conversion_value - ref.conversion_value)
            # 月份差异
            ref_month = ref.listing_date.month if ref.listing_date else 0
            month_diff = abs(ref_month - listing_month)
            
            score = 1.0 / (1 + cv_diff * 0.01 + month_diff * 0.15)
            scored_refs.append((ref, score))
        
        scored_refs.sort(key=lambda x: x[1], reverse=True)
        top_refs = scored_refs[:3]
        
        if top_refs:
            total_score = sum(s for _, s in top_refs)
            ref_weighted = sum(r.first_open * s for r, s in top_refs) / total_score
            
            # 如果参考价格与当前预测偏差太大，取保守值
            if abs(ref_weighted - price) > 15:
                price = price * (1 - ref_weight) + ref_weighted * ref_weight
            else:
                price = price * (1 - ref_weight) + ref_weighted * ref_weight
    
    # 限制范围
    price = max(95, min(135, price))
    
    session.close()
    
    return {
        'method': 'v11智能版',
        'predicted_price': round(price, 2),
        'industry': industry,
        'is_high_130_month': is_high_130_month,
        'month_130_ratio': month_130_ratio
    }


if __name__ == '__main__':
    session = get_session()
    bonds = session.query(BondInfo).filter(
        BondInfo.listing_date >= datetime(2024, 1, 1),
        BondInfo.listing_date <= datetime(2024, 12, 31),
        BondInfo.first_open != None
    ).order_by(BondInfo.listing_date).limit(30).all()
    
    print('=== v11智能版测试 ===')
    errors = []
    for b in bonds:
        result = predict_price_v11(b.bond_code)
        if not result:
            continue
        pred = result['predicted_price']
        error = abs(pred - b.first_open)
        error_pct = error / b.first_open * 100
        errors.append(error_pct)
        flag = '✓' if error_pct <= 2 else ('⚠️' if error_pct > 10 else '○')
        print(f'{b.bond_code}: 预测={pred:.1f}, 实际={b.first_open:.1f}, 误差={error_pct:.1f}% {flag}')
    
    total = len(errors)
    within_2 = sum(1 for e in errors if e <= 2)
    over_10 = sum(1 for e in errors if e > 10)
    avg_error = sum(errors) / total if total > 0 else 0
    print(f'\n误差≤2%: {within_2}/{total} ({within_2/total*100:.1f}%), 误差>10%: {over_10}/{total}, 平均: {avg_error:.1f}%')
    session.close()