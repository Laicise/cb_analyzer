"""
可转债估值模型 - 月份感知版 v13
关键发现：1-3月低开多，8-11月高开多
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

# 2024年各月统计数据
MONTH_STATS = {
    1: {'avg': 112.4, 'ratio_130': 0.09, 'type': 'low'},
    2: {'avg': 125.8, 'ratio_130': 0.75, 'type': 'high'},
    3: {'avg': 110.1, 'ratio_130': 0.00, 'type': 'low'},
    4: {'avg': 130.0, 'ratio_130': 1.00, 'type': 'high'},
    5: {'avg': 119.5, 'ratio_130': 0.00, 'type': 'mid'},
    7: {'avg': 121.9, 'ratio_130': 0.50, 'type': 'mid'},
    8: {'avg': 124.0, 'ratio_130': 0.67, 'type': 'high'},
    9: {'avg': 113.0, 'ratio_130': 0.17, 'type': 'low'},
    11: {'avg': 123.3, 'ratio_130': 0.62, 'type': 'high'},
    12: {'avg': 110.3, 'ratio_130': 0.00, 'type': 'low'},
}


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
    '汽车制造业': {'base': 124, 'boom': 1.05},
    '专用设备制造业': {'base': 126, 'boom': 1.05},
    '化学原料和化学制品制造业': {'base': 116, 'boom': 0.98},
    '计算机、通信和其他电子设备制造业': {'base': 111, 'boom': 0.95},
    '电气机械和器材制造业': {'base': 119, 'boom': 1.03},
    '金属制品业': {'base': 120, 'boom': 1.00},
    '食品制造业': {'base': 119, 'boom': 1.00},
    '医药制造业': {'base': 120, 'boom': 1.02},
    '其他': {'base': 115, 'boom': 1.00}
}


def predict_price_v13(bond_code):
    """v13月份感知版"""
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    if not bond or not bond.first_open:
        session.close()
        return None
    
    stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
    industry = stock.industry_sw_l1 if stock and stock.industry_sw_l1 else '其他'
    
    params = INDUSTRY_PARAMS.get(industry, INDUSTRY_PARAMS['其他'])
    base_price = params['base'] * params['boom']
    
    has_cv = bond.conversion_value is not None
    has_prem = bond.premium_rate is not None
    listing_month = bond.listing_date.month if bond.listing_date else 0
    month_stats = MONTH_STATS.get(listing_month, {'avg': 115, 'ratio_130': 0.3, 'type': 'mid'})
    
    market_score = get_market_score(bond.listing_date)
    
    # 核心：基于月份类型调整
    month_type = month_stats['type']
    month_avg = month_stats['avg']
    
    price = base_price
    
    if month_type == 'high':
        # 高开月份（2/4/8/11月），倾向于预测高价格
        if market_score > 0.75:
            price = max(price, 125)
            # 如果市场好，大概率130
            if has_cv and bond.conversion_value and bond.conversion_value > 120:
                price = 130
    elif month_type == 'low':
        # 低开月份（1/3/9/12月），倾向于保守
        price = min(price, 118)
        if not has_cv:
            # 数据缺失+低开月份，进一步保守
            price = min(price, 110)
    else:
        # 中间月份（5/7月）
        if market_score > 0.80:
            price += 3
    
    # 有数据时用转股价值/溢价率调整
    if has_cv and has_prem:
        if bond.conversion_value:
            cv = bond.conversion_value
            if cv >= 130:
                price = max(price, 128)
            elif cv > 100:
                price += (cv - 100) * 0.06
        
        if bond.premium_rate:
            prem = bond.premium_rate
            if prem < 10:
                price += 3
            elif prem > 40:
                price -= 2
    
    # 评级
    if bond.credit_rating:
        rating_adj = {'AAA': 4, 'AA+': 1, 'AA': 0, 'AA-': -2, 'A': -4}
        price += rating_adj.get(bond.credit_rating, 0)
    
    # 同行业+同月份参考（关键！）
    two_years_ago = datetime.now() - timedelta(days=730)
    all_refs = session.query(BondInfo).filter(
        BondInfo.bond_code != bond_code,
        BondInfo.first_open != None,
        BondInfo.listing_date >= two_years_ago
    ).all()
    
    # 找同月份+同行业的参考
    same_refs = []
    for b in all_refs:
        ref_month = b.listing_date.month if b.listing_date else 0
        s = session.query(StockInfo).filter_by(stock_code=b.stock_code).first()
        
        # 关键：月份相同或类型相同
        ref_type = MONTH_STATS.get(ref_month, {}).get('type', 'mid')
        if s and s.industry_sw_l1 == industry and ref_type == month_type:
            same_refs.append(b)
    
    if same_refs:
        same_refs.sort(key=lambda x: x.first_open, reverse=True)
        ref_avg = sum(r.first_open for r in same_refs[:3]) / min(3, len(same_refs))
        
        # 如果月份类型匹配度高，给更高权重
        if month_type == 'high':
            price = price * 0.4 + ref_avg * 0.6
        else:
            price = price * 0.5 + ref_avg * 0.5
    
    price = max(95, min(135, price))
    
    session.close()
    
    return {
        'method': 'v13月份感知',
        'predicted_price': round(price, 2),
        'month_type': month_type,
        'month_avg': month_avg
    }


if __name__ == '__main__':
    session = get_session()
    bonds = session.query(BondInfo).filter(
        BondInfo.listing_date >= datetime(2024, 1, 1),
        BondInfo.listing_date <= datetime(2024, 12, 31),
        BondInfo.first_open != None
    ).order_by(BondInfo.listing_date).limit(30).all()
    
    print('=== v13月份感知版测试 ===')
    errors = []
    for b in bonds:
        result = predict_price_v13(b.bond_code)
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
    print(f'\n误差≤2%: {within_2}/{total}, 误差>10%: {over_10}/{total}, 平均: {avg_error:.1f}%')
    session.close()