"""
可转债估值模型 - 修复版 v10
修复2024年下半年130元参考偏差问题
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
    '汽车制造业': [124, 35],
    '专用设备制造业': [126, 30],
    '化学原料和化学制品制造业': [116, 40],
    '计算机、通信和其他电子设备制造业': [111, 45],
    '电气机械和器材制造业': [119, 35],
    '金属制品业': [120, 38],
    '食品制造业': [119, 30],
    '医药制造业': [120, 32],
    '有色金属冶炼和压延加工业': [111, 40],
    '橡胶和塑料制品业': [124, 35],
    '铁路、船舶、航空航天和其他运输设备制造业': [107, 45],
    '纺织业': [129, 25],
    '仪器仪表制造业': [130, 20],
    '互联网和相关服务': [130, 25],
    '其他': [115, 40]
}


def predict_price_v10(bond_code):
    """v10修复版"""
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    if not bond or not bond.first_open:
        session.close()
        return None
    
    stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
    industry = stock.industry_sw_l1 if stock and stock.industry_sw_l1 else '其他'
    
    params = INDUSTRY_PARAMS.get(industry, INDUSTRY_PARAMS['其他'])
    base_price, prem_tolerance = params
    
    # 核心：判断目标债券是否可能130元
    is_likely_130 = False
    if bond.conversion_value and bond.conversion_value > 130:
        is_likely_130 = True
    if bond.premium_rate and bond.premium_rate < 10:
        is_likely_130 = True
    
    market_score = get_market_score(bond.listing_date)
    
    price = base_price
    
    # 市场情绪
    if market_score > 0.85:
        price += 5
    elif market_score > 0.78:
        price += 2
    elif market_score < 0.70:
        price -= 4
    
    # 转股价值
    if bond.conversion_value:
        cv = bond.conversion_value
        if cv > 100:
            price += (cv - 100) * 0.12
    
    # 溢价率
    if bond.premium_rate:
        prem = bond.premium_rate
        if prem < 15:
            price += 3
    
    # 评级
    if bond.credit_rating:
        rating_adj = {'AAA': 4, 'AA+': 1, 'AA': 0, 'AA-': -2}
        price += rating_adj.get(bond.credit_rating, 0)
    
    # 关键修复：排除2024年9-12月的130元高开参考
    # 只用2024年前8个月或2022-2023年的数据
    
    # 历史参考（2022-2023）
    all_refs = session.query(BondInfo).filter(
        BondInfo.bond_code != bond_code,
        BondInfo.first_open != None,
        BondInfo.listing_date < datetime(2024, 9, 1)  # 2024年9月前
    ).all()
    
    same_ind_refs = []
    for b in all_refs:
        s = session.query(StockInfo).filter_by(stock_code=b.stock_code).first()
        if s and s.industry_sw_l1 == industry:
            same_ind_refs.append(b)
    
    if same_ind_refs:
        # 转股价值差异计算
        scored_refs = []
        for ref in same_ind_refs:
            cv_diff = 0
            if bond.conversion_value and ref.conversion_value:
                cv_diff = abs(bond.conversion_value - ref.conversion_value)
            score = 1.0 / (1 + cv_diff * 0.01)
            scored_refs.append((ref, score))
        
        scored_refs.sort(key=lambda x: x[1], reverse=True)
        top_refs = scored_refs[:3]
        
        if top_refs:
            total_score = sum(s for _, s in top_refs)
            ref_weighted = sum(r.first_open * s for r, s in top_refs) / total_score
            price = price * 0.6 + ref_weighted * 0.4
    
    price = max(95, min(135, price))
    
    session.close()
    
    return {
        'method': 'v10(排除高开偏差)',
        'predicted_price': round(price, 2),
        'industry': industry,
        'is_likely_130': is_likely_130
    }


# 测试
if __name__ == '__main__':
    session = get_session()
    bonds = session.query(BondInfo).filter(
        BondInfo.listing_date >= datetime(2024, 1, 1),
        BondInfo.listing_date <= datetime(2024, 12, 31),
        BondInfo.first_open != None
    ).order_by(BondInfo.listing_date).limit(30).all()
    
    print('=== v10修复版测试 ===')
    errors = []
    for b in bonds:
        result = predict_price_v10(b.bond_code)
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