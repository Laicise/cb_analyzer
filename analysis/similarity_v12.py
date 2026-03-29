"""
可转债估值模型 - 鲁棒版 v12
处理数据缺失问题：40%转债缺少转股价值和溢价率
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


# 行业参数（带数据完整性调整）
INDUSTRY_PARAMS = {
    '汽车制造业': {'base': 124, 'boom': 1.05, 'has_data_ratio': 0.6},
    '专用设备制造业': {'base': 126, 'boom': 1.05, 'has_data_ratio': 0.5},
    '化学原料和化学制品制造业': {'base': 116, 'boom': 0.98, 'has_data_ratio': 0.7},
    '计算机、通信和其他电子设备制造业': {'base': 111, 'boom': 0.95, 'has_data_ratio': 0.5},
    '电气机械和器材制造业': {'base': 119, 'boom': 1.03, 'has_data_ratio': 0.6},
    '金属制品业': {'base': 120, 'boom': 1.00, 'has_data_ratio': 0.7},
    '食品制造业': {'base': 119, 'boom': 1.00, 'has_data_ratio': 0.5},
    '医药制造业': {'base': 120, 'boom': 1.02, 'has_data_ratio': 0.4},
    '有色金属冶炼和压延加工业': {'base': 111, 'boom': 0.98, 'has_data_ratio': 0.6},
    '橡胶和塑料制品业': {'base': 124, 'boom': 1.00, 'has_data_ratio': 0.6},
    '铁路、船舶、航空航天和其他运输设备制造业': {'base': 107, 'boom': 1.00, 'has_data_ratio': 0.5},
    '纺织业': {'base': 129, 'boom': 1.00, 'has_data_ratio': 0.5},
    '仪器仪表制造业': {'base': 130, 'boom': 1.00, 'has_data_ratio': 0.4},
    '互联网和相关服务': {'base': 130, 'boom': 1.00, 'has_data_ratio': 0.3},
    '其他': {'base': 115, 'boom': 1.00, 'has_data_ratio': 0.5}
}


def predict_price_v12(bond_code):
    """v12鲁棒版 - 处理数据缺失"""
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    if not bond or not bond.first_open:
        session.close()
        return None
    
    stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
    industry = stock.industry_sw_l1 if stock and stock.industry_sw_l1 else '其他'
    
    params = INDUSTRY_PARAMS.get(industry, INDUSTRY_PARAMS['其他'])
    base_price = params['base']
    boom_coeff = params['boom']
    data_ratio = params['has_data_ratio']
    
    # 数据完整性检查
    has_cv = bond.conversion_value is not None
    has_prem = bond.premium_rate is not None
    
    market_score = get_market_score(bond.listing_date)
    listing_month = bond.listing_date.month if bond.listing_date else 0
    
    price = base_price * boom_coeff
    
    # 根据数据完整性调整权重
    if has_cv and has_prem:
        # 数据完整，按正常逻辑
        if market_score > 0.85:
            price += 5
        elif market_score > 0.75:
            price += 2
        elif market_score < 0.70:
            price -= 3
        
        if bond.conversion_value and bond.conversion_value >= 130:
            price = max(price, 128)
        
        if bond.premium_rate:
            if bond.premium_rate < 10:
                price += 4
            elif bond.premium_rate < 20:
                price += 1
    else:
        # 数据缺失，依赖行业基准和月份特征
        # 关键：如果数据缺失，行业基准更重要
        
        # 1月/7月/8月等高130月份，提高预测
        if listing_month in [1, 2, 4, 8, 11]:
            # 这些月份130元概率高
            if market_score > 0.75:
                price += 3
        
        # 如果发行规模小，可能被爆炒
        if bond.issue_size and bond.issue_size < 5:
            price += 2
    
    # 评级（始终有效）
    if bond.credit_rating:
        rating_adj = {'AAA': 4, 'AA+': 1, 'AA': 0, 'AA-': -2, 'A': -4}
        price += rating_adj.get(bond.credit_rating, 0)
    
    # 同行业参考（更保守）
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
        # 按月份接近程度排序
        same_ind_refs.sort(key=lambda x: abs((x.listing_date.month if x.listing_date else 0) - listing_month))
        
        # 取最近3个月上市的
        recent_refs = same_ind_refs[:3]
        if recent_refs:
            ref_prices = [r.first_open for r in recent_refs]
            ref_avg = sum(ref_prices) / len(ref_prices)
            
            # 数据缺失时，更依赖历史参考
            if not (has_cv and has_prem):
                price = price * 0.3 + ref_avg * 0.7
            else:
                price = price * 0.6 + ref_avg * 0.4
    
    price = max(95, min(135, price))
    
    session.close()
    
    return {
        'method': 'v12鲁棒版(数据缺失处理)',
        'predicted_price': round(price, 2),
        'industry': industry,
        'has_cv': has_cv,
        'has_prem': has_prem
    }


if __name__ == '__main__':
    session = get_session()
    bonds = session.query(BondInfo).filter(
        BondInfo.listing_date >= datetime(2024, 1, 1),
        BondInfo.listing_date <= datetime(2024, 12, 31),
        BondInfo.first_open != None
    ).order_by(BondInfo.listing_date).limit(30).all()
    
    print('=== v12鲁棒版测试 ===')
    errors = []
    for b in bonds:
        result = predict_price_v12(b.bond_code)
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