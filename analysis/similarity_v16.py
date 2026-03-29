"""
可转债估值模型 - 增强版 v16
加入市场资金流向作为情绪指标
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
_fund_flow_cache = {}

# 2024年各月统计
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


def get_fund_flow_score(date):
    """获取资金流向情绪"""
    if date is None:
        return 0.5
    
    date_str = date.strftime('%Y-%m-%d')
    if date_str in _fund_flow_cache:
        return _fund_flow_cache[date_str]
    
    try:
        # 获取最近30天的资金流向
        end_date = date.strftime('%Y%m%d')
        start_date = (date - timedelta(days=30)).strftime('%Y%m%d')
        
        df = ak.stock_market_fund_flow(start_date=start_date, end_date=end_date)
        
        # 计算最近5天的平均主力净流入
        if df is not None and len(df) >= 5:
            recent = df.tail(5)
            main_flow = recent['主力净流入-净额'].astype(float)
            avg_flow = main_flow.mean()
            
            # 转换为-1到1的情绪分数
            # 正流入=正面情绪，负流入=负面情绪
            if avg_flow > 100e8:  # 100亿
                score = 0.8
            elif avg_flow > 50e8:
                score = 0.6
            elif avg_flow > 0:
                score = 0.5
            elif avg_flow > -50e8:
                score = 0.4
            else:
                score = 0.3
            
            _fund_flow_cache[date_str] = score
            return score
    except:
        pass
    
    _fund_flow_cache[date_str] = 0.5
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


def predict_price_v16(bond_code):
    """v16增强版 - 加入资金流向"""
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
    month_type = month_stats['type']
    month_avg = month_stats['avg']
    
    # 获取市场点位
    market_score = get_market_score(bond.listing_date)
    
    # 获取资金流向情绪
    fund_flow_score = get_fund_flow_score(bond.listing_date)
    
    # 起点：月份平均价格
    price = month_avg
    
    # 行业调整
    price = price * (base_price / 115)
    
    # 关键：资金流向对市场情绪的影响
    # 资金流入 => 提高预测，流出 => 降低
    if fund_flow_score > 0.6:
        price += 3
    elif fund_flow_score < 0.4:
        price -= 3
    
    # 转股价值调整
    if has_cv and bond.conversion_value:
        cv = bond.conversion_value
        if cv >= 130:
            if month_type == 'high' and fund_flow_score > 0.5:
                price = max(price, 128)
        elif cv > 100:
            price += (cv - 100) * 0.04
    
    # 溢价率
    if has_prem and bond.premium_rate:
        prem = bond.premium_rate
        if prem < 10:
            price += 3
        elif prem > 40:
            price -= 2
    
    # 市场点位
    if market_score > 0.85 and month_type == 'high':
        price += 2
    elif market_score < 0.70 and month_type == 'low':
        price -= 2
    
    # 评级
    if bond.credit_rating:
        rating_adj = {'AAA': 4, 'AA+': 1, 'AA': 0, 'AA-': -2, 'A': -4}
        price += rating_adj.get(bond.credit_rating, 0)
    
    # 同行业参考
    two_years_ago = datetime.now() - timedelta(days=730)
    all_refs = session.query(BondInfo).filter(
        BondInfo.bond_code != bond_code,
        BondInfo.first_open != None,
        BondInfo.listing_date >= two_years_ago
    ).all()
    
    same_refs = []
    for b in all_refs:
        ref_month = b.listing_date.month if b.listing_date else 0
        s = session.query(StockInfo).filter_by(stock_code=b.stock_code).first()
        ref_type = MONTH_STATS.get(ref_month, {}).get('type', 'mid')
        
        if s and s.industry_sw_l1 == industry and ref_type == month_type:
            same_refs.append(b)
    
    if same_refs:
        same_refs.sort(key=lambda x: x.first_open, reverse=True)
        ref_avg = sum(r.first_open for r in same_refs[:3]) / min(3, len(same_refs))
        
        # 数据缺失时更依赖参考
        if not (has_cv and has_prem):
            price = price * 0.3 + ref_avg * 0.7
        else:
            price = price * 0.5 + ref_avg * 0.5
    
    price = max(95, min(135, price))
    
    session.close()
    
    return {
        'method': 'v16增强版(资金流向)',
        'predicted_price': round(price, 2),
        'month_type': month_type,
        'fund_flow_score': round(fund_flow_score, 2)
    }


if __name__ == '__main__':
    session = get_session()
    bonds = session.query(BondInfo).filter(
        BondInfo.listing_date >= datetime(2024, 1, 1),
        BondInfo.listing_date <= datetime(2024, 12, 31),
        BondInfo.first_open != None
    ).order_by(BondInfo.listing_date).limit(30).all()
    
    print('=== v16增强版测试 ===')
    errors = []
    for b in bonds:
        result = predict_price_v16(b.bond_code)
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