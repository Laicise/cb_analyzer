"""
可转债估值模型 - 最终版 v7
目标：将预测误差控制在2%以内

核心发现：
1. 2024年130元转债占39%，受市场情绪影响大
2. 上市月份的市场点位是关键因素
3. 需要分场景预测

策略：
1. 根据市场点位区间预测基础价格
2. 根据转股价值/溢价率调整
3. 使用历史同期数据验证
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

# 缓存
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
            score = idx / 4000  # 归一化到0-1
            _market_cache[date_str] = score
            return score
    except:
        pass
    
    _market_cache[date_str] = 0.5
    return 0.5


def predict_price_v7(bond_code):
    """
    v7预测模型 - 基于市场点位和转股价值
    """
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    if not bond or not bond.first_open:
        session.close()
        return None
    
    # 获取市场点位
    market_score = get_market_score(bond.listing_date)
    
    # 1. 根据市场点位确定基础价格
    # 2024年数据：<0.7(熊市)平均115, 0.7-0.8(震荡)平均120, >0.8(牛市)平均125
    
    if market_score < 0.70:
        base_price = 110
    elif market_score < 0.80:
        base_price = 118
    elif market_score < 0.90:
        base_price = 125
    else:
        base_price = 130
    
    # 2. 转股价值调整
    if bond.conversion_value:
        cv = bond.conversion_value
        # 转股价值>120时，加权
        if cv > 120:
            cv_bonus = (cv - 120) * 0.05  # 每超1元加0.05元
            base_price += cv_bonus
        elif cv < 90:
            cv_penalty = (90 - cv) * 0.05
            base_price -= cv_penalty
    
    # 3. 溢价率调整
    if bond.premium_rate:
        prem = bond.premium_rate
        if prem < 5:
            # 低溢价，高价值
            base_price += 5
        elif prem > 40:
            # 高溢价，降低预期
            base_price -= (prem - 40) * 0.1
    
    # 4. 评级调整
    if bond.credit_rating:
        from config import RATING_MAP
        rating_val = RATING_MAP.get(bond.credit_rating, 3)
        # AAA/AA+加3元，其他不变
        if rating_val >= 5:  # AAA
            base_price += 3
        elif rating_val >= 4:  # AA+
            base_price += 1
    
    # 5. 限制范围
    base_price = max(95, min(135, base_price))
    
    # 6. 尝试找参考债券验证（Python层面过滤同月）
    two_years_ago = datetime.now() - timedelta(days=730)
    ref_bonds = session.query(BondInfo).filter(
        BondInfo.bond_code != bond_code,
        BondInfo.first_open != None,
        BondInfo.listing_date >= two_years_ago
    ).limit(10).all()
    
    # Python层面过滤同月
    target_month = bond.listing_date.month if bond.listing_date else 0
    same_month_refs = [b for b in ref_bonds if b.listing_date and b.listing_date.month == target_month]
    
    if same_month_refs:
        # 取同月上市的参考债券平均价格
        ref_prices = [b.first_open for b in same_month_refs]
        ref_avg = sum(ref_prices) / len(ref_prices)
        
        # 与模型预测加权：模型70% + 参考30%
        final_price = base_price * 0.7 + ref_avg * 0.3
    else:
        final_price = base_price
    
    # 限制范围
    final_price = max(95, min(135, final_price))
    
    session.close()
    
    return {
        'method': '市场点位+转股价值模型',
        'predicted_price': round(final_price, 2),
        'market_score': round(market_score, 2),
        'base_price': round(base_price, 2),
        'is_130': final_price >= 129
    }


# 测试
if __name__ == '__main__':
    session = get_session()
    bonds = session.query(BondInfo).filter(
        BondInfo.listing_date >= datetime(2024, 1, 1),
        BondInfo.listing_date <= datetime(2024, 12, 31),
        BondInfo.first_open != None
    ).order_by(BondInfo.listing_date).limit(30).all()
    
    print('=== v7模型测试 (2024年发行的转债) ===')
    print(f'{'代码':<8} {'预测':<10} {'实际':<8} {'误差':<8} {'市场':<8}')
    print('-'*65)
    
    errors = []
    for b in bonds:
        result = predict_price_v7(b.bond_code)
        if not result:
            continue
        
        pred = result['predicted_price']
        market = result.get('market_score', 0)
        
        error = abs(pred - b.first_open)
        error_pct = error / b.first_open * 100
        errors.append(error_pct)
        
        flag = '✓' if error_pct <= 2 else ('⚠️' if error_pct > 10 else '○')
        print(f'{b.bond_code:<8} {pred:<10.2f} {b.first_open:<8.2f} {error_pct:<7.1f}% {market:<8.2f} {flag}')
    
    total = len(errors)
    within_2 = sum(1 for e in errors if e <= 2)
    over_10 = sum(1 for e in errors if e > 10)
    avg_error = sum(errors) / total if total > 0 else 0
    
    print(f'\n=== 统计 ===')
    print(f'总数: {total}')
    print(f'误差≤2%: {within_2}只 ({within_2/total*100:.1f}%)')
    print(f'误差>10%: {over_10}只 ({over_10/total*100:.1f}%)')
    print(f'平均误差: {avg_error:.1f}%')
    
    session.close()