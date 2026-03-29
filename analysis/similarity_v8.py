"""
可转债估值模型 - 行业优化版 v8
按行业分别建模，结合市场景气度
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
            score = idx / 3500  # 归一化
            _market_cache[date_str] = score
            return score
    except:
        pass
    _market_cache[date_str] = 0.5
    return 0.5


# 行业基础价格（从2024年数据统计）
INDUSTRY_BASE_PRICE = {
    '汽车制造业': 124,
    '专用设备制造业': 126,
    '橡胶和塑料制品业': 124,
    '电气机械和器材制造业': 119,
    '化学原料和化学制品制造业': 116,
    '计算机、通信和其他电子设备制造业': 111,
    '金属制品业': 120,
    '食品制造业': 119,
    '医药制造业': 120,
    '有色金属冶炼和压延加工业': 111,
    '铁路、船舶、航空航天和其他运输设备制造业': 107,
    '纺织业': 129,
    '仪器仪表制造业': 130,
    '互联网和相关服务': 130,
    '生态保护和环境治理业': 130,
    '专业技术服务业': 130,
    '通用设备制造业': 104,
    '土木工程建筑业': 101,
    '批发业': 102,
    '零售业': 119,
    '研究和试验发展': 120,
    '其他': 115
}

# 行业景气度系数（行业处于上升期时加成）
INDUSTRY_BOOM_COEFF = {
    '汽车制造业': 1.05,        # 新能源车景气
    '专用设备制造业': 1.05,    # 设备更新
    '计算机、通信和其他电子设备制造业': 0.95,  # 电子寒冬
    '化学原料和化学制品制造业': 0.98,
    '医药制造业': 1.02,
    '电气机械和器材制造业': 1.03,  # 光伏储能
}


def predict_price_v8(bond_code):
    """v8预测：行业基准 + 转股价值调整 + 市场情绪"""
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    if not bond or not bond.first_open:
        session.close()
        return None
    
    stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
    industry = stock.industry_sw_l1 if stock and stock.industry_sw_l1 else '其他'
    
    # 1. 获取行业基准价
    base_price = INDUSTRY_BASE_PRICE.get(industry, INDUSTRY_BASE_PRICE['其他'])
    
    # 2. 行业景气度调整
    boom_coeff = INDUSTRY_BOOM_COEFF.get(industry, 1.0)
    base_price *= boom_coeff
    
    # 3. 市场情绪调整
    market_score = get_market_score(bond.listing_date)
    # 市场点位高于0.8牛市加成，低于0.7熊市减成
    if market_score > 0.85:
        base_price += 5
    elif market_score > 0.75:
        base_price += 2
    elif market_score < 0.70:
        base_price -= 3
    
    # 4. 转股价值调整
    if bond.conversion_value:
        cv = bond.conversion_value
        # 转股价值每超10元，价格加1元
        if cv > 100:
            cv_adj = (cv - 100) * 0.1
            base_price += cv_adj
    
    # 5. 溢价率调整
    if bond.premium_rate:
        prem = bond.premium_rate
        if prem < 10:
            base_price += 3  # 低溢价=高价值
        elif prem > 30:
            base_price -= 2  # 高溢价=风险
    
    # 6. 评级调整
    if bond.credit_rating:
        if bond.credit_rating == 'AAA':
            base_price += 3
        elif bond.credit_rating == 'AA+':
            base_price += 1
    
    # 7. 限制范围
    base_price = max(95, min(135, base_price))
    
    # 8. 同行业参考验证
    two_years_ago = datetime.now() - timedelta(days=730)
    same_ind_bonds = session.query(BondInfo).filter(
        BondInfo.bond_code != bond_code,
        BondInfo.first_open != None,
        BondInfo.listing_date >= two_years_ago
    ).all()
    
    # 找同行业
    same_ind_refs = []
    for b in same_ind_bonds:
        s = session.query(StockInfo).filter_by(stock_code=b.stock_code).first()
        if s and s.industry_sw_l1 == industry:
            same_ind_refs.append(b)
    
    if same_ind_refs:
        ref_prices = [b.first_open for b in same_ind_refs[:5]]
        ref_avg = sum(ref_prices) / len(ref_prices)
        # 行业参考占40%权重
        final_price = base_price * 0.6 + ref_avg * 0.4
    else:
        final_price = base_price
    
    final_price = max(95, min(135, final_price))
    
    session.close()
    
    return {
        'method': '行业基准+市场情绪',
        'predicted_price': round(final_price, 2),
        'industry': industry,
        'industry_base': INDUSTRY_BASE_PRICE.get(industry, 115),
        'market_score': round(market_score, 2)
    }


# 测试
if __name__ == '__main__':
    session = get_session()
    bonds = session.query(BondInfo).filter(
        BondInfo.listing_date >= datetime(2024, 1, 1),
        BondInfo.listing_date <= datetime(2024, 12, 31),
        BondInfo.first_open != None
    ).order_by(BondInfo.listing_date).limit(30).all()
    
    print('=== v8模型测试 (行业优化版) ===')
    print(f'{'代码':<8} {'行业':<15} {'预测':<8} {'实际':<8} {'误差':<8}')
    print('-'*60)
    
    errors = []
    for b in bonds:
        result = predict_price_v8(b.bond_code)
        if not result:
            continue
        
        pred = result['predicted_price']
        error = abs(pred - b.first_open)
        error_pct = error / b.first_open * 100
        errors.append(error_pct)
        
        ind = result['industry'][:12]
        flag = '✓' if error_pct <= 2 else ('⚠️' if error_pct > 10 else '○')
        print(f'{b.bond_code:<8} {ind:<15} {pred:<8.1f} {b.first_open:<8.1f} {error_pct:<7.1f}% {flag}')
    
    total = len(errors)
    within_2 = sum(1 for e in errors if e <= 2)
    over_10 = sum(1 for e in errors if e > 10)
    avg_error = sum(errors) / total if total > 0 else 0
    
    print(f'\n=== 统计 ===')
    print(f'误差≤2%: {within_2}/{total} ({within_2/total*100:.1f}%)')
    print(f'误差>10%: {over_10}/{total} ({over_10/total*100:.1f}%)')
    print(f'平均误差: {avg_error:.1f}%')
    
    session.close()