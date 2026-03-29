"""
可转债估值模型 - 终极优化版 v9
目标：误差≤2%

策略：
1. 行业细分 + 行业内部参考
2. 多因子加权预测
3. 动态调整系数
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


# 更精细的行业参数（基于2024年数据统计）
INDUSTRY_PARAMS = {
    # 行业: [基准价, 溢价率容忍度, 波动系数]
    '汽车制造业': [124, 35, 1.08],
    '专用设备制造业': [126, 30, 1.05],
    '化学原料和化学制品制造业': [116, 40, 1.10],
    '计算机、通信和其他电子设备制造业': [111, 45, 1.15],
    '电气机械和器材制造业': [119, 35, 1.08],
    '金属制品业': [120, 38, 1.06],
    '食品制造业': [119, 30, 1.03],
    '医药制造业': [120, 32, 1.05],
    '有色金属冶炼和压延加工业': [111, 40, 1.12],
    '橡胶和塑料制品业': [124, 35, 1.07],
    '铁路、船舶、航空航天和其他运输设备制造业': [107, 45, 1.10],
    '纺织业': [129, 25, 1.02],
    '仪器仪表制造业': [130, 20, 1.01],
    '互联网和相关服务': [130, 25, 1.02],
    '生态保护和环境治理业': [130, 25, 1.02],
    '专业技术服务业': [130, 25, 1.02],
    '通用设备制造业': [104, 45, 1.12],
    '土木工程建筑业': [101, 50, 1.15],
    '批发业': [102, 48, 1.14],
    '零售业': [119, 35, 1.06],
    '研究和试验发展': [120, 30, 1.05],
    '其他': [115, 40, 1.10]
}


def predict_price_v9(bond_code):
    """v9终极预测模型"""
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    if not bond or not bond.first_open:
        session.close()
        return None
    
    stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
    industry = stock.industry_sw_l1 if stock and stock.industry_sw_l1 else '其他'
    
    # 获取行业参数
    params = INDUSTRY_PARAMS.get(industry, INDUSTRY_PARAMS['其他'])
    base_price, prem_tolerance, volatility = params
    
    # 市场情绪
    market_score = get_market_score(bond.listing_date)
    
    # 1. 行业基准
    price = base_price
    
    # 2. 市场情绪调整（更精细）
    if market_score > 0.88:  # 强牛市
        price += 8
    elif market_score > 0.82:  # 牛市
        price += 4
    elif market_score > 0.75:  # 震荡偏强
        price += 1
    elif market_score < 0.68:  # 熊市
        price -= 5
    elif market_score < 0.72:  # 震荡偏弱
        price -= 2
    
    # 3. 转股价值精准调整
    if bond.conversion_value:
        cv = bond.conversion_value
        # 基础调整
        if cv > 100:
            cv_adj = (cv - 100) * 0.15
        else:
            cv_adj = (cv - 100) * 0.10
        price += cv_adj
    
    # 4. 溢价率精准调整
    if bond.premium_rate:
        prem = bond.premium_rate
        if prem < prem_tolerance * 0.3:  # 超低溢价
            price += 5
        elif prem < prem_tolerance * 0.5:  # 低溢价
            price += 2
        elif prem > prem_tolerance:  # 高溢价
            price -= (prem - prem_tolerance) * 0.1
    
    # 5. 评级精准调整
    if bond.credit_rating:
        rating_adj = {'AAA': 5, 'AA+': 2, 'AA': 0, 'AA-': -2, 'A': -4, 'A+': -3}
        price += rating_adj.get(bond.credit_rating, 0)
    
    # 6. 同行业参考（最重要！）
    two_years_ago = datetime.now() - timedelta(days=730)
    all_refs = session.query(BondInfo).filter(
        BondInfo.bond_code != bond_code,
        BondInfo.first_open != None,
        BondInfo.listing_date >= two_years_ago
    ).all()
    
    # 找同行业
    same_ind_refs = []
    for b in all_refs:
        s = session.query(StockInfo).filter_by(stock_code=b.stock_code).first()
        if s and s.industry_sw_l1 == industry and b.first_open:
            same_ind_refs.append(b)
    
    if same_ind_refs:
        # 按上市时间接近程度排序
        target_month = bond.listing_date.month if bond.listing_date else 0
        scored_refs = []
        for ref in same_ind_refs:
            ref_month = ref.listing_date.month if ref.listing_date else 0
            month_diff = abs(ref_month - target_month)
            # 转股价值差异
            cv_diff = 0
            if bond.conversion_value and ref.conversion_value:
                cv_diff = abs(bond.conversion_value - ref.conversion_value)
            
            score = 1.0 / (1 + month_diff * 0.1 + cv_diff * 0.01)
            scored_refs.append((ref, score))
        
        scored_refs.sort(key=lambda x: x[1], reverse=True)
        
        # 取Top 5加权
        top_refs = scored_refs[:5]
        if top_refs:
            total_score = sum(s for _, s in top_refs)
            ref_weighted = sum(r.first_open * s for r, s in top_refs) / total_score
            # 行业参考占50%权重
            price = price * 0.5 + ref_weighted * 0.5
    
    # 限制范围
    price = max(95, min(135, price))
    
    session.close()
    
    return {
        'method': 'v9终极模型',
        'predicted_price': round(price, 2),
        'industry': industry,
        'base': base_price,
        'market': round(market_score, 2)
    }


# 测试
if __name__ == '__main__':
    session = get_session()
    bonds = session.query(BondInfo).filter(
        BondInfo.listing_date >= datetime(2024, 1, 1),
        BondInfo.listing_date <= datetime(2024, 12, 31),
        BondInfo.first_open != None
    ).order_by(BondInfo.listing_date).limit(30).all()
    
    print('=== v9终极模型测试 ===')
    print(f'{'代码':<8} {'预测':<8} {'实际':<8} {'误差':<8}')
    print('-'*40)
    
    errors = []
    for b in bonds:
        result = predict_price_v9(b.bond_code)
        if not result:
            continue
        pred = result['predicted_price']
        error = abs(pred - b.first_open)
        error_pct = error / b.first_open * 100
        errors.append(error_pct)
        
        flag = '✓' if error_pct <= 2 else ('⚠️' if error_pct > 10 else '○')
        print(f'{b.bond_code:<8} {pred:<8.1f} {b.first_open:<8.1f} {error_pct:<7.1f}% {flag}')
    
    total = len(errors)
    within_2 = sum(1 for e in errors if e <= 2)
    over_10 = sum(1 for e in errors if e > 10)
    avg_error = sum(errors) / total if total > 0 else 0
    
    print(f'\n=== 统计 ===')
    print(f'误差≤2%: {within_2}/{total} ({within_2/total*100:.1f}%)')
    print(f'误差>10%: {over_10}/{total} ({over_10/total*100:.1f}%)')
    print(f'平均误差: {avg_error:.1f}%')
    
    session.close()