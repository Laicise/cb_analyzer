"""
相似度匹配模型（最终版）
包含：首日价格参考 + 市场景气度因子
"""
import pandas as pd
import numpy as np
from db.models import get_session, BondInfo, StockInfo, BondDaily
from config import SIMILARITY_WEIGHTS, RATING_MAP
import akshare as ak
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 缓存市场指数数据
_market_cache = {}


def get_market_index(date):
    """
    获取上证指数收盘价作为市场景气度指标
    返回: 指数值, 景气度(0-1, 越高越景气)
    """
    if date is None:
        return None, 0.5
    
    # 格式化日期
    if isinstance(date, str):
        date_str = date[:7]  # YYYY-MM
    else:
        date_str = date.strftime('%Y-%m')
    
    if date_str in _market_cache:
        return _market_cache[date_str]
    
    try:
        # 获取上证指数月线
        df = ak.index_zh_a_hist(
            symbol='000001', 
            period='monthly', 
            start_date='20180101', 
            end_date='20251231'
        )
        
        # 找到对应月份的指数
        df['月份'] = df['日期'].astype(str).str[:7]
        
        if date_str in df['月份'].values:
            idx_value = df[df['月份'] == date_str]['收盘'].values[0]
            
            # 计算景气度（相对于历史均值）
            # 简化的景气度计算：取历史数据的分位数
            hist_mean = df['收盘'].mean()
            hist_std = df['收盘'].std()
            
            if hist_std > 0:
                # 标准化到0-1
                z_score = (idx_value - hist_mean) / hist_std
                # 转映射到0-1区间
                market_score = 1 / (1 + np.exp(-z_score))  # sigmoid
            else:
                market_score = 0.5
            
            _market_cache[date_str] = (idx_value, market_score)
            return idx_value, market_score
    except:
        pass
    
    _market_cache[date_str] = (None, 0.5)
    return None, 0.5


def get_industry_score(industry1, industry2):
    if not industry1 or not industry2:
        return 0
    if industry1 == industry2:
        return 1.0
    common_words = set(industry1) & set(industry2)
    if len(common_words) > 0:
        return 0.5
    return 0


def get_business_score(business1, business2):
    if not business1 or not business2:
        return 0
    words1 = set(business1.replace(',', '').replace('。', '').split())
    words2 = set(business2.replace(',', '').replace('。', '').split())
    if not words1 or not words2:
        return 0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0


def get_rating_score(rating1, rating2):
    if not rating1 or not rating2:
        return 0
    r1 = RATING_MAP.get(rating1, 0)
    r2 = RATING_MAP.get(rating2, 0)
    diff = abs(r1 - r2)
    return max(0, 1 - diff / 5)


def get_value_ratio_score(val1, val2):
    if val1 is None or val2 is None:
        return 0
    try:
        v1, v2 = float(val1), float(val2)
        if v1 == 0 or v2 == 0:
            return 0
        return min(abs(v1), abs(v2)) / max(abs(v1), abs(v2))
    except:
        return 0


def get_value_diff_score(val1, val2, max_diff=30):
    if val1 is None or val2 is None:
        return 0
    try:
        diff = abs(float(val1) - float(val2))
        return max(0, 1 - diff / max_diff)
    except:
        return 0


def find_similar_bonds(bond_code, top_n=5):
    """查找最相似的已上市可转债"""
    session = get_session()
    
    new_bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    if not new_bond:
        session.close()
        return None
    
    new_stock = session.query(StockInfo).filter_by(stock_code=new_bond.stock_code).first()
    
    new_data = {
        'bond_code': new_bond.bond_code,
        'bond_name': new_bond.bond_name,
        'stock_name': new_bond.stock_name,
        'industry': new_stock.industry_sw_l1 if new_stock else '',
        'business': new_stock.business if new_stock else '',
        'rating': new_bond.credit_rating,
        'issue_size': new_bond.issue_size,
        'conversion_value': new_bond.conversion_value,
        'premium_rate': new_bond.premium_rate,
        'listing_date': new_bond.listing_date,
    }
    
    # 获取目标债券上市时的市场景气度
    target_market_idx, target_market_score = get_market_index(new_bond.listing_date)
    new_data['market_index'] = target_market_idx
    new_data['market_score'] = target_market_score
    
    # 判断是否为银行转债（转股价值为None）
    is_bank_bond = new_bond.conversion_value is None
    
    bonds = session.query(BondInfo).filter(
        BondInfo.bond_code != bond_code,
        BondInfo.first_open != None
    ).all()
    
    # 如果目标债券是银行转债，则优先匹配银行转债
    if is_bank_bond:
        # 过滤出银行转债（转股价值为None的）
        bank_bonds = [b for b in bonds if b.conversion_value is None]
        if bank_bonds:
            bonds = bank_bonds
    
    similarities = []
    
    for bond in bonds:
        stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
        
        # 计算各维度相似度
        scores = {
            'industry': get_industry_score(new_data['industry'], stock.industry_sw_l1 if stock else ''),
            'business': get_business_score(new_data['business'], stock.business if stock else ''),
            'rating': get_rating_score(new_data['rating'], bond.credit_rating),
            'conversion_value': get_value_ratio_score(new_data['conversion_value'], bond.conversion_value),
            'premium_rate': get_value_diff_score(new_data['premium_rate'], bond.premium_rate, max_diff=15),
        }
        
        # 市场景气度差异
        ref_market_idx, ref_market_score = get_market_index(bond.listing_date)
        if target_market_score and ref_market_score:
            market_diff = abs(target_market_score - ref_market_score)
            scores['market'] = 1 - market_diff
        else:
            scores['market'] = 0.5
        
        total_score = sum(scores[key] * SIMILARITY_WEIGHTS.get(key, 0.1) for key in SIMILARITY_WEIGHTS)
        
        similarities.append({
            'bond_code': bond.bond_code,
            'bond_name': bond.bond_name,
            'stock_name': bond.stock_name,
            'industry': stock.industry_sw_l1 if stock else '',
            'rating': bond.credit_rating,
            'conversion_value': bond.conversion_value,
            'premium_rate': bond.premium_rate,
            'first_open': bond.first_open,
            'first_close': bond.first_close,
            'listing_date': bond.listing_date,
            'market_index': ref_market_idx,
            'market_score': ref_market_score,
            'similarity': round(total_score, 3),
            'scores': scores
        })
    
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    session.close()
    return similarities[:top_n]


def predict_price_v3(bond_code):
    """
    基于首日价格 + 市场景气度的预测方法
    """
    similar_bonds = find_similar_bonds(bond_code, top_n=3)
    
    if not similar_bonds:
        return None
    
    session = get_session()
    new_bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    # 获取目标债券上市时的市场景气度
    target_market_idx, target_market_score = get_market_index(new_bond.listing_date)
    
    predictions = []
    
    for ref in similar_bonds:
        ref_price = ref['first_open']
        if not ref_price:
            continue
        
        base_price = ref_price
        adjustments = []
        
        # 获取参考债券上市时的市场景气度
        ref_market_idx, ref_market_score = get_market_index(ref['listing_date'])
        
        # 1. 转股价值差异调整
        new_cv = new_bond.conversion_value or 100
        ref_cv = ref['conversion_value'] or 100
        if ref_cv > 0:
            cv_diff = new_cv - ref_cv
            cv_adjustment = cv_diff * 0.3
            if abs(cv_adjustment) > 0.5:
                adjustments.append(('转股价值差异', cv_diff, round(cv_adjustment, 2)))
            base_price += cv_adjustment
        
        # 2. 溢价率差异调整
        new_premium = new_bond.premium_rate or 15
        ref_premium = ref['premium_rate'] or 15
        premium_diff = (new_premium - ref_premium) / 100 * ref_cv
        adjustments.append(('溢价率差异', new_premium - ref_premium, round(premium_diff, 2)))
        base_price += premium_diff
        
        # 3. 评级差异调整
        new_rating_val = RATING_MAP.get(new_bond.credit_rating, 3)
        ref_rating_val = RATING_MAP.get(ref['rating'], 3)
        rating_diff = new_rating_val - ref_rating_val
        rating_adjustment = rating_diff * 1.0
        if rating_diff != 0:
            adjustments.append((f'评级差异({rating_diff}级)', rating_diff, round(rating_adjustment, 2)))
        base_price += rating_adjustment
        
        # 4. 市场景气度调整（核心新增！）
        ref_market_score = ref.get('market_score', 0.5)
        if target_market_score and ref_market_score:
            market_diff = target_market_score - ref_market_score
            # 景气度每高1%，价格增加0.5%
            market_adjustment = base_price * market_diff * 0.5
            if abs(market_adjustment) > 0.5:
                adj_name = f'市场景气度({"高点" if market_diff > 0 else "低点"})'
                adjustments.append((adj_name, f'{market_diff*100:+.1f}%', round(market_adjustment, 2)))
            base_price += market_adjustment
        
        predictions.append({
            'ref_bond': ref['bond_name'],
            'ref_price': ref_price,
            'predicted_price': round(base_price, 2),
            'adjustments': adjustments,
            'similarity': ref['similarity'],
            'market_info': {
                'target_score': target_market_score,
                'ref_score': ref_market_score
            }
        })
    
    session.close()
    
    if predictions:
        total_weight = sum(p['similarity'] for p in predictions)
        weighted_price = sum(p['predicted_price'] * p['similarity'] for p in predictions) / total_weight if total_weight > 0 else 0
        
        prices = [p['predicted_price'] for p in predictions]
        
        return {
            'price_range': (min(prices), max(prices)),
            'avg_price': round(weighted_price, 2),
            'reference_bonds': predictions,
            'market_info': {
                'target_index': target_market_idx,
                'target_score': target_market_score
            }
        }
    
    return None


def get_prediction_report_v3(bond_code):
    """生成预测报告"""
    session = get_session()
    
    new_bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    new_stock = session.query(StockInfo).filter_by(stock_code=new_bond.stock_code).first()
    
    print(f"\n{'='*70}")
    print(f"可转债估值预测分析 (含市场景气度): {new_bond.bond_name} ({bond_code})")
    print(f"{'='*70}")
    
    print(f"\n【目标债券信息】")
    print(f"  债券名称: {new_bond.bond_name}")
    print(f"  正股名称: {new_bond.stock_name}")
    print(f"  上市日期: {new_bond.listing_date}")
    print(f"  转股价值: {new_bond.conversion_value}元")
    print(f"  溢价率: {new_bond.premium_rate}%")
    print(f"  评级: {new_bond.credit_rating}")
    
    # 显示市场景气度
    target_market_idx, target_market_score = get_market_index(new_bond.listing_date)
    if target_market_idx:
        market_level = "牛市" if target_market_score > 0.6 else "熊市" if target_market_score < 0.4 else "震荡市"
        print(f"  市场景气度: {target_market_idx:.0f} ({market_level}, 得分:{target_market_score:.2f})")
    
    if new_bond.first_open:
        print(f"  实际首日开盘: {new_bond.first_open}元")
    
    result = predict_price_v3(bond_code)
    
    if result:
        print(f"\n【参考债券及计算过程】")
        
        for i, ref in enumerate(result['reference_bonds'], 1):
            market_info = ref.get('market_info', {})
            ref_score = market_info.get('ref_score', 0.5)
            ref_level = "牛市" if ref_score > 0.6 else "熊市" if ref_score < 0.4 else "震荡市"
            
            print(f"\n  参考{i}: {ref['ref_bond']}")
            print(f"    首日开盘: {ref['ref_price']}元 (上市时市场:{ref_level})")
            print(f"    相似度: {ref['similarity']:.1%}")
            print(f"    调整项:")
            for adj_name, adj_val, adj_price in ref['adjustments']:
                sign = '+' if adj_price >= 0 else ''
                print(f"      {adj_name}: {adj_val} → {sign}{adj_price:.2f}元")
            print(f"    预测价: {ref['predicted_price']}元")
        
        print(f"\n{'='*70}")
        print(f"【预测结果】")
        print(f"  加权预测价: {result['avg_price']} 元")
        print(f"  价格区间: {result['price_range'][0]:.2f} ~ {result['price_range'][1]:.2f} 元")
        
        if new_bond.first_open:
            error = abs(result['avg_price'] - new_bond.first_open)
            error_pct = error / new_bond.first_open * 100
            print(f"\n  实际首日: {new_bond.first_open}元")
            print(f"  预测误差: {error:.2f}元 ({error_pct:.1f}%)")
        print(f"{'='*70}")
    
    session.close()


if __name__ == '__main__':
    get_prediction_report_v3('110074')