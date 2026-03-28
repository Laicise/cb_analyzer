"""
相似度匹配模型（修正版）
基于首日价格预测，而非当前价格
"""
import pandas as pd
import numpy as np
from db.models import get_session, BondInfo, StockInfo, BondDaily
from config import SIMILARITY_WEIGHTS, RATING_MAP
import warnings
warnings.filterwarnings('ignore')


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
    }
    
    # 查找有首日价格的债券
    bonds = session.query(BondInfo).filter(
        BondInfo.bond_code != bond_code,
        BondInfo.first_open != None,
        BondInfo.conversion_value != None
    ).all()
    
    similarities = []
    
    for bond in bonds:
        stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
        
        scores = {
            'industry': get_industry_score(new_data['industry'], stock.industry_sw_l1 if stock else ''),
            'business': get_business_score(new_data['business'], stock.business if stock else ''),
            'rating': get_rating_score(new_data['rating'], bond.credit_rating),
            'conversion_value': get_value_ratio_score(new_data['conversion_value'], bond.conversion_value),
            'premium_rate': get_value_diff_score(new_data['premium_rate'], bond.premium_rate, max_diff=15),
        }
        
        total_score = sum(scores[key] * SIMILARITY_WEIGHTS.get(key, 0.1) for key in SIMILARITY_WEIGHTS)
        
        similarities.append({
            'bond_code': bond.bond_code,
            'bond_name': bond.bond_name,
            'stock_name': bond.stock_name,
            'industry': stock.industry_sw_l1 if stock else '',
            'rating': bond.credit_rating,
            'conversion_value': bond.conversion_value,
            'premium_rate': bond.premium_rate,
            # 使用首日开盘价作为参考！
            'first_open': bond.first_open,
            'first_close': bond.first_close,
            'listing_date': bond.listing_date,
            'similarity': round(total_score, 3),
            'scores': scores
        })
    
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    session.close()
    return similarities[:top_n]


def predict_price_v2(bond_code):
    """
    基于首日价格的预测方法（修正版）
    """
    similar_bonds = find_similar_bonds(bond_code, top_n=3)
    
    if not similar_bonds:
        return None
    
    session = get_session()
    new_bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    predictions = []
    
    for ref in similar_bonds:
        # 使用首日开盘价作为参考基准！
        ref_price = ref['first_open']
        if not ref_price:
            continue
        
        base_price = ref_price
        adjustments = []
        
        # 1. 转股价值差异调整
        new_cv = new_bond.conversion_value or 100
        ref_cv = ref['conversion_value'] or 100
        if ref_cv > 0:
            cv_diff = new_cv - ref_cv
            cv_adjustment = cv_diff * 0.3  # 调整系数0.3（更保守）
            if abs(cv_adjustment) > 0.5:
                adjustments.append(('转股价值差异', cv_diff, round(cv_adjustment, 2)))
            base_price += cv_adjustment
        
        # 2. 溢价率差异调整
        new_premium = new_bond.premium_rate or 15
        ref_premium = ref['premium_rate'] or 15
        # 溢价率差异影响
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
        
        predictions.append({
            'ref_bond': ref['bond_name'],
            'ref_price': ref_price,
            'predicted_price': round(base_price, 2),
            'adjustments': adjustments,
            'similarity': ref['similarity']
        })
    
    session.close()
    
    if predictions:
        total_weight = sum(p['similarity'] for p in predictions)
        weighted_price = sum(p['predicted_price'] * p['similarity'] for p in predictions) / total_weight if total_weight > 0 else 0
        
        prices = [p['predicted_price'] for p in predictions]
        
        return {
            'price_range': (min(prices), max(prices)),
            'avg_price': round(weighted_price, 2),
            'reference_bonds': predictions
        }
    
    return None


def get_prediction_report_v2(bond_code):
    """生成预测报告"""
    session = get_session()
    
    new_bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    new_stock = session.query(StockInfo).filter_by(stock_code=new_bond.stock_code).first()
    
    print(f"\n{'='*60}")
    print(f"可转债预测分析: {new_bond.bond_name} ({bond_code})")
    print(f"{'='*60}")
    
    print(f"\n【目标债券指标】")
    print(f"  正股名称: {new_bond.stock_name}")
    print(f"  所处行业: {new_stock.industry_sw_l1 if new_stock else '未知'}")
    print(f"  信用评级: {new_bond.credit_rating or '未知'}")
    print(f"  转股价值: {new_bond.conversion_value}元")
    print(f"  转股溢价率: {new_bond.premium_rate}%")
    print(f"  发行规模: {new_bond.issue_size}亿元")
    
    if new_bond.first_open:
        print(f"  实际首日开盘: {new_bond.first_open}元")
    
    result = predict_price_v2(bond_code)
    
    if result:
        print(f"\n【参考债券对比分析】(基于首日价格)")
        
        for i, ref in enumerate(result['reference_bonds'], 1):
            print(f"\n  参考{i}: {ref['ref_bond']}")
            print(f"    首日开盘价: {ref['ref_price']}元")
            print(f"    相似度: {ref['similarity']:.1%}")
            print(f"    量化调整:")
            for adj_name, adj_val, adj_price in ref['adjustments']:
                sign = '+' if adj_price > 0 else ''
                print(f"      - {adj_name}: {adj_val:+.2f} → {sign}{adj_price:.2f}元")
            print(f"    ─────────────────────────────")
            print(f"    预测价格: {ref['predicted_price']}元")
        
        print(f"\n{'='*60}")
        print(f"【预测结果】")
        print(f"  价格区间: {result['price_range'][0]:.2f} ~ {result['price_range'][1]:.2f} 元")
        print(f"  加权均价: {result['avg_price']} 元")
        
        if new_bond.first_open:
            error = abs(result['avg_price'] - new_bond.first_open)
            error_pct = error / new_bond.first_open * 100
            print(f"\n  实际首日开盘: {new_bond.first_open} 元")
            print(f"  预测误差: {error:.2f} 元 ({error_pct:.1f}%)")
        print(f"{'='*60}")
    
    session.close()


# 测试
if __name__ == '__main__':
    get_prediction_report_v2('110074')