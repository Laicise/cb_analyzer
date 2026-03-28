"""
相似度匹配模型（量化评估版）
参考表3-7的量化评估方法：基于多维度因素对比推算目标债券价格
"""
import pandas as pd
import numpy as np
from db.models import get_session, BondInfo, StockInfo, BondDaily
from config import SIMILARITY_WEIGHTS, RATING_MAP
import warnings
warnings.filterwarnings('ignore')


def get_industry_score(industry1, industry2):
    """计算行业相似度"""
    if not industry1 or not industry2:
        return 0
    if industry1 == industry2:
        return 1.0
    common_words = set(industry1) & set(industry2)
    if len(common_words) > 0:
        return 0.5
    return 0


def get_business_score(business1, business2):
    """计算主营业务相似度"""
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
    """计算信用评级相似度"""
    if not rating1 or not rating2:
        return 0
    r1 = RATING_MAP.get(rating1, 0)
    r2 = RATING_MAP.get(rating2, 0)
    diff = abs(r1 - r2)
    return max(0, 1 - diff / 5)


def get_value_ratio_score(val1, val2):
    """计算数值比率相似度"""
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
    """计算数值差异得分"""
    if val1 is None or val2 is None:
        return 0
    try:
        diff = abs(float(val1) - float(val2))
        return max(0, 1 - diff / max_diff)
    except:
        return 0


def find_similar_bonds(new_bond_code, top_n=5):
    """查找最相似的已上市可转债"""
    session = get_session()
    
    new_bond = session.query(BondInfo).filter_by(bond_code=new_bond_code).first()
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
        'coupon_rate': new_bond.coupon_rate,
    }
    
    bonds = session.query(BondInfo).filter(
        BondInfo.bond_code != new_bond_code,
        BondInfo.listing_date != None,
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
        
        latest_price = session.query(BondDaily).filter_by(bond_code=bond.bond_code).order_by(BondDaily.trade_date.desc()).first()
        
        similarities.append({
            'bond_code': bond.bond_code,
            'bond_name': bond.bond_name,
            'stock_name': bond.stock_name,
            'industry': stock.industry_sw_l1 if stock else '',
            'rating': bond.credit_rating,
            'conversion_value': bond.conversion_value,
            'premium_rate': bond.premium_rate,
            'coupon_rate': bond.coupon_rate,
            'current_price': latest_price.close_price if latest_price else None,
            'listing_date': bond.listing_date,
            'similarity': round(total_score, 3),
            'scores': scores
        })
    
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    session.close()
    return similarities[:top_n]


def predict_price_quantitative(new_bond_code):
    """
    量化评估预测方法（参考表3-7的逻辑）
    
    考虑因素：
    1. 转股价值差异
    2. 票息差异（6年总票息）
    3. 上市时间差异
    4. 信用评级差异
    5. 行业差异
    """
    similar_bonds = find_similar_bonds(new_bond_code, top_n=3)
    
    if not similar_bonds:
        return None
    
    session = get_session()
    new_bond = session.query(BondInfo).filter_by(bond_code=new_bond_code).first()
    
    # 目标债券的6年总票息（估算）
    # 假设票息结构：0.5%+0.5%+1.0%+1.5%+2.0%+2.5% = 8%
    new_total_coupon = 8.0  # 默认6年票息总和
    if new_bond.coupon_rate:
        new_total_coupon = new_bond.coupon_rate * 6
    
    predictions = []
    
    for ref in similar_bonds:
        ref_price = ref['current_price']
        if not ref_price:
            continue
        
        base_price = ref_price
        adjustments = []
        
        # 1. 转股价值差异调整（最重要）
        new_cv = new_bond.conversion_value or 0
        ref_cv = ref['conversion_value'] or 0
        if ref_cv > 0:
            cv_diff = new_cv - ref_cv
            cv_adjustment = cv_diff * 1.0  # 转股价值每差1元，价格调整1元
            if abs(cv_adjustment) > 0.1:
                adjustments.append(('转股价值差异', cv_diff, cv_adjustment))
            base_price += cv_adjustment
        
        # 2. 票息差异调整
        ref_total_coupon = ref['coupon_rate'] * 6 if ref['coupon_rate'] else 8.0
        coupon_diff = new_total_coupon - ref_total_coupon
        coupon_adjustment = coupon_diff * 1.0  # 票息每差1元，价格调整1元
        if abs(coupon_adjustment) > 0.1:
            adjustments.append(('6年票息差异', coupon_diff, coupon_adjustment))
        base_price += coupon_adjustment
        
        # 3. 上市时间差异调整
        # 上市时间短于参考债券，价格应略低
        if new_bond.listing_date and ref['listing_date']:
            days_diff = (new_bond.listing_date - ref['listing_date']).days
            # 每个月约影响0.2元
            time_adjustment = days_diff / 30 * 0.2
            if abs(time_adjustment) > 0.1:
                adjustments.append(('上市时间差异(天)', days_diff, time_adjustment))
            base_price += time_adjustment
        
        # 4. 评级差异调整
        new_rating_val = RATING_MAP.get(new_bond.credit_rating, 0)
        ref_rating_val = RATING_MAP.get(ref['rating'], 0)
        rating_diff = new_rating_val - ref_rating_val
        rating_adjustment = rating_diff * 0.5
        if rating_diff != 0:
            adjustments.append((f'评级差异({rating_diff}级)', rating_diff, rating_adjustment))
        base_price += rating_adjustment
        
        # 5. 行业差异调整
        industry_same = 1 if new_bond.credit_rating == ref['rating'] and ref['scores']['industry'] == 1.0 else 0
        # 行业相同不加减价，不同则略低
        
        predictions.append({
            'ref_bond': ref['bond_name'],
            'ref_price': ref['current_price'],
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
            'reference_bonds': predictions,
            'learning_factors': learn_from_history()  # 添加学习因子
        }
    
    return None


def learn_from_history():
    """从历史预测误差中学习，调整预测参数"""
    session = get_session()
    
    # 获取所有有预测和实际价格的债券
    bonds = session.query(BondInfo).filter(
        BondInfo.predicted_price != None,
        BondInfo.first_open != None
    ).all()
    
    if len(bonds) < 3:
        session.close()
        return None
    
    # 计算各调整因子的误差
    cv_errors = []
    rating_errors = []
    time_errors = []
    
    for bond in bonds:
        if bond.predicted_price and bond.first_open:
            error = bond.predicted_price - bond.first_open
            
            # 转股价值影响因子（当前为1.0）
            if bond.conversion_value:
                cv_errors.append(error / bond.conversion_value)
            
            # 评级影响因子（当前为0.5）
            if bond.credit_rating:
                rating_errors.append(error / RATING_MAP.get(bond.credit_rating, 1))
    
    # 计算调整系数
    factors = {}
    
    if cv_errors and len(cv_errors) > 0:
        avg_cv_error = sum(cv_errors) / len(cv_errors)
        # 如果预测普遍偏高/偏低，调整系数
        factors['cv_factor'] = 1.0 + avg_cv_error
        factors['cv_sample_count'] = len(cv_errors)
    
    if rating_errors and len(rating_errors) > 0:
        avg_rating_error = sum(rating_errors) / len(rating_errors)
        factors['rating_factor'] = 0.5 + avg_rating_error
        factors['rating_sample_count'] = len(rating_errors)
    
    session.close()
    
    return factors if factors else None


def apply_learning_factors(predicted_price, learning_factors, conversion_value=None, credit_rating=None):
    """应用学习到的因子来调整预测价格"""
    if not learning_factors:
        return predicted_price
    
    adjusted_price = predicted_price
    
    # 应用转股价值因子调整
    cv_factor = learning_factors.get('cv_factor', 1.0)
    if cv_factor and conversion_value:
        # 只做轻微调整
        adjustment = (cv_factor - 1.0) * conversion_value * 0.1
        adjusted_price += adjustment
    
    # 应用评级因子调整
    rating_factor = learning_factors.get('rating_factor', 1.0)
    if rating_factor and credit_rating:
        rating_val = RATING_MAP.get(credit_rating, 0)
        adjustment = (rating_factor - 1.0) * rating_val * 0.1
        adjusted_price += adjustment
    
    return round(adjusted_price, 2)


def get_confidence_level(similarity_score):
    """
    根据相似度得分给出可信度评级
    参考标准：
    - 相似度 >= 70%: 高可信度 (A)
    - 相似度 >= 50%: 中可信度 (B)
    - 相似度 >= 30%: 低可信度 (C)
    - 相似度 < 30%: 可信度不足 (D)
    """
    if similarity_score >= 0.70:
        return 'A', '高可信度', '市场存在高度相似的可转债，评估价格参考性强'
    elif similarity_score >= 0.50:
        return 'B', '中等可信度', '市场存在较相似的可转债，评估价格有一定参考性'
    elif similarity_score >= 0.30:
        return 'C', '较低可信度', '市场相似可转债较少，评估价格参考性有限'
    else:
        return 'D', '可信度不足', '市场缺乏相似可转债，建议观望'


def get_prediction_report(new_bond_code):
    """生成完整的预测报告（含可信度评级）"""
    session = get_session()
    
    new_bond = session.query(BondInfo).filter_by(bond_code=new_bond_code).first()
    new_stock = session.query(StockInfo).filter_by(stock_code=new_bond.stock_code).first()
    
    print("\n" + "="*65)
    print(f"可转债量化估值分析: {new_bond.bond_name} ({new_bond_code})")
    print("="*65)
    
    print(f"\n【目标债券关键指标】")
    print(f"  正股名称: {new_bond.stock_name}")
    print(f"  所处行业: {new_stock.industry_sw_l1 if new_stock else '未知'}")
    print(f"  信用评级: {new_bond.credit_rating or '未知'}")
    print(f"  转股价值: {new_bond.conversion_value}元")
    print(f"  转股溢价率: {new_bond.premium_rate}%")
    print(f"  发行规模: {new_bond.issue_size}亿元")
    
    # 获取预测结果
    result = predict_price_quantitative(new_bond_code)
    
    if result:
        # 计算可信度
        avg_similarity = sum(p['similarity'] for p in result['reference_bonds']) / len(result['reference_bonds'])
        conf_level, conf_title, conf_desc = get_confidence_level(avg_similarity)
        
        print(f"\n【参考债券对比分析】(量化评估方法)")
        
        for i, ref in enumerate(result['reference_bonds'], 1):
            print(f"\n  ┌─ 参考{i}: {ref['ref_bond']}")
            print(f"  │  当前价格: {ref['ref_price']}元")
            print(f"  │  相似度: {ref['similarity']:.1%}")
            print(f"  │  量化调整:")
            for adj_name, adj_val, adj_price in ref['adjustments']:
                sign = '+' if adj_price > 0 else ''
                print(f"  │    - {adj_name}: {adj_val:+.2f} → {sign}{adj_price:.2f}元")
            print(f"  │  ─────────────────────────────")
            print(f"  └─ 预测价格: {ref['predicted_price']}元")
        
        print(f"\n{'='*65}")
        print(f"【量化评估预测结果】")
        print(f"  价格区间: {result['price_range'][0]:.2f} ~ {result['price_range'][1]:.2f} 元")
        print(f"  加权均价: {result['avg_price']} 元")
        print(f"{'='*65}")
        
        # 显示可信度评级
        print(f"\n【可信度评级】")
        print(f"  ┌─────────────────────────────────────────┐")
        print(f"  │  可信度等级: [{conf_level}] {conf_title:<12} │")
        print(f"  │  平均相似度: {avg_similarity:.1%} {' '*30}│")
        print(f"  │  评估说明: {conf_desc:<35}│")
        print(f"  └─────────────────────────────────────────┘")
        
        # 与参考债券的对比
        if result['reference_bonds']:
            main_ref = result['reference_bonds'][0]
            diff = result['avg_price'] - main_ref['ref_price']
            print(f"\n【分析结论】")
            if diff > 0:
                print(f"  预测价格比参考债券({main_ref['ref_bond']})高 {abs(diff):.2f} 元")
            elif diff < 0:
                print(f"  预测价格比参考债券({main_ref['ref_bond']})低 {abs(diff):.2f} 元")
            else:
                print(f"  预测价格与参考债券({main_ref['ref_bond']})基本持平")
    else:
        print("\n无法生成预测报告")
    
    session.close()


if __name__ == '__main__':
    get_prediction_report('110074')