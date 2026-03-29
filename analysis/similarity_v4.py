"""
相似度匹配模型（最终版 v4）
- 行业和主营业务权重提升
- 行业+主营业务均不匹配则相似度=0
- 设置相似度阈值，取Top 2
- 无参考时使用直接估值法（纯债价值+期权价值）
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

# 相似度阈值（高于此值才可作为参考）
SIMILARITY_THRESHOLD = 0.15

# 近两年发行的转债（用于计算权重调整）- 但不限制匹配范围
RECENT_BONDS_START = '2024-01-01'
# 保留历史数据用于匹配，但参考权重来自近两年数据


def get_market_index(date):
    """获取上证指数收盘价作为市场景气度指标"""
    if date is None:
        return None, 0.5
    
    if isinstance(date, str):
        date_str = date[:7]
    else:
        date_str = date.strftime('%Y-%m')
    
    if date_str in _market_cache:
        return _market_cache[date_str]
    
    try:
        df = ak.index_zh_a_hist(
            symbol='000001', 
            period='monthly', 
            start_date='20180101', 
            end_date='20251231'
        )
        
        df['月份'] = df['日期'].astype(str).str[:7]
        
        if date_str in df['月份'].values:
            idx_value = df[df['月份'] == date_str]['收盘'].values[0]
            hist_mean = df['收盘'].mean()
            hist_std = df['收盘'].std()
            
            if hist_std > 0:
                z_score = (idx_value - hist_mean) / hist_std
                market_score = 1 / (1 + np.exp(-z_score))
            else:
                market_score = 0.5
            
            _market_cache[date_str] = (idx_value, market_score)
            return idx_value, market_score
    except:
        pass
    
    _market_cache[date_str] = (None, 0.5)
    return None, 0.5


def get_industry_score(industry1, industry2):
    """行业匹配得分 - 精确匹配优先"""
    if not industry1 or not industry2:
        return 0
    
    # 完全相同
    if industry1 == industry2:
        return 1.0
    
    # 清理常见词汇
    ignore_words = {'和', '其他', '制造业', '业', '的', '及', '以', '于'}
    
    words1 = set(industry1) - ignore_words
    words2 = set(industry2) - ignore_words
    
    # 如果有公共非忽略字符，得0.5分
    common = words1 & words2
    if common:
        # 但要检查是否是主要行业词（至少2个字符）
        main_common = [w for w in common if len(w) >= 2]
        if main_common:
            return 0.5
    
    return 0


def get_business_score(business1, business2):
    """主营业务匹配得分"""
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
    """评级匹配得分"""
    if not rating1 or not rating2:
        return 0
    r1 = RATING_MAP.get(rating1, 0)
    r2 = RATING_MAP.get(rating2, 0)
    diff = abs(r1 - r2)
    return max(0, 1 - diff / 5)


def get_value_ratio_score(val1, val2):
    """转股价值比例得分"""
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
    """转股价值差异得分"""
    if val1 is None or val2 is None:
        return 0
    try:
        diff = abs(float(val1) - float(val2))
        return max(0, 1 - diff / max_diff)
    except:
        return 0


def find_similar_bonds_v4(bond_code, top_n=2):
    """
    查找最相似的可转债（v4版）
    - 行业权重提升到20%
    - 主营业务权重提升到15%
    - 行业+主营业务均不匹配则相似度=0
    - 取Top 2
    """
    session = get_session()
    
    new_bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    if not new_bond:
        session.close()
        return None
    
    new_stock = session.query(StockInfo).filter_by(stock_code=new_bond.stock_code).first()
    
    # 获取目标债券的行业和主营业务
    target_industry = new_stock.industry_sw_l1 if new_stock else ''
    target_business = new_stock.business if new_stock else ''
    
    # 判断是否为银行转债
    is_bank_bond = new_bond.conversion_value is None
    
    # 获取目标债券上市时的市场景气度
    target_market_idx, target_market_score = get_market_index(new_bond.listing_date)
    
    # 查询参考债券（不限制时间范围，但优先使用近两年的数据计算权重）
    bonds = session.query(BondInfo).filter(
        BondInfo.bond_code != bond_code,
        BondInfo.first_open != None
    ).all()
    
    # 如果是银行转债，优先匹配银行转债
    if is_bank_bond:
        bank_bonds = [b for b in bonds if b.conversion_value is None]
        if bank_bonds:
            bonds = bank_bonds
    
    similarities = []
    
    for bond in bonds:
        stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
        
        ref_industry = stock.industry_sw_l1 if stock else ''
        ref_business = stock.business if stock else ''
        
        # 行业得分
        industry_score = get_industry_score(target_industry, ref_industry)
        # 主营业务得分
        business_score = get_business_score(target_business, ref_business)
        
        # 核心规则：如果行业和主营业务均不匹配，相似度为0
        if industry_score == 0 and business_score == 0:
            continue
        
        # 调整后的权重（行业20%，主营业务15%）
        weights = {
            'industry': 0.20,
            'business': 0.15,
            'rating': 0.10,
            'conversion_value': 0.20,
            'premium_rate': 0.10,
            'market': 0.10,
            'issue_size': 0.05,
            'years_to_expiry': 0.05,
            'coupon_rate': 0.05,
        }
        
        # 计算各维度得分
        scores = {
            'industry': industry_score,
            'business': business_score,
            'rating': get_rating_score(new_bond.credit_rating, bond.credit_rating),
            'conversion_value': get_value_ratio_score(new_bond.conversion_value, bond.conversion_value),
            'premium_rate': get_value_diff_score(new_bond.premium_rate, bond.premium_rate, max_diff=15),
            'issue_size': get_value_diff_score(new_bond.issue_size, bond.issue_size, max_diff=50),
            'coupon_rate': get_value_diff_score(new_bond.coupon_rate, bond.coupon_rate, max_diff=2),
        }
        
        # 市场景气度差异
        ref_market_idx, ref_market_score = get_market_index(bond.listing_date)
        if target_market_score and ref_market_score:
            market_diff = abs(target_market_score - ref_market_score)
            scores['market'] = 1 - market_diff
        else:
            scores['market'] = 0.5
        
        # 剩余年限
        if new_bond.expiry_date and bond.expiry_date:
            try:
                new_years = (new_bond.expire_date - new_bond.listing_date).days / 365
                ref_years = (bond.expire_date - bond.listing_date).days / 365
                scores['years_to_expiry'] = max(0, 1 - abs(new_years - ref_years) / 3)
            except:
                scores['years_to_expiry'] = 0.5
        else:
            scores['years_to_expiry'] = 0.5
        
        # 计算总相似度
        total_score = sum(scores[key] * weights.get(key, 0.1) for key in weights)
        
        # 只有高于阈值的才加入
        if total_score >= SIMILARITY_THRESHOLD:
            similarities.append({
                'bond_code': bond.bond_code,
                'bond_name': bond.bond_name,
                'stock_name': bond.stock_name,
                'industry': ref_industry,
                'business': ref_business,
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


def calculate_pure_bond_value(bond):
    """
    计算纯债价值
    纯债价值 = Σ(票息 / (1+r)^t) + 本金 / (1+r)^T
    r: 参考相似期限的国债收益率（简化使用3%）
    """
    # 简化计算：假设票息率近似
    coupon = bond.coupon_rate or 0.5  # 默认0.5%
    years = 6  # 假设剩余6年
    r = 0.03  # 参考利率3%
    
    # 简化公式：纯债价值 ≈ 票息总和 + 面值折现
    coupon_value = coupon * years
    face_discount = 100 / ((1 + r) ** years)
    
    return round(coupon_value + face_discount, 2)


def calculate_option_value(bond, market_score):
    """
    计算期权价值（简化版）
    基于转股价值和市场情绪估算
    """
    if bond.conversion_value is None:
        # 银行转债：期权价值较低
        return 10.0
    
    cv = bond.conversion_value
    
    # 简单估算：转股价值越高，期权价值越高
    # 市场情绪好时，期权价值更高
    base_option = (cv - 100) / 10 if cv > 100 else 5
    
    # 市场调整
    market_factor = 0.8 + market_score * 0.4  # 0.8 ~ 1.2
    
    option_value = base_option * market_factor
    
    return max(5, min(30, round(option_value, 2)))  # 限制在5-30元


def direct_valuation(bond_code):
    """
    直接估值法：纯债价值 + 期权价值
    当没有可参考的相似债券时使用
    """
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    if not bond:
        session.close()
        return None
    
    # 获取市场景气度
    market_idx, market_score = get_market_index(bond.listing_date)
    
    # 计算纯债价值
    pure_bond_value = calculate_pure_bond_value(bond)
    
    # 计算期权价值
    option_value = calculate_option_value(bond, market_score)
    
    # 总价值
    total_value = pure_bond_value + option_value
    
    result = {
        'pure_bond_value': pure_bond_value,
        'option_value': option_value,
        'total_value': round(total_value, 2),
        'market_index': market_idx,
        'market_score': round(market_score, 2),
        'method': '直接估值法(纯债+期权)'
    }
    
    session.close()
    return result


def predict_price_v4(bond_code):
    """
    预测价格（v4版）
    - 如果有相似债券（阈值>=0.25），使用相似债券法
    - 否则使用直接估值法
    """
    # 先尝试相似债券匹配
    similar_bonds = find_similar_bonds_v4(bond_code, top_n=2)
    
    if not similar_bonds or len(similar_bonds) == 0:
        # 没有可参考的相似债券，使用直接估值法
        return direct_valuation(bond_code)
    
    session = get_session()
    new_bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    target_market_idx, target_market_score = get_market_index(new_bond.listing_date)
    
    predictions = []
    
    for ref in similar_bonds:
        ref_price = ref['first_open']
        if not ref_price:
            continue
        
        base_price = ref_price
        adjustments = []
        
        # 转股价值差异调整
        new_cv = new_bond.conversion_value or 100
        ref_cv = ref['conversion_value'] or 100
        if ref_cv > 0:
            cv_diff = new_cv - ref_cv
            cv_adjustment = cv_diff * 0.3
            if abs(cv_adjustment) > 0.5:
                adjustments.append(('转股价值差异', cv_diff, round(cv_adjustment, 2)))
            base_price += cv_adjustment
        
        # 溢价率差异调整
        new_premium = new_bond.premium_rate or 15
        ref_premium = ref['premium_rate'] or 15
        premium_diff = (new_premium - ref_premium) / 100 * ref_cv
        adjustments.append(('溢价率差异', new_premium - ref_premium, round(premium_diff, 2)))
        base_price += premium_diff
        
        # 评级差异调整
        new_rating_val = RATING_MAP.get(new_bond.credit_rating, 3)
        ref_rating_val = RATING_MAP.get(ref['rating'], 3)
        rating_diff = new_rating_val - ref_rating_val
        rating_adjustment = rating_diff * 1.0
        if rating_diff != 0:
            adjustments.append((f'评级差异({rating_diff}级)', rating_diff, round(rating_adjustment, 2)))
        base_price += rating_adjustment
        
        # 市场景气度调整
        ref_market_score = ref.get('market_score', 0.5)
        if target_market_score and ref_market_score:
            market_diff = target_market_score - ref_market_score
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
            'method': '相似债券法'
        }
    
    return None


if __name__ == '__main__':
    # 测试
    import sys
    sys.path.insert(0, '.')
    
    # 测试浦发转债
    print("="*60)
    print("测试: 110059 浦发转债")
    print("="*60)
    
    result = predict_price_v4('110059')
    if result:
        print(f"\n方法: {result.get('method', '未知')}")
        
        if result.get('method') == '相似债券法':
            print(f"预测价格: {result['avg_price']}元")
            print(f"价格区间: {result['price_range']}")
            
            # 获取实际价格对比
            from db.models import get_session, BondInfo
            session = get_session()
            bond = session.query(BondInfo).filter_by(bond_code='110059').first()
            if bond and bond.first_open:
                error = abs(result['avg_price'] - bond.first_open)
                error_pct = error / bond.first_open * 100
                print(f"实际开盘: {bond.first_open}元")
                print(f"误差: {error:.2f}元 ({error_pct:.1f}%)")
            session.close()
        else:
            print(f"纯债价值: {result['pure_bond_value']}元")
            print(f"期权价值: {result['option_value']}元")
            print(f"估值结果: {result['total_value']}元")