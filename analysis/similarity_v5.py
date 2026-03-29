"""
可转债估值模型 - 优化版 v5
- 修复行业匹配逻辑（不允许跨大门类）
- 降低调整系数
- 限制参考债券的上市时间范围
- 添加更多验证
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

# 行业门类映射（用于判断是否跨门类）
INDUSTRY_CATEGORIES = {
    '农林牧渔': ['农业', '林业', '牧业', '渔业', '农副食品', '畜禽', '饲料', '种植', '养殖'],
    '采掘': ['煤炭', '石油', '天然气', '有色金属', '黑色金属', '采矿'],
    '化工': ['化工', '石油化工', '化学', '新材料', '精细化工', '农药', '染料'],
    '工业': ['工业', '制造', '设备', '机械', '汽车', '船舶', '航空航天', '电气', '仪器', '专用设备'],
    '消费': ['食品', '饮料', '纺织', '服装', '家电', '家居', '造纸', '印刷', '日用'],
    '医药': ['医药', '医疗', '生物', '中药', '化学制药', '医疗器械', '疫苗'],
    '电子': ['电子', '半导体', '通信', '计算机', '软件', '互联网', 'IT', '光电', 'LED', 'PCB'],
    '公用事业': ['电力', '燃气', '水务', '环保', '供热', '供电'],
    '建筑': ['建筑', '地产', '房地产', '建材', '园林', '装饰', '工程'],
    '金融': ['银行', '证券', '保险', '信托', '租赁', '担保'],
    '服务': ['商贸', '零售', '批发', '物流', '运输', '旅游', '餐饮', '传媒', '娱乐'],
    '其他': [],
}

# 门类关键词映射
CATEGORY_KEYWORDS = {}
for cat, keywords in INDUSTRY_CATEGORIES.items():
    for kw in keywords:
        CATEGORY_KEYWORDS[kw] = cat

def get_industry_category(industry):
    """获取行业所属门类"""
    if not industry:
        return '其他'
    for kw, cat in CATEGORY_KEYWORDS.items():
        if kw in industry:
            return cat
    return '其他'

def get_industry_score_v2(industry1, industry2):
    """行业匹配得分 - 严格版（不允许跨门类）"""
    if not industry1 or not industry2:
        return 0
    
    # 完全相同
    if industry1 == industry2:
        return 1.0
    
    # 获取门类
    cat1 = get_industry_category(industry1)
    cat2 = get_industry_category(industry2)
    
    # 不同门类，不匹配
    if cat1 != cat2:
        return 0
    
    # 同门类但不同细分，得0.5分
    return 0.5


# 相似度阈值
SIMILARITY_THRESHOLD = 0.15

_market_cache = {}

def get_market_index(date):
    """获取市场景气度"""
    if date is None:
        return None, 0.5
    
    if isinstance(date, str):
        date_str = date[:7]
    else:
        date_str = date.strftime('%Y-%m')
    
    if date_str in _market_cache:
        return _market_cache[date_str]
    
    try:
        df = ak.index_zh_a_hist(symbol='000001', period='monthly', start_date='20180101', end_date='20251231')
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


def get_rating_score(rating1, rating2):
    if not rating1 or not rating2:
        return 0
    from config import RATING_MAP
    r1 = RATING_MAP.get(rating1, 0)
    r2 = RATING_MAP.get(rating2, 0)
    diff = abs(r1 - r2)
    return max(0, 1 - diff / 5)


def find_similar_bonds_v5(bond_code, top_n=2):
    """查找相似债券（v5严格版）"""
    session = get_session()
    
    new_bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    if not new_bond:
        session.close()
        return None
    
    new_stock = session.query(StockInfo).filter_by(stock_code=new_bond.stock_code).first()
    
    target_industry = new_stock.industry_sw_l1 if new_stock else ''
    target_category = get_industry_category(target_industry)
    
    is_bank_bond = new_bond.conversion_value is None
    
    target_market_idx, target_market_score = get_market_index(new_bond.listing_date)
    
    # 查询参考债券（限制在近两年内发行的）
    two_years_ago = datetime.now() - timedelta(days=730)
    bonds = session.query(BondInfo).filter(
        BondInfo.bond_code != bond_code,
        BondInfo.first_open != None,
        BondInfo.listing_date >= two_years_ago  # 仅用近两年数据
    ).all()
    
    if is_bank_bond:
        bank_bonds = [b for b in bonds if b.conversion_value is None]
        if bank_bonds:
            bonds = bank_bonds
    
    similarities = []
    
    for bond in bonds:
        stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
        
        ref_industry = stock.industry_sw_l1 if stock else ''
        ref_category = get_industry_category(ref_industry)
        
        # 行业得分（严格版）
        industry_score = get_industry_score_v2(target_industry, ref_industry)
        
        # 如果行业不匹配，直接跳过
        if industry_score == 0:
            continue
        
        # 权重
        weights = {
            'industry': 0.25,    # 提高行业权重
            'rating': 0.15,
            'conversion_value': 0.20,
            'premium_rate': 0.15,
            'market': 0.10,
            'issue_size': 0.10,
            'years_to_expiry': 0.05,
        }
        
        scores = {
            'industry': industry_score,
            'rating': get_rating_score(new_bond.credit_rating, bond.credit_rating),
            'conversion_value': get_value_ratio_score(new_bond.conversion_value, bond.conversion_value),
            'premium_rate': get_value_diff_score(new_bond.premium_rate, bond.premium_rate, max_diff=20),
            'issue_size': get_value_diff_score(new_bond.issue_size, bond.issue_size, max_diff=30),
        }
        
        # 市场景气度
        ref_market_idx, ref_market_score = get_market_index(bond.listing_date)
        if target_market_score and ref_market_score:
            market_diff = abs(target_market_score - ref_market_score)
            scores['market'] = 1 - market_diff
        else:
            scores['market'] = 0.5
        
        # 剩余年限
        if new_bond.expiry_date and bond.expiry_date:
            try:
                new_years = (new_bond.expiry_date - new_bond.listing_date).days / 365
                ref_years = (bond.expiry_date - bond.listing_date).days / 365
                scores['years_to_expiry'] = max(0, 1 - abs(new_years - ref_years) / 3)
            except:
                scores['years_to_expiry'] = 0.5
        else:
            scores['years_to_expiry'] = 0.5
        
        total_score = sum(scores[key] * weights.get(key, 0.1) for key in weights)
        
        if total_score >= SIMILARITY_THRESHOLD:
            similarities.append({
                'bond_code': bond.bond_code,
                'bond_name': bond.bond_name,
                'industry': ref_industry,
                'category': ref_category,
                'first_open': bond.first_open,
                'listing_date': bond.listing_date,
                'market_score': ref_market_score,
                'similarity': round(total_score, 3),
                'scores': scores
            })
    
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    session.close()
    return similarities[:top_n]


def calculate_pure_bond_value(bond):
    """计算纯债价值"""
    coupon = bond.coupon_rate or 0.5
    years = 6
    r = 0.03
    coupon_value = coupon * years
    face_discount = 100 / ((1 + r) ** years)
    return round(coupon_value + face_discount, 2)


def calculate_option_value(bond, market_score):
    """计算期权价值"""
    if bond.conversion_value is None:
        return 10.0
    
    cv = bond.conversion_value
    base_option = (cv - 100) / 10 if cv > 100 else 5
    market_factor = 0.8 + market_score * 0.4
    option_value = base_option * market_factor
    return max(5, min(30, round(option_value, 2)))


def direct_valuation_v2(bond_code):
    """直接估值法"""
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    if not bond:
        session.close()
        return None
    
    market_idx, market_score = get_market_index(bond.listing_date)
    pure_bond_value = calculate_pure_bond_value(bond)
    option_value = calculate_option_value(bond, market_score)
    total_value = pure_bond_value + option_value
    
    result = {
        'pure_bond_value': pure_bond_value,
        'option_value': option_value,
        'total_value': round(total_value, 2),
        'market_score': round(market_score, 2),
        'method': '直接估值法(纯债+期权)'
    }
    
    session.close()
    return result


def predict_price_v5(bond_code):
    """预测价格v5"""
    similar_bonds = find_similar_bonds_v5(bond_code, top_n=2)
    
    if not similar_bonds or len(similar_bonds) == 0:
        return direct_valuation_v2(bond_code)
    
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
        
        # 转股价值调整（系数降低为0.2）
        new_cv = new_bond.conversion_value or 100
        # 从BondInfo获取参考债券的转股价值
        ref_bond = session.query(BondInfo).filter_by(bond_code=ref['bond_code']).first()
        ref_cv = ref_bond.conversion_value if ref_bond and ref_bond.conversion_value else 100
        if ref_cv > 0:
            cv_diff = new_cv - ref_cv
            cv_adjustment = cv_diff * 0.2  # 降低系数
            if abs(cv_adjustment) > 0.5:
                adjustments.append(('转股价值差异', cv_diff, round(cv_adjustment, 2)))
            base_price += cv_adjustment
        
        # 溢价率调整（系数降低为0.15）
        new_premium = new_bond.premium_rate or 15
        ref_premium = ref_bond.premium_rate if ref_bond and ref_bond.premium_rate else 15
        premium_diff = (new_premium - ref_premium) / 100 * ref_cv * 0.15  # 降低系数
        adjustments.append(('溢价率差异', new_premium - ref_premium, round(premium_diff, 2)))
        base_price += premium_diff
        
        # 评级调整
        from config import RATING_MAP
        new_rating_val = RATING_MAP.get(new_bond.credit_rating, 3)
        ref_rating_val = RATING_MAP.get(ref_bond.credit_rating if ref_bond else '', 3)
        rating_diff = new_rating_val - ref_rating_val
        rating_adjustment = rating_diff * 0.5  # 降低系数
        if rating_diff != 0:
            adjustments.append((f'评级差异({rating_diff}级)', rating_diff, round(rating_adjustment, 2)))
        base_price += rating_adjustment
        
        # 市场景气度调整（系数降低为0.2）
        ref_market_score = ref.get('market_score', 0.5)
        if target_market_score and ref_market_score:
            market_diff = target_market_score - ref_market_score
            market_adjustment = base_price * market_diff * 0.2  # 降低系数
            if abs(market_adjustment) > 0.5:
                adj_name = f'市场景气度({"高点" if market_diff > 0 else "低点"})'
                adjustments.append((adj_name, f'{market_diff*100:+.1f}%', round(market_adjustment, 2)))
            base_price += market_adjustment
        
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
            'reference_bonds': predictions,
            'method': '相似债券法'
        }
    
    return None


# 测试
if __name__ == '__main__':
    from datetime import timedelta
    
    # 测试2024年转债
    start_2024 = datetime(2024, 1, 1)
    end_2024 = datetime(2024, 12, 31)
    
    session = get_session()
    bonds = session.query(BondInfo).filter(
        BondInfo.listing_date >= start_2024,
        BondInfo.listing_date <= end_2024,
        BondInfo.first_open != None
    ).order_by(BondInfo.listing_date).limit(20).all()
    
    print('=== v5模型测试 (2024年发行的转债) ===')
    print(f'{'代码':<8} {'方法':<18} {'预测':<10} {'实际':<8} {'误差':<8}')
    print('-'*65)
    
    errors = []
    for b in bonds:
        result = predict_price_v5(b.bond_code)
        if not result:
            continue
        
        method = result.get('method', '未知')[:16]
        pred = result['avg_price'] if method == '相似债券法' else result['total_value']
        
        if pred:
            error = abs(pred - b.first_open)
            error_pct = error / b.first_open * 100
            errors.append((b.bond_code, error_pct))
            
            flag = '⚠️' if error_pct > 10 else '✓'
            print(f'{b.bond_code:<8} {method:<18} {pred:<10.2f} {b.first_open:<8.2f} {error_pct:<7.1f}% {flag}')
    
    total = len(errors)
    over_10 = sum(1 for e in errors if e[1] > 10)
    avg_error = sum(e[1] for e in errors) / total if total > 0 else 0
    
    print(f'\n=== 统计 ===')
    print(f'总数: {total}, 误差>10%: {over_10}只 ({over_10/total*100:.1f}%), 平均误差: {avg_error:.1f}%')
    
    session.close()