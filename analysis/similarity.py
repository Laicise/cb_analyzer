"""
相似度匹配模型（ML优化版）
===========================
改进：
1. 从历史数据学习最优相似度权重
2. 整合市场情绪因子
3. 使用ML置信度加权
"""
import numpy as np
from db.models import get_session, BondInfo, StockInfo, BondDaily
from config import SIMILARITY_WEIGHTS, RATING_MAP, SIMILARITY_CONFIG
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 全局变量存储学习到的权重
_learned_weights = None
_learned_weights_cache_time = None
LEARNING_CACHE_HOURS = 24  # 权重学习结果缓存24小时


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


def get_market_sentiment(listing_date):
    """获取市场情绪因子"""
    if listing_date is None:
        return {'sentiment': 0.5, 'change_pct': 0}

    # 默认返回中性情绪
    # 实际可接入真实市场数据
    return {'sentiment': 0.5, 'change_pct': 0}


def get_batch_info(session, listing_date):
    """获取同批次新债信息"""
    if listing_date is None:
        return {'batch_count': 1, 'batch_avg_premium': 20, 'batch_heat': 0.5}

    start = listing_date - timedelta(days=7)
    end = listing_date + timedelta(days=7)

    batch_bonds = session.query(BondInfo).filter(
        BondInfo.listing_date >= start,
        BondInfo.listing_date <= end,
        BondInfo.bond_code != None
    ).all()

    count = len(batch_bonds)
    if count > 1:
        premiums = [b.premium_rate for b in batch_bonds if b.premium_rate]
        avg_premium = sum(premiums) / len(premiums) if premiums else 20
        heat = min(1.0, count / 5)
    else:
        avg_premium = 20
        heat = 0.3

    return {'batch_count': count, 'batch_avg_premium': avg_premium, 'batch_heat': heat}


def learn_optimal_weights():
    """从历史数据学习最优相似度权重

    使用线性回归分析各相似度因子与预测误差的关系
    """
    global _learned_weights, _learned_weights_cache_time

    # 检查缓存
    if _learned_weights is not None and _learned_weights_cache_time is not None:
        cache_age = (datetime.now() - _learned_weights_cache_time).total_seconds() / 3600
        if cache_age < LEARNING_CACHE_HOURS:
            return _learned_weights

    session = get_session()

    # 获取有完整数据的债券
    bonds = session.query(BondInfo).filter(
        BondInfo.first_open != None,
        BondInfo.conversion_value != None,
        BondInfo.premium_rate != None
    ).all()

    if len(bonds) < 30:
        session.close()
        return None  # 数据不足

    # 收集特征和目标
    X_features = []
    y_errors = []

    for i, bond in enumerate(bonds):
        if i == 0:
            continue

        # 获取历史参考债券（使用之前的样本）
        ref_bonds = [b for b in bonds[:i] if b.first_open is not None]
        if not ref_bonds:
            continue

        # 找最相似的参考债券
        new_stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()

        best_similarity = -1
        best_ref = None

        for ref in ref_bonds:
            ref_stock = session.query(StockInfo).filter_by(stock_code=ref.stock_code).first()

            scores = {
                'industry': get_industry_score(
                    new_stock.industry_sw_l1 if new_stock else '',
                    ref_stock.industry_sw_l1 if ref_stock else ''
                ),
                'business': get_business_score(
                    new_stock.business if new_stock else '',
                    ref_stock.business if ref_stock else ''
                ),
                'rating': get_rating_score(bond.credit_rating, ref.credit_rating),
                'conversion_value': get_value_ratio_score(bond.conversion_value, ref.conversion_value),
                'premium_rate': get_value_diff_score(bond.premium_rate, ref.premium_rate, max_diff=15),
            }

            total_sim = sum(scores[k] * SIMILARITY_WEIGHTS.get(k, 0.1) for k in SIMILARITY_WEIGHTS)

            if total_sim > best_similarity:
                best_similarity = total_sim
                best_ref = ref

        if best_ref and best_ref.first_open:
            # 特征：各相似度分项得分
            new_stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
            ref_stock = session.query(StockInfo).filter_by(stock_code=best_ref.stock_code).first()

            feature_vec = [
                get_industry_score(new_stock.industry_sw_l1 if new_stock else '',
                                  ref_stock.industry_sw_l1 if ref_stock else ''),
                get_business_score(new_stock.business if new_stock else '',
                                 ref_stock.business if ref_stock else ''),
                get_rating_score(bond.credit_rating, best_ref.credit_rating),
                get_value_ratio_score(bond.conversion_value, best_ref.conversion_value),
                get_value_diff_score(bond.premium_rate, best_ref.premium_rate, max_diff=15),
                best_ref.first_open / 100.0,  # 归一化的参考价格
            ]

            # 目标：预测误差
            pred_price = best_ref.first_open
            actual_price = bond.first_open
            error = (pred_price - actual_price) / actual_price  # 相对误差

            X_features.append(feature_vec)
            y_errors.append(error)

    session.close()

    if len(X_features) < 30:
        return None

    X = np.array(X_features)
    y = np.array(y_errors)

    # 使用Ridge回归学习权重
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # 提取学习到的权重
    # 注意：最后一个特征(参考价格)不参与权重计算
    feature_names = ['industry', 'business', 'rating', 'conversion_value', 'premium_rate', 'ref_price_factor']
    raw_weights = model.coef_[:5]

    # 将负权重归零并归一化
    min_w = min(raw_weights.min(), 0)
    if min_w < 0:
        raw_weights = raw_weights - min_w  # 移除非负权重

    weight_sum = raw_weights.sum()
    if weight_sum > 0:
        normalized_weights = raw_weights / weight_sum
    else:
        normalized_weights = np.ones(5) / 5

    learned_weights = {
        'industry': float(normalized_weights[0]),
        'business': float(normalized_weights[1]),
        'rating': float(normalized_weights[2]),
        'conversion_value': float(normalized_weights[3]),
        'premium_rate': float(normalized_weights[4]),
    }

    # 缓存结果
    _learned_weights = learned_weights
    _learned_weights_cache_time = datetime.now()

    print(f"[相似度优化] 学习到的权重: {learned_weights}")

    return learned_weights


def get_effective_weights():
    """获取当前使用的权重（学习到的或默认的）"""
    learned = learn_optimal_weights()
    if learned is not None:
        return learned
    return SIMILARITY_WEIGHTS


def find_similar_bonds(new_bond_code, top_n=5):
    """查找最相似的已上市可转债（ML优化版）"""
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
        'listing_date': new_bond.listing_date,
    }

    bonds = session.query(BondInfo).filter(
        BondInfo.bond_code != new_bond_code,
        BondInfo.listing_date != None,
        BondInfo.conversion_value != None,
        BondInfo.first_open != None  # 必须有首日开盘价
    ).all()

    # 批量查询StockInfo避免N+1问题
    stock_codes = list(set([b.stock_code for b in bonds if b.stock_code]))
    stocks = session.query(StockInfo).filter(StockInfo.stock_code.in_(stock_codes)).all()
    stock_map = {s.stock_code: s for s in stocks}

    # 获取有效权重
    weights = get_effective_weights()

    # 市场情绪
    market_info = get_market_sentiment(new_bond.listing_date)
    batch_info = get_batch_info(session, new_bond.listing_date)

    similarities = []

    for bond in bonds:
        stock = stock_map.get(bond.stock_code)

        scores = {
            'industry': get_industry_score(new_data['industry'], stock.industry_sw_l1 if stock else ''),
            'business': get_business_score(new_data['business'], stock.business if stock else ''),
            'rating': get_rating_score(new_data['rating'], bond.credit_rating),
            'conversion_value': get_value_ratio_score(new_data['conversion_value'], bond.conversion_value),
            'premium_rate': get_value_diff_score(new_data['premium_rate'], bond.premium_rate, max_diff=15),
        }

        # 使用学习到的权重
        total_score = sum(scores[key] * weights.get(key, 0.1) for key in weights)

        # 市场情绪调整因子
        sentiment_adjustment = 1.0 + (market_info.get('sentiment', 0.5) - 0.5) * 0.1

        # 同批次调整
        batch_adjustment = 1.0 + (batch_info['batch_heat'] - 0.5) * 0.05

        # 综合相似度
        adjusted_score = total_score * sentiment_adjustment * batch_adjustment

        latest_price = session.query(BondDaily).filter_by(
            bond_code=bond.bond_code
        ).order_by(BondDaily.trade_date.desc()).first()

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
            'first_open': bond.first_open,  # 首日开盘价作为参考
            'listing_date': bond.listing_date,
            'similarity': round(adjusted_score, 3),
            'base_similarity': round(total_score, 3),
            'scores': scores
        })

    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    session.close()
    return similarities[:top_n]


def predict_price_similarity(bond_code, method='weighted'):
    """
    基于相似度匹配预测价格（ML优化版）

    method:
    - 'weighted': 加权平均（考虑相似度和置信度）
    - 'adjusted': 调整后的参考价格
    """
    similar_bonds = find_similar_bonds(bond_code, top_n=5)

    if not similar_bonds:
        return None

    session = get_session()
    new_bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()

    # 基础市场调整
    market_info = get_market_sentiment(new_bond.listing_date)
    sentiment_factor = 1.0 + (market_info.get('sentiment', 0.5) - 0.5) * 0.2

    predictions = []

    for ref in similar_bonds:
        ref_price = ref['first_open']  # 使用首日开盘价作为基准
        if not ref_price:
            continue

        base_price = ref_price

        # 1. 转股价值差异调整
        cv_diff = (new_bond.conversion_value or 0) - (ref['conversion_value'] or 0)
        cv_adjustment = cv_diff * 0.8  # 转股价值每差1元，价格调整0.8元

        # 2. 溢价率差异调整
        prem_diff = (new_bond.premium_rate or 0) - (ref['premium_rate'] or 0)
        prem_adjustment = prem_diff * 0.3  # 溢价率每差1%，价格调整0.3元

        # 3. 评级差异调整
        new_rating_val = RATING_MAP.get(new_bond.credit_rating, 0)
        ref_rating_val = RATING_MAP.get(ref['rating'], 0)
        rating_diff = new_rating_val - ref_rating_val
        rating_adjustment = rating_diff * 0.5

        # 4. 规模差异调整（小盘债通常溢价更高）
        size_diff = (new_bond.issue_size or 10) - (ref.get('issue_size', 10) or 10)
        size_adjustment = -size_diff * 0.1 if size_diff < 0 else size_diff * 0.05

        # 5. 市场情绪调整
        sentiment_adjustment = (sentiment_factor - 1.0) * base_price

        # 最终预测价格
        predicted = base_price + cv_adjustment + prem_adjustment + rating_adjustment + size_adjustment + sentiment_adjustment
        predicted = max(ML_CONFIG['pred_min'], min(ML_CONFIG['pred_max'], predicted))

        # 置信度（基于相似度）
        confidence = ref['similarity']

        predictions.append({
            'ref_bond': ref['bond_name'],
            'ref_first_open': ref['first_open'],
            'predicted_price': round(predicted, 2),
            'confidence': confidence,
            'adjustments': {
                'cv': cv_adjustment,
                'premium': prem_adjustment,
                'rating': rating_adjustment,
                'size': size_adjustment,
                'sentiment': sentiment_adjustment
            }
        })

    session.close()

    if not predictions:
        return None

    if method == 'weighted':
        # 置信度加权平均
        total_conf = sum(p['confidence'] for p in predictions)
        weighted_price = sum(p['predicted_price'] * p['confidence'] for p in predictions) / total_conf

        return {
            'predicted_price': round(weighted_price, 2),
            'method': 'similarity_weighted',
            'references': predictions,
            'avg_confidence': round(sum(p['confidence'] for p in predictions) / len(predictions), 3)
        }
    else:
        # 简单平均
        avg_price = sum(p['predicted_price'] for p in predictions) / len(predictions)
        return {
            'predicted_price': round(avg_price, 2),
            'method': 'similarity_adjusted',
            'references': predictions
        }


def get_confidence_level(similarity_score):
    """根据相似度得分给出可信度评级"""
    if similarity_score >= 0.70:
        return 'A', '高可信度', '市场存在高度相似的可转债，评估价格参考性强'
    elif similarity_score >= 0.50:
        return 'B', '中等可信度', '市场存在较相似的可转债，评估价格有一定参考性'
    elif similarity_score >= 0.30:
        return 'C', '较低可信度', '市场相似可转债较少，评估价格参考性有限'
    else:
        return 'D', '可信度不足', '市场缺乏相似可转债，建议观望'


# 导入配置（放在最后避免循环引用）
from config import ML_CONFIG


def get_prediction_report(bond_code):
    """生成完整的相似度预测报告"""
    session = get_session()

    new_bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    new_stock = session.query(StockInfo).filter_by(stock_code=new_bond.stock_code).first()

    print("\n" + "="*70)
    print(f"可转债相似度分析: {new_bond.bond_name} ({bond_code})")
    print("="*70)

    print(f"\n【目标债券】")
    print(f"  正股: {new_bond.stock_name}")
    print(f"  行业: {new_stock.industry_sw_l1 if new_stock else '未知'}")
    print(f"  评级: {new_bond.credit_rating or '未知'}")
    print(f"  转股价值: {new_bond.conversion_value}元")
    print(f"  溢价率: {new_bond.premium_rate}%")
    print(f"  发行规模: {new_bond.issue_size}亿元")

    # 获取预测
    result = predict_price_similarity(bond_code)

    if result:
        avg_sim = result.get('avg_confidence', 0)
        conf_level, conf_title, conf_desc = get_confidence_level(avg_sim)

        print(f"\n【参考债券】(使用ML优化权重)")
        for i, ref in enumerate(result['references'], 1):
            print(f"\n  {i}. {ref['ref_bond']}")
            print(f"     参考价: {ref['ref_first_open']:.2f}元 | 相似度: {ref['confidence']:.1%}")
            print(f"     调整: 转股={ref['adjustments']['cv']:+.2f}, 溢价={ref['adjustments']['premium']:+.2f}, "
                  f"评级={ref['adjustments']['rating']:+.2f}, 规模={ref['adjustments']['size']:+.2f}")
            print(f"     预测: {ref['predicted_price']:.2f}元")

        print(f"\n{'='*70}")
        print(f"【相似度预测结果】")
        print(f"  预测价格: {result['predicted_price']:.2f}元")
        print(f"  方法: {result['method']}")
        print(f"  平均置信度: {avg_sim:.1%}")
        print(f"  可信度等级: [{conf_level}] {conf_title}")
        print("="*70)
    else:
        print("\n无法生成预测报告")

    session.close()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        get_prediction_report(sys.argv[1])
    else:
        get_prediction_report('110074')
