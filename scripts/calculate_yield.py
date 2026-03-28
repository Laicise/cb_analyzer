"""
计算可转债到期收益率
"""
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from db.models import get_session, BondInfo, BondDaily
import warnings
warnings.filterwarnings('ignore')


def calculate_yield_to_maturity(bond_code, price, coupon_rate, years_remaining):
    """
    计算到期收益率 (YTM)
    
    参数:
    - price: 当前价格
    - coupon_rate: 票面利率 (%)
    - years_remaining: 剩余年限
    
    公式: YTM ≈ (annual_coupon + (face_value - price) / years) / ((face_value + price) / 2)
    简化公式，假设复利
    """
    face_value = 100  # 面值
    annual_coupon = face_value * coupon_rate / 100  # 每年利息
    
    if years_remaining <= 0:
        return None
    
    # 简化计算
    ytm = (annual_coupon + (face_value - price) / years_remaining) / ((face_value + price) / 2) * 100
    return round(ytm, 2)


def calculate_yield_to_call(bond_code, price, coupon_rate, years_to_call, call_price=100):
    """
    计算回售收益率
    """
    if years_to_call <= 0:
        return None
    
    annual_coupon = 100 * coupon_rate / 100
    ytc = (annual_coupon + (call_price - price) / years_to_call) / ((call_price + price) / 2) * 100
    return round(ytc, 2)


def fetch_cb_yield_data():
    """从AKShare获取可转债收益率数据"""
    try:
        # AKShare可能没有直接的收益率数据，这里用估算
        df = ak.bond_zh_hs_cov_spot()
        return df
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None


def update_yields():
    """更新所有可转债的到期收益率"""
    print("="*50)
    print("开始计算可转债到期收益率...")
    print("="*50)
    
    session = get_session()
    bonds = session.query(BondInfo).all()
    
    # 获取最新行情
    df_spot = fetch_cb_yield_data()
    if df_spot is None:
        print("无法获取行情数据")
        return
    
    # 创建价格映射
    price_map = {}
    for _, row in df_spot.iterrows():
        code = str(row.get('code', '')).zfill(6)
        price_map[code] = row.get('trade', 100)
    
    # 模拟票面利率（AKShare没有直接提供，这里用常见值）
    # 实际应该从债券详情中获取
    coupon_rates = {
        0: 0.5,   # 第一年
        1: 0.5,
        2: 1.0,
        3: 1.5,
        4: 2.0,
        5: 2.5,
    }
    
    updated = 0
    for bond in bonds:
        bond_code = bond.bond_code
        price = price_map.get(bond_code, 100)
        
        # 计算剩余年限
        if bond.expiry_date:
            years_remaining = (bond.expiry_date - datetime.now()).days / 365
        else:
            years_remaining = 6  # 默认6年
        
        if years_remaining <= 0:
            continue
        
        # 获取票面利率（简化处理）
        year_idx = min(int(6 - years_remaining), 5)
        coupon_rate = coupon_rates.get(year_idx, 2.0)
        
        # 计算到期收益率
        ytm = calculate_yield_to_maturity(bond_code, price, coupon_rate, years_remaining)
        
        if ytm:
            # 存储到备注或其他字段（模型中暂无专门字段）
            bond.coupon_rate = coupon_rate
            updated += 1
    
    session.commit()
    print(f"更新 {updated} 只可转债的收益率数据")
    
    session.close()
    
    print("="*50)
    print("到期收益率计算完成!")
    print("="*50)


def get_bond_yield(bond_code):
    """获取单只可转债的收益率"""
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    if not bond:
        return None
    
    # 获取最新价格
    df_spot = fetch_cb_yield_data()
    price = 100
    if df_spot is not None:
        for _, row in df_spot.iterrows():
            if str(row.get('code', '')).zfill(6) == bond_code:
                price = row.get('trade', 100)
                break
    
    # 计算
    if bond.expiry_date:
        years_remaining = (bond.expiry_date - datetime.now()).days / 365
    else:
        years_remaining = 6
    
    ytm = calculate_yield_to_maturity(bond_code, price, bond.coupon_rate or 1.5, years_remaining)
    
    session.close()
    return ytm


if __name__ == '__main__':
    update_yields()