"""
获取可转债历史数据
从AKShare获取每只可转债的历史K线，提取首日开盘价
"""
import akshare as ak
import pandas as pd
from datetime import datetime
from db.models import get_session, BondInfo, BondDaily, UpdateLog
import warnings
warnings.filterwarnings('ignore')


def fetch_bond_daily_history(bond_code):
    """获取单只可转债的历史K线"""
    try:
        # 拼接股票代码
        symbol = f'sh{bond_code}' if bond_code.startswith('1') else f'sz{bond_code}'
        df = ak.bond_zh_hs_cov_daily(symbol=symbol)
        if df is not None and len(df) > 0:
            return df
    except Exception as e:
        pass
    return None


def get_first_day_price(bond_code):
    """获取可转债首日开盘价"""
    df = fetch_bond_daily_history(bond_code)
    if df is not None and len(df) > 0:
        first_row = df.iloc[0]
        return {
            'first_date': first_row['date'],
            'first_open': first_row['open'],
            'first_high': first_row['high'],
            'first_low': first_row['low'],
            'first_close': first_row['close'],
            'first_volume': first_row['volume']
        }
    return None


def update_all_bonds_first_day():
    """更新所有可转债的首日数据"""
    print("="*50)
    print("开始获取可转债首日数据...")
    print("="*50)
    
    session = get_session()
    
    # 获取所有已上市的债券
    bonds = session.query(BondInfo).filter(
        BondInfo.listing_date != None
    ).all()
    
    total = len(bonds)
    success = 0
    failed = 0
    
    for i, bond in enumerate(bonds):
        print(f"[{i+1}/{total}] 获取 {bond.bond_name} ({bond.bond_code})...")
        
        first_data = get_first_day_price(bond.bond_code)
        
        if first_data:
            # 存储首日数据（更新到bond_info表）
            bond.first_date = first_data['first_date']
            bond.first_open = first_data['first_open']
            bond.first_close = first_data['first_close']
            success += 1
            print(f"  首日: {first_data['first_date']} 开盘:{first_data['first_open']} 收盘:{first_data['first_close']}")
        else:
            failed += 1
            print(f"  获取失败")
        
        # 每10条提交一次
        if (i + 1) % 10 == 0:
            session.commit()
    
    session.commit()
    print(f"\n完成: 成功 {success} 条, 失败 {failed} 条")
    
    # 记录日志
    log = UpdateLog(
        task_name='fetch_first_day',
        status='success' if failed == 0 else 'partial',
        message=f'获取首日数据: 成功 {success}, 失败 {failed}',
        records_count=success
    )
    session.add(log)
    session.commit()
    session.close()


def get_prediction_history():
    """获取历史预测记录及实际价格"""
    session = get_session()
    
    bonds = session.query(BondInfo).filter(
        BondInfo.first_open != None,
        BondInfo.predicted_price != None
    ).all()
    
    print("\n历史预测记录:")
    print(f"{'债券代码':<10} {'债券名称':<12} {'预测价格':<10} {'实际首日':<10} {'误差率':<10}")
    print("-"*60)
    
    total_error = 0
    count = 0
    
    for bond in bonds:
        if bond.predicted_price and bond.first_open:
            error = abs(bond.predicted_price - bond.first_open) / bond.first_open * 100
            total_error += error
            count += 1
            print(f"{bond.bond_code:<10} {bond.bond_name:<12} {bond.predicted_price:<10.2f} {bond.first_open:<10.2f} {error:<10.2f}%")
    
    if count > 0:
        avg_error = total_error / count
        print("-"*60)
        print(f"平均误差: {avg_error:.2f}%")
    
    session.close()


if __name__ == '__main__':
    update_all_bonds_first_day()