"""
批量获取所有可转债历史首日数据
"""
import akshare as ak
import pandas as pd
from datetime import datetime
from db.models import get_session, BondInfo, UpdateLog
import time
import warnings
warnings.filterwarnings('ignore')


def fetch_bond_daily_history(symbol):
    """获取单只可转债的历史K线"""
    try:
        df = ak.bond_zh_hs_cov_daily(symbol=symbol)
        if df is not None and len(df) > 0:
            return df
    except:
        pass
    return None


def update_all_first_day_prices():
    """批量更新所有可转债的首日价格"""
    print("="*60)
    print("开始批量获取历史首日数据...")
    print("="*60)
    
    session = get_session()
    
    # 获取所有已上市的债券
    bonds = session.query(BondInfo).filter(
        BondInfo.listing_date != None
    ).all()
    
    total = len(bonds)
    success = 0
    failed = 0
    skipped = 0
    
    for i, bond in enumerate(bonds):
        # 跳过已有首日数据的
        if bond.first_open:
            skipped += 1
            continue
            
        print(f"[{i+1}/{total}] {bond.bond_name} ({bond.bond_code})...", end=" ")
        
        # 构造证券代码
        if bond.bond_code.startswith('1'):
            symbol = f'sh{bond.bond_code}'
        else:
            symbol = f'sz{bond.bond_code}'
        
        df = fetch_bond_daily_history(symbol)
        
        if df is not None and len(df) > 0:
            first_row = df.iloc[0]
            
            # 解析日期
            first_date = None
            try:
                first_date = pd.to_datetime(first_row['date'])
            except:
                pass
            
            bond.first_date = first_date
            bond.first_open = first_row['open']
            bond.first_close = first_row['close']
            bond.first_high = first_row['high']
            bond.first_low = first_row['low']
            
            success += 1
            print(f"✓ {first_date.strftime('%Y-%m-%d') if first_date else 'N/A'} 开盘:{first_row['open']:.2f}")
        else:
            failed += 1
            print("✗ 获取失败")
        
        # 每20条提交一次，避免长时间锁表
        if (i + 1) % 20 == 0:
            session.commit()
            print(f"\n--- 已提交 {i+1} 条 ---\n")
        
        # 避免请求过快
        time.sleep(0.3)
    
    session.commit()
    
    print(f"\n{'='*60}")
    print(f"完成! 成功: {success}, 失败: {failed}, 跳过: {skipped}")
    print(f"{'='*60}")
    
    # 记录日志
    log = UpdateLog(
        task_name='batch_fetch_first_day',
        status='success' if failed == 0 else 'partial',
        message=f'批量获取首日数据: 成功{success}, 失败{failed}, 跳过{skipped}',
        records_count=success
    )
    session.add(log)
    session.commit()
    session.close()


def show_first_day_statistics():
    """显示首日数据统计"""
    session = get_session()
    
    total = session.query(BondInfo).filter(BondInfo.listing_date != None).count()
    with_first_day = session.query(BondInfo).filter(BondInfo.first_open != None).count()
    
    print(f"\n=== 首日数据统计 ===")
    print(f"已上市可转债: {total}")
    print(f"已有首日数据: {with_first_day}")
    print(f"缺失: {total - with_first_day}")
    
    # 显示最新的几条
    print(f"\n最近上市的几只:")
    recent = session.query(BondInfo).filter(
        BondInfo.first_open != None
    ).order_by(BondInfo.listing_date.desc()).limit(5).all()
    
    for bond in recent:
        print(f"  {bond.bond_name}: 首日开盘 {bond.first_open:.2f}元 ({bond.first_date.strftime('%Y-%m-%d') if bond.first_date else 'N/A'})")
    
    session.close()


if __name__ == '__main__':
    update_all_first_day_prices()
    show_first_day_statistics()