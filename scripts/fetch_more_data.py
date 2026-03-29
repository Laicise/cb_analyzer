"""
扩展数据获取脚本
1. 获取历史转债首日开盘价
2. 获取正股基本信息
3. 尝试获取市场情绪指标
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak
import pandas as pd
from db.models import get_session, BondInfo, StockInfo, UpdateLog
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def fetch_historical_bonds():
    """获取历史转债数据（2018-2023）"""
    print("=== 获取历史转债数据 ===")
    
    # 使用东方财富的转债列表
    try:
        df = ak.bond_zh_cov_info()
        print(f"获取到 {len(df)} 条转债数据")
        
        # 筛选已上市的
        df = df[df['LISTING_DATE'].notna()]
        
        session = get_session()
        count = 0
        for _, row in df.iterrows():
            bond_code = row['SECURITY_CODE']
            listing_date = row['LISTING_DATE']
            
            # 只处理2018-2023年的
            if listing_date and '2018' <= str(listing_date)[:4] <= '2023':
                # 检查是否已存在
                existing = session.query(BondInfo).filter_by(bond_code=bond_code).first()
                if not existing:
                    bond = BondInfo(
                        bond_code=bond_code,
                        bond_name=row.get('SECURITY_NAME_ABBR', ''),
                        stock_code=row.get('CONVERT_STOCK_CODE', ''),
                        listing_date=listing_date,
                        issue_size=row.get('ACTUAL_ISSUE_SCALE'),
                        credit_rating=row.get('RATING'),
                        par_value=row.get('PAR_VALUE', 100),
                        coupon_rate=row.get('COUPON_IR')
                    )
                    session.add(bond)
                    count += 1
        
        session.commit()
        print(f"新增 {count} 条历史转债记录")
        session.close()
        
    except Exception as e:
        print(f"获取历史转债失败: {e}")


def fetch_stock_fundamentals():
    """获取正股基本面数据"""
    print("\n=== 获取正股基本面数据 ===")
    
    session = get_session()
    
    # 获取所有没有基本面的正股
    bonds = session.query(BondInfo).filter(
        BondInfo.stock_code != None
    ).all()
    
    for bond in bonds[:50]:  # 每次处理50只
        if not bond.stock_code:
            continue
        
        # 检查是否已有
        existing = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
        if existing:
            continue
        
        try:
            # 获取公司概况
            df = ak.stock_individual_info_em(symbol=bond.stock_code)
            if df is not None and len(df) > 0:
                info = dict(zip(df['item'], df['value']))
                
                stock = StockInfo(
                    stock_code=bond.stock_code,
                    stock_name=info.get('股票简称', ''),
                    industry_sw_l1=info.get('行业', ''),
                    listing_date=info.get('上市时间', ''),
                    total_shares=info.get('总股本'),
                    float_shares=info.get('流通股'),
                    total_market_cap=info.get('总市值'),
                    float_market_cap=info.get('流通市值')
                )
                session.add(stock)
                print(f"获取 {bond.stock_code} 基本面成功")
                break  # 每次只获取一只，避免被限流
        except Exception as e:
            print(f"获取 {bond.stock_code} 失败: {e}")
            break
    
    session.commit()
    session.close()


def fetch_market_sentiment():
    """尝试获取市场情绪指标"""
    print("\n=== 获取市场情绪指标 ===")
    
    try:
        # 获取近期新股申购数据
        df = ak.stock_ipo_history(start_date='20240101', end_date='20241231')
        if df is not None:
            print(f"新股申购数据字段: {df.columns.tolist()}")
            print(df.tail())
    except Exception as e:
        print(f"获取新股申购数据失败: {e}")
    
    try:
        # 获取投资者情绪指数
        df = ak.stock_market_sentiment()
        print(f"市场情绪数据: {df.columns.tolist()}")
    except Exception as e:
        print(f"获取市场情绪失败: {e}")


if __name__ == '__main__':
    # fetch_historical_bonds()
    fetch_stock_fundamentals()
    # fetch_market_sentiment()