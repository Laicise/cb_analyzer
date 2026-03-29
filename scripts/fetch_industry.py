"""
获取正股行业数据
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak
from db.models import get_session, BondInfo, StockInfo
import time

def fetch_industry_for_stock(stock_code):
    """获取单个股票的行业"""
    try:
        df = ak.stock_individual_info_em(symbol=stock_code)
        for _, row in df.iterrows():
            if row['item'] == '行业':
                return row['value']
    except:
        pass
    return None

def update_all_industries():
    """更新所有正股的行业数据"""
    session = get_session()
    
    # 获取所有有正股但行业为空的
    stocks = session.query(StockInfo).filter(
        StockInfo.industry_sw_l1 == None
    ).all()
    
    # 或者获取所有转债的正股
    if len(stocks) == 0:
        bonds = session.query(BondInfo).all()
        stock_codes = set([b.stock_code for b in bonds])
        
        for stock_code in list(stock_codes)[:100]:  # 限制数量
            # 检查是否已存在
            existing = session.query(StockInfo).filter_by(stock_code=stock_code).first()
            if existing:
                continue
            
            industry = fetch_industry_for_stock(stock_code)
            if industry:
                # 创建或更新StockInfo
                if not existing:
                    new_stock = StockInfo(stock_code=stock_code, industry_sw_l1=industry)
                    session.add(new_stock)
                else:
                    existing.industry_sw_l1 = industry
            
            print(f"  {stock_code}: {industry}")
            time.sleep(0.5)  # 避免请求过快
    
    session.commit()
    session.close()
    print("行业数据更新完成!")

if __name__ == '__main__':
    update_all_industries()