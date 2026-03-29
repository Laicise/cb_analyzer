"""
正股基本面数据批量获取 - 高效版
使用 stock_a_indicator_lg 一次性获取所有股票的PE/PB等指标
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak
import pandas as pd
from db.models import get_session, StockInfo, BondInfo
import warnings, time
warnings.filterwarnings('ignore')


def fetch_all_stock_indicators():
    """一次性获取所有股票的财务指标（PE/PB/市值等）"""
    print("=== 批量获取正股基本面数据 ===")
    print("这可能需要30-60秒...")
    
    try:
        # 东方财富股票指标（包含PE/PB/ROE/市值等）
        df = ak.stock_a_indicator_lg()
        print(f"获取到 {len(df)} 只股票的指标数据")
        print(f"列名: {df.columns.tolist()[:15]}...")
        
        return df
    except Exception as e:
        print(f"获取失败: {e}")
        return None


def parse_indicators(df):
    """解析指标数据，提取有用的字段"""
    if df is None:
        return {}
    
    # 找到关键列
    # 典型列: 代码, 名称, 总市值, 流通市值, PE, PB, ROE等
    cols = df.columns.tolist()
    
    # 查找相关列
    stock_code_col = None
    pe_col = None
    pb_col = None
    mktcap_col = None
    
    for c in cols:
        cl = c.lower()
        if '代码' in c or 'code' in cl:
            stock_code_col = c
        if '市盈率' in c or 'PE' in c.upper() or '盈率' in c:
            pe_col = c
        if '市净率' in c or 'PB' in c.upper() or '净率' in c:
            pb_col = c
        if '总市值' in c or '市值' in c:
            mktcap_col = c
    
    print(f"关键列: code={stock_code_col}, pe={pe_col}, pb={pb_col}, mktcap={mktcap_col}")
    
    result = {}
    for _, row in df.iterrows():
        code = str(row.get(stock_code_col, '')).zfill(6)
        if len(code) != 6:
            continue
        
        pe_val = row.get(pe_col)
        pb_val = row.get(pb_col)
        mktcap = row.get(mktcap_col)
        
        # 转换PE
        pe = None
        if pe_val is not None and str(pe_val) not in ['', '-', 'nan', 'None']:
            try:
                pe = float(pe_val)
                if pe <= 0 or pe > 1000:
                    pe = None
            except:
                pass
        
        # 转换PB
        pb = None
        if pb_val is not None and str(pb_val) not in ['', '-', 'nan', 'None']:
            try:
                pb = float(pb_val)
                if pb <= 0 or pb > 50:
                    pb = None
            except:
                pass
        
        # 转换市值（亿元）
        mktcap_val = None
        if mktcap is not None and str(mktcap) not in ['', '-', 'nan', 'None']:
            try:
                mktcap_val = float(mktcap)
                if mktcap_val > 100:  # 转换为亿元
                    mktcap_val = mktcap_val / 100000000
            except:
                pass
        
        result[code] = {'pe': pe, 'pb': pb, 'total_market_cap': mktcap_val}
    
    return result


def update_stock_info(indicators):
    """更新数据库中的正股信息"""
    session = get_session()
    
    stocks = session.query(StockInfo).all()
    updated = 0
    
    for stock in stocks:
        code = stock.stock_code
        if not code:
            continue
        
        # 标准化代码
        code = code.zfill(6)
        info = indicators.get(code, {})
        
        modified = False
        if info.get('pe') and stock.pe is None:
            stock.pe = info['pe']
            modified = True
        if info.get('pb') and stock.pb is None:
            stock.pb = info['pb']
            modified = True
        if info.get('total_market_cap') and stock.total_market_cap is None:
            stock.total_market_cap = info['total_market_cap']
            modified = True
        
        if modified:
            updated += 1
            if updated % 100 == 0:
                session.commit()
    
    session.commit()
    
    # 统计
    total = session.query(StockInfo).count()
    has_pe = session.query(StockInfo).filter(StockInfo.pe != None).count()
    has_pb = session.query(StockInfo).filter(StockInfo.pb != None).count()
    has_mktcap = session.query(StockInfo).filter(StockInfo.total_market_cap != None).count()
    
    print(f"\n更新完成: 本次更新 {updated} 条")
    print(f"总正股数: {total}")
    print(f"有PE: {has_pe} ({has_pe/total*100:.0f}%)")
    print(f"有PB: {has_pb} ({has_pb/total*100:.0f}%)")
    print(f"有市值: {has_mktcap} ({has_mktcap/total*100:.0f}%)")
    
    session.close()


def run():
    """主流程"""
    df = fetch_all_stock_indicators()
    if df is not None:
        indicators = parse_indicators(df)
        print(f"解析到 {len(indicators)} 只股票的数据")
        if indicators:
            update_stock_info(indicators)


if __name__ == '__main__':
    run()