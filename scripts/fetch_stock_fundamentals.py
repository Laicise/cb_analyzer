"""
获取正股基本面数据
从AKShare获取PE、PB、市值、营收、净利润等关键财务指标
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak
import pandas as pd
from datetime import datetime
from db.models import get_session, StockInfo, BondInfo, UpdateLog
import warnings
warnings.filterwarnings('ignore')


def fetch_stock_essential_info(stock_code):
    """获取个股基本面核心指标"""
    try:
        # 东方财富个股信息
        df = ak.stock_individual_info_em(symbol=stock_code)
        if df is None or len(df) == 0:
            return None
        
        info = dict(zip(df['item'], df['value']))
        
        result = {
            'total_market_cap': None,  # 总市值(亿元)
            'float_market_cap': None,   # 流通市值(亿元)
            'pe': None,                 # 市盈率
            'pb': None,                 # 市净率
            'roe': None,                # 净资产收益率
            'revenue_q': None,          # 季度营收(万元)
            'profit_q': None,           # 季度净利润(万元)
            'listing_days': None,       # 上市天数
        }
        
        # 总市值 - 可能带有不同单位
        mcap = info.get('总市值', info.get('总股本(亿股)'))
        if mcap:
            try:
                val = float(str(mcap).replace(',', ''))
                if val > 100000:  # 万元转亿元
                    val = val / 10000
                result['total_market_cap'] = val
            except:
                pass
        
        # 流通市值
        fcap = info.get('流通市值', info.get('流通股本(亿股)'))
        if fcap:
            try:
                val = float(str(fcap).replace(',', ''))
                if val > 100000:
                    val = val / 10000
                result['float_market_cap'] = val
            except:
                pass
        
        # 市盈率
        pe = info.get('市盈率(动态)', info.get('市盈率', ''))
        if pe and str(pe) not in ['', '-', '无', 'null', 'None']:
            try:
                result['pe'] = float(pe)
            except:
                pass
        
        # 市净率
        pb = info.get('市净率', '')
        if pb and str(pb) not in ['', '-', '无', 'null', 'None']:
            try:
                result['pb'] = float(pb)
            except:
                pass
        
        return result
        
    except Exception as e:
        return None


def fetch_stock_financial_indicator(stock_code):
    """获取个股财务指标（ROE、营收增长等）"""
    try:
        # 使用财务指标API
        df = ak.stock_financial_analysis_indicator(symbol=stock_code, start_date='20200101', end_date='20251231')
        if df is not None and len(df) > 0:
            latest = df.iloc[0]
            return {
                'roe': safe_float(latest.get('净资产收益率(%)')),
                'revenue_growth': safe_float(latest.get('营业收入同比增长率(%)')),
                'profit_growth': safe_float(latest.get('净利润同比增长率(%)')),
                'gross_margin': safe_float(latest.get('销售毛利率(%)')),
                'debt_ratio': safe_float(latest.get('资产负债率(%)')),
            }
    except:
        pass
    return {}


def safe_float(val, default=None):
    """安全转换浮点数"""
    if val is None or pd.isna(val):
        return default
    try:
        return float(val)
    except:
        return default


def fetch_stock_financial_summary(stock_code):
    """获取个股财务摘要"""
    try:
        df = ak.stock_financial_abstract(symbol=stock_code)
        if df is not None and len(df) > 0:
            latest = df.iloc[0]
            return {
                'total_revenue': safe_float(latest.get('营业总收入(万元)')),
                'net_profit': safe_float(latest.get('净利润(万元)')),
                'total_assets': safe_float(latest.get('资产总计(万元)')),
                'total_debt': safe_float(latest.get('负债合计(万元)')),
            }
    except:
        pass
    return {}


def fetch_all_stock_fundamentals():
    """批量获取所有正股基本面数据"""
    session = get_session()
    
    bonds = session.query(BondInfo).filter(
        BondInfo.stock_code != None
    ).all()
    
    total = len(bonds)
    print(f"需要处理 {total} 只正股的基本面数据")
    
    success = 0
    failed = 0
    skip = 0
    
    for i, bond in enumerate(bonds):
        stock_code = bond.stock_code
        if not stock_code:
            continue
        
        # 每50条提交一次
        if i > 0 and i % 50 == 0:
            session.commit()
            print(f"已提交 {i}/{total}")
        
        stock = session.query(StockInfo).filter_by(stock_code=stock_code).first()
        if not stock:
            stock = StockInfo(stock_code=stock_code, stock_name=bond.stock_name)
            session.add(stock)
        
        # 获取核心指标
        essential = fetch_stock_essential_info(stock_code)
        if essential:
            if stock.total_market_cap is None:
                stock.total_market_cap = essential.get('total_market_cap')
            if stock.pe is None:
                stock.pe = essential.get('pe')
            if stock.pb is None:
                stock.pb = essential.get('pb')
            
            # 设置上市天数
            if stock.listing_date is None and bond.listing_date:
                stock.listing_date = bond.listing_date
            if stock.listing_date:
                stock.listing_days = (datetime.now() - stock.listing_date).days
            
            success += 1
            print(f"[{i+1}/{total}] {stock_code} {bond.stock_name} - PE:{essential.get('pe','N/A')} PB:{essential.get('pb','N/A')} 市值:{essential.get('total_market_cap','N/A')}亿")
        else:
            failed += 1
        
        # 避免请求过快
        if i % 5 == 0:
            import time
            time.sleep(0.3)
    
    session.commit()
    
    # 记录日志
    log = UpdateLog(
        task_name='fetch_stock_fundamentals',
        status='success' if failed == 0 else 'partial',
        message=f'获取正股基本面: 成功{success}, 失败{failed}, 跳过{skip}',
        records_count=success
    )
    session.add(log)
    session.commit()
    
    print(f"\n基本面获取完成: 成功 {success} 条, 失败 {failed} 条")
    session.close()


def update_missing_stock_info():
    """补充缺失的正股基本信息"""
    session = get_session()
    
    # 找出没有PE/PB的正股
    stocks_need_update = session.query(StockInfo).filter(
        StockInfo.pe == None
    ).all()
    
    print(f"需要补充 {len(stocks_need_update)} 只正股的基本面数据")
    
    for i, stock in enumerate(stocks_need_update):
        essential = fetch_stock_essential_info(stock.stock_code)
        if essential:
            if essential.get('total_market_cap'):
                stock.total_market_cap = essential.get('total_market_cap')
            if essential.get('pe'):
                stock.pe = essential.get('pe')
            if essential.get('pb'):
                stock.pb = essential.get('pb')
            
            if i % 20 == 0:
                session.commit()
                print(f"已更新 {i} 条")
    
    session.commit()
    print(f"补充完成")
    session.close()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'update':
        update_missing_stock_info()
    else:
        fetch_all_stock_fundamentals()