"""
获取正股信息
从AKShare获取正股行业分类和主营业务
"""
import akshare as ak
import pandas as pd
from datetime import datetime
from db.models import get_session, StockInfo, BondInfo, UpdateLog
from config import AKSHARE_STOCK_PROFILE_FUNC
import warnings
warnings.filterwarnings('ignore')


def fetch_stock_profile(stock_code):
    """获取单只股票的公司概况（含主营业务）"""
    try:
        df = ak.stock_profile_cninfo(symbol=stock_code)
        if df is not None and len(df) > 0:
            row = df.iloc[0]
            # 转换日期
            listing_date = None
            ld = row.get('上市日期')
            if ld:
                try:
                    listing_date = pd.to_datetime(ld)
                except:
                    pass
            return {
                'stock_name': row.get('A股简称', ''),
                'industry_sw': row.get('所属行业', ''),  # 申万行业
                'business': row.get('主营业务', ''),
                'listing_date': listing_date
            }
    except Exception as e:
        print(f"获取股票 {stock_code} 信息失败: {e}")
    return None


def get_stock_industry_sw(stock_code):
    """获取申万行业分类"""
    try:
        # 使用东方财富的行业分类
        df = ak.stock_board_industry_cons_em(symbol=stock_code)
        if df is not None and len(df) > 0:
            return df.iloc[0].get('所属板块', '')
    except:
        pass
    return None


def fetch_all_stock_info():
    """获取所有正股的信息"""
    session = get_session()
    
    # 获取所有未更新主营业务的正股
    bonds = session.query(BondInfo).all()
    total = len(bonds)
    print(f"共需处理 {total} 只正股")
    
    success_count = 0
    failed_count = 0
    
    for i, bond in enumerate(bonds):
        stock_code = bond.stock_code
        if not stock_code:
            continue
            
        print(f"[{i+1}/{total}] 处理 {stock_code} {bond.stock_name}...")
        
        # 检查是否已存在
        existing = session.query(StockInfo).filter_by(stock_code=stock_code).first()
        
        # 获取公司概况
        profile = fetch_stock_profile(stock_code)
        
        if profile:
            if existing:
                existing.stock_name = profile['stock_name']
                existing.industry_sw_l1 = profile['industry_sw']
                existing.business = profile['business']
                existing.listing_date = profile['listing_date']
            else:
                stock = StockInfo(
                    stock_code=stock_code,
                    stock_name=profile['stock_name'],
                    industry_sw_l1=profile['industry_sw'],
                    business=profile['business'],
                    listing_date=profile['listing_date']
                )
                session.add(stock)
            success_count += 1
        else:
            failed_count += 1
        
        # 每处理10条提交一次
        if (i + 1) % 10 == 0:
            session.commit()
            print(f"已提交 {i+1} 条")
    
    session.commit()
    print(f"处理完成: 成功 {success_count} 条, 失败 {failed_count} 条")
    
    # 记录日志
    log = UpdateLog(
        task_name='fetch_stock_info',
        status='success' if failed_count == 0 else 'partial',
        message=f'获取正股信息: 成功 {success_count}, 失败 {failed_count}',
        records_count=success_count
    )
    session.add(log)
    session.commit()
    session.close()


def get_industry_list():
    """获取申万行业列表"""
    print("正在获取申万行业列表...")
    df = ak.stock_board_industry_name_em()
    print(f"获取到 {len(df)} 个行业")
    return df


if __name__ == '__main__':
    # 先测试单只股票
    # profile = fetch_stock_profile('600519')
    # print(profile)
    
    # 批量获取所有正股信息
    fetch_all_stock_info()