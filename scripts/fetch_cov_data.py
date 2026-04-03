"""
获取可转债数据
从AKShare获取可转债实时行情和基本信息，并保存到数据库
"""
import akshare as ak
import pandas as pd
from datetime import datetime
from db.models import get_session, BondInfo, BondDaily, UpdateLog
from config import AKSHARE_COV_SPOT_FUNC, AKSHARE_COV_INFO_FUNC
import warnings
warnings.filterwarnings('ignore')


def fetch_cov_spot():
    """获取可转债实时行情"""
    print("正在获取可转债实时行情...")
    df = ak.bond_zh_hs_cov_spot()
    print(f"获取到 {len(df)} 条实时行情数据")
    return df


def fetch_cov_info():
    """获取可转债基本信息（含转股价值、溢价率）"""
    print("正在获取可转债基本信息...")
    df = ak.bond_zh_cov()
    print(f"获取到 {len(df)} 条基本信息数据")
    return df


def save_bond_info(df_info):
    """保存可转债基本信息"""
    session = get_session()
    saved_count = 0
    try:
        for _, row in df_info.iterrows():
            bond_code = str(row.get('债券代码', '')).zfill(6)

            # 检查是否已存在
            existing = session.query(BondInfo).filter_by(bond_code=bond_code).first()

            # 转换日期
            listing_date = None
            if pd.notna(row.get('上市时间')):
                try:
                    listing_date = pd.to_datetime(row.get('上市时间'))
                except:
                    pass

            # 到期日（假设6年期限）
            expiry_date = None
            if listing_date:
                from datetime import timedelta
                expiry_date = listing_date + timedelta(days=365*6)

            # 获取转股价值、溢价率
            conversion_value = row.get('转股价值')
            premium_rate = row.get('转股溢价率')

            if existing:
                # 更新
                existing.bond_name = row.get('债券简称', '')
                existing.stock_code = str(row.get('正股代码', '')).zfill(6)
                existing.stock_name = row.get('正股简称', '')
                existing.issue_price = 100.0  # 发行价默认为100
                existing.issue_size = row.get('发行规模')
                existing.listing_date = listing_date
                existing.expiry_date = expiry_date
                existing.credit_rating = row.get('信用评级')
                existing.conversion_price = row.get('转股价')
                # 保存转股价值到新字段（如果模型中有）
                existing.conversion_value = conversion_value
                existing.premium_rate = premium_rate
            else:
                # 新增
                bond = BondInfo(
                    bond_code=bond_code,
                    bond_name=row.get('债券简称', ''),
                    stock_code=str(row.get('正股代码', '')).zfill(6),
                    stock_name=row.get('正股简称', ''),
                    issue_price=100.0,
                    issue_size=row.get('发行规模'),
                    listing_date=listing_date,
                    expiry_date=expiry_date,
                    credit_rating=row.get('信用评级'),
                    conversion_price=row.get('转股价'),
                    conversion_value=conversion_value,
                    premium_rate=premium_rate
                )
                session.add(bond)
                saved_count += 1

        session.commit()
        print(f"新增 {saved_count} 条可转债基本信息")

        # 记录日志
        log = UpdateLog(
            task_name='fetch_bond_info',
            status='success',
            message=f'获取可转债基本信息 {len(df_info)} 条',
            records_count=len(df_info)
        )
        session.add(log)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def save_bond_daily(df_spot):
    """保存可转债每日行情"""
    session = get_session()
    today = datetime.now().date()
    saved_count = 0
    try:
        for _, row in df_spot.iterrows():
            # 过滤非交易数据
            if row.get('trade', 0) == 0:
                continue

            bond_code = str(row.get('code', '')).zfill(6)

            daily = BondDaily(
                bond_code=bond_code,
                trade_date=datetime.now(),
                open_price=row.get('open', 0),
                high_price=row.get('high', 0),
                low_price=row.get('low', 0),
                close_price=row.get('trade', 0),
                volume=int(row.get('volume', 0)),
                amount=row.get('amount', 0),
                change_percent=row.get('changepercent', 0)
            )
            session.add(daily)
            saved_count += 1

        session.commit()
        print(f"保存 {saved_count} 条每日行情")

        # 记录日志
        log = UpdateLog(
            task_name='fetch_bond_daily',
            status='success',
            message=f'保存可转债每日行情 {saved_count} 条',
            records_count=saved_count
        )
        session.add(log)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def update_all():
    """更新所有可转债数据"""
    print("="*50)
    print("开始更新可转债数据...")
    print("="*50)
    
    try:
        # 获取实时行情
        df_spot = fetch_cov_spot()
        save_bond_daily(df_spot)
        
        # 获取基本信息（包含转股价值、溢价率）
        df_info = fetch_cov_info()
        save_bond_info(df_info)
        
        print("="*50)
        print("数据更新完成!")
        print("="*50)
    except Exception as e:
        print(f"更新失败: {str(e)}")
        session = get_session()
        log = UpdateLog(
            task_name='update_all',
            status='failed',
            message=str(e),
            records_count=0
        )
        session.add(log)
        session.commit()
        session.close()


if __name__ == '__main__':
    update_all()