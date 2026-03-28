"""
修正历史数据获取脚本
根据债券代码判断交易所：深圳以12/13开头，上海以11开头
"""
import akshare as ak
import time
from db.models import get_session, BondInfo
import warnings
warnings.filterwarnings('ignore')


def get_bond_symbol(code):
    """根据债券代码获取交易所代码"""
    # 深圳: 12xxxx, 13xxxx 开头
    # 上海: 11xxxx, 13xxxx 
    if code.startswith('12') or code.startswith('13'):
        return f'sz{code}'
    else:
        return f'sh{code}'


def fetch_first_day(symbol):
    """获取首日数据"""
    try:
        df = ak.bond_zh_hs_cov_daily(symbol=symbol)
        if df is not None and len(df) > 0:
            first = df.iloc[0]
            return {
                'date': first['date'],
                'open': first['open'],
                'close': first['close'],
                'high': first['high'],
                'low': first['low']
            }
    except:
        pass
    return None


def continue_fetch_history(limit=100):
    """继续获取缺失的首日数据"""
    print("="*60)
    print("继续获取历史首日数据...")
    print("="*60)
    
    session = get_session()
    
    # 获取缺失首日数据的债券
    bonds = session.query(BondInfo).filter(
        BondInfo.listing_date != None,
        BondInfo.first_open == None
    ).limit(limit).all()
    
    total = len(bonds)
    success = 0
    failed = 0
    
    print(f"需要获取: {total} 条")
    
    for i, bond in enumerate(bonds):
        print(f"[{i+1}/{total}] {bond.bond_name} ({bond.bond_code})...", end=" ")
        
        symbol = get_bond_symbol(bond.bond_code)
        data = fetch_first_day(symbol)
        
        if data:
            bond.first_date = data['date']
            bond.first_open = data['open']
            bond.first_close = data['close']
            bond.first_high = data['high']
            bond.first_low = data['low']
            success += 1
            print(f"✓ {data['open']}")
        else:
            failed += 1
            print("✗")
        
        if (i + 1) % 20 == 0:
            session.commit()
        
        time.sleep(0.3)
    
    session.commit()
    session.close()
    
    print(f"\n完成! 成功: {success}, 失败: {failed}")
    return success, failed


if __name__ == '__main__':
    continue_fetch_history(200)