"""
定向获取正股基本面数据
只获取数据库中有可转债的正股，减少API调用次数
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak
from db.models import get_session, StockInfo, BondInfo
from sqlalchemy import create_engine, text
from config import DB_PATH
import warnings
warnings.filterwarnings('ignore')

_engine = None

def get_engine_local():
    global _engine
    if _engine is None:
        _engine = create_engine(f'sqlite:///{DB_PATH}')
    return _engine


def fetch_stock_essential_batch(codes):
    """批量获取正股核心指标（使用东方财富实时行情）"""
    if not codes:
        return {}
    
    print(f"开始获取 {len(codes)} 只正股的基本面数据...")
    
    # 分批获取（每批100只）
    batch_size = 100
    results = {}
    
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        try:
            # 使用实时行情接口
            df = ak.stock_zh_a_spot_em()
            if df is None or len(df) == 0:
                continue
            
            # 找到代码列
            code_col = [c for c in df.columns if '代码' in c][0]
            name_col = [c for c in df.columns if '名称' in c and '股票' not in c][0]
            
            # 找PE/PB列（可能有多个，优先选动态）
            pe_cols = [c for c in df.columns if ('市盈' in c or 'PE' in c.upper()) and '动' in c]
            pb_cols = [c for c in df.columns if '净率' in c or 'PB' in c.upper()]
            mktcap_cols = [c for c in df.columns if '市值' in c and '流' not in c]
            
            if not pe_cols:
                pe_cols = [c for c in df.columns if '市盈' in c or 'PE' in c.upper()]
            if not pb_cols:
                pb_cols = [c for c in df.columns if '净率' in c or 'PB' in c.upper()]
            
            pe_col = pe_cols[0] if pe_cols else None
            pb_col = pb_cols[0] if pb_cols else None
            mktcap_col = mktcap_cols[0] if mktcap_cols else None
            
            print(f"  批次 {i//batch_size + 1}/{(len(codes)+batch_size-1)//batch_size}: PE列={pe_col[:10] if pe_col else None}, PB列={pb_col[:10] if pb_col else None}")
            
            for _, row in df.iterrows():
                code = str(row[code_col]).zfill(6)
                if code in codes:
                    pe_val = row[pe_col] if pe_col else None
                    pb_val = row[pb_col] if pb_col else None
                    mktcap_val = row[mktcap_col] if mktcap_col else None
                    
                    # 处理无效值
                    pe = None
                    if pe_val is not None and str(pe_val) not in ['', '-', 'None', 'nan']:
                        try:
                            pe_f = float(pe_val)
                            if 0 < pe_f < 2000:
                                pe = pe_f
                        except:
                            pass
                    
                    pb = None
                    if pb_val is not None and str(pb_val) not in ['', '-', 'None', 'nan']:
                        try:
                            pb_f = float(pb_val)
                            if 0 < pb_f < 100:
                                pb = pb_f
                        except:
                            pass
                    
                    mktcap = None
                    if mktcap_val is not None and str(mktcap_val) not in ['', '-', 'None', 'nan']:
                        try:
                            mktcap_f = float(mktcap_val)
                            if mktcap_f > 0:
                                # 转为亿元
                                if mktcap_f > 1e8:  # 超过1亿则假设单位是元
                                    mktcap = mktcap_f / 1e8
                                else:
                                    mktcap = mktcap_f
                        except:
                            pass
                    
                    results[code] = {'pe': pe, 'pb': pb, 'total_market_cap': mktcap}
            
            # 每批次之间暂停
            import time
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  批次 {i//batch_size+1} 失败: {e}")
            import time
            time.sleep(1)
    
    print(f"获取到 {len(results)} 只股票的基本面数据")
    return results


def update_database(indicators):
    """更新数据库"""
    engine = get_engine_local()
    
    updated_pe = 0
    updated_pb = 0
    updated_mktcap = 0
    
    for code, vals in indicators.items():
        pe = vals.get('pe')
        pb = vals.get('pb')
        mktcap = vals.get('total_market_cap')
        
        set_clauses = []
        params = {}
        if pe is not None:
            set_clauses.append('pe = :pe')
            params['pe'] = pe
        if pb is not None:
            set_clauses.append('pb = :pb')
            params['pb'] = pb
        if mktcap is not None:
            set_clauses.append('total_market_cap = :total_market_cap')
            params['total_market_cap'] = mktcap
        
        if not set_clauses:
            continue
        
        params['code'] = code
        sql = f"UPDATE stock_info SET {', '.join(set_clauses)} WHERE stock_code = :code"
        
        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            if result.rowcount > 0:
                if pe: updated_pe += 1
                if pb: updated_pb += 1
                if mktcap: updated_mktcap += 1
    
    print(f"\n更新统计: PE更新{updated_pe}条, PB更新{updated_pb}条, 市值更新{updated_mktcap}条")
    return updated_pe, updated_pb, updated_mktcap


def run():
    """主流程"""
    session = get_session()
    
    # 获取所有有可转债的正股代码
    bonds = session.query(BondInfo.stock_code).distinct().all()
    stock_codes = [str(b[0]).zfill(6) for b in bonds if b[0]]
    session.close()
    
    print(f"共 {len(stock_codes)} 只正股需要获取基本面")
    
    # 批量获取
    indicators = fetch_stock_essential_batch(stock_codes)
    
    if indicators:
        update_database(indicators)
    
    # 最终统计
    session = get_session()
    total = session.query(StockInfo).count()
    has_pe = session.query(StockInfo).filter(StockInfo.pe != None).count()
    has_pb = session.query(StockInfo).filter(StockInfo.pb != None).count()
    has_mktcap = session.query(StockInfo).filter(StockInfo.total_market_cap != None).count()
    session.close()
    
    print(f"\n=== 最终统计 ===")
    print(f"总正股: {total}")
    print(f"有PE: {has_pe} ({has_pe/total*100:.0f}%)")
    print(f"有PB: {has_pb} ({has_pb/total*100:.0f}%)")
    print(f"有市值: {has_mktcap} ({has_mktcap/total*100:.0f}%)")


if __name__ == '__main__':
    run()