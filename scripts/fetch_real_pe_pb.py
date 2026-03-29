"""
PE/PB数据获取 - 使用东方财富实时行情（包含PE/PB列）
一次性获取全市场数据，存到数据库
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak
import pandas as pd
from db.models import get_session, StockInfo, BondInfo
from sqlalchemy import create_engine, text
from config import DB_PATH
import warnings, time
warnings.filterwarnings('ignore')

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(f'sqlite:///{DB_PATH}')
    return _engine


def fetch_pe_pb_from_spot():
    """从东方财富实时行情获取PE/PB（包含 市盈率-动态、市净率 列）"""
    print("=== 获取全市场PE/PB数据 ===")
    print("注意：这需要60-90秒...")
    
    try:
        df = ak.stock_zh_a_spot_em()
        print(f"获取到 {len(df)} 只股票")
        
        # 找关键列
        cols = df.columns.tolist()
        code_col = [c for c in cols if '代码' in c][0]
        
        pe_col = [c for c in cols if '市盈' in c and '动态' in c]
        pb_col = [c for c in cols if '净率' in c]
        
        if not pe_col or not pb_col:
            print(f"未找到PE/PB列: pe={pe_col}, pb={pb_col}")
            return {}
        
        pe_col = pe_col[0]
        pb_col = pb_col[0]
        
        print(f"PE列: {pe_col}, PB列: {pb_col}")
        
        results = {}
        for _, row in df.iterrows():
            code = str(row[code_col]).zfill(6)
            
            # PE处理
            pe_raw = row[pe_col]
            pe = None
            if pe_raw is not None and str(pe_raw) not in ['', '-', 'None', 'nan']:
                try:
                    pe_f = float(pe_raw)
                    if 0 < pe_f < 500:  # 合理范围
                        pe = pe_f
                except:
                    pass
            
            # PB处理
            pb_raw = row[pb_col]
            pb = None
            if pb_raw is not None and str(pb_raw) not in ['', '-', 'None', 'nan']:
                try:
                    pb_f = float(pb_raw)
                    if 0 < pb_f < 50:
                        pb = pb_f
                except:
                    pass
            
            results[code] = {'pe': pe, 'pb': pb}
        
        has_pe = sum(1 for v in results.values() if v['pe'] is not None)
        has_pb = sum(1 for v in results.values() if v['pb'] is not None)
        print(f"有效数据: PE={has_pe}只, PB={has_pb}只")
        
        return results
        
    except Exception as e:
        print(f"获取失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def update_database(results):
    """更新数据库"""
    engine = get_engine()
    
    updated_pe = 0
    updated_pb = 0
    
    for code, vals in results.items():
        pe = vals.get('pe')
        pb = vals.get('pb')
        
        if pe is None and pb is None:
            continue
        
        params = {'code': code}
        set_clauses = []
        if pe is not None:
            set_clauses.append('pe = :pe')
            params['pe'] = pe
        if pb is not None:
            set_clauses.append('pb = :pb')
            params['pb'] = pb
        
        if not set_clauses:
            continue
        
        sql = f"UPDATE stock_info SET {', '.join(set_clauses)} WHERE stock_code = :code"
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql), params)
                if result.rowcount > 0:
                    if pe: updated_pe += 1
                    if pb: updated_pb += 1
        except:
            pass
    
    print(f"数据库更新: PE更新{updated_pe}条, PB更新{updated_pb}条")
    return updated_pe, updated_pb


def show_stats():
    """显示统计"""
    session = get_session()
    total = session.query(StockInfo).count()
    has_pe = session.query(StockInfo).filter(StockInfo.pe != None).count()
    has_pb = session.query(StockInfo).filter(StockInfo.pb != None).count()
    has_mktcap = session.query(StockInfo).filter(StockInfo.total_market_cap != None).count()
    
    # 看看PE分布
    stocks_with_pe = session.query(StockInfo).filter(StockInfo.pe != None).all()
    if stocks_with_pe:
        pes = [s.pe for s in stocks_with_pe if s.pe]
        print(f"\nPE分布(前10):")
        for s in sorted(stocks_with_pe, key=lambda x: x.pe or 0)[:10]:
            print(f"  {s.stock_code}: PE={s.pe}, PB={s.pb}")
    
    session.close()
    
    print(f"\n=== 最终统计 ===")
    print(f"总正股: {total}")
    print(f"有PE: {has_pe} ({has_pe/total*100:.0f}%)")
    print(f"有PB: {has_pb} ({has_pb/total*100:.0f}%)")
    print(f"有市值: {has_mktcap} ({has_mktcap/total*100:.0f}%)")


def run():
    """主流程"""
    results = fetch_pe_pb_from_spot()
    
    if results:
        update_database(results)
    
    show_stats()


if __name__ == '__main__':
    run()