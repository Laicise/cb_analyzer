"""
可转债分析主程序 - 完整版
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.models import init_db, get_session, BondInfo, StockInfo, BondDaily, PredictionRecord, UpdateLog
from scripts.fetch_cov_data import update_all as fetch_cov
from scripts.fetch_stock_info import fetch_all_stock_info
from scripts.calculate_yield import update_yields
from analysis.similarity_v3 import predict_price_v3, find_similar_bonds
from analysis.similarity import get_confidence_level
from analysis.ml_model import train_model, predict_with_ml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def init():
    """初始化数据库"""
    print("="*50)
    print("初始化数据库...")
    print("="*50)
    init_db()
    print("初始化完成!")


def update_data():
    """更新所有数据"""
    print("="*50)
    print("开始更新数据...")
    print("="*50)
    
    print("\n[1/4] 获取可转债数据...")
    fetch_cov()
    
    print("\n[2/4] 获取正股信息...")
    fetch_all_stock_info()
    
    print("\n[3/4] 计算到期收益率...")
    update_yields()
    
    print("\n[4/4] 获取历史首日数据...")
    print("  (跳过历史数据获取，请使用 scripts/continue_fetch.py)")
    
    print("\n" + "="*50)
    print("数据更新完成!")
    print("="*50)


def show_bonds(limit=10):
    """显示可转债列表"""
    session = get_session()
    bonds = session.query(BondInfo).limit(limit).all()
    
    print(f"\n{'代码':<8} {'名称':<12} {'正股':<10} {'行业':<15} {'现价':<8} {'首日':<8} {'评级':<6}")
    print("-"*75)
    
    for bond in bonds:
        stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
        industry = stock.industry_sw_l1 if stock and stock.industry_sw_l1 else '-'
        
        latest = session.query(BondDaily).filter_by(bond_code=bond.bond_code).order_by(BondDaily.trade_date.desc()).first()
        price = f"{latest.close_price:.2f}" if latest and latest.close_price else '-'
        
        first = f"{bond.first_open:.2f}" if bond.first_open else '-'
        
        print(f"{bond.bond_code:<8} {bond.bond_name:<12} {bond.stock_name:<10} {industry:<15} {price:<8} {first:<8} {bond.credit_rating or '-':<6}")
    
    session.close()


def predict_new_bond(bond_code):
    """预测新债券价格 - 显示完整分析逻辑"""
    print("\n" + "="*70)
    print(f"可转债估值预测分析: {bond_code}")
    print("="*70)
    
    # 先显示目标债券信息
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    if not bond:
        print(f"未找到债券 {bond_code}")
        session.close()
        return
    
    stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
    
    print(f"\n{'='*70}")
    print(f"【目标债券信息】")
    print(f"{'='*70}")
    print(f"  债券代码: {bond.bond_code}")
    print(f"  债券名称: {bond.bond_name}")
    print(f"  正股名称: {bond.stock_name}")
    print(f"  所属行业: {stock.industry_sw_l1 if stock and stock.industry_sw_l1 else '未知'}")
    print(f"  发行规模: {bond.issue_size}亿元")
    print(f"  信用评级: {bond.credit_rating or '未知'}")
    print(f"  转股价: {bond.conversion_price}元")
    print(f"  转股价值: {bond.conversion_value}元")
    print(f"  转股溢价率: {bond.premium_rate}%")
    
    if bond.first_open:
        print(f"\n  ★ 实际首日开盘价: {bond.first_open}元 (用于对比)")
    
    # 获取相似债券
    similar = find_similar_bonds(bond_code, top_n=5)
    
    if not similar or len(similar) == 0:
        print("\n未找到相似的可转债进行参考")
        session.close()
        return
    
    print(f"\n{'='*70}")
    print(f"【相似债券匹配】")
    print(f"{'='*70}")
    
    for i, s in enumerate(similar, 1):
        print(f"\n  参考{i}: {s['bond_name']} (代码: {s['bond_code']})")
        print(f"    - 相似度: {s['similarity']:.1%}")
        print(f"    - 行业: {s['industry']}")
        print(f"    - 评级: {s['rating']}")
        print(f"    - 转股价值: {s['conversion_value']}元")
        print(f"    - 溢价率: {s['premium_rate']}%")
        print(f"    - 首日开盘价: {s['first_open']}元")
    
    # 进行预测并显示计算过程
    result = predict_price_v3(bond_code)
    
    if result:
        print(f"\n{'='*70}")
        print(f"【预测价格计算过程】")
        print(f"{'='*70}")
        
        for i, ref in enumerate(result['reference_bonds'], 1):
            print(f"\n  参考{chr(64+i)}: {ref['ref_bond']} (首日开盘: {ref['ref_price']}元)")
            print(f"    计算公式: 预测价 = 参考价 + 各项调整")
            print(f"    初始价格: {ref['ref_price']}元")
            
            for adj_name, adj_val, adj_price in ref['adjustments']:
                sign = '+' if adj_price >= 0 else ''
                print(f"    + {adj_name}: {adj_val:+.2f} → {sign}{adj_price:.2f}元")
            
            print(f"    ─────────────────────────")
            print(f"    = 预测价格: {ref['predicted_price']}元")
        
        # 计算加权平均
        total_weight = sum(p['similarity'] for p in result['reference_bonds'])
        print(f"\n  【加权平均计算】")
        print(f"    加权均价 = Σ(预测价 × 相似度) / Σ(相似度)")
        weighted = sum(p['predicted_price'] * p['similarity'] for p in result['reference_bonds']) / total_weight
        print(f"    = {weighted:.2f}元")
        
        # 可信度
        avg_sim = sum(p['similarity'] for p in result['reference_bonds']) / len(result['reference_bonds'])
        conf, title, desc = get_confidence_level(avg_sim)
        
        print(f"\n{'='*70}")
        print(f"【预测结果】")
        print(f"{'='*70}")
        print(f"  预测开盘价区间: {result['price_range'][0]:.2f} ~ {result['price_range'][1]:.2f} 元")
        print(f"  加权预测价格: {result['avg_price']:.2f} 元")
        print(f"  可信度评级: [{conf}] {title}")
        
        if bond.first_open:
            error = abs(result['avg_price'] - bond.first_open)
            error_pct = error / bond.first_open * 100
            print(f"\n  ★ 实际首日开盘: {bond.first_open}元")
            print(f"  ★ 预测误差: {error:.2f}元 ({error_pct:.1f}%)")
        
        # 保存预测结果
        record = PredictionRecord(
            bond_code=bond_code,
            bond_name=bond.bond_name,
            predict_date=datetime.now(),
            predicted_price=result['avg_price'],
            confidence_level=conf,
            reference_bonds=str([p['ref_bond'] for p in result['reference_bonds']]),
            status='pending'
        )
        if bond.first_open:
            record.actual_price = bond.first_open
            record.error_rate = error_pct
            record.status = 'confirmed'
        session.add(record)
        
        if bond:
            bond.predicted_price = result['avg_price']
        
        session.commit()
        print(f"\n  (预测结果已保存到数据库)")
    
    session.close()


def stats():
    """显示统计信息"""
    session = get_session()
    
    bond_count = session.query(BondInfo).count()
    stock_count = session.query(StockInfo).count()
    daily_count = session.query(BondDaily).count()
    first_day_count = session.query(BondInfo).filter(BondInfo.first_open != None).count()
    pred_total = session.query(PredictionRecord).count()
    pred_confirmed = session.query(PredictionRecord).filter_by(status='confirmed').count()
    
    print("\n" + "="*50)
    print("数据统计")
    print("="*50)
    print(f"可转债数量: {bond_count}")
    print(f"正股信息数量: {stock_count}")
    print(f"行情记录数量: {daily_count}")
    print(f"首日数据数量: {first_day_count}")
    print(f"\n预测记录:")
    print(f"  总预测数: {pred_total}")
    print(f"  已确认: {pred_confirmed}")
    
    # 误差统计
    records = session.query(PredictionRecord).filter(
        PredictionRecord.error_rate != None
    ).all()
    
    if records:
        errors = [r.error_rate for r in records]
        avg_error = sum(errors) / len(errors)
        print(f"\n预测误差统计:")
        print(f"  平均误差: {avg_error:.2f}%")
        excellent = len([e for e in errors if e < 10])
        good = len([e for e in errors if 10 <= e < 20])
        print(f"  优秀(<10%): {excellent}, 良好(10-20%): {good}")
    
    # 行业分布
    print("\n行业分布 (Top 10):")
    stocks = session.query(StockInfo).filter(StockInfo.industry_sw_l1 != None).all()
    industry_count = {}
    for stock in stocks:
        if stock.industry_sw_l1:
            industry_count[stock.industry_sw_l1] = industry_count.get(stock.industry_sw_l1, 0) + 1
    
    for ind, cnt in sorted(industry_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ind}: {cnt}")
    
    session.close()


def menu():
    """主菜单"""
    while True:
        print("\n" + "="*70)
        print("可转债量化估值分析系统 v3.0")
        print("="*70)
        print("1. 初始化数据库")
        print("2. 更新所有数据")
        print("3. 查看可转债列表")
        print("4. 预测新债价格(显示分析过程)")
        print("5. 数据统计")
        print("6. 机器学习模型训练")
        print("0. 退出")
        
        choice = input("\n请选择: ").strip()
        
        if choice == '1':
            init()
        elif choice == '2':
            update_data()
        elif choice == '3':
            show_bonds()
        elif choice == '4':
            code = input("请输入债券代码: ").strip()
            predict_new_bond(code)
        elif choice == '5':
            stats()
        elif choice == '6':
            print("\n训练机器学习模型...")
            result = train_model()
            if result:
                print(f"模型训练完成! 测试集MAE: {result['test_mae']:.2f}元")
        elif choice == '0':
            print("再见!")
            break
        else:
            print("无效选择")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='可转债量化估值分析系统')
    parser.add_argument('command', nargs='?', help='命令')
    parser.add_argument('--code', help='债券代码')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        init()
    elif args.command == 'update':
        update_data()
    elif args.command == 'show':
        show_bonds()
    elif args.command == 'predict':
        if args.code:
            predict_new_bond(args.code)
        else:
            print("请指定债券代码: --code 110074")
    elif args.command == 'stats':
        stats()
    elif args.command == 'train':
        result = train_model()
        if result:
            print(f"模型训练完成! MAE: {result['test_mae']:.2f}元")
    else:
        menu()