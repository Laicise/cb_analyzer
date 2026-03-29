"""
可转债分析主程序 - 完整版 v4.0
集成：相似度匹配 + 正股基本面 + ML模型预测
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.models import init_db, get_session, BondInfo, StockInfo, BondDaily, PredictionRecord, UpdateLog
from scripts.fetch_cov_data import update_all as fetch_cov
from scripts.fetch_stock_info import fetch_all_stock_info
from scripts.fetch_stock_fundamentals import fetch_all_stock_fundamentals, update_missing_stock_info
from scripts.calculate_yield import update_yields
from scripts.continue_fetch import continue_fetch_history
from analysis.similarity import find_similar_bonds, get_confidence_level
from analysis.ml_model_v5 import train_ensemble_v5, predict_price_v5
from config import RATING_MAP
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def init():
    """初始化数据库"""
    print("="*60)
    print("初始化数据库...")
    print("="*60)
    init_db()
    print("初始化完成!")


def update_data():
    """更新所有数据"""
    print("="*60)
    print("开始更新数据...")
    print("="*60)
    
    print("\n[1/5] 获取可转债数据...")
    fetch_cov()
    
    print("\n[2/5] 获取正股基本信息...")
    fetch_all_stock_info()
    
    print("\n[3/5] 获取正股基本面数据...")
    fetch_all_stock_fundamentals()
    
    print("\n[4/5] 计算到期收益率...")
    update_yields()
    
    print("\n[5/5] 补全缺失基本面...")
    update_missing_stock_info()
    
    print("\n" + "="*60)
    print("数据更新完成!")
    print("="*60)


def show_bonds(limit=10):
    """显示可转债列表"""
    session = get_session()
    bonds = session.query(BondInfo).limit(limit).all()
    
    print(f"\n{'代码':<8} {'名称':<10} {'正股':<10} {'行业':<12} {'转股价值':<10} {'溢价率':<8} {'现价':<8} {'首日':<8}")
    print("-"*80)
    
    for bond in bonds:
        stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
        industry = stock.industry_sw_l1 if stock and stock.industry_sw_l1 else '-'
        
        latest = session.query(BondDaily).filter_by(bond_code=bond.bond_code).order_by(BondDaily.trade_date.desc()).first()
        price = f"{latest.close_price:.2f}" if latest and latest.close_price else '-'
        
        cv = f"{bond.conversion_value:.1f}" if bond.conversion_value else '-'
        prem = f"{bond.premium_rate:.1f}%" if bond.premium_rate else '-'
        first = f"{bond.first_open:.2f}" if bond.first_open else '-'
        
        print(f"{bond.bond_code:<8} {bond.bond_name:<10} {bond.stock_name:<10} {industry:<12} {cv:<10} {prem:<8} {price:<8} {first:<8}")
    
    session.close()


def predict_bond(bond_code, method='all'):
    """预测债券价格"""
    print("\n" + "="*70)
    print(f"可转债价格预测: {bond_code}")
    print("="*70)
    
    session = get_session()
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    
    if not bond:
        print(f"未找到债券 {bond_code}")
        session.close()
        return
    
    stock = session.query(StockInfo).filter_by(stock_code=bond.stock_code).first()
    
    print(f"\n【目标债券信息】")
    print(f"  债券名称: {bond.bond_name}")
    print(f"  正股名称: {bond.stock_name}")
    print(f"  所属行业: {stock.industry_sw_l1 if stock and stock.industry_sw_l1 else '未知'}")
    print(f"  信用评级: {bond.credit_rating or '未知'}")
    print(f"  转股价: {bond.conversion_price}元")
    print(f"  转股价值: {bond.conversion_value}元")
    print(f"  转股溢价率: {bond.premium_rate}%")
    print(f"  发行规模: {bond.issue_size}亿元")
    
    if bond.first_open:
        print(f"\n  ★ 实际首日开盘价: {bond.first_open}元")
    
    results = {}
    
    # 1. 相似度匹配预测
    if method in ['all', 'similarity']:
        print(f"\n{'='*70}")
        print(f"【方法1: 相似度匹配】")
        print(f"{'='*70}")
        
        similar = find_similar_bonds(bond_code, top_n=5)
        if similar:
            print(f"找到 {len(similar)} 个相似债券:")
            total_sim = 0
            weighted_price = 0
            for i, s in enumerate(similar, 1):
                if s['current_price']:
                    print(f"  {i}. {s['bond_name']} 相似度={s['similarity']:.1%} 现价={s['current_price']:.2f}")
                    total_sim += s['similarity']
                    weighted_price += s['first_open'] * s['similarity'] if s.get('first_open') else s['current_price'] * s['similarity']
            
            if total_sim > 0:
                sim_pred = weighted_price / total_sim
                results['similarity'] = round(sim_pred, 2)
                print(f"\n  相似度匹配预测: {results['similarity']}元")
        else:
            print("未找到相似债券")
    
    # 2. ML模型预测
    if method in ['all', 'ml']:
        print(f"\n{'='*70}")
        print(f"【方法2: 机器学习模型 v5】")
        print(f"{'='*70}")
        
        ml_result = predict_price_v5(bond_code)
        if ml_result:
            results['ml'] = ml_result['predicted_price']
            print(f"  预测价格: {ml_result['predicted_price']}元")
            print(f"  ├─ 线性回归: {ml_result.get('lr', 'N/A')}元")
            print(f"  ├─ K近邻: {ml_result.get('knn', 'N/A')}元")
            print(f"  └─ 梯度提升: {ml_result.get('gb', 'N/A')}元")
            print(f"  模型MAE: {ml_result['mae']:.2f}元 (PE来源: {ml_result.get('pe_source', 'N/A')})")
        else:
            print("ML模型不可用（数据不足）")
    
    # 综合预测
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"【综合预测】")
        print(f"{'='*70}")
        
        avg_price = sum(results.values()) / len(results)
        print(f"  相似度匹配: {results.get('similarity', 'N/A')}元")
        print(f"  机器学习: {results.get('ml', 'N/A')}元")
        print(f"  ─────────────────────────")
        print(f"  综合预测: {round(avg_price, 2)}元")
        
        # 保存预测
        record = PredictionRecord(
            bond_code=bond_code,
            bond_name=bond.bond_name,
            predict_date=datetime.now(),
            predicted_price=round(avg_price, 2),
            confidence_level='B',
            reference_bonds=str(list(results.keys())),
            status='pending'
        )
        if bond.first_open:
            error = abs(avg_price - bond.first_open) / bond.first_open * 100
            record.actual_price = bond.first_open
            record.error_rate = error
            record.status = 'confirmed'
        
        session.add(record)
        bond.predicted_price = avg_price
        session.commit()
        print(f"\n  (预测已保存)")
    elif len(results) == 1:
        final_price = list(results.values())[0]
        print(f"\n最终预测: {final_price}元")
        
        record = PredictionRecord(
            bond_code=bond_code,
            bond_name=bond.bond_name,
            predict_date=datetime.now(),
            predicted_price=final_price,
            confidence_level='C',
            status='pending'
        )
        if bond.first_open:
            record.actual_price = bond.first_open
            record.error_rate = abs(final_price - bond.first_open) / bond.first_open * 100
            record.status = 'confirmed'
        
        session.add(record)
        bond.predicted_price = final_price
        session.commit()
    
    if bond.first_open and results:
        print(f"\n★ 对比实际: {bond.first_open}元, 误差: {abs(round(avg_price,2) - bond.first_open):.2f}元 ({abs(round(avg_price,2) - bond.first_open)/bond.first_open*100:.1f}%)")
    
    session.close()


def train_ml():
    """训练机器学习模型"""
    print("\n" + "="*60)
    print("开始训练机器学习模型...")
    print("="*60)
    
    result = train_ensemble_v5()
    
    if result:
        print(f"\n✓ 模型训练完成!")
        print(f"  验证集MAE: {result['mae_ensemble']:.2f}元")
        print(f"  验证集R²: {result['r2_ensemble']:.4f}")
        print(f"  模型已保存到 models/ 目录")
    else:
        print("\n✗ 模型训练失败（数据不足）")


def backtest():
    """历史回测"""
    print("\n" + "="*60)
    print("开始历史回测...")
    print("="*60)
    
    evaluate_on_history()


def stats():
    """显示统计信息"""
    session = get_session()
    
    bond_count = session.query(BondInfo).count()
    stock_count = session.query(StockInfo).count()
    first_day_count = session.query(BondInfo).filter(BondInfo.first_open != None).count()
    
    # 基本面覆盖
    has_pe = session.query(StockInfo).filter(StockInfo.pe != None).count()
    has_pb = session.query(StockInfo).filter(StockInfo.pb != None).count()
    
    # 预测统计
    pred_total = session.query(PredictionRecord).count()
    pred_confirmed = session.query(PredictionRecord).filter_by(status='confirmed').count()
    
    print("\n" + "="*60)
    print("数据统计")
    print("="*60)
    print(f"可转债数量: {bond_count}")
    print(f"正股信息数量: {stock_count}")
    print(f"  - 有PE数据: {has_pe}")
    print(f"  - 有PB数据: {has_pb}")
    print(f"首日数据数量: {first_day_count}")
    print(f"预测记录: {pred_total} (已确认 {pred_confirmed})")
    
    # 预测误差统计
    records = session.query(PredictionRecord).filter(
        PredictionRecord.error_rate != None,
        PredictionRecord.error_rate < 50  # 排除异常值
    ).all()
    
    if records:
        errors = [r.error_rate for r in records]
        avg_error = sum(errors) / len(errors)
        within_5 = sum(1 for e in errors if e <= 5)
        within_10 = sum(1 for e in errors if e <= 10)
        
        print(f"\n预测误差统计:")
        print(f"  平均误差: {avg_error:.2f}%")
        print(f"  误差≤5%: {within_5}条 ({within_5/len(errors)*100:.0f}%)")
        print(f"  误差≤10%: {within_10}条 ({within_10/len(errors)*100:.0f}%)")
    
    session.close()


def menu():
    """主菜单"""
    while True:
        print("\n" + "="*70)
        print("可转债量化估值分析系统 v4.0")
        print("  特性: 正股基本面 + 集成ML模型")
        print("="*70)
        print("1. 初始化数据库")
        print("2. 更新所有数据")
        print("3. 查看可转债列表")
        print("4. 预测新债价格")
        print("5. 数据统计")
        print("6. 训练机器学习模型")
        print("7. 历史回测")
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
            predict_bond(code)
        elif choice == '5':
            stats()
        elif choice == '6':
            train_ml()
        elif choice == '7':
            backtest()
        elif choice == '0':
            print("再见!")
            break
        else:
            print("无效选择")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='可转债量化估值分析系统 v4.0')
    parser.add_argument('command', nargs='?', help='命令')
    parser.add_argument('--code', help='债券代码')
    parser.add_argument('--method', default='all', help='预测方法: all/similarity/ml')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        init()
    elif args.command == 'update':
        update_data()
    elif args.command == 'show':
        show_bonds()
    elif args.command == 'predict':
        if args.code:
            predict_bond(args.code, args.method)
        else:
            print("请指定债券代码: --code 110074")
    elif args.command == 'stats':
        stats()
    elif args.command == 'train':
        train_ml()
    elif args.command == 'backtest':
        backtest()
    elif args.command == 'fetch':
        # 补历史数据
        print("开始批量获取历史数据...")
        continue_fetch_history()
    else:
        menu()