"""
定时任务脚本（无需额外依赖）
使用系统cron或手动运行
"""
import time
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.models import get_session, BondInfo, UpdateLog
from scripts.fetch_cov_data import fetch_cov_spot, fetch_cov_info, save_bond_info, save_bond_daily
from analysis.ml_model import train_model


def job_update_data():
    """数据更新任务"""
    print(f"\n{'='*50}")
    print(f"任务: 数据更新 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*50}")
    
    try:
        df_spot = fetch_cov_spot()
        save_bond_daily(df_spot)
        df_info = fetch_cov_info()
        save_bond_info(df_info)
        print("✓ 数据更新完成")
        return True
    except Exception as e:
        print(f"✗ 数据更新失败: {e}")
        return False


def job_retrain_model():
    """重新训练模型任务"""
    print(f"\n{'='*50}")
    print(f"任务: 训练模型 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*50}")
    
    try:
        result = train_model()
        if result:
            print(f"✓ 模型训练完成! MAE: {result['test_mae']:.2f}元")
            return True
        else:
            print("✗ 模型训练失败")
            return False
    except Exception as e:
        print(f"✗ 模型训练失败: {e}")
        return False


def job_check_new_bonds():
    """检查新债任务"""
    print(f"\n{'='*50}")
    print(f"任务: 检查新债 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*50}")
    
    session = get_session()
    
    thirty_days_ago = datetime.now() - timedelta(days=30)
    new_bonds = session.query(BondInfo).filter(
        BondInfo.listing_date >= thirty_days_ago,
        BondInfo.predicted_price == None
    ).all()
    
    if new_bonds:
        print(f"发现 {len(new_bonds)} 只新债未预测:")
        for bond in new_bonds:
            print(f"  - {bond.bond_name} ({bond.bond_code})")
    else:
        print("✓ 没有发现需要预测的新债")
    
    session.close()
    return True


def job_statistics():
    """统计任务"""
    print(f"\n{'='*50}")
    print(f"任务: 统计 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*50}")
    
    session = get_session()
    
    total = session.query(BondInfo).count()
    with_first = session.query(BondInfo).filter(BondInfo.first_open != None).count()
    with_pred = session.query(BondInfo).filter(BondInfo.predicted_price != None).count()
    
    print(f"可转债总数: {total}")
    print(f"首日数据: {with_first} ({with_first/total*100:.1f}%)")
    print(f"已预测: {with_pred}")
    
    session.close()
    return True


def run_all_jobs():
    """运行所有任务"""
    print("="*60)
    print("可转债分析系统 - 批量任务执行")
    print("="*60)
    
    results = []
    
    results.append(("数据更新", job_update_data()))
    results.append(("训练模型", job_retrain_model()))
    results.append(("检查新债", job_check_new_bonds()))
    results.append(("统计", job_statistics()))
    
    print("\n" + "="*60)
    print("任务执行结果")
    print("="*60)
    for name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def run_scheduler_demo():
    """模拟定时任务（每隔一段时间执行）"""
    import argparse
    
    parser = argparse.ArgumentParser(description='可转债定时任务')
    parser.add_argument('--interval', type=int, default=3600, help='执行间隔(秒)，默认1小时')
    parser.add_argument('--count', type=int, default=3, help='执行次数')
    args = parser.parse_args()
    
    print(f"启动定时任务 (间隔:{args.interval}秒, 次数:{args.count})")
    print("按 Ctrl+C 停止\n")
    
    for i in range(args.count):
        print(f"\n=== 第 {i+1}/{args.count} 次执行 ===")
        run_all_jobs()
        
        if i < args.count - 1:
            print(f"\n等待 {args.interval} 秒...")
            time.sleep(args.interval)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true', help='只运行一次所有任务')
    parser.add_argument('--interval', type=int, default=60, help='测试间隔秒数')
    parser.add_argument('--count', type=int, default=1, help='执行次数')
    args = parser.parse_args()
    
    if args.once:
        run_all_jobs()
    else:
        run_scheduler_demo()