"""
保存预测结果到数据库
"""
from db.models import get_session, BondInfo, PredictionRecord
from datetime import datetime


def save_prediction(bond_code, bond_name, predicted_price, confidence_level, reference_bonds, avg_similarity):
    """保存预测结果"""
    session = get_session()
    
    # 检查是否已存在预测记录
    existing = session.query(PredictionRecord).filter_by(bond_code=bond_code).first()
    
    if existing:
        # 更新
        existing.predicted_price = predicted_price
        existing.confidence_level = confidence_level
        existing.reference_bonds = str(reference_bonds)
        existing.status = 'pending'
    else:
        # 新增
        record = PredictionRecord(
            bond_code=bond_code,
            bond_name=bond_name,
            predict_date=datetime.now(),
            predicted_price=predicted_price,
            confidence_level=confidence_level,
            reference_bonds=str(reference_bonds),
            status='pending'
        )
        session.add(record)
    
    session.commit()
    session.close()


def update_actual_price(bond_code, actual_price, actual_date):
    """更新实际首日价格"""
    session = get_session()
    
    # 更新预测记录
    record = session.query(PredictionRecord).filter_by(bond_code=bond_code).first()
    if record:
        record.actual_price = actual_price
        record.actual_date = actual_date
        if record.predicted_price:
            record.error_rate = abs(record.predicted_price - actual_price) / actual_price * 100
        record.status = 'confirmed'
    
    # 同时更新bond_info表
    bond = session.query(BondInfo).filter_by(bond_code=bond_code).first()
    if bond:
        bond.first_open = actual_price
        bond.first_date = actual_date
        bond.actual_price = actual_price
        if bond.predicted_price:
            bond.error_rate = abs(bond.predicted_price - actual_price) / actual_price * 100
    
    session.commit()
    session.close()


def get_prediction_statistics():
    """获取预测统计"""
    session = get_session()
    
    total = session.query(PredictionRecord).count()
    confirmed = session.query(PredictionRecord).filter_by(status='confirmed').count()
    pending = session.query(PredictionRecord).filter_by(status='pending').count()
    
    # 计算平均误差
    records = session.query(PredictionRecord).filter(
        PredictionRecord.error_rate != None
    ).all()
    
    if records:
        avg_error = sum(r.error_rate for r in records) / len(records)
    else:
        avg_error = 0
    
    print(f"\n=== 预测统计 ===")
    print(f"总预测数: {total}")
    print(f"已确认: {confirmed}")
    print(f"待确认: {pending}")
    print(f"平均误差: {avg_error:.2f}%")
    
    # 显示误差分布
    if records:
        print(f"\n误差分布:")
        excellent = len([r for r in records if r.error_rate < 5])
        good = len([r for r in records if 5 <= r.error_rate < 10])
        fair = len([r for r in records if 10 <= r.error_rate < 20])
        poor = len([r for r in records if r.error_rate >= 20])
        print(f"  优秀 (<5%): {excellent}")
        print(f"  良好 (5-10%): {good}")
        print(f"  一般 (10-20%): {fair}")
        print(f"  较差 (>20%): {poor}")
    
    session.close()


if __name__ == '__main__':
    get_prediction_statistics()