"""
数据库模型定义
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

from config import DB_URL, DB_PATH

Base = declarative_base()


class BondInfo(Base):
    """可转债基本信息"""
    __tablename__ = 'bond_info'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bond_code = Column(String(10), unique=True, index=True, comment='债券代码')
    bond_name = Column(String(50), comment='债券简称')
    stock_code = Column(String(10), comment='正股代码')
    stock_name = Column(String(50), comment='正股名称')
    issue_price = Column(Float, comment='发行价')
    issue_size = Column(Float, comment='发行规模(亿元)')
    listing_date = Column(DateTime, comment='上市日期')
    expiry_date = Column(DateTime, comment='到期日期')
    coupon_rate = Column(Float, comment='票面利率')
    credit_rating = Column(String(10), comment='信用评级')
    conversion_price = Column(Float, comment='转股价')
    conversion_value = Column(Float, comment='转股价值')
    premium_rate = Column(Float, comment='转股溢价率(%)')
    total_return = Column(Float, comment='本息合计')
    # 预测和实际价格
    predicted_price = Column(Float, comment='预测开盘价')
    first_date = Column(DateTime, comment='首日上市日期')
    first_open = Column(Float, comment='首日开盘价')
    first_close = Column(Float, comment='首日收盘价')
    first_high = Column(Float, comment='首日最高价')
    first_low = Column(Float, comment='首日最低价')
    error_rate = Column(Float, comment='预测误差率(%)')
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class BondDaily(Base):
    """可转债每日行情"""
    __tablename__ = 'bond_daily'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bond_code = Column(String(10), index=True, comment='债券代码')
    trade_date = Column(DateTime, index=True, comment='交易日期')
    open_price = Column(Float, comment='开盘价')
    high_price = Column(Float, comment='最高价')
    low_price = Column(Float, comment='最低价')
    close_price = Column(Float, comment='收盘价')
    volume = Column(Integer, comment='成交量')
    amount = Column(Float, comment='成交额')
    change_percent = Column(Float, comment='涨跌幅')
    created_at = Column(DateTime, default=datetime.now)


class StockInfo(Base):
    """正股信息"""
    __tablename__ = 'stock_info'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), unique=True, index=True, comment='股票代码')
    stock_name = Column(String(50), comment='股票名称')
    industry_sw_l1 = Column(String(50), comment='申万一级行业')
    industry_sw_l2 = Column(String(50), comment='申万二级行业')
    industry_sw_l3 = Column(String(50), comment='申万三级行业')
    industry_em = Column(String(50), comment='东财行业')
    business = Column(Text, comment='主营业务')
    pb = Column(Float, comment='市净率')
    pe = Column(Float, comment='市盈率')
    listing_date = Column(DateTime, comment='上市日期')
    total_market_cap = Column(Float, comment='总市值(亿元)')
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class PredictionRecord(Base):
    """预测记录表"""
    __tablename__ = 'prediction_record'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bond_code = Column(String(10), index=True, comment='债券代码')
    bond_name = Column(String(50), comment='债券名称')
    predict_date = Column(DateTime, comment='预测日期')
    predicted_price = Column(Float, comment='预测开盘价')
    confidence_level = Column(String(10), comment='可信度等级')
    reference_bonds = Column(Text, comment='参考债券')
    actual_price = Column(Float, comment='实际开盘价')
    actual_date = Column(DateTime, comment='实际上市日期')
    error_rate = Column(Float, comment='误差率(%)')
    status = Column(String(20), comment='状态(pending/confirmed)')
    created_at = Column(DateTime, default=datetime.now)


class UpdateLog(Base):
    """更新日志"""
    __tablename__ = 'update_log'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_name = Column(String(50), comment='任务名称')
    status = Column(String(20), comment='状态(success/failed)')
    message = Column(Text, comment='详细信息')
    records_count = Column(Integer, comment='处理记录数')
    created_at = Column(DateTime, default=datetime.now)


def get_engine():
    """获取数据库引擎"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    engine = create_engine(DB_URL, echo=False)
    return engine


def get_session():
    """获取数据库会话"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    """初始化数据库表"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"数据库初始化完成: {DB_PATH}")


if __name__ == '__main__':
    init_db()