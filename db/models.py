"""
数据库模型定义 - v2
扩展正股基本面字段
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
    
    # 扩展字段
    par_value = Column(Float, default=100, comment='面值')
    market_value = Column(Float, comment='市值(亿元)')
    
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
    """正股信息（含基本面）"""
    __tablename__ = 'stock_info'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), unique=True, index=True, comment='股票代码')
    stock_name = Column(String(50), comment='股票名称')
    
    # 行业分类
    industry_sw_l1 = Column(String(50), comment='申万一级行业')
    industry_sw_l2 = Column(String(50), comment='申万二级行业')
    industry_sw_l3 = Column(String(50), comment='申万三级行业')
    industry_em = Column(String(50), comment='东财行业')
    
    # 基本面指标
    pe = Column(Float, comment='市盈率(动态)')
    pb = Column(Float, comment='市净率')
    roe = Column(Float, comment='净资产收益率(%)')
    revenue_growth = Column(Float, comment='营收同比增长率(%)')
    profit_growth = Column(Float, comment='净利润同比增长率(%)')
    gross_margin = Column(Float, comment='销售毛利率(%)')
    debt_ratio = Column(Float, comment='资产负债率(%)')
    
    # 市值
    total_market_cap = Column(Float, comment='总市值(亿元)')
    float_market_cap = Column(Float, comment='流通市值(亿元)')
    total_shares = Column(Float, comment='总股本(亿股)')
    float_shares = Column(Float, comment='流通股本(亿股)')
    
    # 财务数据（万元）
    total_revenue = Column(Float, comment='营业总收入(万元)')
    net_profit = Column(Float, comment='净利润(万元)')
    total_assets = Column(Float, comment='资产总计(万元)')
    total_debt = Column(Float, comment='负债合计(万元)')
    
    # 上市信息
    listing_date = Column(DateTime, comment='上市日期')
    listing_days = Column(Integer, comment='上市天数')
    
    # 主营业务
    business = Column(Text, comment='主营业务')
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class PredictionRecord(Base):
    """预测记录表"""
    __tablename__ = 'prediction_record'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bond_code = Column(String(10), index=True, comment='债券代码')
    bond_name = Column(String(50), comment='债券名称')
    predict_date = Column(DateTime, comment='预测日期')
    
    # 多种预测方法
    predicted_price = Column(Float, comment='综合预测开盘价')
    price_sim = Column(Float, comment='相似度匹配预测价')
    price_ml = Column(Float, comment='机器学习预测价')
    
    confidence_level = Column(String(10), comment='可信度等级')
    reference_bonds = Column(Text, comment='参考债券(JSON)')
    method = Column(String(20), comment='预测方法')
    
    # 结果
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
    status = Column(String(20), comment='状态(success/failed/partial)')
    message = Column(Text, comment='详细信息')
    records_count = Column(Integer, comment='处理记录数')
    created_at = Column(DateTime, default=datetime.now)


class ModelMeta(Base):
    """模型元数据表"""
    __tablename__ = 'model_meta'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), comment='模型名称')
    version = Column(String(20), comment='模型版本')
    train_date = Column(DateTime, comment='训练日期')
    train_size = Column(Integer, comment='训练样本数')
    val_mae = Column(Float, comment='验证集MAE')
    val_r2 = Column(Float, comment='验证集R²')
    feature_names = Column(Text, comment='特征名称(JSON)')
    weights = Column(Text, comment='模型权重(JSON)')
    model_config = Column(Text, comment='模型配置(JSON)')
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


def migrate_db():
    """数据库迁移（添加新字段）"""
    from sqlalchemy import inspect
    
    engine = get_engine()
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    if 'stock_info' in existing_tables:
        # 检查stock_info表的新字段
        columns = [c['name'] for c in inspector.get_columns('stock_info')]
        
        # 需要添加的字段
        new_columns = {
            'pe': Float,
            'pb': Float,
            'roe': Float,
            'revenue_growth': Float,
            'profit_growth': Float,
            'gross_margin': Float,
            'debt_ratio': Float,
            'total_market_cap': Float,
            'float_market_cap': Float,
            'total_shares': Float,
            'float_shares': Float,
            'total_revenue': Float,
            'net_profit': Float,
            'total_assets': Float,
            'total_debt': Float,
            'listing_days': Integer,
        }
        
        from sqlalchemy import text
        with engine.connect() as conn:
            for col_name, col_type in new_columns.items():
                if col_name not in columns:
                    try:
                        conn.execute(text(f"ALTER TABLE stock_info ADD COLUMN {col_name} {col_type}"))
                        print(f"  + 添加字段: {col_name}")
                    except:
                        pass
            conn.commit()
    
    if 'prediction_record' not in existing_tables:
        Base.metadata.create_all(engine)
        print("  + 创建 prediction_record 表")
    
    if 'model_meta' not in existing_tables:
        Base.metadata.create_all(engine)
        print("  + 创建 model_meta 表")
    
    print("数据库迁移完成")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'migrate':
        migrate_db()
    else:
        init_db()