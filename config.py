"""
配置文件
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据库配置
DB_PATH = os.path.join(BASE_DIR, 'data', 'cb_data.db')
DB_URL = f'sqlite:///{DB_PATH}'

# 数据文件路径
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 可转债数据源
AKSHARE_COV_SPOT_FUNC = 'bond_zh_hs_cov_spot'  # 实时行情
AKSHARE_COV_INFO_FUNC = 'bond_zh_cov'          # 基本信息
AKSHARE_STOCK_PROFILE_FUNC = 'stock_profile_cninfo'  # 正股详细信息
AKSHARE_INDUSTRY_FUNC = 'stock_board_industry_name_em'  # 申万行业

# 更新频率（秒）
UPDATE_INTERVAL = 300  # 5分钟更新一次行情

# 相似度匹配配置（量化评估版权重）
SIMILARITY_WEIGHTS = {
    'industry': 0.20,          # 行业权重
    'business': 0.10,         # 主营业务权重
    'rating': 0.15,           # 信用评级权重
    'conversion_value': 0.30, # 转股价值权重（最重要）
    'premium_rate': 0.25,     # 转债溢价率权重
}

# 信用评级映射
RATING_MAP = {
    'AAA': 5, 'AA+': 4, 'AA': 3, 'AA-': 2, 'A+': 1, 'A': 0,
    'A-': -1, 'BBB+': -2, 'BBB': -3, 'BBB-': -4, 'BB': -5, 'B': -6
}