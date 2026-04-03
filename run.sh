#!/bin/bash
# 可转债预测工具 - Linux启动器
# 一键启动GUI应用

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  可转债价格预测工具 v1.0"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到python3，请先安装Python 3.10+"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import tkinter; from db.models import *; from analysis.ml_model_v6 import *" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ 缺少依赖，正在安装..."
    pip3 install akshare numpy sqlalchemy --break-system-packages 2>/dev/null
fi

echo "✅ 依赖就绪"
echo ""
echo "启动中..."
python3 desktop_app.py