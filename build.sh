#!/bin/bash
# 打包脚本 - 可转债预测工具 Linux版
# 生成可执行文件，无需Python环境即可运行

APP_NAME="CBpredictor"
DIST_DIR="dist"
BUILD_DIR="build"

echo "开始打包 $APP_NAME ..."

# 检查pyinstaller
if ! command -v pyinstaller &> /dev/null; then
    echo "安装 pyinstaller..."
    pip3 install pyinstaller
fi

# 清理旧文件
rm -rf $BUILD_DIR $DIST_DIR

# 使用spec文件打包
pyinstaller $APP_NAME.spec --noconfirm --onedir

echo ""
echo "=========================="
echo "打包完成!"
echo "=========================="
echo "输出目录: $DIST_DIR/$APP_NAME"
echo ""
echo "运行方式:"
echo "  cd $DIST_DIR/$APP_NAME"
echo "  ./$APP_NAME"
echo ""
echo "分享给其他Linux用户:"
echo "  将整个 $APP_NAME 目录打包发送即可"