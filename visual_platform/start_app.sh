#!/bin/bash
# 智能癌症生存预测平台启动脚本

echo "🔬 启动癌症生存分析智能预测平台..."
echo "======================================"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装Python3"
    exit 1
fi

# 检查Streamlit
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "❌ Streamlit 未安装，正在安装..."
    pip3 install streamlit
fi

# 检查其他依赖
echo "📦 检查依赖包..."
python3 -c "
import subprocess
import sys

packages = ['pandas', 'numpy', 'plotly', 'matplotlib', 'seaborn', 'lifelines', 'scikit-survival']
missing = []

for package in packages:
    try:
        __import__(package)
        print(f'✅ {package}')
    except ImportError:
        missing.append(package)
        print(f'❌ {package} (缺失)')

if missing:
    print(f'\n正在安装缺失的包: {\" \".join(missing)}')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
"

echo ""
echo "🚀 启动Streamlit应用..."
echo "浏览器将自动打开: http://localhost:8501"
echo ""
echo "功能说明:"
echo "📊 数据概览 - 查看数据集基本信息"
echo "🔬 模型对比 - 比较不同模型性能"
echo "📈 生存分析 - 生存曲线分析"
echo "⚠️ 风险分析 - 风险分层分析"
echo "🎯 交互预测 - 智能风险预测 (新功能!)"
echo ""

# 启动Streamlit
streamlit run streamlit_app.py --server.port 8501