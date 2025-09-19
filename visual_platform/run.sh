#!/bin/bash

# 癌症生存分析可视化平台启动脚本

echo "======================================"
echo "癌症生存分析模型对比可视化平台"
echo "======================================"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3环境"
    exit 1
fi

# 检查当前目录
if [ ! -f "streamlit_app.py" ]; then
    echo "错误: 请在visual_platform目录下运行此脚本"
    exit 1
fi

# 检查并安装依赖
echo "检查依赖包..."
if [ -f "requirements.txt" ]; then
    echo "安装依赖包..."
    pip3 install -r requirements.txt
else
    echo "警告: 未找到requirements.txt文件"
fi

# 检查数据文件
echo "检查数据文件..."
DATA_DIR="../data/processed"
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 未找到数据目录 $DATA_DIR"
    echo "请先运行数据预处理和模型训练notebooks"
    exit 1
fi

# 检查必要的数据文件
REQUIRED_FILES=(
    "modeling_data.csv"
    "train_data.csv" 
    "test_data.csv"
    "deepsurv_predictions.csv"
    "cox_predictions.csv"
    "rsf_predictions.csv"
    "comprehensive_evaluation_results.csv"
    "preprocessors.pkl"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "错误: 以下数据文件缺失:"
    printf '%s\n' "${MISSING_FILES[@]}"
    echo ""
    echo "请先运行以下notebooks生成数据:"
    echo "1. notebooks/1_data_preprocessing.ipynb"
    echo "2. notebooks/2_deepsurv_model.ipynb"
    echo "3. notebooks/3_traditional_models.ipynb" 
    echo "4. notebooks/4_model_evaluation.ipynb"
    exit 1
fi

echo "✅ 所有数据文件检查完成"

# 启动Streamlit应用
echo ""
echo "启动可视化平台..."
echo "请在浏览器中访问: http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止应用"
echo ""

streamlit run streamlit_app.py --server.port 8501 --server.address localhost