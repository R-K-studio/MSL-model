#!/bin/bash
# 启动手语识别后端API服务器

echo "检查依赖..."
pip install -q flask flask-cors 2>/dev/null || echo "请先安装依赖: pip install -r requirements_api.txt"

echo "检查模型文件..."
if [ ! -f "/root/autodl-nus/sign_language_model.pth" ]; then
    echo "警告: 模型文件不存在，请先训练模型"
    echo "运行: python sign_language_recognition.py"
    exit 1
fi

echo "启动API服务器..."
python3 backend_api.py

