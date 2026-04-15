#!/bin/bash
# 下载预训练模型脚本

set -e

MODEL_DIR="/opt/ai-box/models"

echo "创建模型目录..."
mkdir -p $MODEL_DIR
cd $MODEL_DIR

echo "下载MobileNet SSD模型..."

# 下载MobileNet-SSD prototxt文件
if [ ! -f "MobileNetSSD_deploy.prototxt" ]; then
    wget -q "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt" -O MobileNetSSD_deploy.prototxt
    echo "✓ 下载完成: MobileNetSSD_deploy.prototxt"
else
    echo "- 模型已存在: MobileNetSSD_deploy.prototxt"
fi

# 下载MobileNet-SSD caffemodel文件
if [ ! -f "MobileNetSSD_deploy.caffemodel" ]; then
    echo "正在下载模型文件(约23MB)..."
    wget -q "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel" -O MobileNetSSD_deploy.caffemodel
    echo "✓ 下载完成: MobileNetSSD_deploy.caffemodel"
else
    echo "- 模型已存在: MobileNetSSD_deploy.caffemodel"
fi

echo ""
echo "=========================================="
echo "模型下载完成！"
echo "模型目录: $MODEL_DIR"
echo "=========================================="
