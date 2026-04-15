#!/bin/bash
# AI盒子系统安装脚本

set -e

echo "=========================================="
echo "AI边缘计算摄像头分析盒子 - 安装脚本"
echo "=========================================="

INSTALL_DIR="/opt/ai-box"
CONFIG_DIR="/etc/ai-box"
DATA_DIR="/var/lib/ai-box"
LOG_DIR="/var/log/ai-box"

echo "创建目录..."
sudo mkdir -p $INSTALL_DIR
sudo mkdir -p $CONFIG_DIR
sudo mkdir -p $DATA_DIR/data
sudo mkdir -p $LOG_DIR
sudo mkdir -p $INSTALL_DIR/models

echo "复制文件..."
sudo cp -r ./* $INSTALL_DIR/
sudo cp config/settings.yaml $CONFIG_DIR/settings.yaml

echo "安装依赖..."
cd $INSTALL_DIR
sudo pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

echo "设置权限..."
sudo chmod +x $INSTALL_DIR/scripts/*.sh
sudo chown -R $USER:$USER $DATA_DIR
sudo chown -R $USER:$USER $LOG_DIR

echo "安装系统服务..."
sudo cp $INSTALL_DIR/scripts/ai-box.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "启用开机自启..."
sudo systemctl enable ai-box.service

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "启动服务: sudo systemctl start ai-box"
echo "查看状态: sudo systemctl status ai-box"
echo "查看日志: sudo journalctl -u ai-box -f"
echo "Web界面: http://<设备IP>:8080"
echo ""
