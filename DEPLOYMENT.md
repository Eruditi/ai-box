# 部署和GitHub推送指南

## 项目概述

AI边缘计算摄像头分析盒子系统，包含69种智能分析算法，支持从嵌入式设备到高性能服务器的全系列硬件配置。

## 本地部署

### 1. 安装依赖

```bash
# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装系统依赖
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

### 2. 配置文件

```bash
# 复制配置文件
cp config/settings.yaml config/settings.yaml.local

# 根据实际硬件修改配置
vim config/settings.yaml.local
```

### 3. 启动系统

```bash
# 直接运行
python src/main.py

# 或作为系统服务运行
sudo cp scripts/ai-box.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-box
sudo systemctl start ai-box
```

## 推送到GitHub

由于需要GitHub身份验证，您需要手动完成推送操作：

### 1. 克隆仓库

```bash
# 先克隆您的GitHub仓库
git clone https://github.com/Eruditi/ai-box.git ai-box-github
cd ai-box-github
```

### 2. 复制文件

```bash
# 复制所有文件（除了git目录）
cp -r /workspace/ai-box/* .
cp -r /workspace/ai-box/.gitignore .
```

### 3. 提交并推送

```bash
# 初始化git（如果需要）
git init

# 添加远程仓库
git remote add origin https://github.com/Eruditi/ai-box.git

# 设置用户信息
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 添加并提交
git add .
git commit -m "Initial commit: AI边缘计算摄像头分析盒子系统"

# 推送
git push -u origin master
```

## 系统特性

- **69种智能算法**：覆盖传统安防、智能识别、教育应用、生态保护、高空监测等领域
- **高性能设计**：支持336路视频接入，批处理能力
- **混合摄像头管理**：同时支持无人机摄像头和普通摄像头
- **智能算法选择**：根据摄像头类型自动选择合适的算法
- **RAID存储管理**：支持多种RAID级别和数据分层
- **网络接口管理**：支持多址设定、负载均衡、主备模式
- **GB28181协议支持**：支持设备注册和级联

## 技术栈

- Python 3.8+
- OpenCV
- TensorFlow / PyTorch / ONNX Runtime
- TensorRT (高性能推理加速)
- FastAPI (Web界面)
- Docker (容器化部署)
- mdadm (RAID管理)
- netplan (网络配置)

## 文档索引

- [README.md](README.md) - 项目概述
- [算法说明文档.md](算法说明文档.md) - 69种算法详细说明
- [部署使用手册.md](部署使用手册.md) - 标准版部署指南
- [高性能硬件部署手册.md](高性能硬件部署手册.md) - 高性能版部署指南
