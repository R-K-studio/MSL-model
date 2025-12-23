# 手语识别后端API部署指南

本文档提供手语识别后端API的完整部署指南，包括所需文件、安装步骤和配置说明。

## 📋 目录

1. [系统要求](#系统要求)
2. [文件清单](#文件清单)
3. [文件说明](#文件说明)
4. [安装步骤](#安装步骤)
5. [配置说明](#配置说明)
6. [启动服务](#启动服务)
7. [测试验证](#测试验证)
8. [生产环境部署](#生产环境部署)
9. [常见问题](#常见问题)

---

## 系统要求

### 硬件要求
- **CPU**: 4核心以上（推荐8核心+）
- **内存**: 8GB以上（推荐16GB+）
- **GPU**: 可选，支持CUDA的GPU可加速推理
- **存储**: 至少2GB可用空间

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+ / CentOS 7+) 或 macOS / Windows
- **Python**: 3.8 - 3.12
- **CUDA**: 可选，如果使用GPU需要CUDA 11.0+

---

## 文件清单

### 必需文件

部署后端API需要以下文件：

```
项目目录/
├── backend_api.py              # 主API服务器文件 ⭐ 必需
├── sign_language_recognition.py # 模型定义和工具函数 ⭐ 必需
├── sign_language_model.pth     # 训练好的模型文件 ⭐ 必需
├── requirements_api.txt        # Python依赖列表 ⭐ 必需
└── start_backend.sh            # 启动脚本（可选，但推荐）
```

### 可选文件

```
├── test_backend.py             # API测试脚本（用于验证）
├── backend_api_usage.md        # API使用文档
└── README.md                   # 项目说明文档
```

---

## 文件说明

### 1. `backend_api.py` ⭐ 核心文件
- **作用**: Flask API服务器主程序
- **功能**: 
  - 加载训练好的模型
  - 提供REST API接口
  - 处理视频和帧序列
  - 返回JSON格式的预测结果
- **大小**: ~15KB
- **必需**: ✅ 是

### 2. `sign_language_recognition.py` ⭐ 核心文件
- **作用**: 包含模型定义和MediaPipe处理函数
- **功能**:
  - `SignLanguageLSTM`: LSTM模型定义
  - `MediaPipeProcessor`: MediaPipe关键点提取
  - `CONFIG`: 配置参数
- **大小**: ~20KB
- **必需**: ✅ 是

### 3. `sign_language_model.pth` ⭐ 核心文件
- **作用**: 训练好的PyTorch模型文件
- **内容**:
  - 模型权重
  - 手势类别列表
  - 配置参数
- **大小**: ~700KB（取决于类别数量）
- **必需**: ✅ 是
- **获取方式**: 运行 `sign_language_recognition.py` 训练生成

### 4. `requirements_api.txt` ⭐ 必需
- **作用**: Python依赖包列表
- **内容**:
  ```
  flask>=2.0.0
  flask-cors>=3.0.0
  torch>=1.9.0
  opencv-python>=4.5.0
  mediapipe==0.10.13
  numpy>=1.21.0
  ```
- **必需**: ✅ 是

### 5. `start_backend.sh` 推荐
- **作用**: 便捷启动脚本
- **功能**: 自动检查依赖和模型文件
- **必需**: ❌ 否（但推荐使用）

---

## 安装步骤

### 步骤 1: 准备项目目录

```bash
# 创建项目目录
mkdir -p /path/to/sign_language_api
cd /path/to/sign_language_api

# 或者使用现有目录
cd /root/autodl-nus
```

### 步骤 2: 复制必需文件

确保以下文件在同一目录下：

```bash
# 检查必需文件是否存在
ls -lh backend_api.py
ls -lh sign_language_recognition.py
ls -lh sign_language_model.pth
ls -lh requirements_api.txt
```

如果文件不在同一目录，需要复制：

```bash
# 示例：从训练目录复制到部署目录
cp /root/autodl-nus/backend_api.py /path/to/deployment/
cp /root/autodl-nus/sign_language_recognition.py /path/to/deployment/
cp /root/autodl-nus/sign_language_model.pth /path/to/deployment/
cp /root/autodl-nus/requirements_api.txt /path/to/deployment/
```

### 步骤 3: 创建Python虚拟环境（推荐）

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 步骤 4: 安装Python依赖

```bash
# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements_api.txt
```

**安装时间**: 约5-10分钟（取决于网络速度）

**验证安装**:
```bash
python3 -c "import flask, torch, cv2, mediapipe; print('所有依赖已安装')"
```

### 步骤 5: 验证模型文件

```bash
# 检查模型文件
python3 << 'EOF'
import torch
import os

model_path = 'sign_language_model.pth'
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"✓ 模型文件存在")
    print(f"✓ 手势类别数: {len(checkpoint.get('gestures', []))}")
    print(f"✓ 配置: {checkpoint.get('config', {})}")
else:
    print("✗ 模型文件不存在！")
    print("请先运行训练脚本生成模型")
EOF
```

---

## 配置说明

### 模型路径配置

在 `backend_api.py` 的 `main()` 函数中，可以修改模型路径：

```python
# 默认路径
model_path = CONFIG.get('model_save_path', '/root/autodl-nus/sign_language_model.pth')

# 如果模型文件在其他位置，修改为：
model_path = '/path/to/your/sign_language_model.pth'
```

### 服务器配置

在 `backend_api.py` 的最后部分：

```python
# 开发环境（默认）
app.run(host='0.0.0.0', port=5000, debug=True)

# 生产环境（推荐）
app.run(host='0.0.0.0', port=5000, debug=False)
```

### 端口配置

如果需要修改端口：

```python
app.run(host='0.0.0.0', port=8080, debug=False)  # 改为8080端口
```

### CORS配置

如果需要限制跨域访问：

```python
from flask_cors import CORS

# 允许所有来源（默认）
CORS(app)

# 或限制特定来源
CORS(app, origins=["http://localhost:3000", "https://yourdomain.com"])
```

---

## 启动服务

### 方法 1: 使用启动脚本（推荐）

```bash
# 赋予执行权限
chmod +x start_backend.sh

# 启动服务
./start_backend.sh
```

### 方法 2: 直接运行Python

```bash
# 激活虚拟环境（如果使用）
source venv/bin/activate

# 运行API服务器
python3 backend_api.py
```

### 方法 3: 使用systemd（Linux生产环境）

创建服务文件 `/etc/systemd/system/sign-language-api.service`:

```ini
[Unit]
Description=Sign Language Recognition API
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/sign_language_api
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python backend_api.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable sign-language-api
sudo systemctl start sign-language-api
sudo systemctl status sign-language-api
```

### 方法 4: 使用Docker（可选）

创建 `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY backend_api.py .
COPY sign_language_recognition.py .
COPY sign_language_model.pth .

EXPOSE 5000

CMD ["python", "backend_api.py"]
```

构建和运行：
```bash
docker build -t sign-language-api .
docker run -p 5000:5000 sign-language-api
```

---

## 测试验证

### 1. 健康检查

```bash
# 使用curl
curl http://localhost:5000/health

# 预期响应
{
  "status": "healthy",
  "model_loaded": true,
  "num_gestures": 90
}
```

### 2. 使用测试脚本

```bash
# 运行测试脚本
python3 test_backend.py
```

### 3. 手动测试视频预测

```bash
# 使用curl上传视频
curl -X POST \
  -F "video=@/path/to/test_video.mp4" \
  http://localhost:5000/predict

# 使用Python requests
python3 << 'EOF'
import requests

with open('test_video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'video': f}
    )
    print(response.json())
EOF
```

### 4. 测试实时帧处理

```python
import requests
import base64
import cv2

# 读取视频帧
cap = cv2.VideoCapture('test_video.mp4')
frames = []
for i in range(30):
    ret, frame = cap.read()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame)
        frames.append(base64.b64encode(buffer).decode())
cap.release()

# 发送请求
response = requests.post(
    'http://localhost:5000/predict_camera',
    json={'frames': frames}
)
print(response.json())
```

---

## 生产环境部署

### 使用Gunicorn（推荐）

安装Gunicorn:
```bash
pip install gunicorn
```

启动服务:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 backend_api:app
```

参数说明:
- `-w 4`: 4个工作进程
- `-b 0.0.0.0:5000`: 绑定地址和端口
- `backend_api:app`: Flask应用对象

### 使用Nginx反向代理

Nginx配置示例 (`/etc/nginx/sites-available/sign-language-api`):

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 增加超时时间（视频处理可能需要较长时间）
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        
        # 增加请求体大小限制
        client_max_body_size 100M;
    }
}
```

启用配置:
```bash
sudo ln -s /etc/nginx/sites-available/sign-language-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 使用HTTPS（SSL证书）

使用Let's Encrypt:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 常见问题

### Q1: 模型文件不存在

**错误信息**:
```
错误: 模型文件不存在: /root/autodl-nus/sign_language_model.pth
```

**解决方法**:
1. 检查模型文件路径是否正确
2. 如果模型在其他位置，修改 `backend_api.py` 中的路径
3. 如果模型未训练，先运行 `sign_language_recognition.py` 训练模型

### Q2: 依赖安装失败

**错误信息**:
```
ERROR: Could not find a version that satisfies the requirement mediapipe==0.10.13
```

**解决方法**:
```bash
# 升级pip
pip install --upgrade pip

# 尝试安装其他版本
pip install mediapipe==0.10.13 --no-cache-dir

# 或使用conda
conda install -c conda-forge mediapipe
```

### Q3: MediaPipe导入错误

**错误信息**:
```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

**解决方法**:
```bash
# 卸载并重新安装指定版本
pip uninstall mediapipe
pip install mediapipe==0.10.13
```

### Q4: 端口已被占用

**错误信息**:
```
OSError: [Errno 98] Address already in use
```

**解决方法**:
```bash
# 查找占用端口的进程
lsof -i :5000
# 或
netstat -tulpn | grep 5000

# 杀死进程
kill -9 <PID>

# 或修改端口
# 在backend_api.py中修改 port=5000 为其他端口
```

### Q5: CORS错误

**错误信息**:
```
Access to fetch at 'http://localhost:5000/predict' from origin 'http://localhost:3000' has been blocked by CORS policy
```

**解决方法**:
确保 `flask-cors` 已安装并正确配置：
```python
from flask_cors import CORS
CORS(app)  # 允许所有来源
```

### Q6: 内存不足

**错误信息**:
```
RuntimeError: CUDA out of memory
```

**解决方法**:
1. 减少并发请求数量
2. 使用CPU模式（在代码中设置 `device='cpu'`）
3. 增加系统内存或使用GPU

### Q7: 视频处理超时

**解决方法**:
1. 增加Nginx超时时间（见Nginx配置）
2. 使用异步处理（需要修改代码）
3. 限制视频文件大小

### Q8: 模型加载慢

**解决方法**:
1. 使用GPU加速（如果可用）
2. 预加载模型（在应用启动时加载）
3. 使用模型量化减少模型大小

---

## 文件检查清单

部署前请确认：

- [ ] `backend_api.py` 存在且可读
- [ ] `sign_language_recognition.py` 存在且可读
- [ ] `sign_language_model.pth` 存在且大小正常（>100KB）
- [ ] `requirements_api.txt` 存在
- [ ] Python 3.8+ 已安装
- [ ] 所有依赖已安装（运行 `pip list` 检查）
- [ ] 端口5000未被占用
- [ ] 防火墙规则允许访问（如需要）

---

## 快速部署命令总结

```bash
# 1. 进入项目目录
cd /path/to/sign_language_api

# 2. 创建虚拟环境（可选但推荐）
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements_api.txt

# 4. 验证模型文件
python3 -c "import torch; torch.load('sign_language_model.pth', map_location='cpu')"

# 5. 启动服务
python3 backend_api.py

# 6. 测试（新终端）
curl http://localhost:5000/health
```

---

## 技术支持

如遇到问题，请检查：
1. Python版本是否符合要求
2. 所有依赖是否正确安装
3. 模型文件是否完整
4. 端口是否被占用
5. 查看服务器日志输出

---

**部署完成后，API将在 `http://0.0.0.0:5000` 提供服务**

