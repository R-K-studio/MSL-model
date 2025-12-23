# 手语识别后端API使用说明

## 安装依赖

```bash
pip install -r requirements_api.txt
```

## 启动服务器

```bash
python backend_api.py
```

服务器将在 `http://0.0.0.0:5000` 启动

## API端点

### 1. 健康检查
```
GET /health
```

返回：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_gestures": 90
}
```

### 2. 视频预测 (`/predict`)
上传视频文件，返回完整的手势序列

**请求:**
```
POST /predict
Content-Type: multipart/form-data

video: <视频文件>
```

**响应:**
```json
{
  "tokens": [
    {
      "gloss": "SAYA",
      "translation": "I",
      "confidence": 0.93,
      "temporal": false,
      "start_frame": 10,
      "end_frame": 20,
      "fps": 30.0
    },
    {
      "gloss": "LOVE",
      "translation": "love",
      "confidence": 0.87,
      "temporal": false,
      "start_frame": 30,
      "end_frame": 50,
      "fps": 30.0
    }
  ],
  "sentence": "I love learning"
}
```

### 3. 实时帧处理 (`/predict_camera`)
接收帧序列，返回检测到的手势（如果有）

**请求:**
```
POST /predict_camera
Content-Type: application/json

{
  "frames": [
    "<base64编码的图像1>",
    "<base64编码的图像2>",
    ...
  ]
}
```

**响应（检测到手势）:**
```json
{
  "token": {
    "gloss": "SAYA",
    "translation": "I",
    "confidence": 0.93,
    "temporal": false,
    "start_frame": 5,
    "end_frame": 18,
    "fps": 30.0
  }
}
```

**响应（未检测到）:**
```json
{
  "token": null
}
```

### 4. 帧序列处理（替代接口）(`/predict_frames`)
使用表单数据上传帧序列

**请求:**
```
POST /predict_frames
Content-Type: multipart/form-data

frames: <JSON数组，包含base64编码的图像>
```

## 使用示例

### Python客户端示例

```python
import requests
import base64
import cv2

# 1. 上传视频预测
with open('test_video.mp4', 'rb') as f:
    files = {'video': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    result = response.json()
    print(result)

# 2. 实时帧处理
frames_bgr = []  # 你的OpenCV帧列表
frames_base64 = []

for frame in frames_bgr:
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    frames_base64.append(frame_base64)

response = requests.post(
    'http://localhost:5000/predict_camera',
    json={'frames': frames_base64}
)
result = response.json()
if result['token']:
    print(f"检测到手势: {result['token']['gloss']}")
else:
    print("未检测到手势")
```

### JavaScript/前端示例

```javascript
// 1. 上传视频
const formData = new FormData();
formData.append('video', videoFile);

fetch('http://localhost:5000/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Tokens:', data.tokens);
  console.log('Sentence:', data.sentence);
});

// 2. 实时帧处理
const frames = []; // 你的帧数组（base64编码）
fetch('http://localhost:5000/predict_camera', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ frames: frames })
})
.then(response => response.json())
.then(data => {
  if (data.token) {
    console.log('检测到手势:', data.token.gloss);
    // 根据temporal字段决定生成GIF还是JPEG
    if (data.token.temporal) {
      // 生成GIF: frames[token.start_frame : token.end_frame + 1]
    } else {
      // 生成JPEG: frames[token.start_frame + token.end_frame) / 2]
    }
  } else {
    console.log('未检测到手势，继续缓冲');
  }
});
```

## 配置说明

### 模型路径
默认模型路径在 `CONFIG['model_save_path']`，可以在代码中修改。

### 翻译映射
在 `get_translation()` 函数中添加手势到自然语言的映射。

### 时间模式判断
在 `is_temporal_sign()` 函数中定义哪些手势依赖运动模式。

## 注意事项

1. **帧格式**: 前端发送的帧应该是base64编码的JPEG/PNG图像
2. **帧率**: 默认假设30fps，可以根据实际情况调整
3. **置信度阈值**: 
   - 视频预测: 0.5
   - 实时预测: 0.6（更严格）
4. **序列长度**: 默认30帧，与训练时一致

## 故障排除

### 模型未加载
确保模型文件存在且路径正确：
```bash
ls -lh /root/autodl-nus/sign_language_model.pth
```

### CORS错误
如果前端遇到CORS问题，确保 `flask-cors` 已安装并启用。

### 内存不足
如果处理大量视频时内存不足，可以考虑：
- 减少并发请求
- 降低视频分辨率
- 使用GPU加速

