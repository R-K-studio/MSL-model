#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手语识别后端API
提供视频预测和实时帧处理接口
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import json
from typing import List, Dict, Optional, Tuple
from sign_language_recognition import SignLanguageLSTM, MediaPipeProcessor, CONFIG

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
model = None
gestures = None
label_map = None
config = None
device = None
processor = None

# MediaPipe 初始化
mp_holistic = mp.solutions.holistic


def load_model(model_path: str):
    """加载训练好的模型"""
    global model, gestures, label_map, config, device, processor
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = MediaPipeProcessor()
    
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=device)
    gestures = checkpoint['gestures']
    label_map = checkpoint['label_map']
    config = checkpoint['config']
    
    # 初始化模型
    model = SignLanguageLSTM(
        config['input_size'],
        config['hidden_size'],
        len(gestures)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型加载成功! 支持 {len(gestures)} 个手语类别")
    return model


def extract_keypoints(results) -> np.ndarray:
    """从 MediaPipe 结果中提取关键点"""
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                    for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    
    lh = np.array([[res.x, res.y, res.z] 
                  for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    
    rh = np.array([[res.x, res.y, res.z] 
                  for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    
    return np.concatenate([pose, lh, rh])


def process_frames_to_sequence(frames: List[np.ndarray], max_frames: int = 30) -> Optional[np.ndarray]:
    """将帧序列转换为关键点序列"""
    keypoints_sequence = []
    
    for frame in frames:
        # 处理帧
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = processor.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        # 只保存检测到手部的帧
        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_keypoints(results)
            keypoints_sequence.append(keypoints)
    
    if len(keypoints_sequence) == 0:
        return None
    
    # 填充或截断到固定长度
    if len(keypoints_sequence) < max_frames:
        last_frame = keypoints_sequence[-1]
        keypoints_sequence.extend([last_frame] * (max_frames - len(keypoints_sequence)))
    else:
        keypoints_sequence = keypoints_sequence[:max_frames]
    
    return np.array(keypoints_sequence)


def predict_sequence(keypoints_seq: np.ndarray) -> Tuple[str, float]:
    """预测关键点序列对应的手势"""
    if keypoints_seq is None:
        return None, 0.0
    
    # 转换为tensor
    keypoints_tensor = torch.FloatTensor(keypoints_seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(keypoints_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_gesture = gestures[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_gesture, confidence_score


def is_temporal_sign(gloss: str) -> bool:
    """判断手势是否依赖时间模式（需要运动）"""
    # 可以根据手势名称或特征判断
    # 这里简单实现：某些手势通常是temporal的
    temporal_keywords = ['drive', 'learning', 'walk', 'run', 'fly', 'move', 'go']
    return any(keyword in gloss.lower() for keyword in temporal_keywords)


def get_translation(gloss: str) -> str:
    """获取手势的翻译（保持原样的马来语）"""
    # 直接返回手势名称，不进行翻译
    return gloss


def predict_sign(video_path: str, model_obj=None) -> Dict:
    """
    离线视频处理 - 返回完整的手势序列
    
    Args:
        video_path: 视频文件路径
        model_obj: 模型对象（可选，使用全局模型）
    
    Returns:
        {
            "tokens": [
                {
                    "gloss": "SAYA",
                    "translation": "SAYA",
                    "confidence": 0.93,
                    "temporal": False,
                    "start_frame": 10,
                    "end_frame": 20,
                    "fps": 30.0
                },
                ...
            ],
            "sentence": "I love learning..."
        }
    """
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"tokens": [], "sentence": ""}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps < 1:
        fps = 30.0  # 默认FPS
    
    # 读取所有帧
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    
    if len(all_frames) == 0:
        return {"tokens": [], "sentence": ""}
    
    # 滑动窗口检测手势
    sequence_length = config['sequence_length']
    step_size = max(1, sequence_length // 3)  # 重叠2/3，更细粒度检测
    
    tokens = []
    current_start = 0
    last_gloss = None
    last_end = -1
    
    while current_start + sequence_length <= len(all_frames):
        # 提取当前窗口的帧
        window_frames = all_frames[current_start:current_start + sequence_length]
        
        # 处理帧序列
        keypoints_seq = process_frames_to_sequence(window_frames, sequence_length)
        
        if keypoints_seq is not None:
            # 预测手势
            gloss, confidence = predict_sequence(keypoints_seq)
            
            if gloss and confidence > 0.5:  # 置信度阈值
                end_frame = current_start + sequence_length - 1
                
                # 如果与上一个token相同，扩展范围
                if last_gloss == gloss and current_start <= last_end + step_size:
                    tokens[-1]['end_frame'] = int(end_frame)
                    tokens[-1]['confidence'] = max(tokens[-1]['confidence'], float(confidence))
                else:
                    # 新的手势token
                    token = {
                        "gloss": gloss,
                        "translation": get_translation(gloss),
                        "confidence": float(confidence),
                        "temporal": is_temporal_sign(gloss),
                        "start_frame": int(current_start),
                        "end_frame": int(end_frame),
                        "fps": float(fps)
                    }
                    tokens.append(token)
                    last_gloss = gloss
                    last_end = end_frame
        
        current_start += step_size
    
    # 构建句子
    sentence = " ".join([token['translation'] for token in tokens])
    
    return {
        "tokens": tokens,
        "sentence": sentence
    }


def run_msl_model_on_frames(frames_bgr: List[np.ndarray], model_obj=None) -> Optional[Dict]:
    """
    流式帧处理 - 返回单个token或None
    
    Args:
        frames_bgr: OpenCV BGR格式的帧列表（最近1-2秒的窗口）
        model_obj: 模型对象（可选，使用全局模型）
    
    Returns:
        None - 如果还没有稳定的手势
        或单个token字典:
        {
            "gloss": "SAYA",
            "translation": "SAYA",
            "confidence": 0.93,
            "temporal": True,
            "start_frame": 5,  # 在frames_bgr中的索引
            "end_frame": 18,
            "fps": 30.0
        }
    """
    if len(frames_bgr) < config['sequence_length']:
        return None  # 帧数不足，继续缓冲
    
    # 使用最近的sequence_length帧
    recent_frames = frames_bgr[-config['sequence_length']:]
    
    # 处理帧序列
    keypoints_seq = process_frames_to_sequence(recent_frames, config['sequence_length'])
    
    if keypoints_seq is None:
        return None  # 没有检测到手部
    
    # 预测手势
    gloss, confidence = predict_sequence(keypoints_seq)
    
    if not gloss or confidence < 0.6:  # 需要更高的置信度阈值
        return None
    
    # 确定手势在窗口中的位置
    # 假设手势在最后sequence_length帧中
    start_frame = len(frames_bgr) - config['sequence_length']
    end_frame = len(frames_bgr) - 1
    
    token = {
        "gloss": gloss,
        "translation": get_translation(gloss),
        "confidence": float(confidence),
        "temporal": is_temporal_sign(gloss),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "fps": 30.0  # 假设的FPS
    }
    
    return token


# ==================== Flask API 路由 ====================

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "num_gestures": len(gestures) if gestures else 0
    })


@app.route('/predict', methods=['POST'])
def predict_video():
    """
    上传视频并预测手势序列
    请求: multipart/form-data with 'video' file
    返回: JSON格式的tokens和sentence
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        video_path = tmp_file.name
        video_file.save(video_path)
    
    try:
        # 预测手势
        result = predict_sign(video_path, model)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # 清理临时文件
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    """
    接收帧序列并返回检测到的手势
    请求: JSON with 'frames' (base64 encoded images)
    返回: JSON with token or null
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        if 'frames' not in data:
            return jsonify({"error": "No frames provided"}), 400
        
        # 解码base64图像（如果前端发送的是base64）
        # 或者直接接收numpy数组（如果使用其他格式）
        frames_bgr = []
        
        # 这里需要根据前端实际发送的格式来解析
        # 假设前端发送的是base64编码的图像列表
        import base64
        for frame_data in data['frames']:
            # 解码base64
            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frames_bgr.append(frame)
        
        # 处理帧
        token = run_msl_model_on_frames(frames_bgr, model)
        
        if token is None:
            return jsonify({"token": None})
        else:
            return jsonify({"token": token})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict_frames', methods=['POST'])
def predict_frames():
    """
    接收帧序列的替代接口（使用文件上传）
    请求: multipart/form-data with 'frames' (JSON array of base64 images)
    返回: JSON with token or null
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        if 'frames' not in request.form:
            return jsonify({"error": "No frames provided"}), 400
        
        frames_json = request.form['frames']
        frames_data = json.loads(frames_json)
        
        frames_bgr = []
        import base64
        for frame_data in frames_data:
            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frames_bgr.append(frame)
        
        token = run_msl_model_on_frames(frames_bgr, model)
        
        if token is None:
            return jsonify({"token": None})
        else:
            return jsonify({"token": token})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 主程序 ====================

def main():
    """启动API服务器"""
    global model
    
    # 加载模型
    model_path = CONFIG.get('model_save_path', '/root/autodl-nus/sign_language_model.pth')
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型")
        return
    
    print(f"正在加载模型: {model_path}")
    load_model(model_path)
    
    # 启动Flask服务器
    print("启动API服务器...")
    print("API端点:")
    print("  GET  /health - 健康检查")
    print("  POST /predict - 上传视频预测")
    print("  POST /predict_camera - 流式帧处理")
    print("  POST /predict_frames - 帧序列处理（替代接口）")
    
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()

