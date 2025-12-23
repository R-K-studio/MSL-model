#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试后端API的示例脚本
"""

import requests
import base64
import cv2
import os

# API基础URL
BASE_URL = "http://localhost:5000"


def test_health_check():
    """测试健康检查"""
    print("测试健康检查...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()


def test_predict_video(video_path):
    """测试视频预测"""
    print(f"测试视频预测: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    with open(video_path, 'rb') as f:
        files = {'video': f}
        response = requests.post(f"{BASE_URL}/predict", files=files)
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    
    if 'tokens' in result:
        print(f"检测到 {len(result['tokens'])} 个手势:")
        for i, token in enumerate(result['tokens'], 1):
            print(f"  {i}. {token['gloss']} ({token['translation']}) - "
                  f"置信度: {token['confidence']:.2f} - "
                  f"帧范围: {token['start_frame']}-{token['end_frame']}")
        print(f"完整句子: {result['sentence']}")
    else:
        print(f"错误: {result}")
    print()


def test_predict_camera():
    """测试实时帧处理"""
    print("测试实时帧处理...")
    
    # 创建一个测试视频或使用摄像头
    # 这里使用一个示例：读取几帧测试图像
    test_frames = []
    
    # 如果有测试视频，读取几帧
    test_video = "/root/autodl-tmp/data/polis/polis_1_1_1.mp4"
    if os.path.exists(test_video):
        cap = cv2.VideoCapture(test_video)
        for i in range(30):  # 读取30帧
            ret, frame = cap.read()
            if ret:
                test_frames.append(frame)
            else:
                break
        cap.release()
    
    if len(test_frames) == 0:
        print("无法获取测试帧")
        return
    
    # 转换为base64
    frames_base64 = []
    for frame in test_frames:
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        frames_base64.append(frame_base64)
    
    # 发送请求
    response = requests.post(
        f"{BASE_URL}/predict_camera",
        json={'frames': frames_base64}
    )
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    
    if result.get('token'):
        token = result['token']
        print(f"检测到手势: {token['gloss']} ({token['translation']})")
        print(f"置信度: {token['confidence']:.2f}")
        print(f"帧范围: {token['start_frame']}-{token['end_frame']}")
        print(f"时间模式: {token['temporal']}")
    else:
        print("未检测到手势")
    print()


def main():
    """主函数"""
    print("=" * 60)
    print("手语识别后端API测试")
    print("=" * 60)
    print()
    
    # 测试健康检查
    try:
        test_health_check()
    except Exception as e:
        print(f"健康检查失败: {e}")
        print("请确保API服务器正在运行: python backend_api.py")
        return
    
    # 测试视频预测
    test_video = "/root/autodl-tmp/data/polis/polis_1_1_1.mp4"
    if os.path.exists(test_video):
        try:
            test_predict_video(test_video)
        except Exception as e:
            print(f"视频预测测试失败: {e}")
    
    # 测试实时帧处理
    try:
        test_predict_camera()
    except Exception as e:
        print(f"实时帧处理测试失败: {e}")
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    main()

