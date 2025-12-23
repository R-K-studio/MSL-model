#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手语识别系统 - 推理脚本
用于测试单个视频或实时识别
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from sign_language_recognition import SignLanguageLSTM, MediaPipeProcessor, CONFIG

class SignLanguageInference:
    """手语识别推理器"""
    
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = MediaPipeProcessor()
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.gestures = checkpoint['gestures']
        self.label_map = checkpoint['label_map']
        self.config = checkpoint['config']
        
        # 初始化模型
        self.model = SignLanguageLSTM(
            self.config['input_size'],
            self.config['hidden_size'],
            len(self.gestures)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"模型加载成功! 支持 {len(self.gestures)} 个手语类别")
    
    def predict_video(self, video_path, show_video=False):
        """预测视频中的手语"""
        print(f"处理视频: {video_path}")
        
        # 提取关键点
        keypoints_seq = self.processor.process_video(
            video_path,
            max_frames=self.config['sequence_length']
        )
        
        if keypoints_seq is None:
            print("无法从视频中提取关键点")
            return None
        
        # 预测
        keypoints_tensor = torch.FloatTensor(keypoints_seq).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(keypoints_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_gesture = self.gestures[predicted.item()]
        confidence_score = confidence.item()
        
        print(f"预测结果: {predicted_gesture}")
        print(f"置信度: {confidence_score*100:.2f}%")
        
        # 显示所有类别的概率
        probs = probabilities[0].cpu().numpy()
        top_k = 5
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        print(f"\nTop {top_k} 预测:")
        for idx in top_indices:
            print(f"  {self.gestures[idx]}: {probs[idx]*100:.2f}%")
        
        if show_video:
            self._display_video_with_prediction(video_path, predicted_gesture, confidence_score)
        
        return predicted_gesture, confidence_score
    
    def predict_realtime(self, camera_id=0):
        """实时手语识别（使用摄像头）"""
        print("启动实时手语识别...")
        print("按 'q' 退出")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        sequence = []
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 镜像翻转
                frame = cv2.flip(frame, 1)
                
                # 处理帧
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = holistic.process(frame_rgb)
                frame_rgb.flags.writeable = True
                
                # 绘制关键点
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
                )
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                )
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                )
                
                # 提取关键点
                if results.left_hand_landmarks or results.right_hand_landmarks:
                    keypoints = self.processor.extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-self.config['sequence_length']:]
                    
                    # 当序列长度足够时进行预测
                    if len(sequence) == self.config['sequence_length']:
                        keypoints_array = np.array(sequence)
                        keypoints_tensor = torch.FloatTensor(keypoints_array).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(keypoints_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                        
                        predicted_gesture = self.gestures[predicted.item()]
                        confidence_score = confidence.item()
                        
                        # 在图像上显示结果
                        cv2.putText(
                            frame, 
                            f"{predicted_gesture} ({confidence_score*100:.1f}%)",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                
                # 显示帧数
                cv2.putText(
                    frame,
                    f"Frames: {len(sequence)}/{self.config['sequence_length']}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                cv2.imshow('Sign Language Recognition', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _display_video_with_prediction(self, video_path, gesture, confidence):
        """显示视频并标注预测结果"""
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.putText(
                frame,
                f"Prediction: {gesture} ({confidence*100:.1f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            cv2.imshow('Video with Prediction', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='手语识别推理')
    parser.add_argument('--model', type=str, default=CONFIG['model_save_path'],
                       help='模型文件路径')
    parser.add_argument('--video', type=str, default=None,
                       help='要测试的视频文件路径')
    parser.add_argument('--realtime', action='store_true',
                       help='使用实时摄像头识别')
    parser.add_argument('--camera', type=int, default=0,
                       help='摄像头ID')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        print("请先运行训练脚本生成模型")
        return
    
    # 创建推理器
    inference = SignLanguageInference(args.model)
    
    if args.realtime:
        # 实时识别
        inference.predict_realtime(camera_id=args.camera)
    elif args.video:
        # 测试视频
        if not os.path.exists(args.video):
            print(f"错误: 视频文件不存在: {args.video}")
            return
        inference.predict_video(args.video, show_video=True)
    else:
        print("请指定 --video 或 --realtime 参数")
        print("\n使用示例:")
        print("  python sign_language_inference.py --video /path/to/video.mp4")
        print("  python sign_language_inference.py --realtime")


if __name__ == '__main__':
    main()

