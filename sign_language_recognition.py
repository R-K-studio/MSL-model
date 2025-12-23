#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手语识别系统 - 主训练脚本
使用 MediaPipe 提取关键点，LSTM 模型进行训练和识别
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mediapipe as mp
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
import hashlib

# 配置参数
CONFIG = {
    'data_dir': '/root/autodl-tmp/data',
    'output_dir': '/root/autodl-nus/sign_language_output',
    'train_dataset_dir': '/root/autodl-nus/train_dataset',
    'model_save_path': '/root/autodl-nus/sign_language_model.pth',
    'sequence_length': 30,  # 每个手势的帧数
    'input_size': 258,  # MediaPipe 关键点维度 (33*4 + 21*3 + 21*3)
    'hidden_size': 64,
    'num_epochs': 200,
    'batch_size': 16,
    'learning_rate': 0.001,
    'test_size': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# MediaPipe 初始化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def process_single_video_worker(args):
    """处理单个视频的辅助函数（用于多进程）"""
    video_path, max_frames, label = args
    
    try:
        # 每个进程创建自己的 MediaPipe 处理器
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            holistic.close()
            return None, label, video_path
        
        keypoints_sequence = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True
            
            # 只保存检测到手部的帧
            if results.left_hand_landmarks or results.right_hand_landmarks:
                # 提取关键点
                pose = np.array([[res.x, res.y, res.z, res.visibility] 
                                for res in results.pose_landmarks.landmark]).flatten() \
                    if results.pose_landmarks else np.zeros(33 * 4)
                lh = np.array([[res.x, res.y, res.z] 
                              for res in results.left_hand_landmarks.landmark]).flatten() \
                    if results.left_hand_landmarks else np.zeros(21 * 3)
                rh = np.array([[res.x, res.y, res.z] 
                              for res in results.right_hand_landmarks.landmark]).flatten() \
                    if results.right_hand_landmarks else np.zeros(21 * 3)
                keypoints = np.concatenate([pose, lh, rh])
                
                keypoints_sequence.append(keypoints)
                frame_count += 1
        
        cap.release()
        holistic.close()
        
        if len(keypoints_sequence) == 0:
            return None, label, video_path
        
        # 填充或截断到固定长度
        if len(keypoints_sequence) < max_frames:
            last_frame = keypoints_sequence[-1]
            keypoints_sequence.extend([last_frame] * (max_frames - len(keypoints_sequence)))
        else:
            keypoints_sequence = keypoints_sequence[:max_frames]
        
        return np.array(keypoints_sequence), label, video_path
        
    except Exception as e:
        print(f"处理视频 {video_path} 时出错: {e}")
        return None, label, video_path


class MediaPipeProcessor:
    """MediaPipe 关键点提取器"""
    
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_keypoints(self, results):
        """从 MediaPipe 结果中提取关键点"""
        # Pose landmarks (33 points, 4 values: x, y, z, visibility)
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(33 * 4)
        
        # Left hand landmarks (21 points, 3 values: x, y, z)
        lh = np.array([[res.x, res.y, res.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(21 * 3)
        
        # Right hand landmarks (21 points, 3 values: x, y, z)
        rh = np.array([[res.x, res.y, res.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(21 * 3)
        
        return np.concatenate([pose, lh, rh])
    
    def process_frame(self, frame):
        """处理单帧图像"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return results
    
    def process_video(self, video_path, max_frames=30):
        """处理视频，提取关键点序列"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        keypoints_sequence = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.process_frame(frame)
            
            # 只保存检测到手部的帧
            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = self.extract_keypoints(results)
                keypoints_sequence.append(keypoints)
                frame_count += 1
        
        cap.release()
        
        if len(keypoints_sequence) == 0:
            return None
        
        # 填充或截断到固定长度
        if len(keypoints_sequence) < max_frames:
            # 用最后一帧填充
            last_frame = keypoints_sequence[-1]
            keypoints_sequence.extend([last_frame] * (max_frames - len(keypoints_sequence)))
        else:
            keypoints_sequence = keypoints_sequence[:max_frames]
        
        return np.array(keypoints_sequence)
    
    def __del__(self):
        if hasattr(self, 'holistic'):
            self.holistic.close()


class SignLanguageLSTM(nn.Module):
    """手语识别 LSTM 模型"""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(SignLanguageLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=1)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=1)
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.output_layer(x)
        
        return x


class SignLanguageDataset(Dataset):
    """手语数据集"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class SignLanguageTrainer:
    """手语识别训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(config['train_dataset_dir'], exist_ok=True)
        
        self.gestures = []
        self.label_map = {}
        self.model = None
    
    def load_gestures(self):
        """加载手语类别"""
        data_dir = self.config['data_dir']
        self.gestures = [d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d)) 
                        and not d.startswith('.')]
        self.gestures.sort()
        self.label_map = {gesture: idx for idx, gesture in enumerate(self.gestures)}
        
        print(f"找到 {len(self.gestures)} 个手语类别:")
        for i, gesture in enumerate(self.gestures[:10]):
            print(f"  {i}: {gesture}")
        if len(self.gestures) > 10:
            print(f"  ... 还有 {len(self.gestures) - 10} 个类别")
        
        return self.gestures
    
    def extract_features_from_videos(self, force_reload=False):
        """从视频中提取关键点特征（使用多进程加速）"""
        train_dataset_dir = self.config['train_dataset_dir']
        data_dir = self.config['data_dir']
        
        if not force_reload and os.path.exists(os.path.join(train_dataset_dir, 'features.npy')):
            print("加载已保存的特征数据...")
            sequences = np.load(os.path.join(train_dataset_dir, 'features.npy'))
            labels = np.load(os.path.join(train_dataset_dir, 'labels.npy'))
            return sequences, labels
        
        print("开始从视频提取关键点特征...")
        
        # 检查已处理的视频（使用文件哈希避免重复处理）
        processed_file = os.path.join(train_dataset_dir, 'processed_videos.txt')
        processed_videos = set()
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                processed_videos = set(line.strip() for line in f if line.strip())
            print(f"找到 {len(processed_videos)} 个已处理的视频，将跳过")
        
        # 收集所有需要处理的视频任务
        video_tasks = []
        for gesture in self.gestures:
            gesture_dir = os.path.join(data_dir, gesture)
            if not os.path.exists(gesture_dir):
                continue
            video_files = [f for f in os.listdir(gesture_dir) if f.endswith('.mp4')]
            for video_file in video_files:
                video_path = os.path.join(gesture_dir, video_file)
                # 使用文件路径和修改时间生成唯一标识
                try:
                    file_stat = os.stat(video_path)
                    video_id = f"{video_path}:{file_stat.st_mtime}"
                except:
                    video_id = video_path
                
                # 跳过已处理的视频
                if video_id not in processed_videos:
                    video_tasks.append((
                        video_path,
                        self.config['sequence_length'],
                        self.label_map[gesture]
                    ))
        
        total_videos = len(video_tasks)
        print(f"总共需要处理 {total_videos} 个视频")
        
        if total_videos == 0:
            print("所有视频都已处理，加载已保存的特征...")
            if os.path.exists(os.path.join(train_dataset_dir, 'features.npy')):
                sequences = np.load(os.path.join(train_dataset_dir, 'features.npy'))
                labels = np.load(os.path.join(train_dataset_dir, 'labels.npy'))
                return sequences, labels
            else:
                print("错误: 没有找到已保存的特征文件")
                return None, None
        
        # 使用多进程处理
        num_workers = min(cpu_count(), 8)  # 最多使用 8 个进程
        print(f"使用 {num_workers} 个进程并行处理...")
        
        sequences = []
        labels = []
        processed_list = []
        
        # 使用进程池处理
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_video_worker, video_tasks),
                total=total_videos,
                desc="处理视频"
            ))
        
        # 收集结果并记录已处理的视频
        for keypoints_seq, label, video_path in results:
            # 记录已处理的视频
            try:
                file_stat = os.stat(video_path)
                video_id = f"{video_path}:{file_stat.st_mtime}"
            except:
                video_id = video_path
            processed_list.append(video_id)
            
            if keypoints_seq is not None:
                sequences.append(keypoints_seq)
                labels.append(label)
        
        # 保存已处理的视频列表
        os.makedirs(train_dataset_dir, exist_ok=True)
        with open(processed_file, 'a') as f:
            for video_id in processed_list:
                f.write(f"{video_id}\n")
        
        if len(sequences) == 0:
            print("警告: 没有成功提取任何特征")
            return None, None
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        # 如果已有部分数据，合并新旧数据
        existing_features_file = os.path.join(train_dataset_dir, 'features.npy')
        if os.path.exists(existing_features_file) and not force_reload:
            print("合并新旧特征数据...")
            existing_sequences = np.load(existing_features_file)
            existing_labels = np.load(os.path.join(train_dataset_dir, 'labels.npy'))
            sequences = np.concatenate([existing_sequences, sequences], axis=0)
            labels = np.concatenate([existing_labels, labels], axis=0)
            print(f"合并后总样本数: {len(sequences)}")
        
        # 保存提取的特征
        np.save(os.path.join(train_dataset_dir, 'features.npy'), sequences)
        np.save(os.path.join(train_dataset_dir, 'labels.npy'), labels)
        
        print(f"\n提取完成: {len(sequences)} 个样本")
        print(f"特征形状: {sequences.shape}")
        print(f"标签形状: {labels.shape}")
        
        return sequences, labels
    
    def train(self, sequences, labels):
        """训练模型"""
        print("\n开始训练模型...")
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, 
            test_size=self.config['test_size'], 
            random_state=42,
            stratify=labels
        )
        
        print(f"训练集: {len(X_train)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        # 创建数据加载器
        train_dataset = SignLanguageDataset(X_train, y_train)
        test_dataset = SignLanguageDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # 初始化模型
        num_classes = len(self.gestures)
        self.model = SignLanguageLSTM(
            self.config['input_size'],
            self.config['hidden_size'],
            num_classes
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        # 训练历史
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        
        best_test_acc = 0.0
        
        # 训练循环
        for epoch in range(self.config['num_epochs']):
            # 训练阶段
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for sequences_batch, labels_batch in train_loader:
                sequences_batch = sequences_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
            
            train_loss = epoch_loss / len(train_loader)
            train_acc = 100 * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # 测试阶段
            test_acc = self.evaluate(test_loader)
            test_accuracies.append(test_acc)
            
            # 保存最佳模型
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'gestures': self.gestures,
                    'label_map': self.label_map,
                    'config': self.config
                }, self.config['model_save_path'])
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.config['num_epochs']}] "
                      f"Loss: {train_loss:.4f} "
                      f"Train Acc: {train_acc:.2f}% "
                      f"Test Acc: {test_acc:.2f}%")
        
        # 绘制训练曲线
        self.plot_training_curves(train_losses, train_accuracies, test_accuracies)
        
        print(f"\n训练完成! 最佳测试准确率: {best_test_acc:.2f}%")
        print(f"模型已保存到: {self.config['model_save_path']}")
        
        return self.model
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences_batch, labels_batch in test_loader:
                sequences_batch = sequences_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                outputs = self.model(sequences_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def evaluate_detailed(self, sequences, labels):
        """详细评估模型"""
        print("\n进行详细评估...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, 
            test_size=self.config['test_size'], 
            random_state=42,
            stratify=labels
        )
        
        test_dataset = SignLanguageDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences_batch, labels_batch in test_loader:
                sequences_batch = sequences_batch.to(self.device)
                outputs = self.model(sequences_batch)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels_batch.numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, 
                                     target_names=self.gestures, 
                                     zero_division=0)
        
        print(f"\n测试准确率: {accuracy*100:.2f}%")
        print("\n分类报告:")
        print(report)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(cm, self.gestures)
        
        return accuracy, cm, report
    
    def plot_training_curves(self, train_losses, train_accuracies, test_accuracies):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(train_accuracies, label='Train Accuracy')
        ax2.plot(test_accuracies, label='Test Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'training_curves.png'))
        print(f"训练曲线已保存到: {os.path.join(self.config['output_dir'], 'training_curves.png')}")
        plt.close()
    
    def plot_confusion_matrix(self, cm, gestures):
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=gestures, yticklabels=gestures)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'confusion_matrix.png'))
        print(f"混淆矩阵已保存到: {os.path.join(self.config['output_dir'], 'confusion_matrix.png')}")
        plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("手语识别系统 - 训练脚本")
    print("=" * 60)
    
    # 创建训练器
    trainer = SignLanguageTrainer(CONFIG)
    
    # 加载手语类别
    trainer.load_gestures()
    
    # 提取特征
    sequences, labels = trainer.extract_features_from_videos(force_reload=False)
    
    # 训练模型
    model = trainer.train(sequences, labels)
    
    # 详细评估
    trainer.evaluate_detailed(sequences, labels)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"模型文件: {CONFIG['model_save_path']}")
    print(f"输出目录: {CONFIG['output_dir']}")
    print("=" * 60)


if __name__ == '__main__':
    main()

