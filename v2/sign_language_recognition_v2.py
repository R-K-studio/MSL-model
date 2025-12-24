#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手语识别系统 - 精简类别训练脚本 (v2)

用途：
- 复用 sign_language_recognition.py 中的模型与训练逻辑
- 通过 ENABLED_GESTURES 手动指定要保留的手语类别
- 只在这些类别上重新提取特征并训练一个新的模型
"""

import os
import numpy as np
import torch

from sign_language_recognition import (
    CONFIG as BASE_CONFIG,
    SignLanguageTrainer,
)


# 在这里手动填入你要保留的类别名称（目录名），例如：
# ENABLED_GESTURES = ['polis', 'bas', 'ada', 'abang']
# 为空列表时，行为与原脚本一致：使用所有类别
ENABLED_GESTURES = [
    # 示例：
    # 'polis',
    # 'bas',
    # 'ada',
    'ambil',
    'hujan',
    'lupa',
    'pergi',
    'pukul',
    'kakak',
    'jangan',
    'hari',
    'lemak',
    'marah',
    'minum',
    'tanya',
    'hi',
    'kerrta',
    'bawa',
    'bapa',
    'panas_2',
    'abang',
    'curi',
    'ribut',
    'siapa',
    'apa_khabar',
    'bapa_saudara',
    'lelaki',
    'buat',
    'ayah',
    'payung',
    'main',
]


# 基于原始 CONFIG 创建 v2 专用配置
CONFIG_V2 = BASE_CONFIG.copy()

# 为 v2 使用独立的输出目录、特征目录和模型文件，避免覆盖原有结果
CONFIG_V2['output_dir'] = os.path.join(
    os.path.dirname(BASE_CONFIG.get('output_dir', '.')),
    'sign_language_output_v2',
)
CONFIG_V2['train_dataset_dir'] = os.path.join(
    os.path.dirname(BASE_CONFIG.get('train_dataset_dir', '.')),
    'train_dataset_v2',
)
CONFIG_V2['model_save_path'] = os.path.join(
    os.path.dirname(BASE_CONFIG.get('model_save_path', '.')),
    'sign_language_model_v2.pth',
)


class SignLanguageTrainerV2(SignLanguageTrainer):
    """只在 ENABLED_GESTURES 指定的类别上训练的训练器"""

    def load_gestures(self):
        """
        加载手语类别：
        - 如果 ENABLED_GESTURES 非空：只加载这些类别（且目录存在）
        - 如果 ENABLED_GESTURES 为空：回退到父类逻辑，加载所有类别
        """
        data_dir = self.config['data_dir']

        # 如果没有指定保留列表，保持与原版相同的行为
        if not ENABLED_GESTURES:
            print("ENABLED_GESTURES 为空，加载所有类别（与原脚本一致）")
            return super().load_gestures()

        # 扫描所有子目录
        all_dirs = [
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
            and not d.startswith('.')
        ]

        # 只保留在 ENABLED_GESTURES 中且真实存在的目录
        enabled_set = set(ENABLED_GESTURES)
        self.gestures = [d for d in all_dirs if d in enabled_set]
        self.gestures.sort()
        self.label_map = {gesture: idx for idx, gesture in enumerate(self.gestures)}

        print(f"v2: 仅使用 {len(self.gestures)} 个手语类别进行训练：")
        for i, gesture in enumerate(self.gestures):
            print(f"  {i}: {gesture}")

        if not self.gestures:
            print("⚠️ 警告：ENABLED_GESTURES 中的类别在数据目录中都不存在，请检查名称是否与目录名一致。")

        return self.gestures


def main():
    """v2 主函数：只针对筛选后的类别进行训练与评估"""
    print("=" * 60)
    print("手语识别系统 - 精简类别训练脚本 (v2)")
    print("=" * 60)

    print("\n当前使用的配置 (v2):")
    print(f"  data_dir         : {CONFIG_V2['data_dir']}")
    print(f"  output_dir       : {CONFIG_V2['output_dir']}")
    print(f"  train_dataset_dir: {CONFIG_V2['train_dataset_dir']}")
    print(f"  model_save_path  : {CONFIG_V2['model_save_path']}")
    print(f"  device           : {CONFIG_V2['device']}")

    if ENABLED_GESTURES:
        print("\n保留的类别列表 ENABLED_GESTURES:")
        for g in ENABLED_GESTURES:
            print(f"  - {g}")
    else:
        print("\nENABLED_GESTURES 为空，将使用所有类别（与原版一致）")

    # 创建 v2 训练器
    trainer = SignLanguageTrainerV2(CONFIG_V2)

    # 加载（或筛选后）手语类别
    trainer.load_gestures()

    # 提取特征（仅针对保留类别）
    sequences, labels = trainer.extract_features_from_videos(force_reload=False)

    if sequences is None or labels is None:
        print("\n❌ 特征提取失败，无法继续训练。请检查数据和 ENABLED_GESTURES 设置。")
        return

    # 训练模型
    model = trainer.train(sequences, labels)

    # 详细评估（会生成新的 classification_report、混淆矩阵等）
    trainer.evaluate_detailed(sequences, labels)

    print("\n" + "=" * 60)
    print("v2 训练完成!")
    print(f"模型文件: {CONFIG_V2['model_save_path']}")
    print(f"输出目录: {CONFIG_V2['output_dir']}")
    print("=" * 60)


if __name__ == '__main__':
    main()


