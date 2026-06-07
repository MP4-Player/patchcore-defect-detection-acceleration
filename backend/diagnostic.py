#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Диагностика PatchCore — проверка соответствия патчей позициям на изображении
"""

import cv2
import numpy as np
import torch
from patchcore_runner import PatchCoreRunner
import matplotlib.pyplot as plt

def create_test_image_with_single_hot_patch(patch_idx, size=224):
    """Создаёт изображение, где только один патч (patch_idx) имеет значение."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cell_size = size // 28
    row = patch_idx // 28
    col = patch_idx % 28
    
    y1, y2 = row * cell_size, (row + 1) * cell_size
    x1, x2 = col * cell_size, (col + 1) * cell_size
    img[y1:y2, x1:x2] = [255, 255, 255]
    
    return img

def visualize_patch_correspondence(model_path, device="cuda"):
    print("=" * 60)
    print("ДИАГНОСТИКА PATCHCORE")
    print(f"Модель: {model_path}")
    print("=" * 60)
    
    runner = PatchCoreRunner(model_path, device=device)
    
    # Проверка: загрузилась ли модель
    if runner._memory_bank is None:
        print("\n❌ МОДЕЛЬ НЕ ЗАГРУЗИЛАСЬ!")
        print("   Проверьте формат файла.")
        print("   Запустите check_model.py для диагностики")
        return
    
    print(f"\n✓ Модель загружена: memory_bank shape={runner._memory_bank.shape}")
    print(f"  norm_max={runner._norm_max}")
    
    # Тест с одним активным патчем
    print("\n[Тест] Проверка соответствия патчей...")
    
    test_patches = [0, 27, 28, 55, 756, 783]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, patch_idx in enumerate(test_patches):
        test_img = create_test_image_with_single_hot_patch(patch_idx)
        heatmap, score = runner.infer_with_score(test_img)
        
        # Находим позицию максимума в heatmap
        max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        # Ожидаемая позиция (центр патча)
        cell_size = 224 // 28
        row = patch_idx // 28
        col = patch_idx % 28
        expected_y = row * cell_size + cell_size // 2
        expected_x = col * cell_size + cell_size // 2
        
        axes[idx].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        axes[idx].scatter([expected_x], [expected_y], c='green', s=100, marker='o', label='Ожидаемая')
        axes[idx].scatter([max_pos[1]], [max_pos[0]], c='red', s=100, marker='x', label='Фактическая')
        axes[idx].set_title(f"Патч {patch_idx} (ряд {row}, кол {col})\n"
                           f"max в ({max_pos[0]},{max_pos[1]})", fontsize=9)
        axes[idx].axis('off')
        
        # Проверка совпадения
        distance = np.sqrt((max_pos[0] - expected_y)**2 + (max_pos[1] - expected_x)**2)
        if distance > 20:
            print(f"  ⚠️ Патч {patch_idx}: ожидалось ({expected_y},{expected_x}), "
                  f"получено ({max_pos[0]},{max_pos[1]}), расстояние={distance:.0f}px")
        else:
            print(f"  ✓ Патч {patch_idx}: совпадает (расстояние={distance:.0f}px)")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("diagnostic_result.png", dpi=150, bbox_inches='tight')
    print("\n  Сохранено: diagnostic_result.png")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/patchcore.pt", help="Путь к модели")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    visualize_patch_correspondence(args.model, args.device)