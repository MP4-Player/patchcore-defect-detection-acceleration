# -*- coding: utf-8 -*-
"""
PatchCore inference runner — ПОДДЕРЖКА ВСЕХ ФОРМАТОВ МОДЕЛЕЙ

Поддерживаемые форматы:
1. Bundle dict — {"memory_bank": tensor, "fmap_size": 28, ...}
2. Plain tensor — torch.Tensor [N, feature_dim]
3. Словарь с любым ключом, содержащим тензор признаков
4. Модели из официальных репозиториев PatchCore
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms

log = logging.getLogger(__name__)

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.features = []

        def hook(module, inp, out):
            self.features.append(out)

        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

        self._last_spatial_shape = (28, 28)
        self.to(device)

    def forward(self, x):
        self.features = []
        with torch.no_grad():
            self.model(x)

        avg = torch.nn.AvgPool2d(3, stride=1, padding=1)
        fmap_size = self.features[0].shape[-2]

        resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
        resized = [resize(avg(f)) for f in self.features]

        patch = torch.cat(resized, 1)
        self._last_spatial_shape = (patch.shape[2], patch.shape[3])
        
        patch = patch.reshape(patch.shape[1], -1).T
        patch = patch / patch.norm(dim=1, keepdim=True).clamp(min=1e-8)
        
        return patch

    def get_spatial_shape(self):
        return self._last_spatial_shape


# ---------------------------------------------------------------------------
# PatchCore Runner — с поддержкой всех форматов
# ---------------------------------------------------------------------------

class PatchCoreRunner:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self._backbone: Optional[ResNetFeatureExtractor] = None
        self._memory_bank: Optional[torch.Tensor] = None
        self._fmap_size: int = 28
        self._input_size: int = 224
        self._norm_max: float = 1.0  # Значение по умолчанию
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------
    # Универсальная загрузка модели
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.model_path.exists():
            log.warning("Модель не найдена: %s → STUB-режим", self.model_path)
            return

        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        except Exception as e:
            log.error("Ошибка загрузки: %s → STUB-режим", e)
            return

        # Пытаемся извлечь memory_bank из разных форматов
        memory_bank = self._extract_memory_bank(checkpoint)
        
        if memory_bank is None:
            log.error("Не удалось извлечь memory_bank из файла %s", self.model_path)
            log.info("Поддерживаемые форматы: tensor [N,D] или dict с ключами: memory_bank, features, patch_library")
            return

        self._memory_bank = memory_bank.to(self.device)
        
        # Нормализация memory bank
        norms = torch.norm(self._memory_bank, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
            log.info("Нормализация memory bank...")
            self._memory_bank = self._memory_bank / norms.unsqueeze(1).clamp(min=1e-8)

        # Извлекаем norm_max если есть
        if isinstance(checkpoint, dict):
            self._norm_max = float(checkpoint.get("norm_max", 1.0))
            self._fmap_size = checkpoint.get("fmap_size", 28)
            self._input_size = checkpoint.get("input_size", 224)
        else:
            # Для plain tensor используем значения по умолчанию
            self._norm_max = 1.0
            self._fmap_size = 28

        log.info("PatchCore готов: memory_bank shape=%s, norm_max=%.4f, device=%s",
                 list(self._memory_bank.shape), self._norm_max, self.device)

        self._backbone = ResNetFeatureExtractor(self.device)

    def _extract_memory_bank(self, checkpoint: any) -> Optional[torch.Tensor]:
        """
        Извлекает memory_bank из различных форматов:
        - Прямой тензор
        - Словарь с ключом 'memory_bank'
        - Словарь с ключом 'features'
        - Словарь с ключом 'patch_library'
        - Любой другой ключ, содержащий 2D тензор
        """
        
        # Случай 1: прямой тензор
        if isinstance(checkpoint, torch.Tensor):
            if checkpoint.ndim == 2:
                log.info("Загружен plain tensor формат: shape=%s", list(checkpoint.shape))
                return checkpoint
            else:
                log.warning("Тензор имеет размерность %d, ожидается 2D", checkpoint.ndim)
                return None
        
        # Случай 2: словарь
        if isinstance(checkpoint, dict):
            # Приоритетные ключи
            priority_keys = ['memory_bank', 'memory_bank_features', 'features', 
                           'patch_library', 'bank', 'coreset', 'support_set']
            
            for key in priority_keys:
                if key in checkpoint:
                    tensor = checkpoint[key]
                    if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                        log.info("Загружен формат dict[%s]: shape=%s", key, list(tensor.shape))
                        return tensor
            
            # Если ничего не нашли, ищем любой 2D тензор
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor) and value.ndim == 2:
                    log.info("Найден 2D тензор в ключе '%s': shape=%s", key, list(value.shape))
                    return value
            
            # Проверяем вложенные словари (например, model_state_dict)
            for key, value in checkpoint.items():
                if isinstance(value, dict):
                    result = self._extract_memory_bank(value)
                    if result is not None:
                        return result
        
        return None

    # ------------------------------------------------------------------
    # Инференс
    # ------------------------------------------------------------------

    def infer(self, image: np.ndarray) -> np.ndarray:
        heatmap, _ = self.infer_with_score(image)
        return heatmap

    def infer_with_score(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        orig_h, orig_w = image.shape[:2]

        if self._memory_bank is None or self._backbone is None:
            log.warning("Используется STUB-режим (модель не загружена)")
            return _mock_heatmap(orig_h, orig_w), 0.5

        with self._lock:
            tensor = self._bgr_to_tensor(image)
            feats = self._backbone(tensor)

            distances = torch.cdist(feats, self._memory_bank, p=2.0)
            dist_score, _ = torch.min(distances, dim=1)

            # Anomaly score = mean + 2*std
            anomaly_score = float((dist_score.mean() + 2 * dist_score.std()).cpu())

            # Преобразуем в heatmap
            segm_raw = dist_score.view(self._fmap_size, self._fmap_size).cpu().numpy()

            # Интерполяция до размера оригинала
            segm_resized = cv2.resize(
                segm_raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )

            # Нормализация
            heatmap = self._normalize_heatmap(segm_resized)

        return heatmap, anomaly_score

    def infer_with_embeddings(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray, Tuple[int, int]]:
        """
        Возвращает heatmap + сырые патч-эмбеддинги из backbone ДО сравнения с memory bank.

        Returns:
            heatmap:         np.ndarray [H, W] float32 нормализованная карта аномалий
            anomaly_score:   float
            patch_embeddings: np.ndarray [fmap_h*fmap_w, C] float32
                             C = 1536 (512 из layer2 + 1024 из layer3)
                             fmap_h = fmap_w = 28 для входа 224×224
            spatial_shape:   Tuple[int, int] — (fmap_h, fmap_w) = (28, 28)
        """
        orig_h, orig_w = image.shape[:2]

        if self._memory_bank is None or self._backbone is None:
            dummy_emb = np.zeros((self._fmap_size ** 2, 1536), dtype=np.float32)
            return _mock_heatmap(orig_h, orig_w), 0.5, dummy_emb, (self._fmap_size, self._fmap_size)

        with self._lock:
            tensor = self._bgr_to_tensor(image)
            feats = self._backbone(tensor)           # [H*W, C]

            distances = torch.cdist(feats, self._memory_bank, p=2.0)
            dist_score, _ = torch.min(distances, dim=1)

            anomaly_score = float((dist_score.mean() + 2 * dist_score.std()).cpu())

            segm_raw = dist_score.view(self._fmap_size, self._fmap_size).cpu().numpy()
            segm_resized = cv2.resize(
                segm_raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
            heatmap = self._normalize_heatmap(segm_resized)

            # Сырые эмбеддинги — не клонируем, переводим в numpy здесь (под lock)
            patch_embeddings = feats.cpu().numpy().astype(np.float32)
            spatial_shape = self._backbone.get_spatial_shape()

        return heatmap, anomaly_score, patch_embeddings, spatial_shape

    def infer_batch(self, images: list) -> list:
        if not images:
            return []
        if self._memory_bank is None:
            return [_mock_heatmap(img.shape[0], img.shape[1]) for img in images]

        heatmaps = []
        for img in images:
            hm, _ = self.infer_with_score(img)
            heatmaps.append(hm)
        return heatmaps

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _bgr_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return _transform(pil).unsqueeze(0).to(self.device)

    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Нормализация heatmap с использованием norm_max."""
        if self._norm_max > 0:
            normalized = np.clip(heatmap / self._norm_max, 0.0, 1.0)
            normalized = scipy.ndimage.gaussian_filter(normalized, sigma=2)
            return normalized.astype(np.float32)
        
        # Fallback: min-max
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            return ((heatmap - h_min) / (h_max - h_min)).astype(np.float32)
        return np.zeros_like(heatmap, dtype=np.float32)


# ---------------------------------------------------------------------------
# Mock для отладки
# ---------------------------------------------------------------------------

def _mock_heatmap(h: int, w: int) -> np.ndarray:
    """Синтетический heatmap (виден только для отладки)."""
    rng = np.random.default_rng(seed=42)
    heatmap = np.zeros((h, w), dtype=np.float32)
    for _ in range(rng.integers(4, 12)):
        cx = rng.integers(w // 6, 5 * w // 6)
        cy = rng.integers(h // 6, 5 * h // 6)
        rx = rng.integers(8, min(50, w // 4))
        ry = rng.integers(8, min(50, h // 4))
        ys, xs = np.ogrid[:h, :w]
        mask = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
        heatmap[mask] = np.maximum(heatmap[mask], rng.uniform(0.55, 1.0))
    noise = rng.uniform(0.0, 0.12, (h, w)).astype(np.float32)
    return np.clip(heatmap + noise, 0.0, 1.0)