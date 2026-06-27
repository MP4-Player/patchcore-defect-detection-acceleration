# -*- coding: utf-8 -*-
"""
Классификация дефектов на основе патч-эмбеддингов PatchCore backbone.

Архитектура (значительно лучше геометрических признаков):
  1. Для каждого connected component → найти патчи (28×28 сетка) внутри маски
  2. Усреднить их эмбеддинги [1536-dim] → дефект-вектор
  3. PCA(n=20) для снижения размерности по всему батчу
  4. KMeans/GMM/Hierarchical в PCA-пространстве
  5. UMAP(2D) для интерактивной визуализации с GT-метками

Stateless API: все функции принимают данные как аргументы,
fit-состояние возвращается явно для передачи в параллельные воркеры.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

try:
    import umap as _umap_module
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

log = logging.getLogger(__name__)

DEFECT_TYPE_NAMES = ["color", "cut", "hole", "metal_contamination", "thread"]

# ---------------------------------------------------------------------------
# Шаг 1: извлечение дефект-вектора из патч-эмбеддингов
# ---------------------------------------------------------------------------

def extract_defect_embedding(
    patch_embeddings: np.ndarray,
    spatial_shape: Tuple[int, int],
    orig_h: int,
    orig_w: int,
    contour: np.ndarray,
    min_patches: int = 1,
) -> Optional[np.ndarray]:
    """
    Извлекает усреднённый эмбеддинг для одного connected component.

    Алгоритм:
      1. Рисуем маску компоненты (orig_h × orig_w)
      2. Ресайзим маску до fmap_h × fmap_w (28×28) nearest neighbor
      3. Берём патчи с mask > 0
      4. Усредняем их эмбеддинги → вектор [C]

    Args:
        patch_embeddings: [fmap_h*fmap_w, C] float32 — сырые эмбеддинги
        spatial_shape:    (fmap_h, fmap_w) — обычно (28, 28)
        orig_h, orig_w:   исходный размер изображения
        contour:          контур компоненты из cv2.findContours
        min_patches:      минимальное количество патчей для валидного дефекта

    Returns:
        np.ndarray [C] или None если компонента слишком мала
    """
    fmap_h, fmap_w = spatial_shape

    # Маска для этой компоненты на оригинальном разрешении
    comp_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    cv2.drawContours(comp_mask, [contour], -1, 255, cv2.FILLED)

    # Ресайз до fmap-разрешения (nearest neighbor — сохраняет бинарность)
    mask_small = cv2.resize(comp_mask, (fmap_w, fmap_h), interpolation=cv2.INTER_NEAREST)

    # Индексы патчей внутри маски
    flat_mask = mask_small.flatten()
    patch_indices = np.where(flat_mask > 0)[0]

    if len(patch_indices) < min_patches:
        return None

    # Усреднённый эмбеддинг
    selected = patch_embeddings[patch_indices]  # [n_patches, C]
    defect_vec = selected.mean(axis=0)          # [C]
    return defect_vec.astype(np.float32)


def extract_defect_embedding_from_bbox(
    image_bgr: np.ndarray,
    contour: np.ndarray,
    backbone_fn,
    padding: float = 0.2,
) -> Optional[np.ndarray]:
    """
    Альтернатива: вырезаем патч по bounding box и прогоняем через backbone.
    Используется если доступ к эмбеддингам из основного прохода невозможен.

    Args:
        image_bgr:   оригинальное BGR изображение
        contour:     контур дефекта
        backbone_fn: callable(image_bgr: np.ndarray) -> np.ndarray [H*W, C]
                     например lambda img: runner.infer_with_embeddings(img)[2]
        padding:     процент отступа вокруг bounding box
    """
    x, y, w, h = cv2.boundingRect(contour)
    if w < 4 or h < 4:
        return None

    pad_x = int(w * padding)
    pad_y = int(h * padding)
    H, W = image_bgr.shape[:2]
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    try:
        crop_resized = cv2.resize(crop, (224, 224))
        emb_map = backbone_fn(crop_resized)  # [H*W, C]
        return emb_map.mean(axis=0).astype(np.float32)
    except Exception as e:
        log.debug("bbox embedding failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Шаг 2: сборка матрицы эмбеддингов по батчу изображений
# ---------------------------------------------------------------------------

def collect_batch_embeddings(
    patch_embeddings_list: List[np.ndarray],
    spatial_shapes: List[Tuple[int, int]],
    orig_shapes: List[Tuple[int, int]],
    contours_list: List[List[np.ndarray]],
    gt_labels: Optional[List[Optional[int]]] = None,
    image_paths: Optional[List[str]] = None,
    min_patches: int = 1,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Собирает матрицу дефект-векторов для всего батча изображений.

    Returns:
        emb_matrix:  [N_defects, C] float32 — все дефекты из всех изображений
        meta:        List[Dict] — метаданные каждого дефекта
                     {image_idx, image_path, contour_idx, gt_label, area, centroid}
    """
    n_images = len(patch_embeddings_list)
    gt_labels = gt_labels or [None] * n_images
    image_paths = image_paths or [str(i) for i in range(n_images)]

    all_embeddings: List[np.ndarray] = []
    all_meta: List[Dict] = []

    for img_i, (patch_embs, sp_shape, orig_shape, contours) in enumerate(
        zip(patch_embeddings_list, spatial_shapes, orig_shapes, contours_list)
    ):
        orig_h, orig_w = orig_shape
        for c_i, contour in enumerate(contours):
            vec = extract_defect_embedding(
                patch_embs, sp_shape, orig_h, orig_w, contour, min_patches
            )
            if vec is None:
                continue

            area = float(cv2.contourArea(contour))
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

            all_embeddings.append(vec)
            all_meta.append({
                "image_idx":    img_i,
                "image_path":   image_paths[img_i],
                "contour_idx":  c_i,
                "gt_label":     gt_labels[img_i],
                "area":         round(area, 1),
                "centroid":     [cx, cy],
                "emb_idx":      len(all_embeddings) - 1,
            })

    if not all_embeddings:
        return np.empty((0, patch_embeddings_list[0].shape[1] if patch_embeddings_list else 1536),
                        dtype=np.float32), []

    return np.stack(all_embeddings, axis=0), all_meta


# ---------------------------------------------------------------------------
# Шаг 3: PCA → Clustering → UMAP
# ---------------------------------------------------------------------------

def fit_pca_and_cluster(
    emb_matrix: np.ndarray,
    config: Dict,
    random_state: int = 42,
) -> Tuple[PCA, StandardScaler, Any, np.ndarray, int]:
    """
    Обучает PCA + нормализацию + кластеризатор.

    Args:
        emb_matrix: [N, C] матрица эмбеддингов
        config:     конфигурация (pca_components, method, n_clusters, k_range)

    Returns:
        pca:          обученный sklearn PCA
        scaler:       StandardScaler для нормализации после PCA
        clusterer:    обученный кластеризатор (KMeans / GMM / Hierarchical)
        labels:       np.ndarray [N] метки кластеров
        n_clusters:   итоговое число кластеров
    """
    n = len(emb_matrix)
    if n == 0:
        return None, None, None, np.array([], dtype=int), 0

    pca_n = min(int(config.get("pca_components", 20)), n - 1, emb_matrix.shape[1])
    pca_n = max(1, pca_n)

    method      = config.get("method", "kmeans")
    n_clusters  = config.get("n_clusters", None)
    k_range     = config.get("k_range", [2, 8])

    # PCA
    pca = PCA(n_components=pca_n, random_state=random_state)
    X_pca = pca.fit_transform(emb_matrix)

    # Нормализация
    scaler = StandardScaler()
    X = scaler.fit_transform(X_pca)

    # Определяем K
    if n_clusters is None:
        k = _auto_k(X, k_range, method, random_state)
    else:
        k = max(1, min(int(n_clusters), n))

    if k <= 1 or n <= 1:
        return pca, scaler, None, np.zeros(n, dtype=int), 1

    # Кластеризация
    clusterer, labels = _cluster(X, k, method, random_state)

    return pca, scaler, clusterer, labels, int(max(labels) + 1)


def _auto_k(X: np.ndarray, k_range: List[int], method: str, random_state: int) -> int:
    n = len(X)
    k_min = max(2, k_range[0])
    k_max = max(k_min, min(k_range[1], n - 1))
    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        try:
            _, lbl = _cluster(X, k, method, random_state)
            if len(set(lbl)) < 2:
                continue
            s = float(silhouette_score(X, lbl, sample_size=min(500, n)))
            if s > best_score:
                best_score, best_k = s, k
        except Exception:
            pass
    return best_k


def _cluster(
    X: np.ndarray, k: int, method: str, random_state: int
) -> Tuple[Any, np.ndarray]:
    n = len(X)
    k = max(1, min(k, n))
    if k == 1:
        return None, np.zeros(n, dtype=int)

    if method == "kmeans":
        m = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        lbl = m.fit_predict(X)
    elif method == "gmm":
        m = GaussianMixture(n_components=k, random_state=random_state,
                            covariance_type="full", max_iter=200)
        m.fit(X)
        lbl = m.predict(X)
    elif method == "hierarchical":
        m = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lbl = m.fit_predict(X)
    else:
        raise ValueError(f"Unknown method: {method}")
    return m, lbl.astype(int)


# ---------------------------------------------------------------------------
# Метод 1: Prototype classifier (few-shot, supervised)
# ---------------------------------------------------------------------------

def fit_prototype_classifier(
    X: np.ndarray,
    gt_labels: np.ndarray,
    n_prototypes: int = 3,
) -> Tuple[Dict[int, np.ndarray], Dict]:
    """
    Вычисляет прототипы классов как среднее эмбеддингов первых n_prototypes дефектов.

    Args:
        X:            [N, D] PCA+normalized признаки
        gt_labels:    [N] GT-метки (0..4 для MVTec carpet)
        n_prototypes: сколько примеров брать на класс

    Returns:
        prototypes: {class_id: mean_vector [D]}
        proto_meta: {class_id: {"n_used": int, "indices": List[int]}}
    """
    prototypes: Dict[int, np.ndarray] = {}
    proto_meta: Dict[int, Dict] = {}
    for cls in np.unique(gt_labels):
        cls_int = int(cls)
        if cls_int < 0:
            continue
        idx = np.where(gt_labels == cls_int)[0][:n_prototypes]
        if len(idx) == 0:
            continue
        prototypes[cls_int] = X[idx].mean(axis=0)
        proto_meta[cls_int] = {"n_used": len(idx), "indices": idx.tolist()}
    return prototypes, proto_meta


def predict_prototype(
    X: np.ndarray,
    prototypes: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Классифицирует по cosine similarity к прототипам.
    Возвращает массив предсказанных GT-меток.
    """
    if not prototypes or len(X) == 0:
        return np.zeros(len(X), dtype=int)

    sorted_keys = sorted(prototypes.keys())
    proto_matrix = np.stack([prototypes[k] for k in sorted_keys], axis=0)  # [K, D]

    # L2-нормализация для cosine similarity
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    P_norm = proto_matrix / (np.linalg.norm(proto_matrix, axis=1, keepdims=True) + 1e-8)

    similarity = X_norm @ P_norm.T              # [N, K]
    best_idx   = np.argmax(similarity, axis=1)  # [N]
    return np.array([sorted_keys[i] for i in best_idx], dtype=int)


def prototype_accuracy(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
) -> Tuple[float, Dict[int, float]]:
    """
    Считает accuracy напрямую (pred_labels — это GT-метки, не кластеры).
    Возвращает overall и per-class accuracy.
    """
    valid = gt_labels >= 0
    if not valid.any():
        return 0.0, {}

    p, g = pred_labels[valid], gt_labels[valid]
    overall = float((p == g).mean())

    per_class: Dict[int, float] = {}
    for cls in np.unique(g):
        mask = g == cls
        per_class[int(cls)] = float((p[mask] == cls).mean())

    return round(overall, 4), {k: round(v, 4) for k, v in per_class.items()}


# ---------------------------------------------------------------------------
# Вспомогательная: стратифицированный сплит с обработкой редких классов
# ---------------------------------------------------------------------------

def _prepare_stratified_split(
    X: np.ndarray,
    gt_labels: np.ndarray,
    test_size: float = 0.5,
    random_state: int = 42,
    min_class_samples: int = 2,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    StratifiedShuffleSplit с явной обработкой классов < min_class_samples образцов.

    Классы с недостаточным числом образцов принудительно перемещаются в train,
    а не вызывают тихий обход stratify или ложный Accuracy=100%.

    Returns:
        train_idx:        индексы обучающей выборки
        test_idx:         индексы тестовой выборки (только из стратифицируемых классов)
        excluded_classes: List[int] — классы исключённые из теста
        warnings:         List[str] — текстовые предупреждения для UI
    """
    from sklearn.model_selection import StratifiedShuffleSplit as _SSS

    classes, counts = np.unique(gt_labels, return_counts=True)
    sufficient   = {int(c): int(n) for c, n in zip(classes, counts) if n >= min_class_samples}
    insufficient = {int(c): int(n) for c, n in zip(classes, counts) if n < min_class_samples}

    warnings_list: List[str] = []
    for cls, cnt in insufficient.items():
        name = DEFECT_TYPE_NAMES[cls] if 0 <= cls < len(DEFECT_TYPE_NAMES) else str(cls)
        warnings_list.append(
            f"Класс '{name}' не может быть стратифицирован ({cnt} экз. < {min_class_samples})"
            f" — исключён из теста, перенесён в train"
        )

    excluded_classes = list(insufficient.keys())
    can_stratify = np.array([int(l) in sufficient for l in gt_labels])

    if can_stratify.sum() < 4 or len(sufficient) < 2:
        warnings_list.append(
            "Недостаточно данных для стратифицированного разбиения"
            " — весь датасет используется как train (результат ненадёжен)"
        )
        return np.arange(len(X)), np.array([], dtype=int), excluded_classes, warnings_list

    sub_idx           = np.where(can_stratify)[0]
    forced_train_idx  = np.where(~can_stratify)[0]

    sss = _SSS(n_splits=1, test_size=test_size, random_state=random_state)
    tr_rel, te_rel = next(sss.split(X[sub_idx], gt_labels[sub_idx]))

    train_idx = np.concatenate([forced_train_idx, sub_idx[tr_rel]])
    test_idx  = sub_idx[te_rel]
    return train_idx, test_idx, excluded_classes, warnings_list


# ---------------------------------------------------------------------------
# Метод 2: Linear classifier (supervised upper bound)
# ---------------------------------------------------------------------------

def fit_linear_classifier(
    X: np.ndarray,
    gt_labels: np.ndarray,
    test_size: float = 0.5,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Dict:
    """
    Обучает LogisticRegression на PCA-эмбеддингах, stratified split 50/50.

    Returns dict:
        clf:                     обученный LogisticRegression (picklable)
        train_accuracy:          float
        test_accuracy:           float
        per_class_test:          {class_id: accuracy}
        n_classes_in_test:       int — сколько уникальных классов в тестовой выборке
        excluded_classes:        List[int] — классы исключённые из теста
        stratification_warnings: List[str] — предупреждения о малых классах
        n_train, n_test:         int
        classes:                 List[int]
    """
    from sklearn.linear_model import LogisticRegression

    n = len(X)
    classes = sorted(set(int(c) for c in gt_labels if c >= 0))

    if n < 4 or len(classes) < 2:
        clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
        clf.fit(X, gt_labels)
        pred = clf.predict(X)
        acc = float((pred == gt_labels).mean())
        return {
            "clf": clf, "train_accuracy": round(acc, 4), "test_accuracy": round(acc, 4),
            "per_class_test": {}, "train_idx": list(range(n)), "test_idx": [],
            "n_train": n, "n_test": 0, "classes": classes,
            "n_classes_in_test": 0, "excluded_classes": [],
            "stratification_warnings": ["Недостаточно данных (<4 дефектов или <2 классов)"],
        }

    train_idx, test_idx, excl_cls, strat_warn = _prepare_stratified_split(
        X, gt_labels, test_size=test_size, random_state=random_state
    )

    if len(test_idx) == 0:
        clf = LogisticRegression(max_iter=max_iter, random_state=random_state, C=1.0, solver="lbfgs")
        clf.fit(X, gt_labels)
        pred = clf.predict(X)
        acc = float((pred == gt_labels).mean())
        return {
            "clf": clf, "train_accuracy": round(acc, 4), "test_accuracy": round(acc, 4),
            "per_class_test": {}, "train_idx": list(range(n)), "test_idx": [],
            "n_train": n, "n_test": 0, "classes": classes,
            "n_classes_in_test": 0, "excluded_classes": excl_cls,
            "stratification_warnings": strat_warn,
        }

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = gt_labels[train_idx], gt_labels[test_idx]

    clf = LogisticRegression(max_iter=max_iter, random_state=random_state, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred  = clf.predict(X_test)
    train_acc  = float((train_pred == y_train).mean())
    test_acc   = float((test_pred  == y_test ).mean())

    per_class: Dict[int, float] = {}
    classes_in_test = sorted(set(int(c) for c in y_test))
    for cls in classes_in_test:
        mask = y_test == cls
        if mask.any():
            per_class[int(cls)] = round(float((test_pred[mask] == cls).mean()), 4)

    return {
        "clf":                     clf,
        "train_accuracy":          round(train_acc, 4),
        "test_accuracy":           round(test_acc, 4),
        "per_class_test":          per_class,
        "train_idx":               train_idx.tolist(),
        "test_idx":                test_idx.tolist(),
        "n_train":                 len(train_idx),
        "n_test":                  len(test_idx),
        "classes":                 classes,
        "n_classes_in_test":       len(classes_in_test),
        "excluded_classes":        excl_cls,
        "stratification_warnings": strat_warn,
    }


def predict_linear(
    X: np.ndarray,
    clf: Any,
) -> np.ndarray:
    """Применяет обученный LogisticRegression к новым данным."""
    if clf is None or len(X) == 0:
        return np.zeros(len(X), dtype=int)
    return clf.predict(X).astype(int)


# ---------------------------------------------------------------------------
# Метод 3: KNN classifier
# ---------------------------------------------------------------------------

def fit_knn_classifier(
    X: np.ndarray,
    gt_labels: np.ndarray,
    n_neighbors: int = 5,
    metric: str = "euclidean",
    test_size: float = 0.5,
    random_state: int = 42,
) -> Dict:
    """
    KNN на PCA-эмбеддингах. Stratified split 50/50 с обработкой малых классов.

    Args:
        n_neighbors: k соседей [1,3,5,7,9]
        metric:      "euclidean" | "cosine"

    Returns dict: clf, train_accuracy, test_accuracy, per_class_test,
                  n_classes_in_test, excluded_classes, stratification_warnings, ...
    """
    from sklearn.neighbors import KNeighborsClassifier

    n = len(X)
    classes = sorted(set(int(c) for c in gt_labels if c >= 0))
    k = max(1, min(n_neighbors, n - 1))

    if n < 4 or len(classes) < 2:
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
        clf.fit(X, gt_labels)
        pred = clf.predict(X)
        acc = float((pred == gt_labels).mean())
        return {
            "clf": clf, "train_accuracy": round(acc, 4), "test_accuracy": round(acc, 4),
            "per_class_test": {}, "n_train": n, "n_test": 0, "classes": classes,
            "n_neighbors": k, "metric": metric, "method_type": "knn",
            "n_classes_in_test": 0, "excluded_classes": [],
            "stratification_warnings": ["Недостаточно данных (<4 дефектов или <2 классов)"],
        }

    train_idx, test_idx, excl_cls, strat_warn = _prepare_stratified_split(
        X, gt_labels, test_size=test_size, random_state=random_state
    )

    if len(test_idx) == 0:
        k_safe = max(1, min(k, n - 1))
        clf = KNeighborsClassifier(n_neighbors=k_safe, metric=metric)
        clf.fit(X, gt_labels)
        pred = clf.predict(X)
        acc = float((pred == gt_labels).mean())
        return {
            "clf": clf, "train_accuracy": round(acc, 4), "test_accuracy": round(acc, 4),
            "per_class_test": {}, "n_train": n, "n_test": 0, "classes": classes,
            "n_neighbors": k_safe, "metric": metric, "method_type": "knn",
            "n_classes_in_test": 0, "excluded_classes": excl_cls,
            "stratification_warnings": strat_warn,
        }

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = gt_labels[train_idx], gt_labels[test_idx]

    k_safe = max(1, min(k, len(X_train) - 1))
    clf = KNeighborsClassifier(n_neighbors=k_safe, metric=metric)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred  = clf.predict(X_test)
    train_acc  = float((train_pred == y_train).mean())
    test_acc   = float((test_pred  == y_test ).mean())

    per_class: Dict[int, float] = {}
    classes_in_test = sorted(set(int(c) for c in y_test))
    for cls in classes_in_test:
        mask = y_test == cls
        if mask.any():
            per_class[int(cls)] = round(float((test_pred[mask] == cls).mean()), 4)

    return {
        "clf":                     clf,
        "train_accuracy":          round(train_acc, 4),
        "test_accuracy":           round(test_acc,  4),
        "per_class_test":          per_class,
        "train_idx":               train_idx.tolist(),
        "test_idx":                test_idx.tolist(),
        "n_train":                 len(train_idx),
        "n_test":                  len(test_idx),
        "classes":                 classes,
        "n_neighbors":             k_safe,
        "metric":                  metric,
        "method_type":             "knn",
        "n_classes_in_test":       len(classes_in_test),
        "excluded_classes":        excl_cls,
        "stratification_warnings": strat_warn,
    }


def predict_knn(X: np.ndarray, clf: Any) -> np.ndarray:
    if clf is None or len(X) == 0:
        return np.zeros(len(X), dtype=int)
    return clf.predict(X).astype(int)


# ---------------------------------------------------------------------------
# Метод 4: Random Forest
# ---------------------------------------------------------------------------

def fit_random_forest_classifier(
    X: np.ndarray,
    gt_labels: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    test_size: float = 0.5,
    random_state: int = 42,
) -> Dict:
    """
    RandomForestClassifier на PCA-эмбеддингах. Stratified split с обработкой малых классов.
    Возвращает feature_importance (какие PCA-компоненты значимы).
    """
    from sklearn.ensemble import RandomForestClassifier

    n = len(X)
    classes = sorted(set(int(c) for c in gt_labels if c >= 0))

    if n < 4 or len(classes) < 2:
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1,
        )
        clf.fit(X, gt_labels)
        pred = clf.predict(X)
        acc = float((pred == gt_labels).mean())
        return {
            "clf": clf, "train_accuracy": round(acc, 4), "test_accuracy": round(acc, 4),
            "per_class_test": {}, "n_train": n, "n_test": 0, "classes": classes,
            "feature_importance": clf.feature_importances_.tolist(),
            "n_estimators": n_estimators, "max_depth": max_depth,
            "method_type": "random_forest",
            "n_classes_in_test": 0, "excluded_classes": [],
            "stratification_warnings": ["Недостаточно данных (<4 дефектов или <2 классов)"],
        }

    train_idx, test_idx, excl_cls, strat_warn = _prepare_stratified_split(
        X, gt_labels, test_size=test_size, random_state=random_state
    )

    if len(test_idx) == 0:
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1,
        )
        clf.fit(X, gt_labels)
        pred = clf.predict(X)
        acc = float((pred == gt_labels).mean())
        return {
            "clf": clf, "train_accuracy": round(acc, 4), "test_accuracy": round(acc, 4),
            "per_class_test": {}, "n_train": n, "n_test": 0, "classes": classes,
            "feature_importance": clf.feature_importances_.tolist(),
            "n_estimators": n_estimators, "max_depth": max_depth,
            "method_type": "random_forest",
            "n_classes_in_test": 0, "excluded_classes": excl_cls,
            "stratification_warnings": strat_warn,
        }

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = gt_labels[train_idx], gt_labels[test_idx]

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred  = clf.predict(X_test)
    train_acc  = float((train_pred == y_train).mean())
    test_acc   = float((test_pred  == y_test ).mean())

    per_class: Dict[int, float] = {}
    classes_in_test = sorted(set(int(c) for c in y_test))
    for cls in classes_in_test:
        mask = y_test == cls
        if mask.any():
            per_class[int(cls)] = round(float((test_pred[mask] == cls).mean()), 4)

    return {
        "clf":                     clf,
        "train_accuracy":          round(train_acc, 4),
        "test_accuracy":           round(test_acc,  4),
        "per_class_test":          per_class,
        "train_idx":               train_idx.tolist(),
        "test_idx":                test_idx.tolist(),
        "n_train":                 len(train_idx),
        "n_test":                  len(test_idx),
        "classes":                 classes,
        "feature_importance":      clf.feature_importances_.tolist(),
        "n_estimators":            n_estimators,
        "max_depth":               max_depth,
        "method_type":             "random_forest",
        "n_classes_in_test":       len(classes_in_test),
        "excluded_classes":        excl_cls,
        "stratification_warnings": strat_warn,
    }


def predict_random_forest(X: np.ndarray, clf: Any) -> np.ndarray:
    if clf is None or len(X) == 0:
        return np.zeros(len(X), dtype=int)
    return clf.predict(X).astype(int)


# ---------------------------------------------------------------------------
# Метод 5: Decision Tree
# ---------------------------------------------------------------------------

def fit_decision_tree_classifier(
    X: np.ndarray,
    gt_labels: np.ndarray,
    max_depth: Optional[int] = 5,
    test_size: float = 0.5,
    random_state: int = 42,
) -> Dict:
    """
    DecisionTreeClassifier — интерпретируемая модель. Stratified split с обработкой малых классов.
    Возвращает feature_importance и глубину дерева.
    """
    from sklearn.tree import DecisionTreeClassifier

    n = len(X)
    classes = sorted(set(int(c) for c in gt_labels if c >= 0))

    if n < 4 or len(classes) < 2:
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        clf.fit(X, gt_labels)
        pred = clf.predict(X)
        acc = float((pred == gt_labels).mean())
        return {
            "clf": clf, "train_accuracy": round(acc, 4), "test_accuracy": round(acc, 4),
            "per_class_test": {}, "n_train": n, "n_test": 0, "classes": classes,
            "feature_importance": clf.feature_importances_.tolist(),
            "tree_depth": clf.get_depth(), "max_depth": max_depth,
            "method_type": "decision_tree",
            "n_classes_in_test": 0, "excluded_classes": [],
            "stratification_warnings": ["Недостаточно данных (<4 дефектов или <2 классов)"],
        }

    train_idx, test_idx, excl_cls, strat_warn = _prepare_stratified_split(
        X, gt_labels, test_size=test_size, random_state=random_state
    )

    if len(test_idx) == 0:
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        clf.fit(X, gt_labels)
        pred = clf.predict(X)
        acc = float((pred == gt_labels).mean())
        return {
            "clf": clf, "train_accuracy": round(acc, 4), "test_accuracy": round(acc, 4),
            "per_class_test": {}, "n_train": n, "n_test": 0, "classes": classes,
            "feature_importance": clf.feature_importances_.tolist(),
            "tree_depth": clf.get_depth(), "max_depth": max_depth,
            "method_type": "decision_tree",
            "n_classes_in_test": 0, "excluded_classes": excl_cls,
            "stratification_warnings": strat_warn,
        }

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = gt_labels[train_idx], gt_labels[test_idx]

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred  = clf.predict(X_test)
    train_acc  = float((train_pred == y_train).mean())
    test_acc   = float((test_pred  == y_test ).mean())

    per_class: Dict[int, float] = {}
    classes_in_test = sorted(set(int(c) for c in y_test))
    for cls in classes_in_test:
        mask = y_test == cls
        if mask.any():
            per_class[int(cls)] = round(float((test_pred[mask] == cls).mean()), 4)

    return {
        "clf":                     clf,
        "train_accuracy":          round(train_acc, 4),
        "test_accuracy":           round(test_acc,  4),
        "per_class_test":          per_class,
        "train_idx":               train_idx.tolist(),
        "test_idx":                test_idx.tolist(),
        "n_train":                 len(train_idx),
        "n_test":                  len(test_idx),
        "classes":                 classes,
        "feature_importance":      clf.feature_importances_.tolist(),
        "tree_depth":              clf.get_depth(),
        "max_depth":               max_depth,
        "method_type":             "decision_tree",
        "n_classes_in_test":       len(classes_in_test),
        "excluded_classes":        excl_cls,
        "stratification_warnings": strat_warn,
    }


def predict_decision_tree(X: np.ndarray, clf: Any) -> np.ndarray:
    if clf is None or len(X) == 0:
        return np.zeros(len(X), dtype=int)
    return clf.predict(X).astype(int)


def predict_in_pca_space(
    emb_matrix: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
    clusterer: Any,
    method: str,
    n_clusters_fitted: int,
) -> np.ndarray:
    """Применяет fitted PCA+scaler+clusterer к новым эмбеддингам."""
    n = len(emb_matrix)
    if n == 0 or pca is None:
        return np.array([], dtype=int)
    X_pca = pca.transform(emb_matrix)
    X = scaler.transform(X_pca)
    n_c = max(1, min(n_clusters_fitted, n))
    if n_c <= 1 or clusterer is None:
        return np.zeros(n, dtype=int)
    if method == "kmeans":
        return clusterer.predict(X).astype(int)
    elif method == "gmm":
        return clusterer.predict(X).astype(int)
    elif method == "hierarchical":
        k = max(1, min(int(getattr(clusterer, "n_clusters_", n_clusters_fitted)), n))
        m = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return m.fit_predict(X).astype(int)
    return np.zeros(n, dtype=int)


# ---------------------------------------------------------------------------
# Шаг 4: UMAP / t-SNE визуализация
# ---------------------------------------------------------------------------

def compute_2d_projection(
    emb_matrix: np.ndarray,
    pca: Optional[PCA] = None,
    method: str = "umap",
    random_state: int = 42,
) -> Optional[np.ndarray]:
    """
    Вычисляет 2D проекцию эмбеддингов для визуализации.

    Args:
        emb_matrix: [N, C] или [N, pca_dims]
        pca:        если передан, применяется сначала PCA
        method:     "umap" | "tsne" | "pca2"

    Returns:
        np.ndarray [N, 2] или None
    """
    n = len(emb_matrix)
    if n < 2:
        return None

    X = pca.transform(emb_matrix) if pca is not None else emb_matrix

    try:
        if method == "umap" and UMAP_AVAILABLE:
            reducer = _umap_module.UMAP(
                n_components=2,
                n_neighbors=min(15, n - 1),
                min_dist=0.1,
                random_state=random_state,
                verbose=False,
            )
            return reducer.fit_transform(X).astype(np.float32)

        if method == "tsne" and TSNE_AVAILABLE:
            perp = min(30, max(5, n // 3))
            reducer = TSNE(
                n_components=2, perplexity=perp,
                random_state=random_state, n_iter=500,
            )
            return reducer.fit_transform(X).astype(np.float32)

        # Fallback: первые 2 компоненты PCA
        p2 = PCA(n_components=2, random_state=random_state)
        return p2.fit_transform(X).astype(np.float32)

    except Exception as e:
        log.warning("2D projection failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Hungarian accuracy
# ---------------------------------------------------------------------------

def hungarian_accuracy(
    cluster_labels: np.ndarray,
    gt_labels: np.ndarray,
    n_clusters: int,
    n_gt: int,
) -> Tuple[float, Dict[int, int]]:
    n = len(cluster_labels)
    if n == 0:
        return 0.0, {}
    size = max(n_clusters, n_gt)
    cost = np.zeros((size, size), dtype=np.int64)
    for cl, gt in zip(cluster_labels, gt_labels):
        if gt >= 0:
            cost[int(cl)][int(gt)] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)
    correct = int(cost[row_ind, col_ind].sum())
    accuracy = correct / max(1, sum(1 for g in gt_labels if g >= 0))
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind) if r < n_clusters}
    return round(accuracy, 4), mapping


# ---------------------------------------------------------------------------
# Главная публичная функция — batch embedding classification
# ---------------------------------------------------------------------------

def classify_by_embeddings(
    patch_embeddings_list: List[np.ndarray],
    spatial_shapes: List[Tuple[int, int]],
    orig_shapes: List[Tuple[int, int]],
    contours_list: List[List[np.ndarray]],
    config: Dict,
    gt_labels: Optional[List[Optional[int]]] = None,
    image_paths: Optional[List[str]] = None,
    compute_umap: bool = True,
) -> Dict:
    """
    Полный pipeline embedding-based классификации для батча изображений.

    Работает правильно даже для 1 изображения.

    Args:
        patch_embeddings_list: List[[fmap_h*fmap_w, C]] — эмбеддинги из backbone
        spatial_shapes:        List[(fmap_h, fmap_w)]
        orig_shapes:           List[(h, w)] — исходные размеры изображений
        contours_list:         List[List[contour]] — контуры дефектов
        config: {
            method:          "kmeans" | "gmm" | "hierarchical"
            n_clusters:      int | None (None = авто через silhouette)
            k_range:         [2, 8]
            pca_components:  20
            umap_method:     "umap" | "tsne" | "pca2"
            random_state:    42
        }
        gt_labels:    List[int | None] — GT-метки (0..4 для MVTec carpet)
        image_paths:  List[str] — для отчёта
        compute_umap: bool — вычислять 2D проекцию (медленнее)

    Returns dict:
        overall_accuracy:   float | None
        silhouette:         float | None
        n_clusters:         int
        n_defects:          int
        pca_variance:       List[float] — объяснённая дисперсия по компонентам
        per_image:          List[Dict] — результат per-image (cluster_labels, etc.)
        umap_points:        List[{x, y, cluster, gt_label, image_path, area}] | None
        embedding_dim:      int — C
        pca_components:     int
        time_ms:            float
        method:             str
        fitted_pca:         PCA (для reuse)
        fitted_scaler:      StandardScaler
        fitted_clusterer:   Any
    """
    t0 = time.perf_counter()

    n_images = len(patch_embeddings_list)
    gt_labels_safe = gt_labels or [None] * n_images
    image_paths_safe = image_paths or [str(i) for i in range(n_images)]
    method = config.get("method", "kmeans")
    umap_method = config.get("umap_method", "umap" if UMAP_AVAILABLE else "pca2")
    random_state = int(config.get("random_state", 42))

    # ── 1. Собираем все дефект-эмбеддинги ───────────────────────────────────
    emb_matrix, meta = collect_batch_embeddings(
        patch_embeddings_list, spatial_shapes, orig_shapes,
        contours_list, gt_labels_safe, image_paths_safe,
    )

    embedding_dim = emb_matrix.shape[1] if len(emb_matrix) > 0 else 1536
    n_defects_total = len(emb_matrix)

    if n_defects_total == 0:
        elapsed = (time.perf_counter() - t0) * 1000
        return _empty_emb_result(method, elapsed, n_images, embedding_dim)

    # GT array (нужен для supervised методов)
    gt_arr = np.array([m["gt_label"] if m["gt_label"] is not None else -1 for m in meta])

    # ── 2. PCA (общий для всех методов) ─────────────────────────────────────
    pca_n_cfg = min(int(config.get("pca_components", 20)), n_defects_total - 1,
                   emb_matrix.shape[1])
    pca_n_cfg = max(2, pca_n_cfg)
    pca = PCA(n_components=pca_n_cfg, random_state=random_state)
    X_pca_all = pca.fit_transform(emb_matrix)
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X_pca_all)
    pca_n = pca.n_components_
    pca_variance = pca.explained_variance_ratio_.tolist()

    # ── 3. Классификация — supervised или unsupervised ───────────────────────
    clusterer: Any = None
    labels_all: np.ndarray
    n_clusters: int
    silhouette: Optional[float] = None
    overall_acc: Optional[float] = None
    mapping: Dict[int, int] = {}
    extra: Dict = {}   # дополнительные поля для supervised результатов

    if method == "prototype":
        # ── Prototype: few-shot ────────────────────────────────────────────
        n_proto = int(config.get("n_prototypes", 3))
        valid_mask = gt_arr >= 0
        if not valid_mask.any():
            labels_all = np.zeros(n_defects_total, dtype=int)
            n_clusters = 1
        else:
            prototypes, proto_meta_info = fit_prototype_classifier(
                X_scaled_all[valid_mask], gt_arr[valid_mask], n_proto
            )
            labels_all = predict_prototype(X_scaled_all, prototypes)
            n_clusters = len(prototypes)
            overall_acc, per_class = prototype_accuracy(labels_all, gt_arr)
            mapping = {k: k for k in prototypes}  # прямое соответствие
            extra = {
                "train_accuracy": None, "test_accuracy": overall_acc,
                "per_class_accuracy": per_class,
                "n_prototypes_used": n_proto,
                "prototype_classes": list(prototypes.keys()),
                "method_type": "few-shot",
            }
            clusterer = prototypes  # сохраняем для reuse

    elif method == "linear":
        # ── Linear: supervised upper bound ────────────────────────────────
        valid_mask = gt_arr >= 0
        if not valid_mask.any() or len(set(gt_arr[valid_mask])) < 2:
            labels_all = np.zeros(n_defects_total, dtype=int)
            n_clusters = 1
        else:
            lin = fit_linear_classifier(
                X_scaled_all[valid_mask], gt_arr[valid_mask],
                test_size=float(config.get("test_size", 0.5)),
                random_state=random_state,
            )
            labels_all = np.full(n_defects_total, -1, dtype=int)
            labels_all[valid_mask] = predict_linear(X_scaled_all[valid_mask], lin["clf"])
            labels_all = np.where(labels_all < 0, 0, labels_all)
            n_clusters = len(lin["classes"])
            overall_acc = lin["test_accuracy"]
            mapping = {k: k for k in lin["classes"]}
            extra = {
                "train_accuracy":          lin["train_accuracy"],
                "test_accuracy":           lin["test_accuracy"],
                "per_class_accuracy":      lin["per_class_test"],
                "n_train":                 lin["n_train"],
                "n_test":                  lin["n_test"],
                "method_type":             "supervised",
                "n_classes_in_test":       lin.get("n_classes_in_test", 0),
                "excluded_classes":        lin.get("excluded_classes", []),
                "stratification_warnings": lin.get("stratification_warnings", []),
            }
            clusterer = lin["clf"]

    elif method == "knn":
        # ── KNN classifier ─────────────────────────────────────────────────
        valid_mask = gt_arr >= 0
        if not valid_mask.any() or len(set(gt_arr[valid_mask])) < 2:
            labels_all = np.zeros(n_defects_total, dtype=int)
            n_clusters = 1
        else:
            res = fit_knn_classifier(
                X_scaled_all[valid_mask], gt_arr[valid_mask],
                n_neighbors=int(config.get("n_neighbors", 5)),
                metric=str(config.get("knn_metric", "euclidean")),
                test_size=float(config.get("test_size", 0.5)),
                random_state=random_state,
            )
            labels_all = np.full(n_defects_total, -1, dtype=int)
            labels_all[valid_mask] = predict_knn(X_scaled_all[valid_mask], res["clf"])
            labels_all = np.where(labels_all < 0, 0, labels_all)
            n_clusters = len(res["classes"])
            overall_acc = res["test_accuracy"]
            mapping = {k: k for k in res["classes"]}
            extra = {
                "train_accuracy":          res["train_accuracy"],
                "test_accuracy":           res["test_accuracy"],
                "per_class_accuracy":      res["per_class_test"],
                "n_train":                 res["n_train"],
                "n_test":                  res["n_test"],
                "n_neighbors":             res["n_neighbors"],
                "knn_metric":              res["metric"],
                "method_type":             "knn",
                "n_classes_in_test":       res.get("n_classes_in_test", 0),
                "excluded_classes":        res.get("excluded_classes", []),
                "stratification_warnings": res.get("stratification_warnings", []),
            }
            clusterer = res["clf"]

    elif method == "rf":
        # ── Random Forest ──────────────────────────────────────────────────
        valid_mask = gt_arr >= 0
        if not valid_mask.any() or len(set(gt_arr[valid_mask])) < 2:
            labels_all = np.zeros(n_defects_total, dtype=int)
            n_clusters = 1
        else:
            res = fit_random_forest_classifier(
                X_scaled_all[valid_mask], gt_arr[valid_mask],
                n_estimators=int(config.get("n_estimators", 100)),
                max_depth=config.get("max_depth", None),
                test_size=float(config.get("test_size", 0.5)),
                random_state=random_state,
            )
            labels_all = np.full(n_defects_total, -1, dtype=int)
            labels_all[valid_mask] = predict_random_forest(X_scaled_all[valid_mask], res["clf"])
            labels_all = np.where(labels_all < 0, 0, labels_all)
            n_clusters = len(res["classes"])
            overall_acc = res["test_accuracy"]
            mapping = {k: k for k in res["classes"]}
            extra = {
                "train_accuracy":          res["train_accuracy"],
                "test_accuracy":           res["test_accuracy"],
                "per_class_accuracy":      res["per_class_test"],
                "n_train":                 res["n_train"],
                "n_test":                  res["n_test"],
                "feature_importance":      res["feature_importance"],
                "n_estimators":            res["n_estimators"],
                "max_depth":               res["max_depth"],
                "method_type":             "random_forest",
                "n_classes_in_test":       res.get("n_classes_in_test", 0),
                "excluded_classes":        res.get("excluded_classes", []),
                "stratification_warnings": res.get("stratification_warnings", []),
            }
            clusterer = res["clf"]

    elif method == "dt":
        # ── Decision Tree ──────────────────────────────────────────────────
        valid_mask = gt_arr >= 0
        if not valid_mask.any() or len(set(gt_arr[valid_mask])) < 2:
            labels_all = np.zeros(n_defects_total, dtype=int)
            n_clusters = 1
        else:
            res = fit_decision_tree_classifier(
                X_scaled_all[valid_mask], gt_arr[valid_mask],
                max_depth=config.get("max_depth", 5),
                test_size=float(config.get("test_size", 0.5)),
                random_state=random_state,
            )
            labels_all = np.full(n_defects_total, -1, dtype=int)
            labels_all[valid_mask] = predict_decision_tree(X_scaled_all[valid_mask], res["clf"])
            labels_all = np.where(labels_all < 0, 0, labels_all)
            n_clusters = len(res["classes"])
            overall_acc = res["test_accuracy"]
            mapping = {k: k for k in res["classes"]}
            extra = {
                "train_accuracy":          res["train_accuracy"],
                "test_accuracy":           res["test_accuracy"],
                "per_class_accuracy":      res["per_class_test"],
                "n_train":                 res["n_train"],
                "n_test":                  res["n_test"],
                "feature_importance":      res["feature_importance"],
                "tree_depth":              res["tree_depth"],
                "max_depth":               res["max_depth"],
                "method_type":             "decision_tree",
                "n_classes_in_test":       res.get("n_classes_in_test", 0),
                "excluded_classes":        res.get("excluded_classes", []),
                "stratification_warnings": res.get("stratification_warnings", []),
            }
            clusterer = res["clf"]

    else:
        # ── Unsupervised: kmeans / gmm / hierarchical ──────────────────────
        _, _, clusterer, labels_all, n_clusters = fit_pca_and_cluster(
            emb_matrix, config, random_state
        )
        if n_defects_total > 1 and n_clusters > 1:
            try:
                silhouette = round(float(silhouette_score(
                    X_scaled_all, labels_all, sample_size=min(500, n_defects_total)
                )), 4)
            except Exception:
                pass
        if (gt_arr >= 0).any():
            overall_acc, mapping = hungarian_accuracy(
                labels_all, gt_arr, n_clusters, len(DEFECT_TYPE_NAMES)
            )

    # ── 4. UMAP / t-SNE ─────────────────────────────────────────────────────
    umap_coords: Optional[np.ndarray] = None
    if compute_umap and n_defects_total >= 2:
        umap_coords = compute_2d_projection(emb_matrix, pca, umap_method, random_state)

    umap_points: Optional[List[Dict]] = None
    if umap_coords is not None:
        umap_points = [
            {
                "x":          float(umap_coords[i, 0]),
                "y":          float(umap_coords[i, 1]),
                "cluster":    int(labels_all[i]),
                "gt_label":   meta[i]["gt_label"],
                "gt_name":    DEFECT_TYPE_NAMES[meta[i]["gt_label"]]
                               if meta[i]["gt_label"] is not None else None,
                "image_path": meta[i]["image_path"],
                "area":       meta[i]["area"],
                "centroid":   meta[i]["centroid"],
            }
            for i in range(n_defects_total)
        ]

    # ── 5. Per-image результаты ──────────────────────────────────────────────
    per_image: List[Dict] = []
    for img_i in range(n_images):
        img_indices = [j for j, m in enumerate(meta) if m["image_idx"] == img_i]
        img_labels  = [int(labels_all[j]) for j in img_indices]
        img_gt      = gt_labels_safe[img_i]

        img_acc: Optional[float] = None
        img_map: Dict[int, int] = {}
        if img_gt is not None and len(img_labels) > 0:
            gt_arr_i = np.full(len(img_labels), int(img_gt))
            img_acc, img_map = hungarian_accuracy(
                np.array(img_labels), gt_arr_i, n_clusters, len(DEFECT_TYPE_NAMES)
            )

        per_image.append({
            "image_path":     image_paths_safe[img_i],
            "n_defects":      len(img_indices),
            "cluster_labels": img_labels,
            "gt_label":       img_gt,
            "gt_name":        DEFECT_TYPE_NAMES[img_gt] if img_gt is not None else None,
            "accuracy":       img_acc,
            "mapping":        img_map or mapping,
        })

    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "overall_accuracy":  overall_acc,
        "silhouette":        silhouette,
        "n_clusters":        n_clusters,
        "n_defects":         n_defects_total,
        "pca_variance":      [round(v, 4) for v in pca_variance[:10]],
        "pca_components":    pca_n,
        "embedding_dim":     embedding_dim,
        "method":            method,
        "time_ms":           elapsed,
        "per_image":         per_image,
        "umap_points":       umap_points,
        "umap_method":       umap_method,
        "mapping":           mapping,
        "fitted_pca":        pca,
        "fitted_scaler":     scaler,
        "fitted_clusterer":  clusterer,
        "n_images":          n_images,
        **extra,             # train_accuracy, test_accuracy, per_class_accuracy, ...
    }


# ---------------------------------------------------------------------------
# Stateless per-image (для параллельных воркеров с pre-fitted моделями)
# ---------------------------------------------------------------------------

def classify_single_by_embeddings(
    patch_embeddings: np.ndarray,
    spatial_shape: Tuple[int, int],
    orig_shape: Tuple[int, int],
    contours: List[np.ndarray],
    pca: PCA,
    scaler: StandardScaler,
    clusterer: Any,
    method: str,
    n_clusters: int,
    gt_label: Optional[int] = None,
    image_path: str = "",
) -> Dict:
    """
    Применяет pre-fitted PCA+clusterer к одному изображению.
    Используется в параллельных воркерах (thread/process).

    Все аргументы — picklable (numpy, sklearn).
    """
    orig_h, orig_w = orig_shape
    embeddings = []
    valid_contours = []

    for c in contours:
        vec = extract_defect_embedding(patch_embeddings, spatial_shape, orig_h, orig_w, c)
        if vec is not None:
            embeddings.append(vec)
            valid_contours.append(c)

    if not embeddings or pca is None:
        return {
            "image_path":     image_path,
            "n_defects":      len(contours),
            "cluster_labels": [],
            "gt_label":       gt_label,
            "accuracy":       None,
            "mapping":        {},
        }

    emb_matrix = np.stack(embeddings, axis=0)
    X_pca = pca.transform(emb_matrix)
    X = scaler.transform(X_pca)

    # Dispatch по методу
    if method == "prototype":
        # clusterer — это dict prototypes {class_id: vector}
        labels = predict_prototype(X, clusterer) if clusterer else np.zeros(len(X), dtype=int)
        acc = None
        mp: Dict[int, int] = {k: k for k in (clusterer or {}).keys()}
        if gt_label is not None and len(labels) > 0:
            gt_arr_i = np.full(len(labels), int(gt_label))
            valid = gt_arr_i >= 0
            if valid.any():
                acc = round(float((labels[valid] == gt_arr_i[valid]).mean()), 4)
    elif method == "linear":
        labels = predict_linear(X, clusterer)
        acc = None
        mp = {k: k for k in range(n_clusters)}
        if gt_label is not None and len(labels) > 0:
            gt_arr_i = np.full(len(labels), int(gt_label))
            acc = round(float((labels == gt_arr_i).mean()), 4)
    elif method == "knn":
        labels = predict_knn(X, clusterer)
        acc = None
        mp = {k: k for k in range(n_clusters)}
        if gt_label is not None and len(labels) > 0:
            gt_arr_i = np.full(len(labels), int(gt_label))
            acc = round(float((labels == gt_arr_i).mean()), 4)
    elif method == "rf":
        labels = predict_random_forest(X, clusterer)
        acc = None
        mp = {k: k for k in range(n_clusters)}
        if gt_label is not None and len(labels) > 0:
            gt_arr_i = np.full(len(labels), int(gt_label))
            acc = round(float((labels == gt_arr_i).mean()), 4)
    elif method == "dt":
        labels = predict_decision_tree(X, clusterer)
        acc = None
        mp = {k: k for k in range(n_clusters)}
        if gt_label is not None and len(labels) > 0:
            gt_arr_i = np.full(len(labels), int(gt_label))
            acc = round(float((labels == gt_arr_i).mean()), 4)
    else:
        labels = predict_in_pca_space(emb_matrix, pca, scaler, clusterer, method, n_clusters)
        acc = None
        mp = {}
        if gt_label is not None and len(labels) > 0:
            gt_arr_i = np.full(len(labels), int(gt_label))
            acc, mp = hungarian_accuracy(labels, gt_arr_i, n_clusters, len(DEFECT_TYPE_NAMES))

    return {
        "image_path":     image_path,
        "n_defects":      len(contours),
        "cluster_labels": labels.tolist() if hasattr(labels, "tolist") else list(labels),
        "gt_label":       gt_label,
        "gt_name":        DEFECT_TYPE_NAMES[gt_label] if gt_label is not None else None,
        "accuracy":       acc,
        "mapping":        mp,
    }


# ---------------------------------------------------------------------------
# Worker для ProcessPool (module-level)
# ---------------------------------------------------------------------------

def _emb_process_worker(args: tuple) -> Dict:
    """
    ProcessPool worker: classify_single_by_embeddings с pickle-сериализацией.
    emb_bytes передаются как bytes для экономии overhead.
    """
    import pickle as _pk
    (
        emb_bytes, emb_shape,
        sp_shape, orig_shape,
        contours_pts,        # List[np.ndarray] из tolist()
        pca_bytes, scaler_bytes, clusterer_bytes,
        method, n_clusters,
        gt_label, image_path,
    ) = args

    patch_embs = np.frombuffer(emb_bytes, dtype=np.float32).reshape(emb_shape)
    contours = [np.array(pts, dtype=np.int32) for pts in contours_pts]
    pca       = _pk.loads(pca_bytes) if pca_bytes else None
    scaler    = _pk.loads(scaler_bytes) if scaler_bytes else None
    clusterer = _pk.loads(clusterer_bytes) if clusterer_bytes else None

    return classify_single_by_embeddings(
        patch_embs, sp_shape, orig_shape, contours,
        pca, scaler, clusterer, method, n_clusters, gt_label, image_path,
    )


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def _empty_emb_result(method: str, elapsed: float, n_images: int, emb_dim: int) -> Dict:
    return {
        "overall_accuracy": None,
        "silhouette":       None,
        "n_clusters":       0,
        "n_defects":        0,
        "pca_variance":     [],
        "pca_components":   0,
        "embedding_dim":    emb_dim,
        "method":           method,
        "time_ms":          round(elapsed, 2),
        "per_image":        [],
        "umap_points":      None,
        "umap_method":      None,
        "mapping":          {},
        "fitted_pca":       None,
        "fitted_scaler":    None,
        "fitted_clusterer": None,
        "n_images":         n_images,
    }


def result_to_json_safe(r: Dict) -> Dict:
    """Удаляет sklearn-объекты для JSON сериализации."""
    return {
        k: v for k, v in r.items()
        if k not in ("fitted_pca", "fitted_scaler", "fitted_clusterer")
    }
