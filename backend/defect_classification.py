# -*- coding: utf-8 -*-
"""
ML-классификация дефектов на основе PatchCore anomaly map.

Три метода кластеризации:
  kmeans      — KMeans (default, быстрый, хорошо масштабируется)
  gmm         — GaussianMixture (мягкое присвоение, лучше при перекрытии кластеров)
  hierarchical — AgglomerativeClustering (без предположения о форме кластеров)

Признаки на дефект (connected component):
  max_anomaly_score   — максимальный аномальный score внутри компоненты
  mean_anomaly_score  — средний score
  std_anomaly_score   — стандартное отклонение score
  area                — площадь в пикселях
  equivalent_diameter — эквивалентный диаметр (√(4·area/π))
  aspect_ratio        — соотношение осей эллипса
  solidity            — area / convex_hull_area
  extent              — area / bounding_rect_area
  eccentricity        — эксцентриситет эллипса

Автоподбор K (если n_clusters=None):
  Перебирает K от 2 до min(8, n_defects-1), выбирает по максимальному silhouette score.

Accuracy через Hungarian algorithm:
  Если заданы gt_labels — строит cost matrix и находит оптимальное
  соответствие cluster_id → gt_type_id через linear_sum_assignment.

Функция classify_defects — stateless:
  Передаётся в любой executor без изменений.
  fitted_state=(scaler, model) позволяет переиспользовать обученные модели
  в параллельных воркерах (сериализуются через pickle).
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "max_anomaly_score",
    "mean_anomaly_score",
    "std_anomaly_score",
    "area",
    "equivalent_diameter",
    "aspect_ratio",
    "solidity",
    "extent",
    "eccentricity",
]

DEFECT_TYPE_NAMES: List[str] = [
    "color",
    "cut",
    "hole",
    "metal_contamination",
    "thread",
]

METHOD_LABELS: Dict[str, str] = {
    "kmeans":       "K-Means",
    "gmm":          "GMM (Gaussian Mixture)",
    "hierarchical": "Hierarchical (Ward)",
}


# ---------------------------------------------------------------------------
# Извлечение признаков одного connected component
# ---------------------------------------------------------------------------

def extract_component_features(
    contour: np.ndarray,
    anomaly_map: np.ndarray,
) -> Optional[Dict[str, float]]:
    """
    Извлекает вектор из 9 признаков для одного connected component.

    Args:
        contour:    numpy array контура из cv2.findContours
        anomaly_map: нормализованная карта аномалий [H, W] float32

    Returns:
        dict с 9 признаками или None если компонента слишком мала
    """
    area = float(cv2.contourArea(contour))
    if area < 1.0:
        return None

    h, w = anomaly_map.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)

    pixels = anomaly_map[mask > 0]
    if len(pixels) == 0:
        return None

    max_score  = float(np.max(pixels))
    mean_score = float(np.mean(pixels))
    std_score  = float(np.std(pixels)) if len(pixels) > 1 else 0.0

    equivalent_diameter = math.sqrt(4.0 * area / math.pi)

    bx, by, bw, bh = cv2.boundingRect(contour)
    extent = area / (bw * bh) if bw * bh > 0 else 0.0

    aspect_ratio = 1.0
    eccentricity = 0.0
    if len(contour) >= 5:
        try:
            (_, _), (ma, mi), _ = cv2.fitEllipse(contour)
            major = max(float(ma), float(mi))
            minor = min(float(ma), float(mi))
            aspect_ratio = major / max(minor, 1e-6)
            eccentricity = math.sqrt(max(0.0, 1.0 - (minor / max(major, 1e-6)) ** 2))
        except cv2.error:
            pass

    hull      = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity  = area / hull_area if hull_area > 0 else 0.0

    return {
        "max_anomaly_score":   max_score,
        "mean_anomaly_score":  mean_score,
        "std_anomaly_score":   std_score,
        "area":                area,
        "equivalent_diameter": equivalent_diameter,
        "aspect_ratio":        aspect_ratio,
        "solidity":            solidity,
        "extent":              extent,
        "eccentricity":        eccentricity,
    }


def features_to_vector(feat: Dict[str, float]) -> np.ndarray:
    """Конвертирует dict признаков в numpy вектор в порядке FEATURE_NAMES."""
    return np.array([feat[name] for name in FEATURE_NAMES], dtype=np.float32)


# ---------------------------------------------------------------------------
# Автоподбор S_min / S_max из percentile площадей
# ---------------------------------------------------------------------------

_S_MAX_FALLBACK = 99999.0  # используется вместо inf — валидно в JSON


def auto_thresholds(
    areas: List[float],
    s_min_pct: float = 5.0,
    s_max_pct: float = 95.0,
) -> Tuple[float, float]:
    """
    Вычисляет пороги диаметра из percentile-распределения площадей.

    Returns:
        (s_min_diameter, s_max_diameter) — пороги в единицах диаметра.
        Всегда возвращает конечные числа (нет inf/nan).
    """
    if not areas:
        return 0.0, _S_MAX_FALLBACK
    arr = np.array(areas, dtype=np.float64)
    a_min = float(np.percentile(arr, s_min_pct))
    a_max = float(np.percentile(arr, s_max_pct))
    d_min = math.sqrt(4.0 * max(0.0, a_min) / math.pi)
    d_max = math.sqrt(4.0 * max(0.0, a_max) / math.pi)
    return d_min, max(d_max, d_min + 1.0)


# ---------------------------------------------------------------------------
# Автоподбор K через silhouette score
# ---------------------------------------------------------------------------

def auto_select_k(
    X: np.ndarray,
    k_range: Tuple[int, int] = (2, 8),
    method: str = "kmeans",
    random_state: int = 42,
) -> Tuple[int, float]:
    """
    Перебирает K в диапазоне k_range, выбирает по максимальному silhouette score.

    Args:
        X:            матрица признаков [N, D], уже нормализованная
        k_range:      диапазон (k_min, k_max) включительно
        method:       "kmeans" | "gmm" | "hierarchical"
        random_state: seed для воспроизводимости

    Returns:
        (best_k, best_silhouette)
    """
    n = len(X)
    k_min = max(2, k_range[0])
    k_max = min(k_range[1], n - 1)

    if k_max < k_min:
        return k_min, -1.0

    best_k, best_score = k_min, -1.0

    for k in range(k_min, k_max + 1):
        try:
            labels = _fit_predict(X, k, method, random_state)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels, sample_size=min(1000, len(X)))
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as exc:
            logger.debug("auto_select_k k=%d: %s", k, exc)

    return best_k, best_score


def _fit_predict(
    X: np.ndarray,
    k: int,
    method: str,
    random_state: int = 42,
) -> np.ndarray:
    """Fits and predicts cluster labels for X. Гарантирует k <= n_samples."""
    n = len(X)
    k = max(1, min(k, n))  # guard: k не может превышать число образцов
    if k == 1 or n <= 1:
        return np.zeros(n, dtype=int)

    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        return model.fit_predict(X)
    elif method == "gmm":
        model = GaussianMixture(
            n_components=k, random_state=random_state,
            covariance_type="full", max_iter=200,
        )
        model.fit(X)
        return model.predict(X)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return model.fit_predict(X)
    else:
        raise ValueError(f"Неизвестный метод: {method}")


# ---------------------------------------------------------------------------
# Hungarian matching: cluster → GT label
# ---------------------------------------------------------------------------

def hungarian_accuracy(
    cluster_labels: np.ndarray,
    gt_labels: np.ndarray,
    n_clusters: int,
    n_gt_classes: int,
) -> Tuple[float, Dict[int, int]]:
    """
    Вычисляет accuracy через Hungarian algorithm.

    Строит cost matrix [n_clusters × n_gt_classes], где
    cost[i][j] = -(кол-во дефектов с cluster=i И gt=j).
    Optimal assignment → accuracy.

    Args:
        cluster_labels: целочисленные метки кластеров [N]
        gt_labels:      целочисленные GT-метки [N] (0..n_gt_classes-1)
        n_clusters:     количество кластеров K
        n_gt_classes:   количество GT-классов (обычно 5 для MVTec carpet)

    Returns:
        (accuracy, mapping): accuracy ∈ [0, 1], mapping cluster_id → gt_id
    """
    n = len(cluster_labels)
    if n == 0:
        return 0.0, {}

    size = max(n_clusters, n_gt_classes)
    cost = np.zeros((size, size), dtype=np.int64)
    for cl, gt in zip(cluster_labels, gt_labels):
        cost[int(cl)][int(gt)] += 1

    row_ind, col_ind = linear_sum_assignment(-cost)
    correct = int(cost[row_ind, col_ind].sum())
    accuracy = correct / n
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind) if r < n_clusters}
    return accuracy, mapping


# ---------------------------------------------------------------------------
# Построение обученных моделей (для параллельной передачи)
# ---------------------------------------------------------------------------

def fit_classifier(
    feature_matrix: np.ndarray,
    n_clusters: int,
    method: str = "kmeans",
    random_state: int = 42,
) -> Tuple[StandardScaler, Any]:
    """
    Обучает StandardScaler + clustering model на переданной матрице признаков.

    Returns:
        (fitted_scaler, fitted_model) — оба pickle-совместимы для передачи в ProcessPool.
        Гарантирует n_clusters <= n_samples.
    """
    n = len(feature_matrix)
    n_clusters = max(1, min(n_clusters, n))  # guard

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)

    if n_clusters == 1 or n <= 1:
        return scaler, None

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        model.fit(X)
    elif method == "gmm":
        model = GaussianMixture(
            n_components=n_clusters, random_state=random_state,
            covariance_type="full", max_iter=200,
        )
        model.fit(X)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        model.fit(X)
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    return scaler, model


def predict_with_fitted(
    feature_matrix: np.ndarray,
    scaler: StandardScaler,
    model: Any,
    method: str,
) -> np.ndarray:
    """Применяет уже обученные scaler+model к новым данным. Защита от n_samples < n_clusters."""
    n = len(feature_matrix)
    if n == 0:
        return np.array([], dtype=int)
    X = scaler.transform(feature_matrix)
    if model is None or n == 1:
        return np.zeros(n, dtype=int)

    if method == "kmeans":
        return model.predict(X)  # KMeans.predict работает при любом n_samples
    elif method == "gmm":
        return model.predict(X)  # GMM.predict работает при любом n_samples
    elif method == "hierarchical":
        # AgglomerativeClustering не имеет .predict → пересчитываем с безопасным k
        k_orig = getattr(model, "n_clusters_", None) or getattr(model, "n_clusters", 2)
        k_safe = max(1, min(int(k_orig), n))
        if k_safe <= 1:
            return np.zeros(n, dtype=int)
        new_model = AgglomerativeClustering(n_clusters=k_safe, linkage="ward")
        return new_model.fit_predict(X)
    else:
        raise ValueError(f"Неизвестный метод: {method}")


# ---------------------------------------------------------------------------
# Основная stateless функция
# ---------------------------------------------------------------------------

def classify_defects(
    anomaly_map: np.ndarray,
    binary_mask: np.ndarray,
    config: Dict,
    contours: Optional[List[np.ndarray]] = None,
    gt_label: Optional[int] = None,
    fitted_state: Optional[Tuple[StandardScaler, Any]] = None,
) -> Dict:
    """
    Stateless ML-классификация дефектов для одного изображения.

    Эта функция передаётся в любой executor без изменений:
    - Thread-based: GIL не проблема (sklearn освобождает GIL для KMeans)
    - Process-based: fitted_state сериализуется через pickle
    - Async: запускается через loop.run_in_executor

    Args:
        anomaly_map:   нормализованная карта аномалий [H, W] float32
        binary_mask:   бинарная маска дефектов [H, W] uint8 (255=дефект)
        config:        конфигурация классификации (dict):
                         method: "kmeans" | "gmm" | "hierarchical"
                         n_clusters: int | None (None = автоподбор)
                         k_range: [int, int] (диапазон для автоподбора K)
                         s_min_percentile: float (default 5)
                         s_max_percentile: float (default 95)
                         random_state: int (default 42)
                         score_only: bool (только score-признаки, без геометрии)
        contours:      предвычисленные контуры (если None — вычисляются)
        gt_label:      GT-метка изображения (index в DEFECT_TYPE_NAMES)
        fitted_state:  (scaler, model) — использовать вместо обучения с нуля

    Returns dict:
        n_defects:       int — количество найденных компонент
        n_valid:         int — после фильтрации по s_min/s_max
        cluster_labels:  List[int]
        features:        List[Dict[str, float]] — 9 признаков на дефект
        feature_vectors: List[List[float]] — в порядке FEATURE_NAMES
        silhouette:      float | None
        method:          str
        n_clusters:      int
        time_ms:         float
        s_min_auto:      float
        s_max_auto:      float
        gt_label:        int | None
        mapping:         Dict[int, int] cluster→gt (если gt доступен)
        accuracy_single: float | None (Hungarian accuracy для этого изображения)
        fitted_scaler:   StandardScaler (для переиспользования)
        fitted_model:    clusterer (для переиспользования)
    """
    t0 = time.perf_counter()

    method       = config.get("method", "kmeans")
    n_clusters   = config.get("n_clusters", None)
    k_range      = config.get("k_range", [2, 8])
    s_min_pct    = float(config.get("s_min_percentile", 5.0))
    s_max_pct    = float(config.get("s_max_percentile", 95.0))
    random_state = int(config.get("random_state", 42))
    score_only   = bool(config.get("score_only", False))

    # ── Шаг 1: найти контуры ────────────────────────────────────────────────
    if contours is None:
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = list(contours)

    # ── Шаг 2: извлечь признаки ─────────────────────────────────────────────
    all_features: List[Dict[str, float]] = []
    valid_contours: List[np.ndarray] = []

    for c in contours:
        feat = extract_component_features(c, anomaly_map)
        if feat is not None:
            all_features.append(feat)
            valid_contours.append(c)

    n_defects = len(contours)
    n_valid   = len(all_features)

    # Автоподбор S_min / S_max из площадей
    areas = [f["area"] for f in all_features]
    s_min_auto, s_max_auto = auto_thresholds(areas, s_min_pct, s_max_pct)

    # Фильтруем по диаметру (если S_min/S_max заданы в config)
    config_s_min = config.get("s_min")
    config_s_max = config.get("s_max")
    use_s_min = float(config_s_min) if config_s_min is not None else s_min_auto
    use_s_max = float(config_s_max) if config_s_max is not None else s_max_auto

    filtered_features: List[Dict[str, float]] = []
    filtered_indices:  List[int] = []
    for i, feat in enumerate(all_features):
        d = feat["equivalent_diameter"]
        if use_s_min <= d <= use_s_max:
            filtered_features.append(feat)
            filtered_indices.append(i)

    # Если после фильтрации ничего нет — возвращаем пустой результат
    if not filtered_features:
        elapsed = (time.perf_counter() - t0) * 1_000
        return _empty_result(
            n_defects, n_valid, method, elapsed,
            s_min_auto, s_max_auto, gt_label,
        )

    # Только score-признаки (альтернативный метод)
    if score_only:
        feature_keys = ["max_anomaly_score", "mean_anomaly_score", "std_anomaly_score"]
    else:
        feature_keys = FEATURE_NAMES

    # Матрица признаков [N_valid, D]
    feature_matrix = np.array(
        [[f[k] for k in feature_keys] for f in filtered_features],
        dtype=np.float32,
    )

    # ── Шаг 3: нормализация + кластеризация ─────────────────────────────────
    n_valid_filtered = len(filtered_features)
    sil_score: Optional[float] = None

    if fitted_state is not None:
        fitted_scaler, fitted_model = fitted_state
        try:
            labels = predict_with_fitted(feature_matrix, fitted_scaler, fitted_model, method)
        except Exception:
            fitted_state = None  # fallback: обучаем заново

    if fitted_state is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(feature_matrix)

        # Определяем K
        if n_clusters is None:
            k_auto, sil_score = auto_select_k(
                X, (k_range[0], min(k_range[1], n_valid_filtered - 1)), method, random_state
            )
            k = k_auto
        else:
            k = min(int(n_clusters), n_valid_filtered)
            k = max(1, k)

        if k == 1 or n_valid_filtered <= 1:
            labels = np.zeros(n_valid_filtered, dtype=int)
            fitted_model = None
            fitted_scaler = scaler
        else:
            labels = _fit_predict(X, k, method, random_state)
            fitted_scaler, fitted_model = fit_classifier(
                feature_matrix, k, method, random_state
            )

        n_clusters_used = int(max(labels) + 1) if len(labels) > 0 else 1
    else:
        fitted_scaler, fitted_model = fitted_state
        n_clusters_used = getattr(fitted_model, "n_clusters", None) or getattr(
            fitted_model, "n_components", 1
        )

    # Silhouette (если не вычислен при авто-K)
    if sil_score is None and n_valid_filtered > 1 and len(set(labels)) > 1:
        try:
            X_for_sil = fitted_scaler.transform(feature_matrix)
            sil_score = float(silhouette_score(X_for_sil, labels))
        except Exception:
            pass

    # ── Шаг 4: Hungarian matching (если есть GT) ────────────────────────────
    mapping: Dict[int, int] = {}
    accuracy_single: Optional[float] = None

    if gt_label is not None and n_valid_filtered > 0:
        gt_arr = np.full(n_valid_filtered, int(gt_label), dtype=int)
        n_gt  = len(DEFECT_TYPE_NAMES)
        accuracy_single, mapping = hungarian_accuracy(
            labels, gt_arr, n_clusters_used, n_gt
        )

    elapsed = (time.perf_counter() - t0) * 1_000

    return {
        "n_defects":         n_defects,
        "n_valid":           n_valid_filtered,
        "cluster_labels":    labels.tolist() if hasattr(labels, "tolist") else list(labels),
        "features":          [
            {k: round(v, 6) for k, v in f.items()} for f in filtered_features
        ],
        "feature_vectors":   [
            [round(f[k], 6) for k in feature_keys] for f in filtered_features
        ],
        "feature_names":     feature_keys,
        "silhouette":        round(sil_score, 4) if sil_score is not None else None,
        "method":            method,
        "n_clusters":        n_clusters_used,
        "time_ms":           round(elapsed, 3),
        "s_min_auto":        round(s_min_auto, 2),
        "s_max_auto":        round(s_max_auto, 2),
        "gt_label":          gt_label,
        "mapping":           mapping,
        "accuracy_single":   round(accuracy_single, 4) if accuracy_single is not None else None,
        "fitted_scaler":     fitted_scaler,
        "fitted_model":      fitted_model,
    }


# ---------------------------------------------------------------------------
# Batch-версия: все изображения сразу → fit один раз → predict per image
# ---------------------------------------------------------------------------

def classify_defects_batch(
    anomaly_maps: List[np.ndarray],
    binary_masks: List[np.ndarray],
    config: Dict,
    contours_list: Optional[List[List[np.ndarray]]] = None,
    gt_labels: Optional[List[Optional[int]]] = None,
    image_paths: Optional[List[str]] = None,
) -> Dict:
    """
    Batch ML-классификация: fit scaler+model на ВСЕХ дефектах из всех изображений,
    затем predict для каждого изображения отдельно.

    Vectorized операции: матрица признаков строится для всего батча.

    Args:
        anomaly_maps:   List[np.ndarray] — по одному на каждое изображение
        binary_masks:   List[np.ndarray] — бинарные маски
        config:         словарь конфигурации (см. classify_defects)
        contours_list:  предвычисленные контуры (опционально)
        gt_labels:      GT-метки изображений (индексы в DEFECT_TYPE_NAMES)
        image_paths:    имена/пути для отчёта

    Returns dict:
        per_image:       List[Dict] — результат classify_defects для каждого изображения
        overall_accuracy: float | None — Hungarian accuracy по всему батчу
        silhouette_batch: float | None — silhouette по объединённой матрице
        n_clusters:      int
        method:          str
        time_ms:         float
        speedup_vs_sequential: float — ускорение vs поочерёдная fit+predict
    """
    t0 = time.perf_counter()

    method       = config.get("method", "kmeans")
    n_clusters   = config.get("n_clusters", None)
    k_range      = config.get("k_range", [2, 8])
    s_min_pct    = float(config.get("s_min_percentile", 5.0))
    s_max_pct    = float(config.get("s_max_percentile", 95.0))
    random_state = int(config.get("random_state", 42))
    score_only   = bool(config.get("score_only", False))
    feature_keys = ["max_anomaly_score", "mean_anomaly_score", "std_anomaly_score"] \
                   if score_only else FEATURE_NAMES

    n_images = len(anomaly_maps)
    gt_labels = gt_labels or [None] * n_images
    image_paths = image_paths or [str(i) for i in range(n_images)]
    if contours_list is None:
        contours_list = [None] * n_images

    # ── Шаг 1: извлечь признаки для всего батча ─────────────────────────────
    all_feats_global:    List[Dict[str, float]] = []
    all_gt_global:       List[int] = []
    per_image_feat_idxs: List[List[int]] = []  # индексы в all_feats_global
    per_image_contours:  List[List[np.ndarray]] = []

    for i, (amap, bmask) in enumerate(zip(anomaly_maps, binary_masks)):
        conts = contours_list[i]
        if conts is None:
            conts, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            conts = list(conts)
        per_image_contours.append(conts)

        idxs = []
        for c in conts:
            feat = extract_component_features(c, amap)
            if feat is not None:
                idxs.append(len(all_feats_global))
                all_feats_global.append(feat)
                if gt_labels[i] is not None:
                    all_gt_global.append(int(gt_labels[i]))
                else:
                    all_gt_global.append(-1)
        per_image_feat_idxs.append(idxs)

    # Автоподбор S_min/S_max по всему батчу
    all_areas = [f["area"] for f in all_feats_global]
    s_min_auto, s_max_auto = auto_thresholds(all_areas, s_min_pct, s_max_pct)
    config_s_min = float(config.get("s_min") or s_min_auto)
    config_s_max = float(config.get("s_max") or s_max_auto)

    # Фильтрация
    valid_mask_global = [
        config_s_min <= f["equivalent_diameter"] <= config_s_max
        for f in all_feats_global
    ]
    valid_feats = [f for f, ok in zip(all_feats_global, valid_mask_global) if ok]
    valid_gt    = [g for g, ok in zip(all_gt_global,    valid_mask_global) if ok]

    if not valid_feats:
        elapsed = (time.perf_counter() - t0) * 1_000
        return {
            "per_image":            [_empty_result(0, 0, method, 0.0, s_min_auto, s_max_auto, None)] * n_images,
            "overall_accuracy":     None,
            "silhouette_batch":     None,
            "n_clusters":           0,
            "method":               method,
            "time_ms":              round(elapsed, 3),
            "speedup_vs_sequential": 1.0,
        }

    # ── Шаг 2: fit один раз на всём батче ───────────────────────────────────
    fmat = np.array(
        [[f[k] for k in feature_keys] for f in valid_feats], dtype=np.float32
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(fmat)

    sil_batch: Optional[float] = None
    if n_clusters is None:
        k, sil_batch = auto_select_k(
            X, (k_range[0], min(k_range[1], len(valid_feats) - 1)), method, random_state
        )
    else:
        k = min(int(n_clusters), len(valid_feats))
        k = max(1, k)

    if k <= 1 or len(valid_feats) == 1:
        labels_all = np.zeros(len(valid_feats), dtype=int)
        fitted_model = None
    else:
        labels_all = _fit_predict(X, k, method, random_state)
        _, fitted_model = fit_classifier(fmat, k, method, random_state)

    n_clusters_used = int(max(labels_all) + 1) if len(labels_all) > 0 else 1

    if sil_batch is None and len(valid_feats) > 1 and len(set(labels_all)) > 1:
        try:
            sil_batch = float(silhouette_score(X, labels_all, sample_size=min(1000, len(X))))
        except Exception:
            pass

    # Строим mapping valid_global_idx → label
    valid_idx_counter = 0
    global_label_map: Dict[int, int] = {}
    for gi, ok in enumerate(valid_mask_global):
        if ok:
            global_label_map[gi] = int(labels_all[valid_idx_counter])
            valid_idx_counter += 1

    # ── Шаг 3: Hungarian accuracy по всему батчу ────────────────────────────
    overall_accuracy: Optional[float] = None
    overall_mapping:  Dict[int, int] = {}

    gt_available = [g for g in valid_gt if g >= 0]
    if gt_available and len(gt_available) == len(valid_gt):
        gt_arr = np.array(valid_gt, dtype=int)
        overall_accuracy, overall_mapping = hungarian_accuracy(
            labels_all, gt_arr, n_clusters_used, len(DEFECT_TYPE_NAMES)
        )

    # ── Шаг 4: собрать per-image результаты ─────────────────────────────────
    per_image: List[Dict] = []
    elapsed_extract = (time.perf_counter() - t0) * 1_000  # время до per-image сборки

    for i in range(n_images):
        idxs = per_image_feat_idxs[i]
        img_feats = [all_feats_global[gi] for gi in idxs]
        img_labels = [global_label_map[gi] for gi in idxs if gi in global_label_map]
        img_valid_feats = [all_feats_global[gi] for gi in idxs if gi in global_label_map]

        n_total_i = len(per_image_contours[i])
        n_valid_i = len(img_labels)

        img_areas = [f["area"] for f in img_feats]
        s_min_i, s_max_i = auto_thresholds(img_areas, s_min_pct, s_max_pct)

        img_sil: Optional[float] = None
        if n_valid_i > 1 and len(set(img_labels)) > 1:
            try:
                img_fmat = np.array([[f[k] for k in feature_keys] for f in img_valid_feats], dtype=np.float32)
                Xi = scaler.transform(img_fmat)
                img_sil = float(silhouette_score(Xi, np.array(img_labels)))
            except Exception:
                pass

        img_acc: Optional[float] = None
        img_mapping: Dict[int, int] = {}
        if gt_labels[i] is not None and n_valid_i > 0:
            gt_arr_i = np.full(n_valid_i, int(gt_labels[i]), dtype=int)
            img_acc, img_mapping = hungarian_accuracy(
                np.array(img_labels), gt_arr_i, n_clusters_used, len(DEFECT_TYPE_NAMES)
            )

        per_image.append({
            "image_path":       image_paths[i] if image_paths else str(i),
            "n_defects":        n_total_i,
            "n_valid":          n_valid_i,
            "cluster_labels":   img_labels,
            "features":         [{k: round(v, 6) for k, v in f.items()} for f in img_valid_feats],
            "feature_vectors":  [[round(f[k], 6) for k in feature_keys] for f in img_valid_feats],
            "feature_names":    feature_keys,
            "silhouette":       round(img_sil, 4) if img_sil is not None else None,
            "method":           method,
            "n_clusters":       n_clusters_used,
            "s_min_auto":       round(s_min_i, 2),
            "s_max_auto":       round(s_max_i, 2),
            "gt_label":         gt_labels[i],
            "mapping":          img_mapping or overall_mapping,
            "accuracy_single":  round(img_acc, 4) if img_acc is not None else None,
            "fitted_scaler":    scaler,
            "fitted_model":     fitted_model,
        })

    elapsed = (time.perf_counter() - t0) * 1_000

    # Оценка speedup vs поочерёдная обработка (эмпирическая)
    speedup = round(n_images * elapsed_extract / max(elapsed, 1.0), 3) if n_images > 1 else 1.0

    return {
        "per_image":             per_image,
        "overall_accuracy":      round(overall_accuracy, 4) if overall_accuracy is not None else None,
        "overall_mapping":       overall_mapping,
        "silhouette_batch":      round(sil_batch, 4) if sil_batch is not None else None,
        "n_clusters":            n_clusters_used,
        "method":                method,
        "time_ms":               round(elapsed, 3),
        "speedup_vs_sequential": speedup,
        "n_total_defects":       len(valid_feats),
        "s_min_auto":            round(s_min_auto, 2),
        "s_max_auto":            round(s_max_auto, 2),
        "fitted_scaler":         scaler,
        "fitted_model":          fitted_model,
    }


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _empty_result(
    n_defects: int,
    n_valid: int,
    method: str,
    elapsed: float,
    s_min: float,
    s_max: float,
    gt_label: Optional[int],
) -> Dict:
    """Возвращает пустой результат классификации."""
    return {
        "n_defects":        n_defects,
        "n_valid":          n_valid,
        "cluster_labels":   [],
        "features":         [],
        "feature_vectors":  [],
        "feature_names":    FEATURE_NAMES,
        "silhouette":       None,
        "method":           method,
        "n_clusters":       0,
        "time_ms":          round(elapsed, 3),
        "s_min_auto":       round(s_min, 2),
        "s_max_auto":       round(s_max, 2),
        "gt_label":         gt_label,
        "mapping":          {},
        "accuracy_single":  None,
        "fitted_scaler":    None,
        "fitted_model":     None,
    }


def compare_methods(
    anomaly_map: np.ndarray,
    binary_mask: np.ndarray,
    config: Dict,
    contours: Optional[List[np.ndarray]] = None,
    gt_label: Optional[int] = None,
) -> Dict[str, Dict]:
    """
    Запускает все 3 метода классификации и возвращает сравнение.
    Используется для выбора дефолтного метода.

    Returns:
        dict: {"kmeans": result, "gmm": result, "hierarchical": result,
               "best_method": str, "best_accuracy": float}
    """
    results: Dict[str, Dict] = {}
    for m in ["kmeans", "gmm", "hierarchical"]:
        cfg = {**config, "method": m}
        r = classify_defects(anomaly_map, binary_mask, cfg, contours, gt_label)
        results[m] = r

    # Выбираем лучший по accuracy (если есть GT) или по silhouette
    best_method = "kmeans"
    best_score  = -1.0

    for m, r in results.items():
        score = r.get("accuracy_single") or r.get("silhouette") or 0.0
        if score is not None and score > best_score:
            best_score  = score
            best_method = m

    return {
        **results,
        "best_method":   best_method,
        "best_score":    best_score,
    }


def classify_defects_score_only(
    anomaly_map: np.ndarray,
    binary_mask: np.ndarray,
    config: Dict,
    contours: Optional[List[np.ndarray]] = None,
    gt_label: Optional[int] = None,
) -> Dict:
    """
    Альтернативный метод: классификация только по anomaly score (без геометрии).
    Быстрее, полезен как baseline для сравнения.
    """
    return classify_defects(
        anomaly_map, binary_mask,
        {**config, "score_only": True},
        contours, gt_label,
    )


# ---------------------------------------------------------------------------
# Утилиты для подготовки результата к JSON (убираем sklearn-объекты)
# ---------------------------------------------------------------------------

def _sanitize_value(v: Any) -> Any:
    """Заменяет float('inf')/float('nan') на JSON-совместимые значения."""
    if isinstance(v, float):
        if math.isinf(v) or math.isnan(v):
            return None
    return v


def result_to_json_safe(result: Dict) -> Dict:
    """Удаляет sklearn-объекты и sanitize inf/nan значения для JSON-сериализации."""
    return {
        k: _sanitize_value(v)
        for k, v in result.items()
        if k not in ("fitted_scaler", "fitted_model")
    }


def batch_result_to_json_safe(result: Dict) -> Dict:
    """Версия для batch-результата."""
    clean = {k: v for k, v in result.items() if k not in ("fitted_scaler", "fitted_model")}
    if "per_image" in clean:
        clean["per_image"] = [result_to_json_safe(r) for r in clean["per_image"]]
    return clean


# ---------------------------------------------------------------------------
# Быстрое вычисление benchmark по всем 8 режимам
# ---------------------------------------------------------------------------

def benchmark_classification_modes(
    anomaly_map: np.ndarray,
    binary_mask: np.ndarray,
    config: Dict,
    contours: Optional[List[np.ndarray]] = None,
    n_repeats: int = 5,
) -> Dict[str, float]:
    """
    Замеряет время classify_defects n_repeats раз.
    Используется для benchmark chart в фронтенде.

    Returns:
        {"mean_ms": float, "min_ms": float, "max_ms": float, "std_ms": float}
    """
    times = []
    for _ in range(n_repeats):
        r = classify_defects(anomaly_map, binary_mask, config, contours)
        times.append(r["time_ms"])
    arr = np.array(times)
    return {
        "mean_ms": round(float(arr.mean()), 3),
        "min_ms":  round(float(arr.min()), 3),
        "max_ms":  round(float(arr.max()), 3),
        "std_ms":  round(float(arr.std()), 3),
    }
