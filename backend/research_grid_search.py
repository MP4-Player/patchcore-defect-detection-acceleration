# -*- coding: utf-8 -*-
"""
Joint optimization grid search для Research вкладки.

Архитектура:
  1. Предвычисляем PatchCore heatmaps + patch embeddings для всего датасета (один раз)
  2. Для каждой точки сетки (threshold, S_min, S_max):
     - Бинаризация + извлечение контуров (быстро, без GPU)
  3. Для каждого (pca_n, method):
     - Классификация на эмбеддингах
  4. Результаты кэшируются в JSON — не пересчитываем

Кэш-ключ: "{threshold}_{pca}_{method}_{s_min}_{s_max}_{k_param}_{n_neighbors}_{rs}"
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np


def _imread(path: str) -> Optional[np.ndarray]:
    """cv2.imread с поддержкой Unicode/кириллических путей (Windows)."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

log = logging.getLogger(__name__)

CACHE_FILE           = Path(__file__).parent / "storage" / "grid_search_cache.json"
FULL_GRID_CACHE_FILE = Path(__file__).parent / "storage" / "full_grid_cache.json"
DEFECT_TYPE_NAMES    = ["color", "cut", "hole", "metal_contamination", "thread"]


# ---------------------------------------------------------------------------
# Кэш
# ---------------------------------------------------------------------------

def _load_cache() -> Dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _cache_key(threshold: float, pca_n: int, method: str, s_min: float, s_max: float,
               k_param: int, n_neighbors: int, metric: str,
               n_estimators: int, max_depth: Optional[int], random_state: int) -> str:
    parts = [
        f"thr{threshold:.3f}", f"pca{pca_n}", method,
        f"smin{s_min:.0f}", f"smax{s_max:.0f}", f"k{k_param}",
        f"nn{n_neighbors}", metric,
        f"ne{n_estimators}", f"md{max_depth or 'None'}",
        f"rs{random_state}",
    ]
    raw = "_".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()[:16] + "_" + raw[:60]


def clear_cache() -> int:
    """Очищает кэш, возвращает число удалённых записей."""
    if not CACHE_FILE.exists():
        return 0
    cache = _load_cache()
    n = len(cache)
    CACHE_FILE.write_text("{}", encoding="utf-8")
    return n


def get_all_results() -> List[Dict]:
    """Возвращает все кэшированные результаты."""
    cache = _load_cache()
    return list(cache.values())


# ---------------------------------------------------------------------------
# Предвычисление heatmaps + embeddings (самая дорогая часть)
# ---------------------------------------------------------------------------

def precompute_inference(
    runner,
    image_files: List[Path],
    gt_dir: Path,
) -> List[Dict]:
    """
    Прогоняет PatchCore inference для всех изображений ОДИН раз.
    Возвращает список словарей:
        {path, rel_path, heatmap, patch_embeddings, spatial_shape,
         orig_shape, gt_label}
    """
    from defect_embedding_classifier import DEFECT_TYPE_NAMES as _DN

    results = []
    for img_path in image_files:
        try:
            img_bgr = _imread(str(img_path))
            if img_bgr is None:
                log.warning("Не удалось прочитать: %s", img_path)
                continue

            h, w = img_bgr.shape[:2]
            heatmap, score, patch_embs, sp_shape = runner.infer_with_embeddings(img_bgr)

            # GT label из имени папки
            parent_name = img_path.parent.name
            gt_label = None
            for i, name in enumerate(_DN):
                if name == parent_name:
                    gt_label = i
                    break
            if gt_label is None:
                # Попробовать по grand_truth директории
                try:
                    rel = img_path.relative_to(img_path.parent.parent.parent)
                    parts = rel.parts
                    if len(parts) >= 2:
                        for i, name in enumerate(_DN):
                            if name == parts[0]:
                                gt_label = i
                                break
                except Exception:
                    pass

            results.append({
                "path":             str(img_path),
                "rel_path":         str(img_path.name),
                "heatmap":          heatmap,
                "orig_bgr":         img_bgr,
                "patch_embeddings": patch_embs,
                "spatial_shape":    sp_shape,
                "orig_shape":       (h, w),
                "gt_label":         gt_label,
                "anomaly_score":    float(score),
            })
        except Exception as e:
            log.warning("Inference failed for %s: %s", img_path, e)

    return results


# ---------------------------------------------------------------------------
# Бинаризация + контуры (по threshold / S_min / S_max)
# ---------------------------------------------------------------------------

def extract_contours_from_heatmap(
    heatmap: np.ndarray,
    threshold: float,
    s_min: float,
    s_max: float,
    k_param: int = 5,
) -> List[np.ndarray]:
    """
    Быстрая бинаризация heatmap → контуры дефектов (фильтрация по S_min/S_max).
    """
    binary = (heatmap >= threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_param, k_param))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if s_min <= area <= s_max:
            valid.append(c)
    return valid


# ---------------------------------------------------------------------------
# Один эксперимент (одна точка сетки)
# ---------------------------------------------------------------------------

def run_single_experiment(
    precomp: List[Dict],
    threshold: float,
    pca_n: int,
    method: str,
    s_min: float = 40.0,
    s_max: float = 1500.0,
    k_param: int = 5,
    n_neighbors: int = 5,
    knn_metric: str = "euclidean",
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    compute_umap: bool = False,
) -> Dict:
    """
    Запускает классификацию для одной точки grid search.
    Использует предвычисленные heatmaps/embeddings.
    """
    from defect_embedding_classifier import classify_by_embeddings

    t0 = time.perf_counter()

    patch_embs_list = []
    spatial_shapes  = []
    orig_shapes     = []
    contours_list   = []
    gt_labels       = []
    image_paths     = []

    n_total_contours = 0
    n_images_with_defects = 0
    n_cls = len(DEFECT_TYPE_NAMES)
    class_dist: Dict[int, int] = {i: 0 for i in range(n_cls)}

    for item in precomp:
        contours = extract_contours_from_heatmap(
            item["heatmap"], threshold, s_min, s_max, k_param
        )
        if not contours:
            # Включаем изображение с пустым списком чтобы сохранить GT
            pass
        else:
            n_images_with_defects += 1
        n_total_contours += len(contours)

        # Подсчёт дефектов по классам для предупреждений
        gt = item.get("gt_label")
        if gt is not None and 0 <= int(gt) < n_cls:
            class_dist[int(gt)] += len(contours)

        patch_embs_list.append(item["patch_embeddings"])
        spatial_shapes.append(item["spatial_shape"])
        orig_shapes.append(item["orig_shape"])
        contours_list.append(contours)
        gt_labels.append(item["gt_label"])
        image_paths.append(item["rel_path"])

    # Предупреждения о распределении по классам
    missing_classes = [DEFECT_TYPE_NAMES[i] for i, cnt in class_dist.items() if cnt == 0]
    sparse_classes  = [DEFECT_TYPE_NAMES[i] for i, cnt in class_dist.items() if 0 < cnt < 5]

    if n_total_contours == 0:
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "overall_accuracy":  None,
            "test_accuracy":     None,
            "train_accuracy":    None,
            "n_defects":         0,
            "n_images":          len(precomp),
            "n_images_with_def": 0,
            "pca_components":    pca_n,
            "method":            method,
            "threshold":         threshold,
            "s_min":             s_min,
            "s_max":             s_max,
            "time_ms":           elapsed,
            "error":             "no_defects",
            "class_distribution": {str(k): v for k, v in class_dist.items()},
            "missing_classes":   missing_classes,
            "sparse_classes":    sparse_classes,
        }

    config = {
        "method":          method,
        "pca_components":  pca_n,
        "n_clusters":      len(DEFECT_TYPE_NAMES),
        "n_neighbors":     n_neighbors,
        "knn_metric":      knn_metric,
        "n_estimators":    n_estimators,
        "max_depth":       max_depth,
        "random_state":    random_state,
        "test_size":       0.5,
    }

    try:
        result = classify_by_embeddings(
            patch_embs_list, spatial_shapes, orig_shapes, contours_list,
            config=config,
            gt_labels=gt_labels,
            image_paths=image_paths,
            compute_umap=compute_umap,
        )
    except Exception as e:
        log.error("classify_by_embeddings failed: %s", e)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {"error": str(e), "method": method, "threshold": threshold,
                "pca_components": pca_n, "time_ms": elapsed}

    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    # Точность: для supervised берём test_accuracy, для unsupervised — overall (hungarian)
    acc = result.get("test_accuracy") or result.get("overall_accuracy")

    out: Dict = {
        "overall_accuracy":        result.get("overall_accuracy"),
        "test_accuracy":           result.get("test_accuracy"),
        "train_accuracy":          result.get("train_accuracy"),
        "n_defects":               result.get("n_defects", n_total_contours),
        "n_images":                len(precomp),
        "n_images_with_def":       n_images_with_defects,
        "pca_components":          pca_n,
        "method":                  method,
        "threshold":               threshold,
        "s_min":                   s_min,
        "s_max":                   s_max,
        "time_ms":                 elapsed,
        "accuracy":                acc,
        "method_type":             result.get("method_type", "unknown"),
        "silhouette":              result.get("silhouette"),
        "per_class_accuracy":      result.get("per_class_accuracy", {}),
        # Диагностика распределения и качества разбиения
        "class_distribution":      {str(k): v for k, v in class_dist.items()},
        "missing_classes":         missing_classes,
        "sparse_classes":          sparse_classes,
        "n_classes_in_test":       result.get("n_classes_in_test"),
        "excluded_classes":        result.get("excluded_classes", []),
        "stratification_warnings": result.get("stratification_warnings", []),
    }

    # Feature importance для RF/DT
    if "feature_importance" in result:
        out["feature_importance"] = result["feature_importance"]
    if "n_neighbors" in result:
        out["n_neighbors"] = result["n_neighbors"]
    if "knn_metric" in result:
        out["knn_metric"] = result["knn_metric"]
    if "n_estimators" in result:
        out["n_estimators"] = result["n_estimators"]
    if "tree_depth" in result:
        out["tree_depth"] = result["tree_depth"]

    return out


# ---------------------------------------------------------------------------
# Полный grid search со стримингом прогресса
# ---------------------------------------------------------------------------

def run_grid_search(
    runner,
    image_files: List[Path],
    gt_dir: Path,
    grid: Dict,
    progress_cb: Optional[Callable[[Dict], None]] = None,
    use_cache: bool = True,
    random_state: int = 42,
) -> Generator[Dict, None, None]:
    """
    Запускает полный grid search. Генерирует события прогресса.

    grid:
        thresholds:   [0.70, 0.80, 0.85, 0.90, 0.92, 0.95]
        pca_values:   [20, 35, 50, 75, 100]
        methods:      ["linear", "knn", "rf", "dt", "kmeans"]
        s_min:        40.0
        s_max:        1500.0
        k_param:      5
        n_neighbors:  5
        knn_metric:   "euclidean"
        n_estimators: 100
        max_depth:    null
    """
    thresholds   = grid.get("thresholds",   [0.70, 0.80, 0.85, 0.90])
    pca_values   = grid.get("pca_values",   [20, 35, 50])
    methods      = grid.get("methods",      ["linear", "kmeans"])
    s_min        = float(grid.get("s_min",   40.0))
    s_max        = float(grid.get("s_max",   1500.0))
    k_param      = int(grid.get("k_param",   5))
    n_neighbors  = int(grid.get("n_neighbors", 5))
    knn_metric   = str(grid.get("knn_metric", "euclidean"))
    n_estimators = int(grid.get("n_estimators", 100))
    max_depth    = grid.get("max_depth", None)
    if max_depth is not None:
        max_depth = int(max_depth)

    total = len(thresholds) * len(pca_values) * len(methods)

    yield {"type": "start", "total": total, "message": f"Начинаем grid search: {total} точек"}

    # Предвычисляем heatmaps (один раз для всех изображений)
    yield {"type": "progress", "step": "inference", "message": f"Инференс PatchCore для {len(image_files)} изображений..."}
    t_inf = time.perf_counter()
    precomp = precompute_inference(runner, image_files, gt_dir)
    inf_ms = round((time.perf_counter() - t_inf) * 1000)
    yield {"type": "progress", "step": "inference_done",
           "message": f"Инференс завершён за {inf_ms} мс, {len(precomp)} изображений"}

    cache = _load_cache() if use_cache else {}
    done = 0

    for threshold in thresholds:
        for pca_n in pca_values:
            for method in methods:
                key = _cache_key(
                    threshold, pca_n, method, s_min, s_max,
                    k_param, n_neighbors, knn_metric,
                    n_estimators, max_depth, random_state,
                )

                if use_cache and key in cache:
                    result = cache[key]
                    result["_cached"] = True
                    done += 1
                    yield {
                        "type": "result",
                        "done": done, "total": total,
                        "result": result,
                        "cached": True,
                    }
                    continue

                result = run_single_experiment(
                    precomp, threshold, pca_n, method,
                    s_min=s_min, s_max=s_max, k_param=k_param,
                    n_neighbors=n_neighbors, knn_metric=knn_metric,
                    n_estimators=n_estimators, max_depth=max_depth,
                    random_state=random_state,
                )
                result["_cached"] = False
                result["_key"] = key

                if use_cache and "error" not in result:
                    cache[key] = result
                    _save_cache(cache)

                done += 1
                yield {
                    "type": "result",
                    "done": done, "total": total,
                    "result": result,
                    "cached": False,
                }

    yield {"type": "done", "total": total, "done": done,
           "message": f"Grid search завершён: {done}/{total} точек"}


# ---------------------------------------------------------------------------
# Анализ стабильности: 3 запуска, std accuracy
# ---------------------------------------------------------------------------

def run_stability_check(
    runner,
    image_files: List[Path],
    gt_dir: Path,
    configs: List[Dict],
    n_repeats: int = 3,
) -> List[Dict]:
    """
    Для топ-N конфигураций запускает n_repeats раз с разными random_state.
    Возвращает mean/std accuracy.
    """
    results = []
    precomp = precompute_inference(runner, image_files, gt_dir)

    for cfg in configs:
        accs = []
        for rep in range(n_repeats):
            rs = 42 + rep * 17
            r = run_single_experiment(
                precomp,
                threshold=float(cfg.get("threshold", 0.85)),
                pca_n=int(cfg.get("pca_components", 35)),
                method=str(cfg.get("method", "linear")),
                s_min=float(cfg.get("s_min", 40.0)),
                s_max=float(cfg.get("s_max", 1500.0)),
                k_param=int(cfg.get("k_param", 5)),
                n_neighbors=int(cfg.get("n_neighbors", 5)),
                knn_metric=str(cfg.get("knn_metric", "euclidean")),
                n_estimators=int(cfg.get("n_estimators", 100)),
                max_depth=cfg.get("max_depth"),
                random_state=rs,
            )
            acc = r.get("accuracy")
            if acc is not None:
                accs.append(acc)

        if accs:
            results.append({
                "config":   cfg,
                "n_repeats": len(accs),
                "mean_acc":  round(float(np.mean(accs)), 4),
                "std_acc":   round(float(np.std(accs)), 4),
                "min_acc":   round(float(np.min(accs)), 4),
                "max_acc":   round(float(np.max(accs)), 4),
                "accs":      [round(a, 4) for a in accs],
            })
        else:
            results.append({
                "config": cfg, "n_repeats": 0,
                "mean_acc": None, "std_acc": None, "error": "no_defects",
            })

    return results


# ---------------------------------------------------------------------------
# Confusion matrix из per_image результатов
# ---------------------------------------------------------------------------

def _build_confusion_matrix(
    per_image: List[Dict],
    method: str,
    n_classes: int = 5,
) -> List[List[int]]:
    """
    Строит confusion matrix [GT × Pred] из per_image результатов.
    Supervised (linear/knn/rf/dt/prototype): cluster_labels = predicted GT class.
    Unsupervised (kmeans/gmm/hierarchical): cluster_labels = cluster id, mapping → GT.
    """
    cm = [[0] * n_classes for _ in range(n_classes)]
    supervised = method in ("linear", "knn", "rf", "dt", "prototype")

    for item in per_image:
        gt = item.get("gt_label")
        if gt is None or not (0 <= int(gt) < n_classes):
            continue
        gt = int(gt)
        mapping = item.get("mapping", {})
        for lbl in item.get("cluster_labels", []):
            lbl = int(lbl)
            pred = lbl if supervised else int(mapping.get(lbl, -1))
            if 0 <= pred < n_classes:
                cm[gt][pred] += 1
    return cm


def _class_distribution_from_precomp(
    precomp: List[Dict],
    threshold: float,
    s_min: float,
    s_max: float,
    k_param: int,
    n_classes: int = 5,
) -> Dict[int, int]:
    """Считает сколько дефектов каждого класса при данном threshold."""
    dist = {i: 0 for i in range(n_classes)}
    for item in precomp:
        gt = item.get("gt_label")
        if gt is None or not (0 <= int(gt) < n_classes):
            continue
        contours = extract_contours_from_heatmap(
            item["heatmap"], threshold, s_min, s_max, k_param
        )
        dist[int(gt)] += len(contours)
    return dist


# ---------------------------------------------------------------------------
# Детальный анализ одной конфигурации (с confusion matrix)
# ---------------------------------------------------------------------------

def run_config_detailed(
    precomp: List[Dict],
    threshold: float,
    pca_n: int,
    method: str,
    s_min: float = 40.0,
    s_max: float = 1500.0,
    k_param: int = 5,
    n_neighbors: int = 5,
    knn_metric: str = "euclidean",
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> Dict:
    """
    Расширенная версия run_single_experiment:
    - confusion_matrix [5×5]
    - per_class_accuracy {class_id: float}
    - class_distribution {class_id: n_defects_total}
    - train_dist / test_dist — дефекты в train/test по классам
    - per_class_n_test — сколько тест-дефектов по классу (для предупреждения малой выборки)
    """
    from defect_embedding_classifier import classify_by_embeddings

    t0 = time.perf_counter()
    n_classes = len(DEFECT_TYPE_NAMES)

    # Класс-распределение при данном threshold
    class_dist = _class_distribution_from_precomp(
        precomp, threshold, s_min, s_max, k_param, n_classes
    )

    # Собираем данные для классификации
    patch_embs_list, spatial_shapes, orig_shapes = [], [], []
    contours_list, gt_labels_list, image_paths = [], [], []
    n_total = 0
    for item in precomp:
        contours = extract_contours_from_heatmap(
            item["heatmap"], threshold, s_min, s_max, k_param
        )
        n_total += len(contours)
        patch_embs_list.append(item["patch_embeddings"])
        spatial_shapes.append(item["spatial_shape"])
        orig_shapes.append(item["orig_shape"])
        contours_list.append(contours)
        gt_labels_list.append(item["gt_label"])
        image_paths.append(item["rel_path"])

    if n_total == 0:
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "error": "no_defects",
            "confusion_matrix": [[0]*n_classes for _ in range(n_classes)],
            "per_class_accuracy": {}, "class_distribution": class_dist,
            "per_class_n_test": {}, "accuracy": None, "train_accuracy": None,
            "n_defects": 0, "method": method, "threshold": threshold,
            "pca_components": pca_n, "time_ms": elapsed,
        }

    config = {
        "method": method, "pca_components": pca_n,
        "n_clusters": n_classes,
        "n_neighbors": n_neighbors, "knn_metric": knn_metric,
        "n_estimators": n_estimators, "max_depth": max_depth,
        "random_state": random_state, "test_size": 0.5,
    }

    try:
        result = classify_by_embeddings(
            patch_embs_list, spatial_shapes, orig_shapes, contours_list,
            config=config, gt_labels=gt_labels_list, image_paths=image_paths,
            compute_umap=False,
        )
    except Exception as e:
        log.error("run_config_detailed failed: %s", e)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return {"error": str(e), "method": method, "threshold": threshold,
                "pca_components": pca_n, "time_ms": elapsed,
                "confusion_matrix": [[0]*n_classes for _ in range(n_classes)],
                "per_class_accuracy": {}, "class_distribution": class_dist,
                "per_class_n_test": {}, "accuracy": None}

    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    # Confusion matrix из per_image
    cm = _build_confusion_matrix(result.get("per_image", []), method, n_classes)

    # per_class_n_test: сколько тест-дефектов на класс
    # Суммируем диагональ + все ошибки для каждого GT класса
    per_class_n_test = {i: sum(cm[i]) for i in range(n_classes)}

    per_class_acc = result.get("per_class_accuracy", {})
    # Для unsupervised — вычислим из CM
    if not per_class_acc:
        for i in range(n_classes):
            row_sum = sum(cm[i])
            if row_sum > 0:
                per_class_acc[str(i)] = round(cm[i][i] / row_sum, 4)

    acc = result.get("test_accuracy") or result.get("overall_accuracy")

    return {
        "accuracy":           acc,
        "overall_accuracy":   result.get("overall_accuracy"),
        "test_accuracy":      result.get("test_accuracy"),
        "train_accuracy":     result.get("train_accuracy"),
        "confusion_matrix":   cm,
        "per_class_accuracy": {str(k): v for k, v in per_class_acc.items()},
        "class_distribution": {str(k): v for k, v in class_dist.items()},
        "per_class_n_test":   {str(i): per_class_n_test[i] for i in range(n_classes)},
        "n_defects":          n_total,
        "n_train":            result.get("n_train"),
        "n_test":             result.get("n_test"),
        "method":             method,
        "threshold":          threshold,
        "pca_components":     pca_n,
        "time_ms":            elapsed,
        "method_type":        result.get("method_type", "unknown"),
        "feature_importance": result.get("feature_importance"),
        "silhouette":         result.get("silhouette"),
        "class_names":        DEFECT_TYPE_NAMES,
    }


# ---------------------------------------------------------------------------
# Стабильность: расширенная версия (10 запусков, confusion matrices)
# ---------------------------------------------------------------------------

def run_stability_extended(
    runner,
    image_files: List[Path],
    gt_dir: Path,
    configs: List[Dict],
    n_repeats: int = 10,
) -> List[Dict]:
    """
    Для каждой конфигурации прогоняет n_repeats раз с разными random_state.
    Возвращает mean/std/min/max accuracy + список accuracy по запускам.
    """
    precomp = precompute_inference(runner, image_files, gt_dir)
    results = []

    for cfg in configs:
        accs: List[float] = []
        per_run_cm: List[List[List[int]]] = []
        n_classes = len(DEFECT_TYPE_NAMES)

        for rep in range(n_repeats):
            rs = 42 + rep * 17
            try:
                detail = run_config_detailed(
                    precomp,
                    threshold=float(cfg.get("threshold", 0.85)),
                    pca_n=int(cfg.get("pca_components", 35)),
                    method=str(cfg.get("method", "linear")),
                    s_min=float(cfg.get("s_min", 40.0)),
                    s_max=float(cfg.get("s_max", 1500.0)),
                    k_param=int(cfg.get("k_param", 5)),
                    n_neighbors=int(cfg.get("n_neighbors", 5)),
                    knn_metric=str(cfg.get("knn_metric", "euclidean")),
                    n_estimators=int(cfg.get("n_estimators", 100)),
                    max_depth=cfg.get("max_depth"),
                    random_state=rs,
                )
            except Exception as e:
                log.warning("Stability run %d failed: %s", rep, e)
                continue

            acc = detail.get("accuracy")
            if acc is not None:
                accs.append(acc)
            cm = detail.get("confusion_matrix")
            if cm:
                per_run_cm.append(cm)

        if not accs:
            results.append({
                "config": cfg, "n_repeats": 0,
                "mean_acc": None, "std_acc": None, "error": "no_defects",
                "accs": [], "per_run_cm": [],
            })
            continue

        # Средняя confusion matrix по всем запускам
        mean_cm: List[List[float]] = [[0.0]*n_classes for _ in range(n_classes)]
        if per_run_cm:
            for cm in per_run_cm:
                for i in range(n_classes):
                    for j in range(n_classes):
                        mean_cm[i][j] += cm[i][j] / len(per_run_cm)
            mean_cm = [[round(v, 1) for v in row] for row in mean_cm]

        mean_a = float(np.mean(accs))
        std_a  = float(np.std(accs))
        results.append({
            "config":      cfg,
            "n_repeats":   len(accs),
            "mean_acc":    round(mean_a, 4),
            "std_acc":     round(std_a,  4),
            "min_acc":     round(float(np.min(accs)), 4),
            "max_acc":     round(float(np.max(accs)), 4),
            "accs":        [round(a, 4) for a in accs],
            "mean_cm":     mean_cm,
            "unstable":    std_a > 0.10,
        })

    return results


# ---------------------------------------------------------------------------
# Проблемные пары классов: агрегирование по всем точкам grid search
# ---------------------------------------------------------------------------

def aggregate_problem_pairs(
    results: List[Dict],
    n_classes: int = 5,
    top_n: int = 5,
) -> List[Dict]:
    """
    Агрегирует confusion matrices из кэша grid search (если они есть)
    или из переданного списка результатов с полем `confusion_matrix`.

    Возвращает топ-N пар (gt_class → pred_class) по числу ошибок.
    """
    agg = [[0] * n_classes for _ in range(n_classes)]
    n_matrices = 0

    for r in results:
        cm = r.get("confusion_matrix")
        if not cm:
            continue
        n_matrices += 1
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and j < len(cm[i]):
                    agg[i][j] += cm[i][j]

    if n_matrices == 0:
        return []

    pairs = []
    for i in range(n_classes):
        row_total = sum(agg[i])
        for j in range(n_classes):
            if i != j and agg[i][j] > 0:
                pairs.append({
                    "gt":       i,
                    "pred":     j,
                    "gt_name":  DEFECT_TYPE_NAMES[i],
                    "pred_name": DEFECT_TYPE_NAMES[j],
                    "count":    agg[i][j],
                    "pct":      round(agg[i][j] / row_total, 3) if row_total > 0 else 0.0,
                })

    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_n]


# ---------------------------------------------------------------------------
# Вспомогательная: shape descriptors контуров (solidity, compactness, aspect)
# ---------------------------------------------------------------------------

def _compute_contour_shape_stats(
    items: List[Dict],
    threshold: float,
    k_param: int = 5,
) -> Dict:
    """
    Для всех raw-контуров при данном threshold вычисляет дескрипторы формы:
      - solidity     = area / convex_hull_area  (0-1; тонкие: низкая)
      - compactness  = 4π·area / perimeter²     (0-1; круг=1; тонкая линия → 0)
      - aspect_ratio = max(w,h)/min(w,h) из minAreaRect
    Позволяет понять, можно ли отличить размытые blob-ы от good-ложняков
    по форме, а не только по площади.
    """
    areas: List[float] = []; perimeters: List[float] = []
    solidities: List[float] = []; compactnesses: List[float] = []
    aspect_ratios: List[float] = []

    for item in items:
        binary = (item["heatmap"] >= threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_param, k_param))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
        raw_cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in raw_cnts:
            area = float(cv2.contourArea(c))
            if area < 1:
                continue
            peri = float(cv2.arcLength(c, True))
            areas.append(area)
            perimeters.append(peri)

            hull    = cv2.convexHull(c)
            hull_a  = float(cv2.contourArea(hull))
            solidities.append(area / hull_a if hull_a > 0 else 1.0)

            comp = 4 * math.pi * area / (peri ** 2) if peri > 1 else 0.0
            compactnesses.append(comp)

            if len(c) >= 5:
                _, (rw, rh), _ = cv2.minAreaRect(c)
                mx = max(rw, rh, 1e-3); mn = max(min(rw, rh), 1e-3)
                aspect_ratios.append(mx / mn)
            else:
                bx, by_, bw, bh = cv2.boundingRect(c)
                aspect_ratios.append(max(bw, bh) / max(min(bw, bh), 1))

    def _st(lst: List[float]) -> Dict:
        if not lst:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
        a = np.array(lst, dtype=np.float64)
        return {k: round(float(v), 4) for k, v in [
            ("mean", a.mean()), ("max", a.max()), ("min", a.min()),
            ("p50", np.percentile(a, 50)), ("p90", np.percentile(a, 90)),
            ("p95", np.percentile(a, 95)),
        ]}

    def _hist(lst: List[float], bins: int = 20) -> Dict:
        if not lst:
            return {"counts": [], "edges": []}
        counts, edges = np.histogram(np.array(lst), bins=bins)
        return {"counts": counts.tolist(), "edges": [round(float(e), 2) for e in edges]}

    return {
        "n_contours":    len(areas),
        "threshold":     threshold,
        "area":          _st(areas),
        "perimeter":     _st(perimeters),
        "solidity":      _st(solidities),
        "compactness":   _st(compactnesses),
        "aspect_ratio":  _st(aspect_ratios),
        "area_hist":     _hist(areas, bins=25),
        "solidity_hist": _hist(solidities, bins=10),
        "compact_hist":  _hist(compactnesses, bins=10),
    }


def _detect_n_images(items: List[Dict], threshold: float,
                     s_min: float, s_max: float, k_param: int) -> int:
    """Возвращает число изображений с ≥1 контуром в диапазоне [s_min, s_max]."""
    n = 0
    for item in items:
        binary = (item["heatmap"] >= threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_param, k_param))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if s_min <= float(cv2.contourArea(c)) <= s_max:
                n += 1
                break
    return n


# ---------------------------------------------------------------------------
# ROC-like кривая: Recall(target) vs FPR(good) по сетке (threshold × S_max)
# ---------------------------------------------------------------------------

def diagnose_recall_vs_fpr(
    runner,
    defect_files: List[Path],
    good_files:   List[Path],
    target_class: str = "thread",
    thresholds:   Optional[List[float]] = None,
    s_max_values: Optional[List[float]] = None,
    s_min:        float = 5.0,
    k_param:      int   = 5,
) -> Dict:
    """
    ROC-like анализ: recall(target_class) vs FPR(good) при разных (threshold, S_max).

    Использование:
      - Для каждой точки (threshold, S_max): вычисляем recall на target_class
        и FPR на good-изображениях
      - Строим scatter: x=FPR, y=recall; цвет = threshold, размер ∝ S_max
      - Помогает найти S_max, дающий max recall при допустимом FPR

    Дополнительно:
      - Shape stats (solidity, compactness) для target vs good контуров
        чтобы понять можно ли различить их по форме
      - Baseline FPR на good при текущем S_max=1500
    """
    if thresholds is None:
        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    if s_max_values is None:
        s_max_values = [1_000, 5_000, 10_000, 50_000, 100_000,
                        300_000, 600_000, 1_000_000]

    t0 = time.perf_counter()

    # Предвычисляем heatmaps (два батча: дефекты и good)
    precomp_def  = precompute_inference(runner, defect_files, gt_dir=None)
    precomp_good = precompute_inference(runner, good_files,   gt_dir=None)

    target_idx   = next(
        (i for i, n in enumerate(DEFECT_TYPE_NAMES) if n == target_class), -1
    )
    target_items = [it for it in precomp_def if it.get("gt_label") == target_idx]
    n_target     = len(target_items)
    n_good       = len(precomp_good)

    # Shape stats при минимальном threshold (без S-фильтра)
    min_thr = min(thresholds)
    shape_target = _compute_contour_shape_stats(target_items,   min_thr, k_param)
    shape_good   = _compute_contour_shape_stats(precomp_good,   min_thr, k_param)

    # Baseline FPR на good при текущем S_max=1500
    good_baseline: List[Dict] = []
    for thr in thresholds:
        n_fp = _detect_n_images(precomp_good, thr, s_min, 1500.0, k_param)
        good_baseline.append({
            "threshold": thr,
            "n_fp":      n_fp,
            "fpr":       round(n_fp / n_good, 4) if n_good > 0 else 0.0,
        })

    # Sweep (threshold × S_max) → ROC-like кривая
    curve_points: List[Dict] = []
    for thr in thresholds:
        for s_max in s_max_values:
            n_tgt = _detect_n_images(target_items,   thr, s_min, s_max, k_param)
            n_fp  = _detect_n_images(precomp_good,   thr, s_min, s_max, k_param)
            curve_points.append({
                "threshold": thr,
                "s_max":     int(s_max),
                "recall":    round(n_tgt / n_target, 4) if n_target > 0 else 0.0,
                "fpr":       round(n_fp  / n_good,   4) if n_good   > 0 else 0.0,
                "n_target":  n_tgt,
                "n_fp":      n_fp,
            })

    elapsed = round((time.perf_counter() - t0) * 1000)
    return {
        "target_class":     target_class,
        "n_target_images":  n_target,
        "n_good_images":    n_good,
        "thresholds":       thresholds,
        "s_max_values":     [int(v) for v in s_max_values],
        "s_min":            s_min,
        "curve_points":     curve_points,
        "shape_target":     shape_target,
        "shape_good":       shape_good,
        "good_baseline_fpr": good_baseline,
        "time_ms":          elapsed,
        "class_names":      DEFECT_TYPE_NAMES,
    }


# ---------------------------------------------------------------------------
# Диагностика детектируемости по типам дефектов
# ---------------------------------------------------------------------------

def diagnose_class_detectability(
    runner,
    image_files: List[Path],
    thresholds: Optional[List[float]] = None,
    s_min: float = 40.0,
    s_max: float = 1500.0,
    k_param: int = 5,
) -> Dict:
    """
    Для каждого класса дефектов вычисляет:
      1. Распределение anomaly scores (max per image)
      2. По каждому threshold: % изображений с ≥1 детекцией ДО и ПОСЛЕ S_min/S_max фильтра
      3. Площади/периметры/аспект-отношения контуров до фильтрации

    Позволяет диагностировать почему thread=0 дефектов:
      — слишком низкий anomaly score (не проходит threshold)
      — или слишком тонкие контуры (area < S_min, но perimeter >> S_min)
    """
    if thresholds is None:
        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]

    t0 = time.perf_counter()
    precomp = precompute_inference(runner, image_files, gt_dir=None)

    # Группировка по GT-классу
    by_class: Dict[int, List[Dict]] = {i: [] for i in range(len(DEFECT_TYPE_NAMES))}
    for item in precomp:
        gt = item.get("gt_label")
        if gt is not None and 0 <= int(gt) < len(DEFECT_TYPE_NAMES):
            by_class[int(gt)].append(item)

    per_class: Dict[str, Any] = {}
    recommendations: List[str] = []

    for cls_idx, items in by_class.items():
        cls_name = DEFECT_TYPE_NAMES[cls_idx]
        if not items:
            continue

        # Аномальные оценки (max patch score per image)
        scores = [float(np.max(item["heatmap"])) for item in items]
        score_arr = np.array(scores, dtype=np.float32)

        by_threshold: Dict[str, Any] = {}
        for thr in thresholds:
            thr_key = f"{thr:.3f}"
            n_raw = 0       # изображений с ≥1 контуром без S_min фильтра
            n_filtered = 0  # изображений с ≥1 контуром после S_min/S_max
            areas_raw:        List[float] = []
            areas_filtered:   List[float] = []
            perimeters_raw:   List[float] = []
            aspect_ratios:    List[float] = []
            diagonals_raw:    List[float] = []  # sqrt(w^2+h^2) bounding box

            for item in items:
                hm = item["heatmap"]
                binary = (hm >= thr).astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_param, k_param))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
                raw_cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if raw_cnts:
                    n_raw += 1

                img_has_filtered = False
                for c in raw_cnts:
                    area = float(cv2.contourArea(c))
                    peri = float(cv2.arcLength(c, True))
                    areas_raw.append(round(area, 1))
                    perimeters_raw.append(round(peri, 1))

                    # Аспект-отношение (minAreaRect)
                    if len(c) >= 5:
                        _, (rw, rh), _ = cv2.minAreaRect(c)
                        mx = max(rw, rh, 1e-3)
                        mn = max(min(rw, rh), 1e-3)
                        ar = round(mx / mn, 2)
                    else:
                        bx, by_, bw, bh = cv2.boundingRect(c)
                        ar = round(max(bw, bh) / max(min(bw, bh), 1), 2)
                    aspect_ratios.append(ar)

                    # Диагональ bounding box
                    bx, by_, bw, bh = cv2.boundingRect(c)
                    diag = round(math.sqrt(bw**2 + bh**2), 1)
                    diagonals_raw.append(diag)

                    if s_min <= area <= s_max:
                        areas_filtered.append(round(area, 1))
                        img_has_filtered = True

                if img_has_filtered:
                    n_filtered += 1

            n_imgs = len(items)
            # Статистики по контурам
            def _stats(lst: List[float]) -> Dict:
                if not lst:
                    return {"mean": 0.0, "max": 0.0, "min": 0.0, "p50": 0.0, "p90": 0.0}
                a = np.array(lst)
                return {
                    "mean": round(float(a.mean()), 2),
                    "max":  round(float(a.max()),  2),
                    "min":  round(float(a.min()),  2),
                    "p50":  round(float(np.percentile(a, 50)), 2),
                    "p90":  round(float(np.percentile(a, 90)), 2),
                }

            by_threshold[thr_key] = {
                "threshold":          thr,
                "n_images":           n_imgs,
                "n_raw":              n_raw,
                "n_filtered":         n_filtered,
                "pct_raw":            round(n_raw    / n_imgs, 4) if n_imgs > 0 else 0.0,
                "pct_filtered":       round(n_filtered / n_imgs, 4) if n_imgs > 0 else 0.0,
                "n_contours_raw":     len(areas_raw),
                "area_stats":         _stats(areas_raw),
                "perimeter_stats":    _stats(perimeters_raw),
                "aspect_ratio_stats": _stats(aspect_ratios),
                "diagonal_stats":     _stats(diagonals_raw),
                # Первые 60 значений для гистограммы на фронте
                "areas_raw_sample":   sorted(areas_raw)[:60],
                "aspect_ratios_sample": sorted(aspect_ratios, reverse=True)[:30],
            }

        per_class[cls_name] = {
            "cls_idx":       cls_idx,
            "n_images":      len(items),
            "anomaly_scores": [round(s, 4) for s in scores],
            "score_stats": {
                "mean": round(float(score_arr.mean()), 4),
                "max":  round(float(score_arr.max()),  4),
                "min":  round(float(score_arr.min()),  4),
                "p50":  round(float(np.percentile(score_arr, 50)), 4),
                "p90":  round(float(np.percentile(score_arr, 90)), 4),
                "p95":  round(float(np.percentile(score_arr, 95)), 4),
            },
            "by_threshold": by_threshold,
        }

        # Рекомендации по диагностике
        min_thr = min(thresholds)
        thr_key0 = f"{min_thr:.3f}"
        d0 = by_threshold.get(thr_key0, {})
        n_raw0 = d0.get("n_raw", 0)
        n_fil0 = d0.get("n_filtered", 0)

        if n_fil0 == 0:
            if n_raw0 > 0:
                # Контуры есть, но отфильтрованы по S_min
                ar_mean = d0.get("aspect_ratio_stats", {}).get("mean", 1)
                area_mean = d0.get("area_stats", {}).get("mean", 0)
                peri_mean = d0.get("perimeter_stats", {}).get("mean", 0)
                diag_mean = d0.get("diagonal_stats", {}).get("mean", 0)
                if ar_mean >= 3 or area_mean < s_min:
                    kind = "тонких вытянутых" if ar_mean >= 3 else "мелких"
                    recommendations.append(
                        f"[{cls_name}] Контуры обнаружены ({n_raw0}/{len(items)} изобр. при thr={min_thr}), "
                        f"но отфильтрованы S_min={s_min:.0f}. "
                        f"Площадь: {area_mean:.0f} px², периметр: {peri_mean:.0f} px, "
                        f"диагональ bbox: {diag_mean:.0f} px, аспект: {ar_mean:.1f}. "
                        f"Причина: {kind} контуры. "
                        f"Решения: (1) снизить S_min до {max(5, int(area_mean*0.5))} "
                        f"или (2) фильтровать по периметру/диагонали вместо площади."
                    )
            else:
                # Нет ни одного контура даже без S_min
                max_score = per_class[cls_name]["score_stats"]["max"]
                recommendations.append(
                    f"[{cls_name}] Нет контуров даже без S_min фильтра (thr={min_thr}). "
                    f"Max anomaly score = {max_score:.4f}. "
                    f"Снизить threshold до {max(0.3, round(max_score * 0.85, 2))} "
                    f"чтобы начать детектировать этот класс."
                )

    elapsed = round((time.perf_counter() - t0) * 1000)
    return {
        "class_names":   DEFECT_TYPE_NAMES,
        "thresholds":    thresholds,
        "s_min":         s_min,
        "s_max":         s_max,
        "k_param":       k_param,
        "per_class":     per_class,
        "recommendations": recommendations,
        "n_images_total": len(precomp),
        "time_ms":       elapsed,
    }


# ---------------------------------------------------------------------------
# Корреляция размер дефекта vs стабильность классификации
# ---------------------------------------------------------------------------

def compute_area_stability_correlation(
    runner,
    image_files: List[Path],
    gt_dir: Path,
    threshold: float = 0.85,
    pca_n: int = 35,
    method: str = "linear",
    s_min: float = 40.0,
    s_max: float = 1500.0,
) -> Dict:
    """
    Для каждого дефекта считает: area (пиксели) и точность классификации.
    Возвращает список точек для scatter-plot.
    """
    from defect_embedding_classifier import (
        classify_by_embeddings,
        DEFECT_TYPE_NAMES as _DN,
    )

    precomp = precompute_inference(runner, image_files, gt_dir)

    patch_embs_list, spatial_shapes, orig_shapes = [], [], []
    contours_list, gt_labels_list, image_paths = [], [], []
    area_per_defect = []

    for item in precomp:
        contours = extract_contours_from_heatmap(
            item["heatmap"], threshold, s_min, s_max
        )
        for c in contours:
            area_per_defect.append(float(cv2.contourArea(c)))

        patch_embs_list.append(item["patch_embeddings"])
        spatial_shapes.append(item["spatial_shape"])
        orig_shapes.append(item["orig_shape"])
        contours_list.append(contours)
        gt_labels_list.append(item["gt_label"])
        image_paths.append(item["rel_path"])

    if not any(c for c in contours_list):
        return {"error": "no_defects", "points": []}

    config = {
        "method": method, "pca_components": pca_n,
        "n_clusters": len(_DN), "random_state": 42, "test_size": 0.5,
    }
    result = classify_by_embeddings(
        patch_embs_list, spatial_shapes, orig_shapes, contours_list,
        config=config, gt_labels=gt_labels_list, image_paths=image_paths,
        compute_umap=False,
    )

    # Собираем per-defect accuracy (угадан ли каждый дефект)
    points = []
    defect_idx = 0
    for per_img in result.get("per_image", []):
        labels = per_img.get("cluster_labels", [])
        gt = per_img.get("gt_label")
        mapping = per_img.get("mapping", {})
        for lbl in labels:
            if defect_idx >= len(area_per_defect):
                break
            area = area_per_defect[defect_idx]
            # Для supervised — прямое сравнение; для unsupervised — через mapping
            correct = None
            if gt is not None:
                if method in ("linear", "knn", "rf", "dt", "prototype"):
                    correct = int(lbl) == int(gt)
                else:
                    mapped_gt = mapping.get(int(lbl), -1)
                    correct = mapped_gt == int(gt)
            points.append({"area": area, "correct": correct, "gt_label": gt, "cluster": lbl})
            defect_idx += 1

    return {
        "points":         points,
        "n_defects":      len(points),
        "method":         method,
        "threshold":      threshold,
        "pca_components": pca_n,
    }


# ---------------------------------------------------------------------------
# Full Grid Search: LC detection pipeline × Classification
# ---------------------------------------------------------------------------

def _load_full_grid_cache() -> Dict[str, Any]:
    if FULL_GRID_CACHE_FILE.exists():
        try:
            return json.loads(FULL_GRID_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_full_grid_cache(cache: Dict[str, Any]) -> None:
    FULL_GRID_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    FULL_GRID_CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_full_grid_results() -> List[Dict]:
    """Returns all cached full-grid results."""
    cache = _load_full_grid_cache()
    return list(cache.values())


def clear_full_grid_cache() -> int:
    if not FULL_GRID_CACHE_FILE.exists():
        return 0
    cache = _load_full_grid_cache()
    n = len(cache)
    FULL_GRID_CACHE_FILE.write_text("{}", encoding="utf-8")
    return n


def _detect_calibrated(
    heatmap:            np.ndarray,
    orig_shape:         Tuple[int, int],
    blur_kernel:        int   = 0,
    percentile_clip:    float = 95.0,
    vmin_floor:         float = 0.0,
    gamma:              float = 1.0,
    binary_threshold:   float = 20.0,
    s_min:              int   = 40,
    s_max:              int   = 1_500_000,
    use_edge_mask:      bool  = True,
    k_morph:            int   = 5,
    baseline_map:       Optional[np.ndarray] = None,
    baseline_alpha:     float = 1.0,
    use_baseline_sub:   bool  = False,
    use_comp_sup:       bool  = False,
    suppression_factor: float = 0.3,
    min_comp_area:      int   = 0,
) -> Tuple[str, List]:
    """
    LC-style detection pipeline without visualization.
    Mirrors compute_calibrated_result() steps exactly.
    Returns (verdict, contours).
    """
    H, W = orig_shape

    work = cv2.resize(heatmap.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    if blur_kernel >= 3 and blur_kernel % 2 == 1:
        work = cv2.GaussianBlur(work, (blur_kernel, blur_kernel), 0)

    if use_baseline_sub and baseline_map is not None:
        bm = baseline_map
        if bm.shape != (H, W):
            bm = cv2.resize(bm, (W, H), interpolation=cv2.INTER_LINEAR)
        work = np.clip(work - baseline_alpha * bm, 0.0, None)

    p_clip = float(np.clip(percentile_clip, 50.1, 99.9))
    p_lo   = float(np.percentile(work, 100.0 - p_clip))
    p_hi   = float(np.percentile(work, p_clip))
    norm   = np.clip((work - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)

    if use_edge_mask:
        patch_sz = max(1, H // 28)
        border   = patch_sz * 3
        norm[:border, :]  = 0.0
        norm[-border:, :] = 0.0
        norm[:, :border]  = 0.0
        norm[:, -border:] = 0.0

    floored = np.clip((norm - vmin_floor) / (1.0 - vmin_floor + 1e-8), 0.0, 1.0)
    display = (floored ** (1.0 / max(gamma, 0.1))) * 100.0

    binary = (display > binary_threshold).astype(np.uint8) * 255
    if k_morph > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

    raw_cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in raw_cnts if s_min <= float(cv2.contourArea(c)) <= s_max]

    if use_comp_sup and valid:
        _min_comp = min_comp_area if min_comp_area > 0 else s_min
        if _min_comp > s_min:
            display_u8 = display.astype(np.float32)
            suppressed = []
            for c in valid:
                area = float(cv2.contourArea(c))
                if area < _min_comp:
                    mask_s = np.zeros(display_u8.shape, dtype=np.uint8)
                    cv2.drawContours(mask_s, [c], -1, 1, cv2.FILLED)
                    if np.any(display_u8[mask_s == 1] * suppression_factor > binary_threshold):
                        suppressed.append(c)
                else:
                    suppressed.append(c)
            valid = suppressed

    return ("defect" if valid else "ok"), valid


def run_full_grid_search(
    runner,
    image_files:        List[Path],
    gt_dir:             Path,
    # Detection grid (lists = sweep axes)
    blur_values:        List[int]   = None,
    pct_clip_values:    List[float] = None,
    vmin_values:        List[float] = None,
    gamma_values:       List[float] = None,
    bin_thr_values:     List[float] = None,
    smin_values:        List[int]   = None,
    smax_values:        List[int]   = None,
    # Detection fixed params
    use_edge_mask:      bool  = True,
    k_morph:            int   = 5,
    baseline_map:       Optional[np.ndarray] = None,
    baseline_alpha:     float = 1.0,
    use_baseline_sub:   bool  = False,
    use_comp_sup:       bool  = False,
    suppression_factor: float = 0.3,
    min_comp_area:      int   = 0,
    # Classification grid
    pca_values:         List[int]  = None,
    methods:            List[str]  = None,
    n_neighbors:        int        = 5,
    knn_metric:         str        = "euclidean",
    n_estimators:       int        = 100,
    max_depth:          Optional[int] = None,
    random_state:       int        = 42,
    # Search control
    use_cache:          bool  = True,
    max_combinations:   int   = 1000,
) -> Generator[Dict, None, None]:
    """
    Full grid search: LC detection pipeline × embedding-based classification.

    For each (detection combo) × (classification combo):
      1. Run _detect_calibrated() for all images → contours per image
      2. Compute detection metrics: fp_rate, detection_recall
      3. Call classify_by_embeddings() → clf_accuracy
      4. Cache and yield result
    """
    from itertools import product as iproduct
    from defect_embedding_classifier import classify_by_embeddings
    import random as _rnd

    if blur_values     is None: blur_values     = [0, 5, 7]
    if pct_clip_values is None: pct_clip_values = [90.0, 95.0, 99.0]
    if vmin_values     is None: vmin_values     = [0.0, 0.10, 0.20]
    if gamma_values    is None: gamma_values    = [1.0, 2.0, 3.0]
    if bin_thr_values  is None: bin_thr_values  = [15.0, 25.0, 40.0]
    if smin_values     is None: smin_values     = [10, 40]
    if smax_values     is None: smax_values     = [200_000, 600_000, 1_000_000]
    if pca_values      is None: pca_values      = [20, 50, 100]
    if methods         is None: methods         = ["linear", "knn", "rf"]

    # Build combo lists
    det_keys  = ["blur_kernel", "percentile_clip", "vmin_floor", "gamma",
                 "binary_threshold", "s_min", "s_max"]
    det_lists = [blur_values, pct_clip_values, vmin_values, gamma_values,
                 bin_thr_values, smin_values, smax_values]
    det_combos = list(iproduct(*det_lists))
    cls_combos = list(iproduct(pca_values, methods))

    all_pairs = [(d, c) for d in det_combos for c in cls_combos]
    total = len(all_pairs)

    if max_combinations and total > max_combinations:
        rng = _rnd.Random(random_state)
        all_pairs = rng.sample(all_pairs, max_combinations)
        total = len(all_pairs)

    yield {"type": "start", "total": total,
           "message": f"Full Grid: {total} комбинаций"}

    # Precompute inference (single GPU pass for all images)
    yield {"type": "progress", "step": "inference",
           "message": f"Инференс PatchCore для {len(image_files)} изображений..."}
    t_inf = time.perf_counter()
    precomp = precompute_inference(runner, image_files, gt_dir)
    inf_ms = round((time.perf_counter() - t_inf) * 1000)
    yield {"type": "progress", "step": "inference_done",
           "message": f"Инференс завершён за {inf_ms} мс, {len(precomp)} изображений"}

    n_good   = sum(1 for item in precomp if item["gt_label"] is None)
    n_defect = len(precomp) - n_good

    cache = _load_full_grid_cache() if use_cache else {}
    done  = 0

    # Group all_pairs by detection combo to batch classify calls
    from collections import defaultdict
    by_det: Dict[tuple, List[tuple]] = defaultdict(list)
    for det_vals, cls_vals in all_pairs:
        by_det[det_vals].append(cls_vals)

    for det_vals, cls_list in by_det.items():
        det_params = dict(zip(det_keys, det_vals))

        # Run detection for all images with this detection combo
        contours_per_image = []
        fp_cnt = 0
        tp_cnt = 0
        fn_cnt = 0
        for item in precomp:
            verdict, cnts = _detect_calibrated(
                item["heatmap"], item["orig_shape"],
                blur_kernel=int(det_params["blur_kernel"]),
                percentile_clip=float(det_params["percentile_clip"]),
                vmin_floor=float(det_params["vmin_floor"]),
                gamma=float(det_params["gamma"]),
                binary_threshold=float(det_params["binary_threshold"]),
                s_min=int(det_params["s_min"]),
                s_max=int(det_params["s_max"]),
                use_edge_mask=use_edge_mask,
                k_morph=k_morph,
                baseline_map=baseline_map,
                baseline_alpha=baseline_alpha,
                use_baseline_sub=use_baseline_sub,
                use_comp_sup=use_comp_sup,
                suppression_factor=suppression_factor,
                min_comp_area=min_comp_area,
            )
            contours_per_image.append(cnts)
            is_good   = item["gt_label"] is None
            is_defect = verdict == "defect"
            if is_good and is_defect:
                fp_cnt += 1
            elif not is_good and is_defect:
                tp_cnt += 1
            elif not is_good and not is_defect:
                fn_cnt += 1

        det_recall = tp_cnt / (tp_cnt + fn_cnt + 1e-9) if (tp_cnt + fn_cnt) > 0 else 0.0
        fp_rate    = fp_cnt / (n_good + 1e-9) if n_good > 0 else 0.0
        feasible   = fp_cnt == 0

        n_total_cnts = sum(len(c) for c in contours_per_image)

        # Classify with each classification combo
        for pca_n, method in cls_list:
            # Build cache key
            _key_dict = {**det_params,
                         "pca_n": pca_n, "method": method,
                         "n_neighbors": n_neighbors, "knn_metric": knn_metric,
                         "n_estimators": n_estimators, "max_depth": str(max_depth),
                         "random_state": random_state,
                         "use_edge_mask": use_edge_mask, "k_morph": k_morph,
                         "use_baseline_sub": use_baseline_sub,
                         "use_comp_sup": use_comp_sup}
            _key_raw = json.dumps(_key_dict, sort_keys=True)
            _key = hashlib.md5(_key_raw.encode()).hexdigest()[:16]

            if use_cache and _key in cache:
                entry = cache[_key]
                entry["_cached"] = True
                done += 1
                yield {"type": "result", "done": done, "total": total,
                       "result": entry, "cached": True}
                continue

            clf_accuracy = None
            clf_error    = None

            if n_total_cnts > 0:
                config = {
                    "method":         method,
                    "pca_components": pca_n,
                    "n_clusters":     len(DEFECT_TYPE_NAMES),
                    "n_neighbors":    n_neighbors,
                    "knn_metric":     knn_metric,
                    "n_estimators":   n_estimators,
                    "max_depth":      max_depth,
                    "random_state":   random_state,
                    "test_size":      0.5,
                }
                try:
                    clf_res = classify_by_embeddings(
                        [item["patch_embeddings"] for item in precomp],
                        [item["spatial_shape"] for item in precomp],
                        [item["orig_shape"] for item in precomp],
                        contours_per_image,
                        config=config,
                        gt_labels=[item["gt_label"] for item in precomp],
                        image_paths=[item["rel_path"] for item in precomp],
                        compute_umap=False,
                    )
                    acc = clf_res.get("test_accuracy") or clf_res.get("overall_accuracy")
                    clf_accuracy = round(float(acc), 4) if acc is not None else None
                except Exception as e:
                    clf_error = str(e)

            combined = round(clf_accuracy * (1.0 if feasible else 0.0), 4) \
                if clf_accuracy is not None else None

            entry = {
                "params": {
                    **det_params,
                    "pca_n":  pca_n,
                    "method": method,
                },
                "metrics": {
                    "detection_recall": round(det_recall, 4),
                    "fp_rate":          round(fp_rate, 4),
                    "fp_count":         fp_cnt,
                    "tp_count":         tp_cnt,
                    "fn_count":         fn_cnt,
                    "clf_accuracy":     clf_accuracy,
                    "combined_score":   combined,
                    "feasible":         feasible,
                    "n_defect_contours": n_total_cnts,
                },
                "_key":    _key,
                "_cached": False,
            }
            if clf_error:
                entry["clf_error"] = clf_error

            if use_cache and clf_error is None:
                cache[_key] = entry
                _save_full_grid_cache(cache)

            done += 1
            yield {"type": "result", "done": done, "total": total,
                   "result": entry, "cached": False}

    yield {"type": "done", "total": total, "done": done,
           "message": f"Full Grid завершён: {done}/{total} точек"}
