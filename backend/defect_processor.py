"""
Три версии обработки дефектов. Параллелизм осмысленный:
каждый дефект проходит полный анализ формы (моменты, выпуклая оболочка,
эллипс, кольцевые дескрипторы) — ~1–5 мс на дефект.

Persistent pool (один на весь процесс) устраняет overhead Windows-spawn.
"""

import math
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from defect_classifier import (
    classify_defect_full,
    get_defect_color,
    get_defect_name,
)

try:
    from optimizer.parameter_optimizer import DEFAULT_PARAMS, load_best_params
    _OPTIMIZER_AVAILABLE = True
except ImportError:
    _OPTIMIZER_AVAILABLE = False
    DEFAULT_PARAMS = {
        "threshold":             0.65,
        "min_area":              0,
        "morphology_kernel":     3,
        "morphology_iterations": 1,
        "gaussian_blur_kernel":  0,
    }

# ---------------------------------------------------------------------------
# Параметры бинаризации (загружаются из best_params.json при старте)
# ---------------------------------------------------------------------------

_binarize_params: Dict = DEFAULT_PARAMS.copy()


def reload_binarize_params() -> Dict:
    """Перечитывает best_params.json. При отсутствии файла — возвращает DEFAULT_PARAMS."""
    global _binarize_params
    if _OPTIMIZER_AVAILABLE:
        loaded = load_best_params()
        if loaded:
            _binarize_params = loaded
            return _binarize_params
    _binarize_params = DEFAULT_PARAMS.copy()
    return _binarize_params


def get_binarize_params() -> Dict:
    """Возвращает копию текущих параметров бинаризации."""
    return _binarize_params.copy()


# Загружаем при первом импорте модуля
reload_binarize_params()


# ---------------------------------------------------------------------------
# Цвета классов размера [BGR] (оставлено для обратной совместимости)
# ---------------------------------------------------------------------------

SIZE_CLASS_COLORS: List[Tuple[int, int, int]] = [
    (0,   200,   0),   # 0 — зелёный  (малый)
    (0,   200, 200),   # 1 — жёлтый
    (0,    80, 220),   # 2 — оранжевый
    (0,     0, 220),   # 3 — красный  (крупный)
    (200,   0, 200),   # 4 — магента
    (200, 100,   0),   # 5 — бирюзовый
    (100,   0, 200),   # 6 — фиолетовый
    (200, 200,   0),   # 7 — голубой
    (150, 150, 150),   # 8 — серый
    (255, 255, 255),   # 9 — белый
]


def get_size_class_color(cls: int) -> Tuple[int, int, int]:
    """Возвращает цвет для класса размера."""
    return SIZE_CLASS_COLORS[cls % len(SIZE_CLASS_COLORS)]


# ---------------------------------------------------------------------------
# Обнуление границ и углов
# ---------------------------------------------------------------------------

def zero_edges_and_corners(
    heatmap: np.ndarray,
    border_width: int = 28,
    corner_extra: int = 0,
) -> np.ndarray:
    """
    Обнуляет значения heatmap на границах и в углах изображения.
    
    Args:
        heatmap: входная карта аномалий [H, W]
        border_width: ширина обнуляемой границы в пикселях
        corner_extra: дополнительное обнуление в углах (треугольником)
    
    Returns:
        masked_heatmap: с обнулёнными границами и углами
    """
    h, w = heatmap.shape
    result = heatmap.copy()
    
    # 1. Обнуляем всю границу
    result[:border_width, :] = 0                    # верх
    result[-border_width:, :] = 0                   # низ
    result[:, :border_width] = 0                    # лево
    result[:, -border_width:] = 0                   # право
    
    # 2. Дополнительное обнуление углов (треугольником)
    if corner_extra > 0:
        # Левый верхний угол
        for i in range(min(corner_extra, border_width)):
            result[i, :border_width - i] = 0
        
        # Правый верхний угол
        for i in range(min(corner_extra, border_width)):
            result[i, w - (border_width - i):] = 0
        
        # Левый нижний угол
        for i in range(min(corner_extra, border_width)):
            result[h - 1 - i, :border_width - i] = 0
        
        # Правый нижний угол
        for i in range(min(corner_extra, border_width)):
            result[h - 1 - i, w - (border_width - i):] = 0
    
    return result


def extract_defects_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    edge_zeroing: bool = True,
    border_width: int = 28,
    corner_extra: int = 0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Бинаризация heatmap с обнулением границ и углов.
    
    Пайплайн:
        0. Обнуление границ и углов
        1. Gaussian blur (если gaussian_blur_kernel > 0)
        2. Порог → binary 0/255
        3. Morphology open + close
        4. Фильтр min_area
        5. Удаление контуров, касающихся границ
    """
    h = heatmap.astype(np.float32)

    # 0. Обнуляем границы и углы
    if edge_zeroing:
        h = zero_edges_and_corners(h, border_width=border_width, corner_extra=corner_extra)

    # 1. Gaussian blur
    blur_k = int(_binarize_params.get("gaussian_blur_kernel", 0))
    if blur_k > 0:
        if blur_k % 2 == 0:
            blur_k += 1
        h = cv2.GaussianBlur(h, (blur_k, blur_k), 0)

    # 2. Threshold
    binary = (h > float(threshold)).astype(np.uint8) * 255

    # 3. Morphology
    mk = int(_binarize_params.get("morphology_kernel", 3))
    mi = int(_binarize_params.get("morphology_iterations", 1))
    if mk > 1:
        if mk % 2 == 0:
            mk += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=mi)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=mi)

    # 4. min_area filter
    min_area = float(_binarize_params.get("min_area", 0))
    if min_area > 0:
        conts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(binary)
        for c in conts:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(filtered, [c], -1, 255, cv2.FILLED)
        binary = filtered

    # 5. Удаляем контуры, которые касаются границы
    conts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(binary)
    h_border, w_border = binary.shape
    
    for c in conts:
        touches_border = False
        # Проверяем точки контура
        for point in c.squeeze():
            if len(point.shape) > 1:
                for p in point:
                    if p[0] <= 2 or p[0] >= w_border - 3 or p[1] <= 2 or p[1] >= h_border - 3:
                        touches_border = True
                        break
            else:
                if point[0] <= 2 or point[0] >= w_border - 3 or point[1] <= 2 or point[1] >= h_border - 3:
                    touches_border = True
                    break
            if touches_border:
                break
        
        if not touches_border and cv2.contourArea(c) >= min_area:
            cv2.drawContours(filtered, [c], -1, 255, cv2.FILLED)
    
    binary = filtered
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return list(contours), binary


# ---------------------------------------------------------------------------
# Примитивы
# ---------------------------------------------------------------------------

def compute_defect_size(contour: np.ndarray) -> Dict:
    area = float(cv2.contourArea(contour))
    diameter = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0
    major_axis = diameter
    if len(contour) >= 5:
        try:
            _, axes, _ = cv2.fitEllipse(contour)
            major_axis = float(max(axes))
        except cv2.error:
            pass
    return {"area": area, "diameter": diameter, "major_axis": major_axis}


def classify_defect_size(diameter: float, S_min: float, S_max: float, K: int) -> int:
    """Классифицирует дефект по размеру (0..K-1)."""
    if S_max <= S_min:
        return 0
    norm = max(0.0, min(1.0, (diameter - S_min) / (S_max - S_min)))
    return min(int(norm * K), K - 1)


# ---------------------------------------------------------------------------
# Аннотирование изображения
# ---------------------------------------------------------------------------

def annotate_image(
    image: np.ndarray,
    contours: List[np.ndarray],
    defects: List[Dict],
    K: int,
) -> np.ndarray:
    """
    Аннотирует изображение с отображением типа дефекта и класса размера.
    """
    overlay = image.copy()
    bbox_to_defect: Dict[Tuple, Dict] = {tuple(d["bbox"]): d for d in defects}

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        d = bbox_to_defect.get((x, y, w, h))
        if d is None:
            cv2.drawContours(overlay, [contour], -1, (80, 80, 80), 1)
            continue
        
        # Используем цвет из классификации типа дефекта
        color = d.get("type_color", (0, 200, 0))
        
        # Заливка контура полупрозрачным цветом
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        overlay[mask == 255] = (
            overlay[mask == 255] * 0.45
            + np.array(color, dtype=np.float32) * 0.55
        ).astype(np.uint8)
        
        # Рисуем границу контура
        cv2.drawContours(overlay, [contour], -1, color, 2)
        
        # Отображаем информацию о дефекте
        cx, cy = x + w // 2, y + h // 2
        type_name = d.get("type_name", "дефект")
        size_class = d.get("class", 0)
        confidence = d.get("confidence", 0)
        
        # Показываем тип дефекта и класс размера
        cv2.putText(overlay, f"{type_name} C{size_class}", 
                    (cx - 30, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Показываем уверенность (опционально)
        # cv2.putText(overlay, f"{confidence:.0%}", 
        #             (cx - 30, cy - 8),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    
    return overlay


# ---------------------------------------------------------------------------
# Полный анализ одного дефекта
# ---------------------------------------------------------------------------

def _full_defect_analysis(
    contour: np.ndarray, 
    uid: int,
    S_min: float, 
    S_max: float, 
    K: int
) -> Optional[Dict]:
    """
    Полный геометрический анализ контура с классификацией типа дефекта.
    
    Returns:
        Dict с информацией о дефекте, включая:
        - id: номер дефекта
        - area: площадь
        - diameter: диаметр
        - major_axis: большая ось
        - perimeter: периметр
        - circularity: округлость
        - solidity: заполненность
        - enc_radius: радиус описанной окружности
        - class: класс размера (0..K-1)
        - defect_type: тип дефекта (нитка, пятно, разрыв, дырка, заноза, царапина, вмятина)
        - type_name: русское название типа
        - type_color: цвет для отображения
        - confidence: уверенность классификации
        - size_name: размерная категория (малый/средний/крупный)
        - bbox: bounding box
    """
    area = float(cv2.contourArea(contour))
    if area < 1.0:
        return None

    # Основные метрики
    perimeter = float(cv2.arcLength(contour, True))
    diameter = math.sqrt(4.0 * area / math.pi)

    # Фильтрация по размеру
    if diameter < S_min or diameter > S_max:
        return None

    # Bounding rect
    x, y, w, h = cv2.boundingRect(contour)

    # Выпуклая оболочка
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0
    circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    # Эллипс
    major_axis = diameter
    if len(contour) >= 5:
        try:
            _, (ma, mi), _ = cv2.fitEllipse(contour)
            major_axis = float(max(ma, mi))
        except cv2.error:
            pass

    # Минимальная охватывающая окружность
    _, enc_radius = cv2.minEnclosingCircle(contour)

    # Класс размера (0..K-1)
    size_class = classify_defect_size(diameter, S_min, S_max, K)
    
    # Классификация типа дефекта
    defect_type, type_confidence, size_name, type_details = classify_defect_full(
        contour, size_class, K, area, diameter
    )

    # Моменты и Hu-моменты
    M = cv2.moments(contour)
    hu = cv2.HuMoments(M).flatten()

    return {
        "id": uid,
        "area": round(area, 2),
        "diameter": round(diameter, 2),
        "major_axis": round(major_axis, 2),
        "perimeter": round(perimeter, 2),
        "circularity": round(circularity, 4),
        "solidity": round(solidity, 4),
        "enc_radius": round(float(enc_radius), 2),
        "hu_moments": [round(float(v), 8) for v in hu],
        "class": size_class,                    # класс размера
        "defect_type": defect_type,             # тип дефекта
        "type_name": get_defect_name(defect_type),  # русское название
        "type_color": get_defect_color(defect_type),  # цвет для отображения
        "confidence": round(type_confidence, 2),     # уверенность
        "size_name": size_name,                      # размерная категория
        "bbox": [int(x), int(y), int(w), int(h)],
    }


# ---------------------------------------------------------------------------
# Worker для multiprocessing (module-level — требование Windows/spawn)
# ---------------------------------------------------------------------------

def _worker(args: Tuple) -> Optional[Dict]:
    uid, pts, S_min, S_max, K = args
    contour = np.array(pts, dtype=np.int32)
    return _full_defect_analysis(contour, uid, S_min, S_max, K)


def _worker_init() -> None:
    """
    Инициализатор воркера ProcessPool.
    Скрываем CUDA: воркеры работают ТОЛЬКО на CPU (cv2, numpy, контуры).
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _dummy() -> int:
    """Прогрев воркера (import + init)."""
    import os
    return os.getpid()


# ---------------------------------------------------------------------------
# Persistent CPU pool
# ---------------------------------------------------------------------------

_pools: Dict[int, ProcessPoolExecutor] = {}


def get_or_create_pool(workers: int) -> ProcessPoolExecutor:
    """Возвращает уже запущенный пул. Создаёт + прогревает при первом вызове."""
    if workers not in _pools:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
        )
        # Прогрев: запускаем dummy в каждом воркере
        futs = [pool.submit(_dummy) for _ in range(workers)]
        for f in futs:
            f.result()
        _pools[workers] = pool
    return _pools[workers]


def shutdown_all_pools() -> None:
    for pool in _pools.values():
        pool.shutdown(wait=False)
    _pools.clear()


# ---------------------------------------------------------------------------
# Версия 1 — Sequential
# ---------------------------------------------------------------------------

def sequential_process(
    contours: List[np.ndarray],
    S_min: float, 
    S_max: float, 
    K: int,
) -> Dict:
    t0 = time.perf_counter()
    defects, uid = [], 1
    for c in contours:
        d = _full_defect_analysis(c, uid, S_min, S_max, K)
        if d:
            defects.append(d)
            uid += 1
    elapsed = (time.perf_counter() - t0) * 1_000
    return {
        "defects": defects,
        "n_defects_found": len(contours),
        "n_defects_after_filter": len(defects),
        "time_ms": round(elapsed, 3),
        "speedup": 1.0,
        "mode": "sequential",
        "workers": 1,
    }


# ---------------------------------------------------------------------------
# Версия 2 — Parallel CPU
# ---------------------------------------------------------------------------

def parallel_cpu_process(
    contours: List[np.ndarray],
    S_min: float, 
    S_max: float, 
    K: int,
    workers: int,
    sequential_time_ms: Optional[float] = None,
    pool: Optional[ProcessPoolExecutor] = None,
) -> Dict:
    """
    Параллельная классификация дефектов через ProcessPoolExecutor.
    """
    if pool is None:
        pool = get_or_create_pool(workers)

    args = [(i + 1, c.tolist(), S_min, S_max, K) for i, c in enumerate(contours)]

    t0 = time.perf_counter()
    results = list(pool.map(_worker, args))
    elapsed = (time.perf_counter() - t0) * 1_000

    defects = []
    uid = 1
    for d in results:
        if d is not None:
            d["id"] = uid
            defects.append(d)
            uid += 1

    speedup = None
    if sequential_time_ms and elapsed > 0:
        speedup = round(sequential_time_ms / elapsed, 3)

    return {
        "defects": defects,
        "n_defects_found": len(contours),
        "n_defects_after_filter": len(defects),
        "time_ms": round(elapsed, 3),
        "speedup": speedup,
        "mode": "parallel_cpu",
        "workers": workers,
    }


# ---------------------------------------------------------------------------
# Версия 3 — GPU
# ---------------------------------------------------------------------------

def cpu_heatmap_pipeline(
    heatmap: np.ndarray,
    S_min: float, 
    S_max: float, 
    K: int,
    threshold: float = 0.5,
    border_width: int = 28,
) -> Tuple[float, List[np.ndarray], List[Dict]]:
    """
    Эталонный CPU-пайплайн для heatmap с обнулением границ.
    Возвращает (time_ms, contours, defects).
    """
    t0 = time.perf_counter()
    contours, _ = extract_defects_from_heatmap(
        heatmap, threshold,
        edge_zeroing=True,
        border_width=border_width,
    )
    defects, uid = [], 1
    for c in contours:
        d = _full_defect_analysis(c, uid, S_min, S_max, K)
        if d:
            defects.append(d)
            uid += 1
    elapsed = (time.perf_counter() - t0) * 1_000
    return round(elapsed, 3), contours, defects


def parallel_gpu_process(
    heatmap: np.ndarray,
    S_min: float, 
    S_max: float, 
    K: int,
    threshold: float = 0.5,
    border_width: int = 28,
    cpu_pipeline_time_ms: Optional[float] = None,
) -> Dict:
    """
    GPU-пайплайн с обнулением границ.
    """
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Обнуляем границы на CPU
    h = heatmap.copy()
    h[:border_width, :] = 0
    h[-border_width:, :] = 0
    h[:, :border_width] = 0
    h[:, -border_width:] = 0

    # GPU: threshold + морфология
    t_gpu = torch.from_numpy(h).float().unsqueeze(0).unsqueeze(0).to(device)
    binary = (t_gpu > threshold).float()

    k, pad = 3, 1
    eroded = -F.max_pool2d(-binary, k, stride=1, padding=pad)
    opened = F.max_pool2d(eroded, k, stride=1, padding=pad)
    dilated = F.max_pool2d(opened, k, stride=1, padding=pad)
    closed = -F.max_pool2d(-dilated, k, stride=1, padding=pad)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gpu_done = time.perf_counter()

    binary_cpu = (closed.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects, uid = [], 1
    for c in contours:
        d = _full_defect_analysis(c, uid, S_min, S_max, K)
        if d:
            defects.append(d)
            uid += 1

    elapsed_total = (time.perf_counter() - t0) * 1_000
    elapsed_gpu = (t_gpu_done - t0) * 1_000

    speedup = None
    if cpu_pipeline_time_ms and elapsed_total > 0:
        speedup = round(cpu_pipeline_time_ms / elapsed_total, 3)

    return {
        "defects": defects,
        "n_defects_found": len(contours),
        "n_defects_after_filter": len(defects),
        "time_ms": round(elapsed_total, 3),
        "gpu_time_ms": round(elapsed_gpu, 3),
        "speedup": speedup,
        "mode": "parallel_gpu",
        "workers": None,
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Версия 4 — Batch GPU (все heatmap за один GPU-проход)
# ---------------------------------------------------------------------------

def batch_gpu_pipeline(
    heatmaps: List[np.ndarray],
    S_min: float, 
    S_max: float, 
    K: int,
    threshold: float = 0.5,
    border_width: int = 28,
) -> Tuple[float, float, List[List], List[List]]:
    """
    Складывает все heatmap в тензор [N, 1, H, W] и делает морфологию
    одним вызовом GPU.
    """
    if not heatmaps:
        return 0.0, 0.0, [], []

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Обнуляем границы у каждого heatmap
    masked_heatmaps = []
    for hm in heatmaps:
        h = hm.copy()
        h[:border_width, :] = 0
        h[-border_width:, :] = 0
        h[:, :border_width] = 0
        h[:, -border_width:] = 0
        masked_heatmaps.append(h)

    batch = torch.stack(
        [torch.from_numpy(hm).float().unsqueeze(0) for hm in masked_heatmaps], dim=0
    ).to(device)

    binary = (batch > threshold).float()
    k, pad = 3, 1
    eroded = -F.max_pool2d(-binary, k, stride=1, padding=pad)
    opened = F.max_pool2d(eroded, k, stride=1, padding=pad)
    dilated = F.max_pool2d(opened, k, stride=1, padding=pad)
    closed = -F.max_pool2d(-dilated, k, stride=1, padding=pad)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gpu = time.perf_counter()

    binary_np = (closed.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8) * 255

    all_contours: List[List] = []
    all_defects: List[List] = []
    for i in range(len(heatmaps)):
        contours, _ = cv2.findContours(
            binary_np[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = list(contours)
        defects, uid = [], 1
        for c in contours:
            d = _full_defect_analysis(c, uid, S_min, S_max, K)
            if d:
                defects.append(d)
                uid += 1
        all_contours.append(contours)
        all_defects.append(defects)

    elapsed_total = (time.perf_counter() - t0) * 1_000
    elapsed_gpu = (t_gpu - t0) * 1_000
    return round(elapsed_total, 3), round(elapsed_gpu, 3), all_contours, all_defects