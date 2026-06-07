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
# Цвета классов [BGR]
# ---------------------------------------------------------------------------

CLASS_COLORS: List[Tuple[int, int, int]] = [
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


def get_class_color(cls: int) -> Tuple[int, int, int]:
    return CLASS_COLORS[cls % len(CLASS_COLORS)]


# ---------------------------------------------------------------------------
# Экспоненциальная маска углов
# ---------------------------------------------------------------------------

def create_corner_exponential_mask(
    shape: Tuple[int, int],
    corner_size: int = 30,
    transition_width: int = 10,
    exponent: float = 2.0,
) -> np.ndarray:
    """
    Создаёт маску, где углы полностью отрезаются по экспоненте.
    
    Принцип: для каждого пикселя вычисляется расстояние до ближайшего угла.
    Если расстояние < corner_size → маска = 0 (полное отсечение)
    Если distance в зоне перехода → плавный экспоненциальный подъём
    Если расстояние > corner_size + transition_width → маска = 1
    
    Args:
        shape: (height, width) размер маски
        corner_size: радиус полного отсечения от угла (пиксели)
        transition_width: ширина зоны плавного перехода (пиксели)
        exponent: степень экспоненты (1 = линейный, 2 = квадратичный, 3 = кубический)
    
    Returns:
        mask: float32 маска от 0 до 1
    """
    h, w = shape
    mask = np.ones((h, w), dtype=np.float32)
    
    # Координаты четырёх углов
    corners = [
        (0, 0),           # левый верхний
        (0, w - 1),       # правый верхний
        (h - 1, 0),       # левый нижний
        (h - 1, w - 1),   # правый нижний
    ]
    
    # Для каждого пикселя находим минимальное расстояние до любого угла
    y_coords, x_coords = np.ogrid[:h, :w]
    
    min_dist_to_corner = np.full((h, w), float('inf'), dtype=np.float32)
    
    for cy, cx in corners:
        dist = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
        min_dist_to_corner = np.minimum(min_dist_to_corner, dist)
    
    # Вычисляем маску с экспоненциальным переходом
    corner_zone = corner_size + transition_width
    mask = np.zeros_like(min_dist_to_corner)
    
    # Там где расстояние < corner_size → маска = 0
    mask[min_dist_to_corner < corner_size] = 0.0
    
    # Там где расстояние > corner_zone → маска = 1
    mask[min_dist_to_corner > corner_zone] = 1.0
    
    # В зоне перехода: экспоненциальный подъём
    transition_mask = (min_dist_to_corner >= corner_size) & (min_dist_to_corner <= corner_zone)
    if transition_mask.any():
        t = (min_dist_to_corner[transition_mask] - corner_size) / transition_width
        # Экспоненциальный подъём: t^exponent
        mask[transition_mask] = t ** exponent
    
    return mask


def create_corner_gaussian_mask(
    shape: Tuple[int, int],
    corner_radius: int = 40,
    sigma: float = 12.0,
) -> np.ndarray:
    """
    Альтернативная маска: гауссовское подавление от углов.
    
    Args:
        shape: (height, width)
        corner_radius: радиус подавления
        sigma: параметр Гаусса
    """
    h, w = shape
    y_coords, x_coords = np.ogrid[:h, :w]
    
    # Координаты углов
    corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
    
    mask = np.ones((h, w), dtype=np.float32)
    
    for cy, cx in corners:
        dist = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
        gaussian = np.exp(-(dist ** 2) / (2 * sigma ** 2))
        # Чем ближе к углу, тем меньше вес
        mask = np.minimum(mask, 1 - gaussian * 0.95)
    
    return mask


def apply_corner_mask(
    heatmap: np.ndarray,
    corner_size: int = 35,
    transition_width: int = 15,
    exponent: float = 2.5,
    mask_type: str = "gaussian",  # "exponential" или "gaussian"
) -> np.ndarray:
    """
    Применяет маску углов к heatmap.
    
    Args:
        heatmap: входная карта аномалий
        corner_size: радиус полного отсечения от угла
        transition_width: ширина зоны перехода
        exponent: степень экспоненты (выше = резче переход)
        mask_type: тип маски ("exponential" или "gaussian")
    """
    if corner_size <= 0:
        return heatmap
    
    if mask_type == "gaussian":
        mask = create_corner_gaussian_mask(
            heatmap.shape, 
            corner_radius=corner_size + transition_width,
            sigma=max(5, corner_size / 3)
        )
    else:
        mask = create_corner_exponential_mask(
            heatmap.shape,
            corner_size=corner_size,
            transition_width=transition_width,
            exponent=exponent
        )
    
    return heatmap * mask


# ---------------------------------------------------------------------------
# Примитивы
# ---------------------------------------------------------------------------

def compute_defect_size(contour: np.ndarray) -> Dict:
    area     = float(cv2.contourArea(contour))
    diameter = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0
    major_axis = diameter
    if len(contour) >= 5:
        try:
            _, axes, _ = cv2.fitEllipse(contour)
            major_axis = float(max(axes))
        except cv2.error:
            pass
    return {"area": area, "diameter": diameter, "major_axis": major_axis}


def classify_defect(size: float, S_min: float, S_max: float, K: int) -> int:
    if S_max <= S_min:
        return 0
    norm = max(0.0, min(1.0, (size - S_min) / (S_max - S_min)))
    return min(int(norm * K), K - 1)


def extract_defects_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    # Параметры маски углов
    corner_mask_enabled: bool = True,
    corner_size: int = 35,           # радиус полного отсечения от угла
    corner_transition: int = 15,     # ширина зоны перехода
    corner_exponent: float = 2.5,    # крутизна экспоненты (2-3)
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Бинаризация heatmap с экспоненциальной маской углов.
    
    Пайплайн:
        1. Экспоненциальная маска углов (полное отсечение углов)
        2. Gaussian blur (если gaussian_blur_kernel > 0)
        3. Порог → binary 0/255
        4. Morphology open + close
        5. Фильтр min_area + удаление контуров касающихся границ
    """
    h = heatmap.astype(np.float32)

    # 0. Экспоненциальная маска углов
    if corner_mask_enabled:
        h = apply_corner_mask(
            h,
            corner_size=corner_size,
            transition_width=corner_transition,
            exponent=corner_exponent,
            mask_type="exponential"
        )

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
    
    # 5. Удаляем контуры, которые касаются границ (углы уже отрезаны маской)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(binary)
    h_border, w_border = binary.shape
    border_margin = max(1, corner_size // 10)  # небольшой отступ от границы
    
    for c in contours:
        # Проверяем площадь
        if cv2.contourArea(c) < min_area:
            continue
        
        # Проверяем, касается ли контур границы
        touches_border = False
        for point in c.squeeze():
            if len(point.shape) > 1:
                for p in point:
                    if p[0] <= border_margin or p[0] >= w_border - border_margin - 1:
                        touches_border = True
                        break
                    if p[1] <= border_margin or p[1] >= h_border - border_margin - 1:
                        touches_border = True
                        break
            else:
                if point[0] <= border_margin or point[0] >= w_border - border_margin - 1:
                    touches_border = True
                    break
                if point[1] <= border_margin or point[1] >= h_border - border_margin - 1:
                    touches_border = True
                    break
            if touches_border:
                break
        
        if not touches_border:
            cv2.drawContours(filtered, [c], -1, 255, cv2.FILLED)
    
    binary = filtered
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return list(contours), binary


# ---------------------------------------------------------------------------
# Аннотирование изображения
# ---------------------------------------------------------------------------

def annotate_image(
    image: np.ndarray,
    contours: List[np.ndarray],
    defects: List[Dict],
    K: int,
) -> np.ndarray:
    overlay = image.copy()
    bbox_to_defect: Dict[Tuple, Dict] = {tuple(d["bbox"]): d for d in defects}

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        d = bbox_to_defect.get((x, y, w, h))
        if d is None:
            cv2.drawContours(overlay, [contour], -1, (80, 80, 80), 1)
            continue
        color = get_class_color(d["class"])
        mask  = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        overlay[mask == 255] = (
            overlay[mask == 255] * 0.45
            + np.array(color, dtype=np.float32) * 0.55
        ).astype(np.uint8)
        cv2.drawContours(overlay, [contour], -1, color, 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.putText(overlay, f"C{d['class']}", (cx - 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


# ---------------------------------------------------------------------------
# Полный анализ одного дефекта (общий для всех трёх версий)
# ---------------------------------------------------------------------------

def _full_defect_analysis(contour: np.ndarray, uid: int,
                          S_min: float, S_max: float, K: int) -> Optional[Dict]:
    """
    Полный геометрический анализ контура:
      area, perimeter, diameter, major_axis, circularity, solidity,
      min_enclosing_circle, Hu-моменты → ~1-5 мс на дефект.
    """
    area = float(cv2.contourArea(contour))
    if area < 1.0:
        return None

    # Основные метрики
    perimeter = float(cv2.arcLength(contour, True))
    diameter  = math.sqrt(4.0 * area / math.pi)

    # Фильтрация до тяжёлых вычислений
    if diameter < S_min or diameter > S_max:
        return None

    # Bounding rect
    x, y, w, h = cv2.boundingRect(contour)

    # Выпуклая оболочка
    hull      = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity  = area / hull_area if hull_area > 0 else 0.0
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

    # Моменты и Hu-моменты (7 инвариантов)
    M  = cv2.moments(contour)
    hu = cv2.HuMoments(M).flatten()

    # Класс
    cls = classify_defect(diameter, S_min, S_max, K)

    return {
        "id":           uid,
        "area":         round(area, 2),
        "diameter":     round(diameter, 2),
        "major_axis":   round(major_axis, 2),
        "perimeter":    round(perimeter, 2),
        "circularity":  round(circularity, 4),
        "solidity":     round(solidity, 4),
        "enc_radius":   round(float(enc_radius), 2),
        "hu_moments":   [round(float(v), 8) for v in hu],
        "class":        cls,
        "bbox":         [int(x), int(y), int(w), int(h)],
    }


# ---------------------------------------------------------------------------
# Worker для multiprocessing
# ---------------------------------------------------------------------------

def _worker(args: Tuple) -> Optional[Dict]:
    uid, pts, S_min, S_max, K = args
    contour = np.array(pts, dtype=np.int32)
    return _full_defect_analysis(contour, uid, S_min, S_max, K)


def _worker_init() -> None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _dummy() -> int:
    import os
    return os.getpid()


# ---------------------------------------------------------------------------
# Persistent CPU pool
# ---------------------------------------------------------------------------

_pools: Dict[int, ProcessPoolExecutor] = {}


def get_or_create_pool(workers: int) -> ProcessPoolExecutor:
    if workers not in _pools:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
        )
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
    S_min: float, S_max: float, K: int,
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
    S_min: float, S_max: float, K: int,
    workers: int,
    sequential_time_ms: Optional[float] = None,
    pool: Optional[ProcessPoolExecutor] = None,
) -> Dict:
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
    S_min: float, S_max: float, K: int,
    threshold: float = 0.5,
    corner_size: int = 35,
    corner_transition: int = 15,
    corner_exponent: float = 2.5,
) -> Tuple[float, List[np.ndarray], List[Dict]]:
    """
    Эталонный CPU-пайплайн для heatmap с экспоненциальной маской углов.
    """
    t0 = time.perf_counter()
    contours, _ = extract_defects_from_heatmap(
        heatmap, threshold,
        corner_size=corner_size,
        corner_transition=corner_transition,
        corner_exponent=corner_exponent,
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
    S_min: float, S_max: float, K: int,
    threshold: float = 0.5,
    corner_size: int = 35,
    corner_transition: int = 15,
    corner_exponent: float = 2.5,
    cpu_pipeline_time_ms: Optional[float] = None,
) -> Dict:
    """
    GPU-пайплайн с экспоненциальной маской углов.
    """
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Применяем маску углов на CPU
    h = heatmap.copy()
    h = apply_corner_mask(
        h,
        corner_size=corner_size,
        transition_width=corner_transition,
        exponent=corner_exponent,
        mask_type="exponential"
    )

    # GPU: threshold + морфология
    t_gpu = torch.from_numpy(h).float().unsqueeze(0).unsqueeze(0).to(device)
    binary = (t_gpu > threshold).float()

    k, pad = 3, 1
    eroded  = -F.max_pool2d(-binary,  k, stride=1, padding=pad)
    opened  =  F.max_pool2d(eroded,   k, stride=1, padding=pad)
    dilated = F.max_pool2d(opened,    k, stride=1, padding=pad)
    closed  = -F.max_pool2d(-dilated, k, stride=1, padding=pad)

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

    elapsed_total  = (time.perf_counter() - t0) * 1_000
    elapsed_gpu    = (t_gpu_done - t0) * 1_000

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
# Версия 4 — Batch GPU
# ---------------------------------------------------------------------------

def batch_gpu_pipeline(
    heatmaps: List[np.ndarray],
    S_min: float, S_max: float, K: int,
    threshold: float = 0.5,
    corner_size: int = 35,
    corner_transition: int = 15,
    corner_exponent: float = 2.5,
) -> Tuple[float, float, List[List], List[List]]:
    """
    Batch GPU пайплайн с экспоненциальной маской углов.
    """
    if not heatmaps:
        return 0.0, 0.0, [], []

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Применяем маску к каждому heatmap
    masked_heatmaps = []
    for hm in heatmaps:
        h = hm.copy()
        h = apply_corner_mask(
            h,
            corner_size=corner_size,
            transition_width=corner_transition,
            exponent=corner_exponent,
            mask_type="exponential"
        )
        masked_heatmaps.append(h)

    batch = torch.stack(
        [torch.from_numpy(hm).float().unsqueeze(0) for hm in masked_heatmaps], dim=0
    ).to(device)

    binary  = (batch > threshold).float()
    k, pad  = 3, 1
    eroded  = -F.max_pool2d(-binary,  k, stride=1, padding=pad)
    opened  =  F.max_pool2d(eroded,   k, stride=1, padding=pad)
    dilated =  F.max_pool2d(opened,   k, stride=1, padding=pad)
    closed  = -F.max_pool2d(-dilated, k, stride=1, padding=pad)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gpu = time.perf_counter()

    binary_np = (closed.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8) * 255

    all_contours: List[List] = []
    all_defects:  List[List] = []
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
    elapsed_gpu   = (t_gpu - t0) * 1_000
    return round(elapsed_total, 3), round(elapsed_gpu, 3), all_contours, all_defects