"""
Метрики качества детекции и автоподбор параметров (grid search).

Четыре режима:
  sequential        — последовательный перебор (threshold × S_min × S_max)
  parallel_cpu      — ThreadPoolExecutor по кускам комбинаций
                      (cv2/numpy освобождают GIL → реальный параллелизм без IPC overhead;
                       ProcessPool на Windows: pickle ~250 МБ данных → зависание на 10-30 мин)
  parallel_gpu      — GPU-морфология, последовательный перебор
  parallel_gpu_batch — GPU-батч: все M порогов для одного изображения за 1 GPU-проход
"""

import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def find_gt_mask(image_rel_path: str, gt_dir: Path) -> Optional[np.ndarray]:
    """
    Бинарная маска ground truth для изображения (uint8, 0/1).
    Возвращает None для good-изображений (нет дефектов).
    """
    parts = image_rel_path.replace("\\", "/").split("/")
    if len(parts) < 2:
        return None
    defect_type = parts[0]
    if defect_type == "good":
        return None
    stem = Path(parts[-1]).stem
    mask_path = gt_dir / defect_type / f"{stem}_mask.png"
    if not mask_path.exists():
        return None
    with open(mask_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return None if mask is None else (mask > 127).astype(np.uint8)


# ---------------------------------------------------------------------------
# Пиксельные метрики
# ---------------------------------------------------------------------------

def compute_pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict:
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = int(np.logical_and( p,  g).sum())
    fp = int(np.logical_and( p, ~g).sum())
    fn = int(np.logical_and(~p,  g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou       = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "iou":       round(iou,       4),
    }


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _prep_gt(gt: Optional[np.ndarray], h: int, w: int) -> np.ndarray:
    """GT-маска нужного размера; None → нулевая маска (good-изображение)."""
    if gt is None:
        return np.zeros((h, w), dtype=np.uint8)
    if gt.shape != (h, w):
        return cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
    return gt


def _binary_to_pred_mask(binary_255: np.ndarray, S_min: float, S_max: float) -> np.ndarray:
    """Контуры → предсказанная маска с фильтрацией по размеру."""
    h, w = binary_255.shape
    contours, _ = cv2.findContours(binary_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred = np.zeros((h, w), dtype=np.uint8)
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 1.0:
            continue
        diam = math.sqrt(4.0 * area / math.pi)
        if S_min <= diam <= S_max:
            cv2.drawContours(pred, [c], -1, 1, cv2.FILLED)
    return pred


def _cpu_binary(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    """CPU: threshold → open → close → binary 255."""
    b = (heatmap > threshold).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  k)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    return b


def _gpu_binary(heatmap: np.ndarray, threshold: float, device: torch.device) -> np.ndarray:
    """GPU: threshold → open → close → binary 255 (numpy)."""
    t = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0).to(device)
    b = (t > threshold).float()
    k, pad = 3, 1
    eroded  = -F.max_pool2d(-b,      k, 1, pad)
    opened  =  F.max_pool2d(eroded,  k, 1, pad)
    dilated =  F.max_pool2d(opened,  k, 1, pad)
    closed  = -F.max_pool2d(-dilated,k, 1, pad)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (closed.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255


def _eval_combo(
    threshold: float, S_min: float, S_max: float,
    heatmaps: List[np.ndarray],
    gt_masks:  List[Optional[np.ndarray]],
    use_gpu: bool = False,
    device: Optional[torch.device] = None,
) -> Dict:
    """Оценка одной комбинации параметров на всех изображениях."""
    totals = dict(f1=0.0, iou=0.0, precision=0.0, recall=0.0)
    n = 0
    for hm, gt in zip(heatmaps, gt_masks):
        h, w = hm.shape
        gt_eff = _prep_gt(gt, h, w)
        if use_gpu and device is not None:
            binary = _gpu_binary(hm, threshold, device)
        else:
            binary = _cpu_binary(hm, threshold)
        pred = _binary_to_pred_mask(binary, S_min, S_max)
        m = compute_pixel_metrics(pred, gt_eff)
        for key in totals:
            totals[key] += m[key]
        n += 1
    if n == 0:
        return {"threshold": threshold, "S_min": S_min, "S_max": S_max,
                "f1": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "threshold": threshold, "S_min": S_min, "S_max": S_max,
        **{k: round(v / n, 4) for k, v in totals.items()},
    }


def _all_combos(thresholds, s_mins, s_maxs) -> list:
    return list(product(thresholds, s_mins, s_maxs))


# ---------------------------------------------------------------------------
# Module-level worker (обязательно для ProcessPoolExecutor на Windows/spawn)
# ---------------------------------------------------------------------------

def _chunk_worker(args: Tuple) -> List[Dict]:
    """
    Обрабатывает один кусок (threshold, S_min, S_max) комбинаций.
    Вызывается в дочернем процессе через ProcessPoolExecutor.
    Chunk-подход: heatmaps передаются N_workers раз (не N_combos раз),
    что сокращает IPC overhead в W раз.
    """
    param_chunk, heatmaps_list, gt_masks_list = args
    return [
        _eval_combo(thr, smin, smax, heatmaps_list, gt_masks_list)
        for thr, smin, smax in param_chunk
    ]


def _warmup_worker() -> bool:
    """Прогрев: импорт cv2/numpy в воркере при старте сервера."""
    import cv2 as _cv2
    import numpy as _np
    import math as _m
    return True


# ---------------------------------------------------------------------------
# Режим 1 — Sequential
# ---------------------------------------------------------------------------

def optimize_sequential(
    heatmaps:  List[np.ndarray],
    gt_masks:  List[Optional[np.ndarray]],
    thresholds: List[float],
    s_mins:    List[float],
    s_maxs:    List[float],
) -> Tuple[float, List[Dict]]:
    t0 = time.perf_counter()
    results = [
        _eval_combo(thr, smin, smax, heatmaps, gt_masks)
        for thr, smin, smax in _all_combos(thresholds, s_mins, s_maxs)
    ]
    return round((time.perf_counter() - t0) * 1_000, 3), results


# ---------------------------------------------------------------------------
# Режим 2 — Parallel CPU (ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def optimize_parallel_cpu(
    heatmaps:  List[np.ndarray],
    gt_masks:  List[Optional[np.ndarray]],
    thresholds: List[float],
    s_mins:    List[float],
    s_maxs:    List[float],
    workers:   int,
    pool:      Optional[ProcessPoolExecutor] = None,  # ignored, kept for совместимости
) -> Tuple[float, List[Dict]]:
    """
    Параллельный перебор через ThreadPoolExecutor.

    Почему Thread, а не Process:
      - cv2.morphologyEx, cv2.findContours и numpy освобождают GIL → истинный параллелизм
      - Данные не сериализуются: heatmaps доступны через shared memory потоков
      - ProcessPool на Windows: pickle ~250 МБ float32 данных на каждый воркер = 10-30 мин
      - ThreadPool: нулевой IPC overhead, весь выигрыш от параллелизма

    Разбиение на chunks: каждый поток обрабатывает свой срез комбинаций.
    """
    combos = _all_combos(thresholds, s_mins, s_maxs)
    chunk_size = max(1, math.ceil(len(combos) / workers))
    chunks = [combos[i: i + chunk_size] for i in range(0, len(combos), chunk_size)]

    def _thread_chunk(param_chunk):
        return [
            _eval_combo(thr, smin, smax, heatmaps, gt_masks)
            for thr, smin, smax in param_chunk
        ]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        nested = list(ex.map(_thread_chunk, chunks))
    elapsed = round((time.perf_counter() - t0) * 1_000, 3)

    results = [r for chunk_res in nested for r in chunk_res]
    return elapsed, results


# ---------------------------------------------------------------------------
# Режим 3 — GPU Sequential
# ---------------------------------------------------------------------------

def optimize_gpu_sequential(
    heatmaps:  List[np.ndarray],
    gt_masks:  List[Optional[np.ndarray]],
    thresholds: List[float],
    s_mins:    List[float],
    s_maxs:    List[float],
) -> Tuple[float, List[Dict]]:
    """GPU-морфология для каждой комбинации, перебор последовательный."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    results = [
        _eval_combo(thr, smin, smax, heatmaps, gt_masks, use_gpu=True, device=device)
        for thr, smin, smax in _all_combos(thresholds, s_mins, s_maxs)
    ]
    return round((time.perf_counter() - t0) * 1_000, 3), results


# ---------------------------------------------------------------------------
# Режим 4 — GPU Batch (все пороги для одного изображения за один GPU-проход)
# ---------------------------------------------------------------------------

def optimize_gpu_batch(
    heatmaps:  List[np.ndarray],
    gt_masks:  List[Optional[np.ndarray]],
    thresholds: List[float],
    s_mins:    List[float],
    s_maxs:    List[float],
) -> Tuple[float, List[Dict]]:
    """
    Для каждого изображения — ОДИН GPU-вызов на все M порогов одновременно.
    Тензор [M, 1, H, W] проходит морфологию за один pass.
    Затем CPU: перебор S_min/S_max на уже готовых бинарных масках.

    GPU calls: N_images (не N_images × M_thresholds как в GPU sequential).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = len(thresholds)
    k, pad = 3, 1
    thr_t = torch.tensor(thresholds, dtype=torch.float32)

    # Накопители метрик по каждой комбинации
    accum: Dict[Tuple, Dict] = {
        combo: dict(f1=0.0, iou=0.0, precision=0.0, recall=0.0, n=0)
        for combo in _all_combos(thresholds, s_mins, s_maxs)
    }

    t0 = time.perf_counter()

    for hm, gt in zip(heatmaps, gt_masks):
        h, w = hm.shape
        gt_eff = _prep_gt(gt, h, w)

        # [M, 1, H, W] — все M порогов параллельно на GPU
        hm_t  = torch.from_numpy(hm).float()
        batch = (hm_t.unsqueeze(0) > thr_t.view(M, 1, 1)).float().unsqueeze(1).to(device)

        eroded  = -F.max_pool2d(-batch,   k, 1, pad)
        opened  =  F.max_pool2d(eroded,   k, 1, pad)
        dilated =  F.max_pool2d(opened,   k, 1, pad)
        closed  = -F.max_pool2d(-dilated, k, 1, pad)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # [M, H, W] numpy
        masks_np = (closed.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)

        for i, thr in enumerate(thresholds):
            bin_255 = masks_np[i] * 255
            for smin in s_mins:
                for smax in s_maxs:
                    pred = _binary_to_pred_mask(bin_255, smin, smax)
                    m    = compute_pixel_metrics(pred, gt_eff)
                    a    = accum[(thr, smin, smax)]
                    a["f1"]        += m["f1"]
                    a["iou"]       += m["iou"]
                    a["precision"] += m["precision"]
                    a["recall"]    += m["recall"]
                    a["n"]         += 1

    elapsed = round((time.perf_counter() - t0) * 1_000, 3)

    results = []
    for (thr, smin, smax), a in accum.items():
        n = a["n"]
        if n == 0:
            continue
        results.append({
            "threshold": thr, "S_min": smin, "S_max": smax,
            "f1":        round(a["f1"]        / n, 4),
            "iou":       round(a["iou"]       / n, 4),
            "precision": round(a["precision"] / n, 4),
            "recall":    round(a["recall"]    / n, 4),
        })

    return elapsed, results


# ---------------------------------------------------------------------------
# Утилиты для результатов
# ---------------------------------------------------------------------------

def best_combo(results: List[Dict], metric: str = "f1") -> Dict:
    return max(results, key=lambda r: r[metric]) if results else {}


def top_n(results: List[Dict], metric: str = "f1", n: int = 10) -> List[Dict]:
    return sorted(results, key=lambda r: -r[metric])[:n]
