"""
Умная настройка параметров бинаризации PatchCore heatmap.

Функция потерь (штрафы производственного типа):
    loss = (FP * 10 + FN * 5) / (TP + TN + FP + FN)

    FP × 10 — ложная тревога: остановка конвейера, раздражение операторов
    FN × 5  — пропуск дефекта: уходит дальше по конвейеру

Параметры:
    threshold            — порог бинаризации heatmap [0..1]
    min_area             — минимальная площадь контура в пикселях
    morphology_kernel    — размер ядра морфологии (нечётное, 1 = выкл)
    morphology_iterations — количество итераций морфологии
    gaussian_blur_kernel — размытие heatmap перед порогом (0 = выкл)
"""

import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Сетка поиска (дефолтная)
# ---------------------------------------------------------------------------

DEFAULT_PARAM_GRID: Dict[str, List] = {
    "threshold":             [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "min_area":              [0, 10, 50, 100, 200, 500],
    "morphology_kernel":     [1, 3, 5, 7],
    "morphology_iterations": [1, 2, 3],
    "gaussian_blur_kernel":  [0, 3, 5, 7],
}
# Итого: 7 × 6 × 4 × 3 × 4 = 2016 комбинаций

DEFAULT_PARAMS: Dict = {
    "threshold": 0.5,
    "min_area": 0,
    "morphology_kernel": 3,
    "morphology_iterations": 1,
    "gaussian_blur_kernel": 0,
}

BEST_PARAMS_FILE = Path(__file__).parent / "best_params.json"


# ---------------------------------------------------------------------------
# Применение параметров к heatmap
# ---------------------------------------------------------------------------

def apply_parameters(heatmap: np.ndarray, params: Dict) -> np.ndarray:
    """
    Применяет пайплайн бинаризации к heatmap.

    Порядок:
        1. Gaussian blur (если gaussian_blur_kernel > 0)
        2. Threshold → binary 0/255
        3. Morphology open+close (если morphology_kernel > 1)
        4. Фильтрация контуров по min_area

    Returns:
        binary mask uint8: 0 = норма, 255 = дефект
    """
    h = heatmap.astype(np.float32)

    # 1. Gaussian blur
    blur_k = int(params.get("gaussian_blur_kernel", 0))
    if blur_k > 0:
        if blur_k % 2 == 0:
            blur_k += 1   # ядро должно быть нечётным
        h = cv2.GaussianBlur(h, (blur_k, blur_k), 0)

    # 2. Binarization
    binary = (h > float(params.get("threshold", 0.5))).astype(np.uint8) * 255

    # 3. Morphology
    mk = int(params.get("morphology_kernel", 1))
    mi = int(params.get("morphology_iterations", 1))
    if mk > 1:
        if mk % 2 == 0:
            mk += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=mi)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=mi)

    # 4. min_area filter
    min_area = float(params.get("min_area", 0))
    if min_area > 0:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(binary)
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(filtered, [c], -1, 255, cv2.FILLED)
        binary = filtered

    return binary


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def compute_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """
    Попиксельное сравнение predicted mask (0/255) с GT mask (0/1 или 0/255).

    Returns dict: TP, TN, FP, FN, precision, recall, f1, fpr, fnr
    """
    p = predicted.astype(bool)
    g = ground_truth.astype(bool)

    tp = int(np.logical_and( p,  g).sum())
    fp = int(np.logical_and( p, ~g).sum())
    fn = int(np.logical_and(~p,  g).sum())
    tn = int(np.logical_and(~p, ~g).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    fpr       = fp / (fp + tn) if fp + tn > 0 else 0.0
    fnr       = fn / (fn + tp) if fn + tp > 0 else 0.0

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "fpr":       round(fpr,       6),
        "fnr":       round(fnr,       4),
    }


def compute_loss(metrics: Dict) -> float:
    """
    loss = (FP * 10 + FN * 5) / (TP + TN + FP + FN)

    FP штрафуется в 2× сильнее FN:
    ложная тревога дискредитирует систему быстрее пропуска дефекта.
    """
    total = metrics["TP"] + metrics["TN"] + metrics["FP"] + metrics["FN"]
    if total == 0:
        return 1.0
    return round((metrics["FP"] * 10 + metrics["FN"] * 5) / total, 6)


def _aggregate_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    fpr       = fp / (fp + tn) if fp + tn > 0 else 0.0
    fnr       = fn / (fn + tp) if fn + tp > 0 else 0.0
    total     = tp + tn + fp + fn
    loss      = round((fp * 10 + fn * 5) / total, 6) if total > 0 else 1.0
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "fpr":       round(fpr,       6),
        "fnr":       round(fnr,       4),
        "loss":      loss,
    }


def _prep_gt(gt: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """GT маска нужного размера. None → нулевая (good-изображение)."""
    if gt is None:
        return np.zeros(shape, dtype=np.uint8)
    if gt.shape != shape:
        return cv2.resize(gt, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return gt.astype(np.uint8)


# ---------------------------------------------------------------------------
# Grid Search (оптимизированный)
# ---------------------------------------------------------------------------

def grid_search(
    heatmaps:  List[np.ndarray],
    gt_masks:  List[Optional[np.ndarray]],
    param_grid: Optional[Dict] = None,
    workers:   int = 4,
    progress_callback=None,
) -> Tuple[float, List[Dict]]:
    """
    Перебирает все комбинации параметров и возвращает список результатов,
    отсортированных по loss (от лучшего к худшему).

    Оптимизация скорости:
        Предварительно вычисляем binary base masks для каждой пары
        (gaussian_blur_kernel, threshold) — 4×7 = 28 уникальных вариантов.
        Это устраняет повторное применение blur+threshold для каждой из 2016 комбинаций.

    Returns:
        (elapsed_ms, sorted_results)
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    thresholds   = param_grid["threshold"]
    min_areas    = param_grid["min_area"]
    kernels      = param_grid["morphology_kernel"]
    iters_list   = param_grid["morphology_iterations"]
    blurs        = param_grid["gaussian_blur_kernel"]

    n_combos = len(thresholds) * len(min_areas) * len(kernels) * len(iters_list) * len(blurs)
    log.info("Grid search: %d комбинаций, %d изображений", n_combos, len(heatmaps))

    # Шаг 1: Предвычисляем (blur, threshold) → [binary per image]
    # Только 28 уникальных пар вместо 2016
    base_masks: Dict[Tuple, List[np.ndarray]] = {}
    for blur_k in blurs:
        for thr in thresholds:
            masks = []
            for hm in heatmaps:
                h = hm.astype(np.float32)
                if blur_k > 0:
                    bk = blur_k if blur_k % 2 == 1 else blur_k + 1
                    h = cv2.GaussianBlur(h, (bk, bk), 0)
                masks.append((h > thr).astype(np.uint8) * 255)
            base_masks[(blur_k, thr)] = masks

    # Шаг 2: Перебираем оставшиеся параметры параллельно
    all_combos = list(product(thresholds, min_areas, kernels, iters_list, blurs))

    gt_prepared = [_prep_gt(gt, heatmaps[i].shape) for i, gt in enumerate(gt_masks)]

    def _eval(combo: Tuple) -> Dict:
        thr, min_area, mk, mi, blur_k = combo
        base = base_masks[(blur_k, thr)]

        tp = tn = fp = fn = 0
        for binary, gt in zip(base, gt_prepared):
            # Морфология
            if mk > 1:
                mk_ = mk if mk % 2 == 1 else mk + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk_, mk_))
                morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=mi)
                morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel, iterations=mi)
            else:
                morphed = binary

            # Фильтр min_area
            if min_area > 0:
                contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pred_arr = np.zeros_like(morphed)
                for c in contours:
                    if cv2.contourArea(c) >= min_area:
                        cv2.drawContours(pred_arr, [c], -1, 255, cv2.FILLED)
                p_bool = pred_arr.astype(bool)
            else:
                p_bool = morphed.astype(bool)

            g_bool = gt.astype(bool)
            tp += int(np.logical_and( p_bool,  g_bool).sum())
            fp += int(np.logical_and( p_bool, ~g_bool).sum())
            fn += int(np.logical_and(~p_bool,  g_bool).sum())
            tn += int(np.logical_and(~p_bool, ~g_bool).sum())

        m = _aggregate_metrics(tp, tn, fp, fn)
        return {
            "params": {
                "threshold":             thr,
                "min_area":              min_area,
                "morphology_kernel":     mk,
                "morphology_iterations": mi,
                "gaussian_blur_kernel":  blur_k,
            },
            "metrics": {k: v for k, v in m.items() if k != "loss"},
            "loss":    m["loss"],
        }

    t0 = time.perf_counter()

    chunk_size = max(1, math.ceil(len(all_combos) / workers))
    chunks = [all_combos[i: i + chunk_size] for i in range(0, len(all_combos), chunk_size)]

    _done = [0]
    _lock = threading.Lock()

    def _eval_chunk(chunk):
        res = [_eval(c) for c in chunk]
        with _lock:
            _done[0] += len(chunk)
            if progress_callback is not None:
                try:
                    progress_callback(_done[0], n_combos)
                except Exception:
                    pass
        return res

    with ThreadPoolExecutor(max_workers=workers) as ex:
        nested = list(ex.map(_eval_chunk, chunks))

    results = [r for chunk_res in nested for r in chunk_res]
    elapsed_ms = round((time.perf_counter() - t0) * 1_000, 1)

    results.sort(key=lambda r: r["loss"])
    log.info("Grid search завершён: %.1f мс, лучший loss=%.6f", elapsed_ms, results[0]["loss"])
    return elapsed_ms, results


# ---------------------------------------------------------------------------
# GPU Grid Search
# ---------------------------------------------------------------------------

def gpu_grid_search(
    heatmaps:  List[np.ndarray],
    gt_masks:  List[Optional[np.ndarray]],
    param_grid: Optional[Dict] = None,
    device: str = "cuda",
    exact: bool = True,
    progress_callback=None,
) -> Tuple[float, List[Dict]]:
    """
    GPU-ускоренный grid search через PyTorch.

    exact=True  — min_area через cv2.findContours (точно, медленнее для >0)
    exact=False — аппроксимация морфологическим открытием (быстрее, погрешность ~5%)

    Fallback на CPU grid_search при: отсутствии PyTorch/CUDA, неоднородных размерах.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        log.warning("gpu_grid_search: PyTorch не установлен → CPU fallback")
        return grid_search(heatmaps, gt_masks, param_grid,
                           progress_callback=progress_callback)

    if not torch.cuda.is_available() and device == "cuda":
        log.warning("gpu_grid_search: CUDA недоступна → CPU fallback")
        return grid_search(heatmaps, gt_masks, param_grid,
                           progress_callback=progress_callback)

    if not heatmaps:
        return 0.0, []

    shape0 = heatmaps[0].shape
    if any(h.shape != shape0 for h in heatmaps):
        log.warning("gpu_grid_search: неоднородные размеры heatmap → CPU fallback")
        return grid_search(heatmaps, gt_masks, param_grid,
                           progress_callback=progress_callback)

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    thresholds   = param_grid["threshold"]
    min_areas    = param_grid["min_area"]
    kernels      = param_grid["morphology_kernel"]
    iters_list   = param_grid["morphology_iterations"]
    blurs        = param_grid["gaussian_blur_kernel"]
    N            = len(heatmaps)
    H, W         = shape0
    n_combos     = (len(thresholds) * len(min_areas) * len(kernels)
                    * len(iters_list) * len(blurs))

    log.info("GPU grid search: %d комбинаций, %d изображений, device=%s, exact=%s",
             n_combos, N, device, exact)

    try:
        dev = torch.device(device)
        t0  = time.perf_counter()

        hm_t = torch.tensor(
            np.stack(heatmaps).astype(np.float32), device=dev
        ).unsqueeze(1)  # [N,1,H,W]

        gt_prepared = [_prep_gt(gt, shape0) for gt in gt_masks]
        gt_t = torch.tensor(
            np.stack([g.astype(np.uint8) for g in gt_prepared]).astype(bool),
            device=dev,
        )  # [N,H,W] bool

        # ── GPU helpers ──────────────────────────────────────────────────────
        _gauss_cache: Dict[int, "torch.Tensor"] = {}

        def _gauss_kern(k: int) -> "torch.Tensor":
            if k not in _gauss_cache:
                sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
                x = torch.arange(k, dtype=torch.float32, device=dev) - k // 2
                g = torch.exp(-x ** 2 / (2 * sigma ** 2))
                g /= g.sum()
                _gauss_cache[k] = (g.unsqueeze(0) * g.unsqueeze(1)).view(1, 1, k, k)
            return _gauss_cache[k]

        def _blur_t(x: "torch.Tensor", k: int) -> "torch.Tensor":
            if k <= 0:
                return x
            bk = k | 1
            return F.conv2d(x, _gauss_kern(bk), padding=bk // 2)

        def _erode(x: "torch.Tensor", k: int) -> "torch.Tensor":
            return -F.max_pool2d(-x, k, stride=1, padding=k // 2)

        def _dilate(x: "torch.Tensor", k: int) -> "torch.Tensor":
            return F.max_pool2d(x, k, stride=1, padding=k // 2)

        def _open(x: "torch.Tensor", k: int, n: int) -> "torch.Tensor":
            for _ in range(n): x = _erode(x, k)
            for _ in range(n): x = _dilate(x, k)
            return x

        def _close(x: "torch.Tensor", k: int, n: int) -> "torch.Tensor":
            for _ in range(n): x = _dilate(x, k)
            for _ in range(n): x = _erode(x, k)
            return x

        # ── Precompute (blur, thr) → base binary [N,H,W] ────────────────────
        base_cache: Dict[Tuple, "torch.Tensor"] = {}
        for blur_k in blurs:
            blurred = _blur_t(hm_t, blur_k)
            for thr in thresholds:
                base_cache[(blur_k, thr)] = (blurred.squeeze(1) > thr)  # [N,H,W] bool

        # ── Main loop ────────────────────────────────────────────────────────
        results: List[Dict] = []
        done_cnt = [0]
        any_min_area = any(ma > 0 for ma in min_areas)

        for blur_k in blurs:
            for thr in thresholds:
                base_bool = base_cache[(blur_k, thr)]
                base_f    = base_bool.float().unsqueeze(1)  # [N,1,H,W]

                for mk in kernels:
                    for mi in iters_list:
                        if mk > 1:
                            mk_ = mk | 1
                            morphed_f = _open(base_f, mk_, mi)
                            morphed_f = _close(morphed_f, mk_, mi)
                        else:
                            morphed_f = base_f

                        morphed_bool = morphed_f.squeeze(1) > 0.5  # [N,H,W]

                        # Precompute CPU arrays once per (mk, mi) combo
                        if any_min_area and exact:
                            morphed_cpu = (morphed_bool.cpu().numpy().astype(np.uint8) * 255)
                            gt_cpu = gt_t.cpu().numpy().astype(bool)

                        for min_area in min_areas:
                            if min_area <= 0:
                                tp = int((morphed_bool &  gt_t).sum().item())
                                fp = int((morphed_bool & ~gt_t).sum().item())
                                fn = int((~morphed_bool &  gt_t).sum().item())
                                tn = int((~morphed_bool & ~gt_t).sum().item())

                            elif exact:
                                tp = tn = fp = fn = 0
                                for i in range(N):
                                    m_img = morphed_cpu[i]
                                    cnts, _ = cv2.findContours(
                                        m_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                                    )
                                    pred = np.zeros_like(m_img)
                                    for c in cnts:
                                        if cv2.contourArea(c) >= min_area:
                                            cv2.drawContours(pred, [c], -1, 255, cv2.FILLED)
                                    pb = pred.astype(bool)
                                    gb = gt_cpu[i]
                                    tp += int(( pb &  gb).sum())
                                    fp += int(( pb & ~gb).sum())
                                    fn += int((~pb &  gb).sum())
                                    tn += int((~pb & ~gb).sum())

                            else:
                                r_approx = math.sqrt(min_area / math.pi)
                                ak = max(3, int(2 * r_approx + 1)) | 1
                                ak = min(ak, min(H, W) - 2) | 1
                                flt = (_open(morphed_f, ak, 1).squeeze(1) > 0.5)
                                tp = int((flt &  gt_t).sum().item())
                                fp = int((flt & ~gt_t).sum().item())
                                fn = int((~flt &  gt_t).sum().item())
                                tn = int((~flt & ~gt_t).sum().item())

                            m = _aggregate_metrics(tp, tn, fp, fn)
                            results.append({
                                "params": {
                                    "threshold":             thr,
                                    "min_area":              min_area,
                                    "morphology_kernel":     mk,
                                    "morphology_iterations": mi,
                                    "gaussian_blur_kernel":  blur_k,
                                },
                                "metrics": {k: v for k, v in m.items() if k != "loss"},
                                "loss":    m["loss"],
                            })
                            done_cnt[0] += 1
                            if progress_callback is not None:
                                try:
                                    progress_callback(done_cnt[0], n_combos)
                                except Exception:
                                    pass

        elapsed_ms = round((time.perf_counter() - t0) * 1_000, 1)
        results.sort(key=lambda r: r["loss"])
        log.info("GPU grid search завершён: %.1f мс, лучший loss=%.6f",
                 elapsed_ms, results[0]["loss"] if results else 0)
        return elapsed_ms, results

    except Exception as e:
        log.warning("gpu_grid_search: ошибка (%s) → CPU fallback", e)
        return grid_search(heatmaps, gt_masks, param_grid,
                           progress_callback=progress_callback)


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def find_best_params(results: List[Dict]) -> Dict:
    """Первый элемент отсортированного списка — минимальный loss."""
    if not results:
        return {}
    return results[0]


def top_n_results(results: List[Dict], n: int = 10, worst: bool = False) -> List[Dict]:
    """Топ-N лучших (или худших) комбинаций."""
    if worst:
        return results[-n:][::-1]
    return results[:n]


def save_best_params(best: Dict, path: Optional[Path] = None) -> Path:
    """Сохраняет лучшие параметры в JSON файл."""
    if path is None:
        path = BEST_PARAMS_FILE
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        **best.get("params", {}),
        "loss":      best.get("loss"),
        "precision": best.get("metrics", {}).get("precision"),
        "recall":    best.get("metrics", {}).get("recall"),
        "f1":        best.get("metrics", {}).get("f1"),
        "fpr":       best.get("metrics", {}).get("fpr"),
        "fnr":       best.get("metrics", {}).get("fnr"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("best_params.json сохранён: %s (loss=%.6f)", path, best.get("loss", 0))
    return path


def load_best_params(path: Optional[Path] = None) -> Optional[Dict]:
    """Загружает best_params.json. Возвращает None если файл не найден."""
    if path is None:
        path = BEST_PARAMS_FILE
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "threshold":             float(data.get("threshold", 0.5)),
            "min_area":              float(data.get("min_area", 0)),
            "morphology_kernel":     int(data.get("morphology_kernel", 3)),
            "morphology_iterations": int(data.get("morphology_iterations", 1)),
            "gaussian_blur_kernel":  int(data.get("gaussian_blur_kernel", 0)),
        }
    except Exception as e:
        log.warning("Не удалось загрузить best_params.json: %s", e)
        return None
