# -*- coding: utf-8 -*-
"""
Live Heatmap Calibration — интерактивная настройка параметров детекции.

Архитектура кэширования:
  - Инференс PatchCore (тяжёлый, ~300 мс/img): ОДИН раз → кэш в памяти.
  - Нормализация + threshold + контуры (~5 мс/img): пересчитывается при каждом
    движении ползунков — параллельно для всех 6 эталонных изображений.

КРИТИЧЕСКОЕ ТРЕБОВАНИЕ — Zero False Positives:
  Конфигурация с ANY FP на good-изображениях абсолютно недопустима.
  В auto-calibration: первый же FP → fast-fail, комбинация отбраковывается.
  Результат помечается {"feasible": False, "score": -inf}.
"""
from __future__ import annotations

import base64
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

CALIB_CLASSES  = ["good", "color", "cut", "hole", "metal_contamination", "thread"]
DEFECT_CLASSES = ["color", "cut", "hole", "metal_contamination", "thread"]

DISPLAY_W = 224
DISPLAY_H = 224

BASELINE_PATH = Path(__file__).parent / "models" / "baseline_map.npy"

# ─── Module-level raw heatmap cache ───────────────────────────────────────────

_raw_cache: Dict[str, Dict] = {}
_raw_lock   = threading.Lock()


def _imread_safe(path: Path) -> Optional[np.ndarray]:
    """Unicode-safe imread (Windows Cyrillic paths)."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _b64_jpg(img: np.ndarray, quality: int = 80) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def clear_calib_cache() -> int:
    """Clear raw anomaly map cache. Returns number of cleared entries."""
    with _raw_lock:
        n = len(_raw_cache)
        _raw_cache.clear()
    return n


def _cache_key(mb_ver: str, img_path: Path) -> str:
    return f"{mb_ver}::{img_path.resolve()}"


# ─── Inference with cache ──────────────────────────────────────────────────────

def _get_or_run_inference(runner, img_path: Path, mb_ver: str) -> Optional[Dict]:
    """
    Return cached raw anomaly map or run PatchCore inference and cache result.
    Thread-safe (GPU inference is serialized by runner._lock).
    """
    key = _cache_key(mb_ver, img_path)
    with _raw_lock:
        cached = _raw_cache.get(key)
    if cached is not None:
        return cached

    img = _imread_safe(img_path)
    if img is None:
        log.warning("Cannot read: %s", img_path)
        return None

    try:
        heatmap, score, _patch_embs, _sp = runner.infer_with_embeddings(img)
        entry: Dict = {
            "heatmap":  heatmap.astype(np.float32),
            "orig_bgr": img,
            "score":    float(score),
        }
        with _raw_lock:
            _raw_cache[key] = entry
        return entry
    except Exception as exc:
        log.error("Inference failed for %s: %s", img_path, exc)
        return None


# ─── Baseline map (Stage A: pixel-wise noise floor subtraction) ───────────────

def compute_baseline_map(
    runner,
    good_paths: List[Path],
    mb_ver: str,
    percentile: float = 75.0,
) -> np.ndarray:
    """
    Pixel-wise p75 baseline from good-image raw heatmaps.
    Structural noise (same-location artifacts across frames) is captured here.
    Uses _raw_cache → no redundant GPU inference if images were already processed.
    Returns [H, W] float32.
    """
    maps = []
    for p in good_paths:
        raw = _get_or_run_inference(runner, p, mb_ver)
        if raw is not None:
            maps.append(raw["heatmap"])
    if not maps:
        raise ValueError("Нет good-кадров для baseline — inference вернул None для всех путей")
    stack = np.stack(maps, axis=0)          # [N, H, W]
    return np.percentile(stack, percentile, axis=0).astype(np.float32)


def save_baseline_map(baseline: np.ndarray, path: Path = BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), baseline)
    log.info("Baseline сохранён: %s shape=%s", path, list(baseline.shape))


def load_baseline_map(path: Path = BASELINE_PATH) -> Optional[np.ndarray]:
    """None если файла нет → Stage A автоматически отключена, пайплайн не меняется."""
    try:
        b = np.load(str(path)).astype(np.float32)
        log.info("Baseline загружен: %s shape=%s", path, list(b.shape))
        return b
    except Exception:
        return None


# ─── Core calibration result ───────────────────────────────────────────────────

def compute_calibrated_result(
    raw_map:          np.ndarray,
    orig_bgr:         np.ndarray,
    vmin_floor:       float = 0.0,
    gamma:            float = 1.0,
    binary_threshold: float = 20.0,
    s_min:            int   = 40,
    s_max:            int   = 1_500_000,
    use_edge_mask:    bool  = True,
    k_morph:          int   = 5,
    blur_kernel:      int   = 0,
    percentile_clip:  float = 95.0,
    return_arrays:             bool                 = False,
    # Stage A: pixel-wise baseline subtraction (structural noise floor removal)
    baseline_map:              Optional[np.ndarray] = None,
    baseline_alpha:            float                = 1.0,
    use_baseline_subtraction:  bool                 = False,
    # Stage B: soft component suppression (small contours weakened, not deleted)
    use_component_suppression: bool                 = False,
    suppression_factor:        float                = 0.3,
    min_component_area:        int                  = 0,
) -> Dict:
    """
    Fast post-processing of a cached raw anomaly map. ~5-8 ms per 900×900 image.

    Full pipeline:
      raw_map
        → GaussianBlur(blur_kernel)           # suppress speckle noise
        → percentile_clip([lo, hi])           # per-image adaptive normalization
        → edge_mask (zero-out border)         # suppress PatchCore boundary artifacts
        → vmin_floor clip                     # raise noise floor
        → gamma correction                    # boost anomaly peaks
        → ×100 scale                          # display range 0-100
        → binary_threshold                    # binary mask
        → s_min / s_max contour filter        # size gate

    Parameters:
        blur_kernel:     Gaussian kernel size (0 = off; must be odd: 3/5/7/9).
                         Suppresses high-frequency speckle before normalization.
        percentile_clip: Upper percentile for adaptive clipping (90–99).
                         p_hi = percentile_clip-th percentile (e.g., 95 → 95th).
                         p_lo = (100 - percentile_clip)-th percentile (e.g., 5th).
                         Values outside [p_lo, p_hi] are clipped to 0 / 1.
                         Per-image: not affected by dataset-wide score distribution.
        vmin_floor:      Additional floor on [0,1] after percentile clip (0.0–0.5).
        gamma:           Gamma exponent applied to floored map (>1 boosts peaks).
        binary_threshold: Decision boundary on 0-100 display scale.
        s_min / s_max:   Contour area gate (px²).
        return_arrays:   If True, attach binary_arr and display_arr (for IoU eval).
    """
    H, W = raw_map.shape

    # Step 1: Gaussian blur to suppress high-frequency speckle noise.
    # Applied before percentile computation so blur reduces outlier influence.
    work = raw_map.copy()
    if blur_kernel >= 3 and blur_kernel % 2 == 1:
        work = cv2.GaussianBlur(work, (blur_kernel, blur_kernel), 0)

    # Step 1.5: Pixel-wise baseline subtraction (Stage A).
    # Removes structural noise that repeats at fixed spatial locations across good frames.
    # clip(..., 0, None) preserves only positive excess above the learned noise floor.
    # No-op when use_baseline_subtraction=False or baseline_map is None → identical to current.
    if use_baseline_subtraction and baseline_map is not None:
        bm = baseline_map
        if bm.shape != work.shape:
            bm = cv2.resize(bm, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_LINEAR)
        work = np.clip(work - baseline_alpha * bm, 0.0, None)

    # Step 2: Per-image adaptive percentile clipping.
    # Replaces global min/max: each image sets its own white/black point.
    # Background noise → maps near 0; real anomalies → near 1.
    p_clip = float(np.clip(percentile_clip, 50.1, 99.9))
    p_lo   = float(np.percentile(work, 100.0 - p_clip))
    p_hi   = float(np.percentile(work, p_clip))
    norm   = np.clip((work - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)

    # Step 3: Suppress 3-patch border artifacts AFTER normalization.
    # (Applying mask before normalization would create artificial contrast on uniform images.)
    if use_edge_mask:
        patch_sz = max(1, H // 28)
        border   = patch_sz * 3
        norm[:border, :]  = 0.0
        norm[-border:, :] = 0.0
        norm[:, :border]  = 0.0
        norm[:, -border:] = 0.0

    # Step 4: Floor → gamma → 0-100 scale
    floored  = np.clip((norm - vmin_floor) / (1.0 - vmin_floor + 1e-8), 0.0, 1.0)
    contrast = floored ** (1.0 / max(gamma, 0.1))
    display  = contrast * 100.0

    anomaly_score = float(np.percentile(display, 95))

    # Binary mask
    binary = (display > binary_threshold).astype(np.uint8) * 255
    if k_morph > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Contour filtering
    raw_cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid  = [c for c in raw_cnts if s_min <= float(cv2.contourArea(c)) <= s_max]
    areas  = sorted([round(float(cv2.contourArea(c))) for c in valid], reverse=True)[:20]
    verdict = "defect" if valid else "ok"

    # Step 9.5: Soft component suppression (Stage B).
    # Contours with area in [s_min, min_component_area) are weakened by suppression_factor
    # and re-checked against binary_threshold. Those that still exceed the threshold survive;
    # those that don't are filtered out naturally (not hard-deleted).
    # No-op when use_component_suppression=False → identical to current.
    if use_component_suppression and valid:
        _min_comp = min_component_area if min_component_area > 0 else s_min
        if _min_comp > s_min:
            suppressed: List = []
            for c in valid:
                area = float(cv2.contourArea(c))
                if area < _min_comp:
                    mask_s = np.zeros(display.shape, dtype=np.uint8)
                    cv2.drawContours(mask_s, [c], -1, 1, cv2.FILLED)
                    if np.any(display[mask_s == 1] * suppression_factor > binary_threshold):
                        suppressed.append(c)
                    # else: falls out naturally — no explicit deletion
                else:
                    suppressed.append(c)
            valid   = suppressed
            areas   = sorted([round(float(cv2.contourArea(c))) for c in valid], reverse=True)[:20]
            verdict = "defect" if valid else "ok"

    # Visualizations resized to display dimensions
    dw, dh = DISPLAY_W, DISPLAY_H
    hm_color = cv2.applyColorMap((contrast * 255).astype(np.uint8), cv2.COLORMAP_JET)
    orig_sq  = cv2.resize(orig_bgr, (dw, dh))
    hm_sq    = cv2.resize(hm_color, (dw, dh))
    overlay  = cv2.addWeighted(orig_sq, 0.50, hm_sq, 0.50, 0)
    bin_sq   = cv2.resize(binary, (dw, dh), interpolation=cv2.INTER_NEAREST)
    bin_bgr  = cv2.cvtColor(bin_sq, cv2.COLOR_GRAY2BGR)

    if valid:
        sx = dw / W
        sy = dh / H
        scaled = [(c * np.array([[[sx, sy]]])).astype(np.int32) for c in valid]
        cv2.drawContours(overlay, scaled, -1, (0, 255, 50), 2)
        cv2.drawContours(bin_bgr, scaled, -1, (0, 220, 50), 2)

    result: Dict = {
        "original_b64":  _b64_jpg(orig_sq),
        "overlay_b64":   _b64_jpg(overlay),
        "binary_b64":    _b64_jpg(bin_bgr),
        "anomaly_score": round(anomaly_score, 2),
        "verdict":       verdict,
        "contour_count": len(valid),
        "contour_areas": areas,
        "display_stats": {
            "min": round(float(display.min()), 2),
            "max": round(float(display.max()), 2),
            "p50": round(float(np.percentile(display, 50)), 2),
            "p95": round(float(np.percentile(display, 95)), 2),
            "p99": round(float(np.percentile(display, 99)), 2),
        },
    }
    if return_arrays:
        result["binary_arr"]  = binary
        result["display_arr"] = display
    return result


# ─── Batch calibration (6 reference images) ───────────────────────────────────

def batch_calibrate(
    runner,
    ref_images:                Dict[str, Path],
    mb_ver:                    str,
    vmin_floor:                float,
    gamma:                     float,
    binary_threshold:          float,
    s_min:                     int,
    s_max:                     int,
    use_edge_mask:             bool,
    k_morph:                   int                 = 5,
    blur_kernel:               int                 = 0,
    percentile_clip:           float               = 95.0,
    baseline_map:              Optional[np.ndarray] = None,
    baseline_alpha:            float               = 1.0,
    use_baseline_subtraction:  bool                = False,
    use_component_suppression: bool                = False,
    suppression_factor:        float               = 0.3,
    min_component_area:        int                 = 0,
) -> Dict[str, Dict]:
    """
    Run calibrated visualization on all reference images.
    First call: runs inference (sequential, GPU-safe) and caches raw heatmaps.
    Subsequent calls (slider change): only recomputes normalization + threshold (~5 ms/img).
    """
    _noise_kw = dict(
        blur_kernel=blur_kernel, percentile_clip=percentile_clip,
        baseline_map=baseline_map, baseline_alpha=baseline_alpha,
        use_baseline_subtraction=use_baseline_subtraction,
        use_component_suppression=use_component_suppression,
        suppression_factor=suppression_factor, min_component_area=min_component_area,
    )
    results: Dict[str, Dict] = {}
    for cls_name, img_path in ref_images.items():
        raw = _get_or_run_inference(runner, img_path, mb_ver)
        if raw is None:
            results[cls_name] = {"error": f"Инференс не удался: {img_path.name}"}
            continue
        r = compute_calibrated_result(
            raw["heatmap"], raw["orig_bgr"],
            vmin_floor=vmin_floor,
            gamma=gamma,
            binary_threshold=binary_threshold,
            s_min=s_min,
            s_max=s_max,
            use_edge_mask=use_edge_mask,
            k_morph=k_morph,
            **_noise_kw,
        )
        r["class_name"] = cls_name
        r["image_name"] = img_path.name
        results[cls_name] = r
    return results


# ─── Reference image selection ────────────────────────────────────────────────

def pick_reference_images(
    data_dir: Path,
    n_calib:  int           = 5,
    gt_dir:   Optional[Path] = None,
    img_idx:  int            = 0,
) -> Tuple[Dict[str, Path], Dict[str, List[Dict]]]:
    """
    Select display reference images (1 per class) and calibration samples (N per class).

    Returns:
        display_refs:   {cls: Path}             — 1 image for the live 6-card preview
        calib_samples:  {cls: [{path, gt_mask}]} — images used for auto-calibration IoU
    """
    allowed = {".png", ".jpg", ".jpeg", ".bmp"}
    display_refs:  Dict[str, Path]       = {}
    calib_samples: Dict[str, List[Dict]] = {}

    for cls in CALIB_CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            continue
        imgs = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in allowed)
        if not imgs:
            continue

        disp = imgs[min(img_idx, len(imgs) - 1)]
        display_refs[cls] = disp

        calib_pool = [p for p in imgs if p != disp][:n_calib]
        samples: List[Dict] = []
        for img_path in calib_pool:
            sample: Dict = {"path": img_path, "class_name": cls}
            if cls != "good" and gt_dir is not None:
                stem = img_path.stem
                for pat in (f"{stem}_mask.png", f"{stem}.png"):
                    mask_path = gt_dir / cls / pat
                    if mask_path.exists():
                        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if m is not None:
                            sample["gt_mask"] = (m > 127).astype(np.uint8)
                        break
            samples.append(sample)
        calib_samples[cls] = samples

    return display_refs, calib_samples


def _load_raw_for_samples(runner, samples: List[Dict], mb_ver: str) -> List[Dict]:
    """Run inference for all samples and attach raw heatmaps to each entry."""
    enriched: List[Dict] = []
    for s in samples:
        raw = _get_or_run_inference(runner, s["path"], mb_ver)
        if raw is None:
            continue
        entry = dict(s)
        entry["heatmap"]  = raw["heatmap"]
        entry["orig_bgr"] = raw["orig_bgr"]
        enriched.append(entry)
    return enriched


# ─── IoU metric ───────────────────────────────────────────────────────────────

def compute_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    p = pred_bin > 0
    g = gt_bin   > 0
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or( p, g).sum())
    return inter / (union + 1e-8)


# ─── Parameter evaluation (hard FP=0 constraint) ──────────────────────────────

def _evaluate_params(
    defect_samples:            List[Dict],
    good_samples:              List[Dict],
    vmin_floor:                float,
    gamma:                     float,
    threshold:                 float,
    s_min:                     int,
    s_max:                     int,
    use_edge_mask:             bool                = True,
    k_morph:                   int                 = 5,
    blur_kernel:               int                 = 0,
    percentile_clip:           float               = 95.0,
    baseline_map:              Optional[np.ndarray] = None,
    baseline_alpha:            float               = 1.0,
    use_baseline_subtraction:  bool                = False,
    use_component_suppression: bool                = False,
    suppression_factor:        float               = 0.3,
    min_component_area:        int                 = 0,
) -> Dict:
    """
    Evaluate one parameter combination.

    HARD CONSTRAINT: first FP on any good sample → fast-fail, return -inf.
    Only if ALL good samples pass, compute IoU on defect samples.
    """
    _noise_kw = dict(
        blur_kernel=blur_kernel, percentile_clip=percentile_clip,
        baseline_map=baseline_map, baseline_alpha=baseline_alpha,
        use_baseline_subtraction=use_baseline_subtraction,
        use_component_suppression=use_component_suppression,
        suppression_factor=suppression_factor, min_component_area=min_component_area,
    )

    # Hard FP check — fail-fast on first FP
    for g in good_samples:
        r = compute_calibrated_result(
            g["heatmap"], g["orig_bgr"],
            vmin_floor, gamma, threshold, s_min, s_max, use_edge_mask, k_morph,
            **_noise_kw,
        )
        if r["verdict"] == "defect":
            return {
                "feasible":         False,
                "score":            float("-inf"),
                "fp_count":         1,
                "mean_iou":         0.0,
                "per_class_iou":    {},
                "detected_classes": [],
            }

    # Compute IoU on defect samples
    per_class_ious: Dict[str, List[float]] = {}
    for s in defect_samples:
        cls = s.get("class_name", "?")
        r   = compute_calibrated_result(
            s["heatmap"], s["orig_bgr"],
            vmin_floor, gamma, threshold, s_min, s_max, use_edge_mask, k_morph,
            return_arrays=True, **_noise_kw,
        )
        gt = s.get("gt_mask")
        if gt is not None:
            pred = r.get("binary_arr", np.zeros(s["heatmap"].shape, dtype=np.uint8))
            if pred.shape != gt.shape:
                gt = cv2.resize(gt.astype(np.uint8), (pred.shape[1], pred.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            iou = compute_iou(pred, gt)
        else:
            iou = 1.0 if r["verdict"] == "defect" else 0.0
        per_class_ious.setdefault(cls, []).append(iou)

    per_class_mean = {k: round(float(np.mean(v)), 4) for k, v in per_class_ious.items()}
    all_ious       = [v for vals in per_class_ious.values() for v in vals]
    mean_iou       = float(np.mean(all_ious)) if all_ious else 0.0
    detected       = [c for c, iou in per_class_mean.items() if iou > 0.0]

    return {
        "feasible":         True,
        "score":            mean_iou,
        "fp_count":         0,
        "mean_iou":         round(mean_iou, 4),
        "per_class_iou":    per_class_mean,
        "detected_classes": detected,
    }


# ─── Auto-calibration state ────────────────────────────────────────────────────

_auto_state: Dict[str, Any] = {
    "running":     False,
    "progress":    0,
    "total":       0,
    "status":      "idle",
    "result":      None,
    "log":         [],
    "best_params": None,
    "best_score":  float("-inf"),
}
_auto_lock = threading.Lock()


def get_auto_calib_status() -> Dict:
    with _auto_lock:
        return dict(_auto_state)


def run_auto_calibrate_grid(
    runner,
    data_dir:                  Path,
    gt_dir:                    Optional[Path],
    mb_ver:                    str,
    n_calib:                   int                  = 5,
    n_good:                    int                  = 10,
    vmin_values:               Optional[List[float]] = None,
    gamma_values:              Optional[List[float]] = None,
    thr_values:                Optional[List[float]] = None,
    s_min_values:              Optional[List[int]]   = None,
    s_max_values:              Optional[List[int]]   = None,
    blur_values:               Optional[List[int]]   = None,
    pct_clip_values:           Optional[List[float]] = None,
    use_edge_mask:             bool                 = True,
    k_morph:                   int                  = 5,
    # Phase 2: noise suppression parameters
    baseline_map:              Optional[np.ndarray] = None,
    baseline_alpha_values:     Optional[List[float]] = None,
    suppression_factor_values: Optional[List[float]] = None,
    min_comp_multipliers:      Optional[List[float]] = None,
) -> None:
    """
    Two-phase grid-search auto-calibration (daemon background thread).

    Phase 1: full 7-parameter grid (blur × pct × vmin × gamma × thr × s_min × s_max).
             Noise suppression disabled — finds the best base configuration.
    Phase 2: fixes Phase 1 best_params, searches 40 noise-suppression combinations
             (4 baseline options × 10 component-suppression options).
             Only runs if Phase 1 found at least one feasible configuration.
    """
    if vmin_values     is None: vmin_values     = [0.0, 0.05, 0.10, 0.15, 0.20]
    if gamma_values    is None: gamma_values    = [1.0, 1.5, 2.0, 2.5, 3.0]
    if thr_values      is None: thr_values      = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0]
    if s_min_values    is None: s_min_values    = [5, 20, 40]
    if s_max_values    is None: s_max_values    = [50_000, 200_000, 600_000, 1_000_000]
    if blur_values     is None: blur_values     = [0, 5, 7]
    if pct_clip_values is None: pct_clip_values = [90.0, 95.0, 99.0]
    if baseline_alpha_values     is None: baseline_alpha_values     = [0.5, 1.0, 1.5]
    if suppression_factor_values is None: suppression_factor_values = [0.2, 0.3, 0.5]
    if min_comp_multipliers      is None: min_comp_multipliers      = [2.0, 5.0, 10.0]

    total_p1 = (len(vmin_values) * len(gamma_values) * len(thr_values)
                * len(s_min_values) * len(s_max_values)
                * len(blur_values) * len(pct_clip_values))

    # Phase 2 grid: (1 baseline-off + N_alpha baseline-on) × (1 suppression-off + N_mult×N_factor suppression-on)
    _n_bs  = 1 + (len(baseline_alpha_values) if baseline_map is not None else 0)
    _n_cs  = 1 + len(min_comp_multipliers) * len(suppression_factor_values)
    total_p2 = _n_bs * _n_cs
    total    = total_p1 + total_p2

    with _auto_lock:
        _auto_state.update({
            "running": True, "progress": 0, "total": total,
            "status": "Загрузка эталонов...", "result": None, "log": [],
            "best_params": None, "best_score": float("-inf"),
        })

    def _worker():
        log_lines: List[str] = []
        try:
            # 1. Select reference images
            _display_refs, calib_samples = pick_reference_images(
                data_dir, n_calib=n_calib, gt_dir=gt_dir
            )

            # 2. Sequential inference (GPU safety — runner holds _lock)
            with _auto_lock:
                _auto_state["status"] = "PatchCore инференс (кэш)..."

            defect_loaded: List[Dict] = []
            good_loaded:   List[Dict] = []
            for cls_name, samples in calib_samples.items():
                loaded = _load_raw_for_samples(runner, samples, mb_ver)
                (good_loaded if cls_name == "good" else defect_loaded).extend(loaded)

            good_loaded = good_loaded[:max(n_good, 5)]

            n_def_cls = len({s["class_name"] for s in defect_loaded})
            log_lines += [
                f"Дефектные эталоны: {len(defect_loaded)} шт. ({n_def_cls} классов)",
                f"Good эталоны: {len(good_loaded)} шт.",
                f"Фаза 1: {total_p1} комбинаций | Фаза 2: {total_p2} комбинаций",
            ]
            with _auto_lock:
                _auto_state["status"] = "Фаза 1: grid search..."
                _auto_state["log"]    = log_lines[:]

            # ── PHASE 1: 7-parameter base grid (noise suppression always off) ──────
            best_score     = float("-inf")
            best_params_p1: Optional[Dict] = None
            best_eval:      Optional[Dict] = None
            done           = 0
            feasible_count = 0

            for blur in blur_values:
              for pct in pct_clip_values:
                for vmin in vmin_values:
                  for g in gamma_values:
                    for thr in thr_values:
                      for smin in s_min_values:
                        for smax in s_max_values:
                            ev = _evaluate_params(
                                defect_loaded, good_loaded,
                                vmin, g, thr, smin, smax,
                                use_edge_mask, k_morph,
                                blur_kernel=blur, percentile_clip=pct,
                            )
                            done += 1
                            if ev["feasible"]:
                                feasible_count += 1
                                if ev["score"] > best_score:
                                    best_score  = ev["score"]
                                    best_params_p1 = {
                                        "vmin_floor":       vmin,
                                        "gamma":            g,
                                        "binary_threshold": thr,
                                        "s_min":            smin,
                                        "s_max":            smax,
                                        "use_edge_mask":    use_edge_mask,
                                        "k_morph":          k_morph,
                                        "blur_kernel":      blur,
                                        "percentile_clip":  pct,
                                    }
                                    best_eval = ev
                            if done % 200 == 0:
                                with _auto_lock:
                                    _auto_state["progress"] = done
                                    _auto_state["status"] = (
                                        f"Фаза 1: {done}/{total_p1}, feasible: {feasible_count}"
                                    )

            with _auto_lock:
                _auto_state["progress"] = total_p1

            log_lines.append(
                f"Фаза 1 завершена: {feasible_count}/{total_p1} feasible, "
                f"лучший IoU: {best_score:.4f}" if best_score > float("-inf")
                else f"Фаза 1 завершена: feasible конфигураций не найдено."
            )
            with _auto_lock:
                _auto_state["log"] = log_lines[:]

            # ── PHASE 2: noise suppression grid (fixed base params from Phase 1) ───
            # Best params start as Phase 1 result with noise suppression defaults.
            best_params: Optional[Dict] = None
            if best_params_p1 is not None:
                best_params = {
                    **best_params_p1,
                    "use_baseline_subtraction":  False,
                    "baseline_alpha":             1.0,
                    "use_component_suppression": False,
                    "suppression_factor":         0.3,
                    "min_component_area":         0,
                }

            with _auto_lock:
                _auto_state["status"] = "Фаза 2: шумодав grid..."

            if best_params_p1 is not None:
                # Build Phase 2 option lists
                bs_options: List[Tuple] = [(False, 1.0)]
                if baseline_map is not None:
                    bs_options += [(True, alpha) for alpha in baseline_alpha_values]

                best_smin = best_params_p1["s_min"]
                cs_options: List[Tuple] = [(False, 0, 0.3)]
                for mult in min_comp_multipliers:
                    for sf in suppression_factor_values:
                        cs_options.append((True, int(best_smin * mult), sf))

                p2_feasible = 0
                for (use_bs, alpha) in bs_options:
                    for (use_cs, min_comp, sf) in cs_options:
                        ev2 = _evaluate_params(
                            defect_loaded, good_loaded,
                            best_params_p1["vmin_floor"],
                            best_params_p1["gamma"],
                            best_params_p1["binary_threshold"],
                            best_params_p1["s_min"],
                            best_params_p1["s_max"],
                            best_params_p1["use_edge_mask"],
                            best_params_p1["k_morph"],
                            best_params_p1["blur_kernel"],
                            best_params_p1["percentile_clip"],
                            baseline_map=baseline_map if use_bs else None,
                            baseline_alpha=alpha,
                            use_baseline_subtraction=use_bs,
                            use_component_suppression=use_cs,
                            suppression_factor=sf,
                            min_component_area=min_comp,
                        )
                        done += 1
                        if ev2["feasible"]:
                            p2_feasible += 1
                            feasible_count += 1
                            if ev2["score"] > best_score:
                                best_score = ev2["score"]
                                best_params = {
                                    **best_params_p1,
                                    "use_baseline_subtraction":  use_bs,
                                    "baseline_alpha":             alpha,
                                    "use_component_suppression": use_cs,
                                    "suppression_factor":         sf,
                                    "min_component_area":         min_comp,
                                }
                                best_eval = ev2
                        with _auto_lock:
                            _auto_state["progress"] = done
                            _auto_state["status"] = (
                                f"Фаза 2: {done - total_p1}/{total_p2}, "
                                f"шумодав feasible: {p2_feasible}"
                            )

                noise_improved = (
                    best_params is not None
                    and (best_params.get("use_baseline_subtraction")
                         or best_params.get("use_component_suppression"))
                )
                log_lines.append(
                    f"Фаза 2: {p2_feasible}/{total_p2} feasible. "
                    + ("Шумодав улучшил результат." if noise_improved
                       else "Шумодав не улучшил Phase-1 результат (base params оставлены).")
                )
            else:
                # Phase 1 found nothing → skip Phase 2, advance progress to total
                done = total
                with _auto_lock:
                    _auto_state["progress"] = total

            with _auto_lock:
                _auto_state["progress"] = total

            # ── Build final result ────────────────────────────────────────────────
            if feasible_count == 0 or best_eval is None or best_params is None:
                result: Dict = {
                    "success": False,
                    "error": (
                        "Не найдено конфигураций с FP=0% на good-эталонах. "
                        "Расширьте диапазоны параметров или добавьте больше "
                        "good-эталонов."
                    ),
                    "feasible_count": 0,
                    "total": total,
                }
                log_lines.append("Нет feasible конфигураций при ограничении FP=0.")
            else:
                all_cls    = {s["class_name"] for s in defect_loaded}
                detected   = set(best_eval.get("detected_classes", []))
                undetected = sorted(all_cls - detected)

                undetected_msgs = [
                    f"Класс «{uc}» не детектируется при FP=0% — "
                    "без ложных срабатываний этот класс неразличим."
                    for uc in undetected
                ]
                result = {
                    "success":             True,
                    "best_params":         best_params,
                    "mean_iou":            best_eval["mean_iou"],
                    "per_class_iou":       best_eval["per_class_iou"],
                    "fp_count":            0,
                    "feasible_count":      feasible_count,
                    "total":               total,
                    "undetected_classes":  undetected,
                    "undetected_messages": undetected_msgs,
                }
                log_lines.append(
                    f"Итог: {feasible_count}/{total} feasible, лучший mean IoU: {best_score:.4f}"
                )
                log_lines.extend(undetected_msgs)

            with _auto_lock:
                _auto_state.update({
                    "running":     False,
                    "status":      "done",
                    "result":      result,
                    "log":         log_lines,
                    "best_score":  best_score,
                    "best_params": best_params,
                })
        except Exception as exc:
            log.exception("Auto-calibration failed")
            with _auto_lock:
                _auto_state.update({
                    "running": False,
                    "status": "error",
                    "result": {"success": False, "error": str(exc)},
                })

    threading.Thread(target=_worker, daemon=True, name="auto_calib").start()
