"""
FastAPI — Parallel Defect Classification API.

Endpoints:
  POST /analyze             — одно изображение (3 режима)
  POST /analyze_dir         — директория (7 режимов)
  POST /analyze_all         — все 7 режимов на одном наборе + сравнение
  POST /calibrate           — калибровка image-level gate по good/ изображениям
  GET  /calibrate           — текущий image-level порог
  POST /optimize            — автоподбор threshold/S_min/S_max по GT маскам (1 режим)
  POST /optimize_all        — автоподбор во всех 4 режимах + сравнение скорости
  GET  /images              — список тест-изображений
  GET  /subfolders          — список поддиректорий test/
  GET  /history             — история прогонов
  DELETE /history           — очистить историю
  GET  /modes               — список режимов обработки
  GET  /ui                  — тестовый интерфейс
  GET  /original/{path}     — оригинальное изображение
"""

import base64
import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from defect_processor import (
    annotate_image,
    batch_gpu_pipeline,
    cpu_heatmap_pipeline,
    get_binarize_params,
    get_or_create_pool,
    parallel_cpu_process,
    parallel_gpu_process,
    reload_binarize_params,
    sequential_process,
    shutdown_all_pools,
)
from optimizer.parameter_optimizer import (
    DEFAULT_PARAM_GRID   as BINARIZE_PARAM_GRID,
    find_best_params     as binarize_find_best,
    gpu_grid_search      as binarize_gpu_grid_search,
    grid_search          as binarize_grid_search,
    save_best_params     as binarize_save_params,
    top_n_results        as binarize_top_n,
)
from history_manager import clear_history, get_all_runs, save_run
from metrics_evaluator import (
    best_combo,
    find_gt_mask,
    optimize_gpu_batch,
    optimize_gpu_sequential,
    optimize_parallel_cpu,
    optimize_sequential,
    top_n,
    _cpu_binary,
    _binary_to_pred_mask,
    compute_pixel_metrics,
    _prep_gt,
)
from patchcore_runner import PatchCoreRunner

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR.parent / "data" / "carpet" / "carpet" / "test"
TRAIN_DIR  = BASE_DIR.parent / "data" / "carpet" / "carpet" / "train"
GT_DIR     = BASE_DIR.parent / "data" / "carpet" / "carpet" / "ground_truth"
MODEL_PATH = BASE_DIR / "models" / "patchcore.pt"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

runner = PatchCoreRunner(str(MODEL_PATH), device="cuda")

# ---------------------------------------------------------------------------
# Режимы обработки
# ---------------------------------------------------------------------------

# Режимы, которые используют ThreadPoolExecutor на уровне изображений
_IMAGE_PARALLEL_MODES = frozenset({
    "parallel_images",
    "parallel_images_gpu",
    "parallel_combined",
})

# Для image-parallel режимов: какой per-image режим дефектов использовать
_PER_IMAGE_DEFECT_MODE = {
    "parallel_images":      "sequential",
    "parallel_images_gpu":  "parallel_gpu",
    "parallel_combined":    "parallel_cpu",
}

ALL_MODES: List[str] = [
    "sequential",           # последовательно всё
    "parallel_cpu",         # параллельные дефекты (ProcessPool), seq изображения
    "parallel_images",      # параллельные изображения (ThreadPool), seq дефекты
    "parallel_gpu",         # seq изображения, GPU морфология
    "parallel_images_gpu",  # ThreadPool изображений + GPU морфология
    "parallel_combined",    # ThreadPool изображений + ProcessPool дефектов
    "parallel_combined_gpu",# batch GPU: все heatmap за один GPU-проход
]

MODE_LABELS = {
    "sequential":            "Последовательный",
    "parallel_cpu":          "Параллельные дефекты (CPU)",
    "parallel_images":       "Параллельные изображения (CPU)",
    "parallel_gpu":          "Параллельные дефекты (GPU)",
    "parallel_images_gpu":   "Параллельные изображения (GPU)",
    "parallel_combined":     "Комбинированный CPU (фото+дефекты)",
    "parallel_combined_gpu": "Batch GPU (все фото за раз)",
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

POOL_SIZES = [2, 4]


def _startup() -> None:
    params = reload_binarize_params()
    logger.info("Параметры бинаризации: %s", params)
    for n in POOL_SIZES:
        try:
            logger.info("Прогрев пула workers=%d ...", n)
            get_or_create_pool(n)
            logger.info("Пул workers=%d готов.", n)
        except Exception as e:
            logger.warning("Не удалось создать пул workers=%d: %s", n, e)
    logger.info("Все пулы готовы.")


app = FastAPI(
    title="Defect Classifier API",
    description="Parallel defect classification: PatchCore + CPU/GPU",
    version="2.0.0",
    on_startup=[_startup],
    on_shutdown=[shutdown_all_pools],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Все необработанные исключения → JSON с деталями вместо plain-text 500."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


# ---------------------------------------------------------------------------
# Схемы запросов
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    image_path: str = Field(..., description="Путь относительно test/")
    mode:      Literal["sequential", "parallel_cpu", "parallel_gpu"] = "sequential"
    workers:   int   = Field(default=4,    ge=1,  le=16)
    S_min:     float = Field(default=30.0, ge=0)
    S_max:     float = Field(default=500.0, gt=0)
    K:         int   = Field(default=3,    ge=1,  le=10)
    threshold: float = Field(default=0.6,  ge=0.0, le=1.0)


class AnalyzeDirRequest(BaseModel):
    subfolder:  str   = Field(default="", description="Поддиректория или '' для всех")
    mode:       Literal[
        "sequential",
        "parallel_cpu",
        "parallel_images",
        "parallel_gpu",
        "parallel_images_gpu",
        "parallel_combined",
        "parallel_combined_gpu",
    ] = "sequential"
    workers:    int   = Field(default=4,   ge=1,  le=16)
    S_min:      float = Field(default=30.0, ge=0)
    S_max:      float = Field(default=500.0, gt=0)
    K:          int   = Field(default=3,   ge=1,  le=10)
    threshold:  float = Field(default=0.6,  ge=0.0, le=1.0)
    max_images: int   = Field(default=50,  ge=1,  le=500)


class AnalyzeAllRequest(BaseModel):
    subfolder:  str   = Field(default="")
    workers:    int   = Field(default=4,   ge=1,  le=16)
    S_min:      float = Field(default=30.0, ge=0)
    S_max:      float = Field(default=500.0, gt=0)
    K:          int   = Field(default=3,   ge=1,  le=10)
    threshold:  float = Field(default=0.6,  ge=0.0, le=1.0)
    max_images: int   = Field(default=20,  ge=1,  le=200)


class OptimizeRequest(BaseModel):
    subfolder:  str  = Field(default="")
    mode:       Literal[
        "sequential", "parallel_cpu", "parallel_gpu", "parallel_gpu_batch"
    ] = "sequential"
    workers:    int  = Field(default=4,  ge=1, le=16)
    max_images: int  = Field(default=20, ge=1, le=200)
    K:          int  = Field(default=3,  ge=1, le=10)
    thresholds: List[float] = Field(default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    s_mins:     List[float] = Field(default=[0,5,10,20,30])
    s_maxs:     List[float] = Field(default=[100,200,300,500])


class OptimizeAllRequest(BaseModel):
    subfolder:  str  = Field(default="")
    workers:    int  = Field(default=4,  ge=1, le=16)
    max_images: int  = Field(default=20, ge=1, le=100)
    K:          int  = Field(default=3,  ge=1, le=10)
    thresholds: List[float] = Field(default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    s_mins:     List[float] = Field(default=[0,5,10,20,30])
    s_maxs:     List[float] = Field(default=[100,200,300,500])


class BinarizeOptimizeRequest(BaseModel):
    subfolder:              str            = Field(default="")
    max_images:             int            = Field(default=50,  ge=1, le=200)
    workers:                int            = Field(default=4,   ge=1, le=16)
    use_gpu:                bool           = Field(default=False)
    gpu_exact:              bool           = Field(default=True)
    thresholds:             Optional[List[float]] = None
    min_areas:              Optional[List[int]]   = None
    morphology_kernels:     Optional[List[int]]   = None
    morphology_iterations:  Optional[List[int]]   = None
    gaussian_blur_kernels:  Optional[List[int]]   = None


class BinarizeParamsRequest(BaseModel):
    threshold:             float = Field(default=0.5,  ge=0.0, le=1.0)
    min_area:              float = Field(default=0,    ge=0)
    morphology_kernel:     int   = Field(default=3,    ge=1)
    morphology_iterations: int   = Field(default=1,    ge=1)
    gaussian_blur_kernel:  int   = Field(default=0,    ge=0)


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _read_image(path: Path) -> Optional[np.ndarray]:
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _amdahl_parallel_fraction(speedup: float, workers: int) -> Optional[float]:
    if workers <= 1 or speedup <= 1.0:
        return None
    try:
        p = workers * (speedup - 1.0) / (speedup * (workers - 1))
        return round(max(0.0, min(1.0, p)), 4)
    except ZeroDivisionError:
        return None


def _process_one(image_path: str, image: np.ndarray, req_mode: str,
                 req_workers: int, req_S_min: float, req_S_max: float,
                 req_K: int, req_threshold: float) -> dict:
    """Пайплайн для одного изображения (режимы: sequential / parallel_cpu / parallel_gpu)."""
    heatmap, anomaly_score = runner.infer_with_score(image)

    if req_mode == "sequential":
        _, contours, _ = cpu_heatmap_pipeline(
            heatmap, req_S_min, req_S_max, req_K, req_threshold
        )
        result = sequential_process(contours, req_S_min, req_S_max, req_K)
        ref_time_ms = result["time_ms"]

    elif req_mode == "parallel_cpu":
        _, contours, _ = cpu_heatmap_pipeline(
            heatmap, req_S_min, req_S_max, req_K, req_threshold
        )
        seq = sequential_process(contours, req_S_min, req_S_max, req_K)
        ref_time_ms = seq["time_ms"]
        pool = get_or_create_pool(req_workers)
        result = parallel_cpu_process(
            contours, req_S_min, req_S_max, req_K,
            workers=req_workers,
            sequential_time_ms=ref_time_ms,
            pool=pool,
        )

    else:  # parallel_gpu
        cpu_time_ms, contours, _ = cpu_heatmap_pipeline(
            heatmap, req_S_min, req_S_max, req_K, req_threshold
        )
        ref_time_ms = cpu_time_ms
        result = parallel_gpu_process(
            heatmap, req_S_min, req_S_max, req_K,
            threshold=req_threshold,
            cpu_pipeline_time_ms=cpu_time_ms,
        )

    speedup      = result.get("speedup") or 1.0
    workers_used = req_workers if req_mode == "parallel_cpu" else None
    p_fraction   = _amdahl_parallel_fraction(speedup, workers_used or 1)

    return {
        "timestamp":              datetime.now().isoformat(timespec="seconds"),
        "image":                  image_path,
        "mode":                   req_mode,
        "workers":                workers_used,
        "n_defects_found":        result["n_defects_found"],
        "n_defects_after_filter": result["n_defects_after_filter"],
        "S_min":                  req_S_min,
        "S_max":                  req_S_max,
        "K":                      req_K,
        "threshold":              req_threshold,
        "time_ms":                result["time_ms"],
        "ref_time_ms":            ref_time_ms,
        "speedup":                speedup,
        "parallel_fraction":      p_fraction,
        "device":                 result.get("device"),
        "gpu_time_ms":            result.get("gpu_time_ms"),
        "defects":                result["defects"],
        "_contours":              contours,
        "_heatmap":               heatmap,
        "_image":                 image,
        "anomaly_score":          anomaly_score,
        "classified_as":          "defective" if result["n_defects_after_filter"] > 0 else "normal",
    }


def _process_image_task(args) -> Optional[dict]:
    """
    Вызывается из ThreadPoolExecutor для image-parallel режимов.
    Возвращает entry-dict или None если изображение нечитаемо.
    """
    img_file, mode, workers, S_min, S_max, K, threshold = args
    rel_path = str(img_file.relative_to(DATA_DIR)).replace("\\", "/")
    image = _read_image(img_file)
    if image is None:
        logger.warning("Не удалось прочитать: %s — пропускаем", rel_path)
        return None
    defect_mode = _PER_IMAGE_DEFECT_MODE[mode]
    return _process_one(rel_path, image, defect_mode, workers, S_min, S_max, K, threshold)


def _annotate_and_encode(entry: dict, K: int) -> str:
    annotated = annotate_image(
        entry["_image"].copy(), entry["_contours"], entry["defects"], K
    )
    annotated_small = cv2.resize(annotated, (224, 224))
    _, buf = cv2.imencode(".jpg", annotated_small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _save_history(entry: dict) -> None:
    save_run({k: v for k, v in entry.items() if not k.startswith("_")})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    img_file = DATA_DIR / req.image_path
    if not img_file.exists():
        raise HTTPException(404, f"Файл не найден: {req.image_path}")
    if img_file.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Неподдерживаемый формат.")

    image = _read_image(img_file)
    if image is None:
        raise HTTPException(400, "Не удалось декодировать изображение.")

    entry = _process_one(
        req.image_path, image,
        req.mode, req.workers, req.S_min, req.S_max, req.K, req.threshold,
    )

    annotated = annotate_image(
        entry["_image"].copy(), entry["_contours"], entry["defects"], req.K
    )
    _, buf = cv2.imencode(".png", annotated)
    annotated_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    _save_history(entry)

    return {
        "defects":                entry["defects"],
        "n_defects_found":        entry["n_defects_found"],
        "n_defects_after_filter": entry["n_defects_after_filter"],
        "time_ms":                entry["time_ms"],
        "ref_time_ms":            entry["ref_time_ms"],
        "speedup":                entry["speedup"],
        "parallel_fraction":      entry["parallel_fraction"],
        "device":                 entry.get("device"),
        "annotated_image":        annotated_b64,
    }


@app.post("/analyze_dir")
def analyze_dir(req: AnalyzeDirRequest):
    def _generate():
        try:
            search_root = DATA_DIR / req.subfolder if req.subfolder else DATA_DIR
            if not search_root.exists():
                yield json.dumps({"type": "error", "detail": f"Поддиректория не найдена: {req.subfolder}"}) + "\n"
                return

            all_files: List[Path] = sorted(
                f for f in search_root.rglob("*")
                if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
            )
            if not all_files:
                yield json.dumps({"type": "error", "detail": "Изображения не найдены."}) + "\n"
                return

            files   = all_files[: req.max_images]
            skipped = len(all_files) - len(files)
            logger.info("analyze_dir stream: %d/%d, mode=%s", len(files), len(all_files), req.mode)

            yield json.dumps({"type": "start", "total": len(files), "mode": req.mode}) + "\n"

            total_found = total_filtered = n_processed = 0
            total_time_ms = 0.0
            wall_ms = 0.0

            def _pack(entry: dict) -> dict:
                return {
                    "image":              entry["image"],
                    "n_defects_found":    entry["n_defects_found"],
                    "n_defects_filtered": entry["n_defects_after_filter"],
                    "time_ms":            entry["time_ms"],
                    "speedup":            entry.get("speedup"),
                    "annotated_image":    _annotate_and_encode(entry, req.K),
                    "defects":            entry["defects"],
                    "classified_as":      "defective" if entry["n_defects_after_filter"] > 0 else "normal",
                }

            # ── Batch GPU ─────────────────────────────────────────────────
            if req.mode == "parallel_combined_gpu":
                t_wall = time.perf_counter()
                images_data = []
                for img_file in files:
                    rel = str(img_file.relative_to(DATA_DIR)).replace("\\", "/")
                    img = _read_image(img_file)
                    if img is None:
                        continue
                    hm = runner.infer(img)
                    images_data.append((img_file, rel, img, hm))

                if not images_data:
                    yield json.dumps({"type": "error", "detail": "Нет изображений для обработки."}) + "\n"
                    return

                heatmaps_list = [d[3] for d in images_data]
                batch_ms, gpu_ms, all_contours, all_defects = batch_gpu_pipeline(
                    heatmaps_list, req.S_min, req.S_max, req.K, req.threshold
                )
                wall_ms = round((time.perf_counter() - t_wall) * 1_000, 2)
                seq_total_ms = sum(
                    cpu_heatmap_pipeline(hm, req.S_min, req.S_max, req.K, req.threshold)[0]
                    for hm in heatmaps_list
                )
                speedup    = round(seq_total_ms / batch_ms, 3) if batch_ms > 0 else None
                per_img_ms = round(batch_ms / len(images_data), 3) if images_data else 0.0

                for i, (img_file, rel, img, _) in enumerate(images_data):
                    contours = all_contours[i]
                    defects  = all_defects[i]
                    total_found    += len(contours)
                    total_filtered += len(defects)
                    total_time_ms  += per_img_ms
                    ann = annotate_image(img.copy(), contours, defects, req.K)
                    ann_s = cv2.resize(ann, (224, 224))
                    _, buf = cv2.imencode(".jpg", ann_s, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    r = {
                        "image":              rel,
                        "n_defects_found":    len(contours),
                        "n_defects_filtered": len(defects),
                        "time_ms":            per_img_ms,
                        "speedup":            speedup,
                        "annotated_image":    base64.b64encode(buf.tobytes()).decode("ascii"),
                        "defects":            defects,
                        "classified_as":      "defective" if defects else "normal",
                    }
                    save_run({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "image": rel, "mode": req.mode, "workers": None,
                        "n_defects_found": len(contours), "n_defects_after_filter": len(defects),
                        "S_min": req.S_min, "S_max": req.S_max, "K": req.K, "threshold": req.threshold,
                        "time_ms": per_img_ms,
                        "ref_time_ms": round(seq_total_ms / len(images_data), 3),
                        "speedup": speedup, "gpu_time_ms": round(gpu_ms / len(images_data), 3),
                    })
                    n_processed += 1
                    yield json.dumps({"type": "image", "result": r}) + "\n"

            # ── Image-parallel: as_completed → progressive stream ─────────
            elif req.mode in _IMAGE_PARALLEL_MODES:
                t_wall = time.perf_counter()
                args_list = [
                    (f, req.mode, req.workers, req.S_min, req.S_max, req.K, req.threshold)
                    for f in files
                ]
                with ThreadPoolExecutor(max_workers=req.workers) as ex:
                    future_map = {ex.submit(_process_image_task, a): a for a in args_list}
                    for future in as_completed(future_map):
                        entry = future.result()
                        if entry is None:
                            continue
                        total_found    += entry["n_defects_found"]
                        total_filtered += entry["n_defects_after_filter"]
                        total_time_ms  += entry["time_ms"]
                        n_processed    += 1
                        r = _pack(entry)
                        _save_history(entry)
                        yield json.dumps({"type": "image", "result": r}) + "\n"
                wall_ms = round((time.perf_counter() - t_wall) * 1_000, 2)

            # ── Sequential / parallel_cpu / parallel_gpu ──────────────────
            else:
                t_wall = time.perf_counter()
                for img_file in files:
                    rel = str(img_file.relative_to(DATA_DIR)).replace("\\", "/")
                    img = _read_image(img_file)
                    if img is None:
                        logger.warning("Пропуск: %s", rel)
                        continue
                    entry = _process_one(
                        rel, img, req.mode, req.workers,
                        req.S_min, req.S_max, req.K, req.threshold,
                    )
                    total_found    += entry["n_defects_found"]
                    total_filtered += entry["n_defects_after_filter"]
                    total_time_ms  += entry["time_ms"]
                    n_processed    += 1
                    r = _pack(entry)
                    _save_history(entry)
                    yield json.dumps({"type": "image", "result": r}) + "\n"
                wall_ms = round((time.perf_counter() - t_wall) * 1_000, 2)

            summary = {
                "subfolder":              req.subfolder or "все",
                "mode":                   req.mode,
                "total_images":           n_processed,
                "skipped_limit":          skipped,
                "total_defects_found":    total_found,
                "total_defects_filtered": total_filtered,
                "avg_time_ms":            round(total_time_ms / n_processed, 2) if n_processed else 0,
                "total_time_ms":          round(total_time_ms, 2),
                "wall_time_ms":           wall_ms,
            }
            yield json.dumps({"type": "done", "summary": summary}) + "\n"

        except Exception as e:
            logger.exception("analyze_dir stream error")
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


@app.post("/analyze_all")
def analyze_all(req: AnalyzeAllRequest):
    def _generate():
        try:
            search_root = DATA_DIR / req.subfolder if req.subfolder else DATA_DIR
            if not search_root.exists():
                yield json.dumps({"type": "error", "detail": f"Поддиректория не найдена: {req.subfolder}"}) + "\n"
                return
            all_files: List[Path] = sorted(
                f for f in search_root.rglob("*")
                if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
            )
            if not all_files:
                yield json.dumps({"type": "error", "detail": "Изображения не найдены."}) + "\n"
                return
            files = all_files[: req.max_images]
            logger.info("analyze_all stream: %d изображений, %d режимов", len(files), len(ALL_MODES))
            yield json.dumps({"type": "start", "n_images": len(files), "n_modes": len(ALL_MODES)}) + "\n"

            # ── Pre-computation: GT labels + pixel metrics ────────────────
            image_meta = []
            for img_file in files:
                rel = str(img_file.relative_to(DATA_DIR)).replace("\\", "/")
                img = _read_image(img_file)
                if img is None:
                    continue
                hm = runner.infer(img)
                gt_mask = find_gt_mask(rel, GT_DIR)
                h, w = hm.shape
                gt_eff = _prep_gt(gt_mask, h, w)
                binary = _cpu_binary(hm, req.threshold)
                pred_mask = _binary_to_pred_mask(binary, req.S_min, req.S_max)
                px = compute_pixel_metrics(pred_mask, gt_eff)
                image_meta.append({
                    "gt_label":  1 if gt_mask is not None else 0,
                    "pred_label": 1 if pred_mask.any() else 0,
                    "iou":       px["iou"],
                    "pixel_f1":  px["f1"],
                })

            n_meta = len(image_meta)
            if n_meta > 0:
                gt_defective = sum(m["gt_label"] for m in image_meta)
                tp = sum(1 for m in image_meta if m["gt_label"] == 1 and m["pred_label"] == 1)
                tn = sum(1 for m in image_meta if m["gt_label"] == 0 and m["pred_label"] == 0)
                fp = sum(1 for m in image_meta if m["gt_label"] == 0 and m["pred_label"] == 1)
                fn = sum(1 for m in image_meta if m["gt_label"] == 1 and m["pred_label"] == 0)
                acc  = round((tp + tn) / n_meta, 4)
                prec = round(tp / (tp + fp), 4) if tp + fp > 0 else 0.0
                rec  = round(tp / (tp + fn), 4) if tp + fn > 0 else 0.0
                f1_  = round(2 * prec * rec / (prec + rec), 4) if prec + rec > 0 else 0.0
                mean_iou      = round(sum(m["iou"]      for m in image_meta) / n_meta, 4)
                mean_pixel_f1 = round(sum(m["pixel_f1"] for m in image_meta) / n_meta, 4)
                detection_quality: Optional[dict] = {
                    "gt_defective": gt_defective, "gt_normal": n_meta - gt_defective,
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                    "accuracy": acc, "precision": prec, "recall": rec, "f1": f1_,
                    "mean_pixel_iou": mean_iou, "mean_pixel_f1": mean_pixel_f1,
                    "n_evaluated": n_meta, "threshold_used": req.threshold,
                }
            else:
                detection_quality = None

            yield json.dumps({"type": "detection_quality", "data": detection_quality}) + "\n"

            mode_stats = []
            seq_wall_ms: Optional[float] = None

            for mode in ALL_MODES:
                t_wall = time.perf_counter()
                total_found = total_filtered = n_proc = 0
                total_time_ms = 0.0

                if mode == "parallel_combined_gpu":
                    images_data = []
                    for img_file in files:
                        img = _read_image(img_file)
                        if img is None:
                            continue
                        hm = runner.infer(img)
                        images_data.append((img, hm))
                    if images_data:
                        hm_list = [d[1] for d in images_data]
                        batch_ms, _, all_c, all_d = batch_gpu_pipeline(
                            hm_list, req.S_min, req.S_max, req.K, req.threshold
                        )
                        for i in range(len(images_data)):
                            total_found    += len(all_c[i])
                            total_filtered += len(all_d[i])
                            total_time_ms  += batch_ms / len(images_data)
                            n_proc         += 1

                elif mode in _IMAGE_PARALLEL_MODES:
                    args_list = [
                        (f, mode, req.workers, req.S_min, req.S_max, req.K, req.threshold)
                        for f in files
                    ]
                    with ThreadPoolExecutor(max_workers=req.workers) as ex:
                        entries = list(ex.map(_process_image_task, args_list))
                    for entry in entries:
                        if entry is None:
                            continue
                        total_found    += entry["n_defects_found"]
                        total_filtered += entry["n_defects_after_filter"]
                        total_time_ms  += entry["time_ms"]
                        n_proc         += 1

                else:
                    for img_file in files:
                        img = _read_image(img_file)
                        if img is None:
                            continue
                        rel = str(img_file.relative_to(DATA_DIR)).replace("\\", "/")
                        entry = _process_one(
                            rel, img, mode, req.workers,
                            req.S_min, req.S_max, req.K, req.threshold,
                        )
                        total_found    += entry["n_defects_found"]
                        total_filtered += entry["n_defects_after_filter"]
                        total_time_ms  += entry["time_ms"]
                        n_proc         += 1

                wall_ms = round((time.perf_counter() - t_wall) * 1_000, 2)
                if mode == "sequential":
                    seq_wall_ms = wall_ms
                speedup_vs_seq = (
                    round(seq_wall_ms / wall_ms, 3)
                    if seq_wall_ms and wall_ms > 0 and mode != "sequential"
                    else 1.0
                )
                stat = {
                    "mode": mode, "label": MODE_LABELS[mode],
                    "n_images": n_proc,
                    "total_defects_found": total_found,
                    "total_defects_filtered": total_filtered,
                    "avg_time_ms": round(total_time_ms / n_proc, 3) if n_proc else 0,
                    "total_time_ms": round(total_time_ms, 2),
                    "wall_time_ms": wall_ms,
                    "speedup_vs_sequential": speedup_vs_seq,
                }
                mode_stats.append(stat)
                logger.info("analyze_all [%s]: wall=%.1f мс", mode, wall_ms)
                yield json.dumps({"type": "mode", "data": stat}) + "\n"

            yield json.dumps({
                "type": "done",
                "subfolder": req.subfolder or "все",
                "n_images": len(files),
                "workers": req.workers,
                "modes": mode_stats,
                "detection_quality": detection_quality,
            }) + "\n"

        except Exception as e:
            logger.exception("analyze_all stream error")
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


@app.get("/images")
def get_images():
    if not DATA_DIR.exists():
        return {"images": []}
    images = sorted(
        str(f.relative_to(DATA_DIR)).replace("\\", "/")
        for f in DATA_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    )
    return {"images": images}


@app.get("/subfolders")
def get_subfolders():
    if not DATA_DIR.exists():
        return {"subfolders": []}
    folders = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    return {"subfolders": folders}


@app.get("/history")
def get_history():
    return {"runs": get_all_runs()}


@app.delete("/history")
def delete_history():
    clear_history()
    return {"message": "История очищена"}


class CalibrateRequest(BaseModel):
    source: Literal["test_good", "train_good", "both"] = Field(
        default="test_good",
        description=(
            "test_good — честная калибровка (РЕКОМЕНДУЕТСЯ): test/good/ никогда не видел PatchCore. "
            "train_good — НЕчестная: train/good/ использовался для построения memory bank, "
            "расстояния будут искусственно занижены. "
            "both — оба набора вместе (компромисс, больше данных)."
        ),
    )
    percentile: float = Field(default=95.0, ge=50.0, le=100.0)


@app.post("/calibrate")
def calibrate_image_threshold(req: CalibrateRequest):
    """
    Калибрует image-level порог аномальности.

    Как работает:
    - PatchCore вычисляет max(L2-расстояние до memory bank) для каждого изображения.
    - Нормальные изображения имеют низкий score, дефектные — высокий.
    - Порог = percentile-й процентиль score по нормальным изображениям.
    - После этого в /analyze, /analyze_dir: score < threshold → 0 дефектов.

    Почему test_good, а не train_good:
    - Memory bank PatchCore построен из train/good — их фичи буквально в нём.
    - Расстояния от train/good до memory bank искусственно низкие.
    - test/good — unseen: даёт честный порог разделения норма/дефект.
    """
    # Выбираем директории в зависимости от source
    candidate_dirs: List[Path] = []
    if req.source in ("test_good", "both"):
        candidate_dirs.append(DATA_DIR / "good")
    if req.source in ("train_good", "both"):
        candidate_dirs.append(TRAIN_DIR / "good")

    existing_dirs = [d for d in candidate_dirs if d.exists()]
    if not existing_dirs:
        missing = ", ".join(str(d) for d in candidate_dirs)
        raise HTTPException(404, f"Не найдены директории: {missing}")

    all_files: List[Path] = []
    for d in existing_dirs:
        all_files.extend(
            f for f in sorted(d.iterdir())
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
        )
    if not all_files:
        raise HTTPException(404, "Нет изображений в выбранных good/ директориях")

    # Инференс и сбор scores
    scores_by_source: dict = {"test_good": [], "train_good": []}
    for img_file in all_files:
        img = _read_image(img_file)
        if img is None:
            continue
        _, score = runner.infer_with_score(img)
        src = "train_good" if TRAIN_DIR in img_file.parents else "test_good"
        scores_by_source[src].append(score)

    all_scores = scores_by_source["test_good"] + scores_by_source["train_good"]
    if not all_scores:
        raise HTTPException(500, "Не удалось обработать ни одно изображение")

    scores_arr = np.array(all_scores)
    threshold = float(np.percentile(scores_arr, req.percentile))

    logger.info(
        "Calibrated: threshold=%.4f (p%.0f, source=%s, n=%d, max=%.4f)",
        threshold, req.percentile, req.source, len(all_scores), scores_arr.max(),
    )

    # Отдельная статистика по источникам
    def _stats(lst: list) -> dict:
        if not lst:
            return {"n": 0}
        a = np.array(lst)
        return {
            "n": len(lst),
            "min": round(float(a.min()), 4),
            "max": round(float(a.max()), 4),
            "mean": round(float(a.mean()), 4),
            "std": round(float(a.std()), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
        }

    return {
        "status":     "ok",
        "source":     req.source,
        "percentile": req.percentile,
        "threshold":  round(threshold, 4),
        "gate_enabled": True,
        "total_images": len(all_scores),
        "test_good":  _stats(scores_by_source["test_good"]),
        "train_good": _stats(scores_by_source["train_good"]),
        "warning": (
            "train/good score занижен: memory bank построен из этих изображений — "
            "использовать только для сравнения, не для калибровки."
        ) if req.source != "test_good" else None,
    }


@app.get("/calibrate")
def get_calibration_status():
    """Статус калибровки (только информационный — gate удалён)."""
    return {"gate_enabled": False, "threshold": None}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    """
    Подбор параметров threshold/S_min/S_max по пиксельным метрикам.
    Использует ground truth маски из GT_DIR.
    """
    search_root = DATA_DIR / req.subfolder if req.subfolder else DATA_DIR
    if not search_root.exists():
        raise HTTPException(404, f"Поддиректория не найдена: {req.subfolder}")

    all_files: List[Path] = sorted(
        f for f in search_root.rglob("*")
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    )
    if not all_files:
        raise HTTPException(404, "Изображения не найдены.")

    files = all_files[: req.max_images]

    # Собираем heatmaps и GT маски (только дефектные изображения имеют маски).
    # Если gate откалиброван: нормальные изображения (score < threshold) получают
    # нулевой heatmap — сегментация предсказывает «нет дефектов», что честно
    # соответствует поведению в /analyze и /analyze_all.
    heatmaps: List[np.ndarray] = []
    gt_masks = []
    skipped_gt = 0

    for img_file in files:
        rel = str(img_file.relative_to(DATA_DIR)).replace("\\", "/")
        img = _read_image(img_file)
        if img is None:
            continue
        hm = runner.infer(img)
        gt = find_gt_mask(rel, GT_DIR)
        heatmaps.append(hm)
        gt_masks.append(gt)
        if gt is None:
            skipped_gt += 1

    if not heatmaps:
        raise HTTPException(404, "Нет изображений для оценки.")

    logger.info(
        "optimize: %d изображений (%d без GT), mode=%s, combos=%d",
        len(heatmaps), skipped_gt, req.mode,
        len(req.thresholds) * len(req.s_mins) * len(req.s_maxs),
    )

    if req.mode == "sequential":
        elapsed_ms, results = optimize_sequential(
            heatmaps, gt_masks, req.thresholds, req.s_mins, req.s_maxs
        )
    elif req.mode == "parallel_cpu":
        pool = get_or_create_pool(req.workers)
        elapsed_ms, results = optimize_parallel_cpu(
            heatmaps, gt_masks, req.thresholds, req.s_mins, req.s_maxs,
            workers=req.workers, pool=pool,
        )
    elif req.mode == "parallel_gpu":
        elapsed_ms, results = optimize_gpu_sequential(
            heatmaps, gt_masks, req.thresholds, req.s_mins, req.s_maxs
        )
    else:  # parallel_gpu_batch
        elapsed_ms, results = optimize_gpu_batch(
            heatmaps, gt_masks, req.thresholds, req.s_mins, req.s_maxs
        )

    best = best_combo(results, "f1")
    top = top_n(results, "f1", 10)

    return {
        "mode":          req.mode,
        "n_images":      len(heatmaps),
        "n_with_gt":     len(heatmaps) - skipped_gt,
        "n_combos":      len(results),
        "elapsed_ms":    elapsed_ms,
        "best":          best,
        "top10":         top,
    }


OPTIMIZE_MODES = ["sequential", "parallel_cpu", "parallel_gpu", "parallel_gpu_batch"]
OPTIMIZE_MODE_LABELS = {
    "sequential":         "Последовательный",
    "parallel_cpu":       "Параллельный CPU (ThreadPool, без IPC)",
    "parallel_gpu":       "GPU морфология (последовательный)",
    "parallel_gpu_batch": "GPU батч (все пороги за раз)",
}


@app.post("/optimize_all")
def optimize_all(req: OptimizeAllRequest):
    def _generate():
        try:
            search_root = DATA_DIR / req.subfolder if req.subfolder else DATA_DIR
            if not search_root.exists():
                yield json.dumps({"type": "error", "detail": f"Поддиректория не найдена: {req.subfolder}"}) + "\n"
                return
            all_files: List[Path] = sorted(
                f for f in search_root.rglob("*")
                if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
            )
            if not all_files:
                yield json.dumps({"type": "error", "detail": "Изображения не найдены."}) + "\n"
                return
            files = all_files[: req.max_images]

            heatmaps: List[np.ndarray] = []
            gt_masks = []
            skipped_gt = 0
            for img_file in files:
                rel = str(img_file.relative_to(DATA_DIR)).replace("\\", "/")
                img = _read_image(img_file)
                if img is None:
                    continue
                hm = runner.infer(img)
                gt = find_gt_mask(rel, GT_DIR)
                heatmaps.append(hm)
                gt_masks.append(gt)
                if gt is None:
                    skipped_gt += 1

            if not heatmaps:
                yield json.dumps({"type": "error", "detail": "Нет изображений для оценки."}) + "\n"
                return

            n_combos = len(req.thresholds) * len(req.s_mins) * len(req.s_maxs)
            logger.info("optimize_all stream: %d изображений, combos=%d", len(heatmaps), n_combos)

            yield json.dumps({
                "type": "start",
                "n_modes": len(OPTIMIZE_MODES),
                "n_combos": n_combos,
                "n_images": len(heatmaps),
            }) + "\n"

            pool = get_or_create_pool(req.workers)
            mode_results = []
            seq_elapsed: Optional[float] = None

            for mode in OPTIMIZE_MODES:
                if mode == "sequential":
                    elapsed_ms, results = optimize_sequential(
                        heatmaps, gt_masks, req.thresholds, req.s_mins, req.s_maxs
                    )
                    seq_elapsed = elapsed_ms
                elif mode == "parallel_cpu":
                    elapsed_ms, results = optimize_parallel_cpu(
                        heatmaps, gt_masks, req.thresholds, req.s_mins, req.s_maxs,
                        workers=req.workers, pool=pool,
                    )
                elif mode == "parallel_gpu":
                    elapsed_ms, results = optimize_gpu_sequential(
                        heatmaps, gt_masks, req.thresholds, req.s_mins, req.s_maxs
                    )
                else:
                    elapsed_ms, results = optimize_gpu_batch(
                        heatmaps, gt_masks, req.thresholds, req.s_mins, req.s_maxs
                    )

                best = best_combo(results, "f1")
                speedup = (
                    round(seq_elapsed / elapsed_ms, 3)
                    if seq_elapsed and elapsed_ms > 0 and mode != "sequential"
                    else 1.0
                )
                stat = {
                    "mode": mode, "label": OPTIMIZE_MODE_LABELS[mode],
                    "elapsed_ms": elapsed_ms, "speedup": speedup,
                    "best": best, "top10": top_n(results, "f1", 10),
                }
                mode_results.append(stat)
                logger.info("optimize_all [%s]: %.1f мс, best_f1=%.4f",
                            mode, elapsed_ms, best.get("f1", 0))
                yield json.dumps({"type": "mode", "data": stat}) + "\n"

            yield json.dumps({
                "type": "done",
                "subfolder": req.subfolder or "все",
                "n_images": len(heatmaps),
                "n_with_gt": len(heatmaps) - skipped_gt,
                "n_combos": n_combos,
                "workers": req.workers,
                "modes": mode_results,
            }) + "\n"

        except Exception as e:
            logger.exception("optimize_all stream error")
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


@app.post("/binarize_optimize")
def binarize_optimize(req: BinarizeOptimizeRequest):
    def _generate():
        try:
            search_root = DATA_DIR / req.subfolder if req.subfolder else DATA_DIR
            if not search_root.exists():
                yield json.dumps({"type": "error", "detail": f"Поддиректория не найдена: {req.subfolder}"}) + "\n"
                return
            all_files: List[Path] = sorted(
                f for f in search_root.rglob("*")
                if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
            )
            if not all_files:
                yield json.dumps({"type": "error", "detail": "Изображения не найдены."}) + "\n"
                return

            files = all_files[: req.max_images]
            heatmaps, gt_masks = [], []
            skipped_gt = 0
            for img_file in files:
                img = _read_image(img_file)
                if img is None:
                    continue
                hm  = runner.infer(img)
                rel = str(img_file.relative_to(DATA_DIR)).replace("\\", "/")
                gt  = find_gt_mask(rel, GT_DIR)
                heatmaps.append(hm)
                gt_masks.append(gt)
                if gt is None:
                    skipped_gt += 1

            if not heatmaps:
                yield json.dumps({"type": "error", "detail": "Нет изображений для оценки."}) + "\n"
                return

            param_grid = BINARIZE_PARAM_GRID.copy()
            if req.thresholds:           param_grid["threshold"]             = req.thresholds
            if req.min_areas:            param_grid["min_area"]              = req.min_areas
            if req.morphology_kernels:   param_grid["morphology_kernel"]     = req.morphology_kernels
            if req.morphology_iterations: param_grid["morphology_iterations"] = req.morphology_iterations
            if req.gaussian_blur_kernels: param_grid["gaussian_blur_kernel"]  = req.gaussian_blur_kernels

            n_combos = (
                len(param_grid["threshold"]) * len(param_grid["min_area"])
                * len(param_grid["morphology_kernel"]) * len(param_grid["morphology_iterations"])
                * len(param_grid["gaussian_blur_kernel"])
            )
            logger.info(
                "binarize_optimize stream: %d изображений (%d без GT), %d комбинаций, gpu=%s",
                len(heatmaps), skipped_gt, n_combos, req.use_gpu,
            )
            yield json.dumps({"type": "start", "n_combos": n_combos, "n_images": len(heatmaps)}) + "\n"

            # Queue-based progress from grid_search thread
            q: "queue.Queue" = queue.Queue()
            last_rep = [0]

            def _cb(done: int, total: int) -> None:
                interval = max(1, total // 40)
                if done == total or done - last_rep[0] >= interval:
                    last_rep[0] = done
                    q.put(("progress", done, total))

            def _run() -> None:
                try:
                    if req.use_gpu:
                        res = binarize_gpu_grid_search(
                            heatmaps, gt_masks,
                            param_grid=param_grid,
                            device="cuda",
                            exact=req.gpu_exact,
                            progress_callback=_cb,
                        )
                    else:
                        res = binarize_grid_search(
                            heatmaps, gt_masks,
                            param_grid=param_grid,
                            workers=req.workers,
                            progress_callback=_cb,
                        )
                    q.put(("done", res))
                except Exception as exc:
                    q.put(("error", str(exc)))

            t = threading.Thread(target=_run, daemon=True)
            t.start()

            while True:
                msg = q.get()
                if msg[0] == "progress":
                    yield json.dumps({"type": "progress", "done": msg[1], "total": msg[2]}) + "\n"
                elif msg[0] == "done":
                    elapsed_ms, results = msg[1]
                    best = binarize_find_best(results)
                    binarize_save_params(best)
                    applied = reload_binarize_params()
                    logger.info(
                        "binarize_optimize завершён: %.0f мс, best_loss=%.6f",
                        elapsed_ms, best.get("loss", 0),
                    )
                    yield json.dumps({
                        "type":          "done",
                        "n_images":      len(heatmaps),
                        "n_with_gt":     len(heatmaps) - skipped_gt,
                        "n_combos":      len(results),
                        "elapsed_ms":    elapsed_ms,
                        "best":          best,
                        "top10":         binarize_top_n(results, 10),
                        "applied_params": applied,
                    }) + "\n"
                    break
                elif msg[0] == "error":
                    yield json.dumps({"type": "error", "detail": msg[1]}) + "\n"
                    break

        except Exception as e:
            logger.exception("binarize_optimize stream error")
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


@app.get("/binarize_params")
def get_binarize_params_endpoint():
    """Возвращает текущие активные параметры бинаризации."""
    return get_binarize_params()


@app.post("/binarize_params")
def set_binarize_params_endpoint(req: BinarizeParamsRequest):
    """Вручную задаёт параметры бинаризации и сохраняет best_params.json."""
    from optimizer.parameter_optimizer import BEST_PARAMS_FILE
    import json

    payload = {
        "threshold":             req.threshold,
        "min_area":              req.min_area,
        "morphology_kernel":     req.morphology_kernel,
        "morphology_iterations": req.morphology_iterations,
        "gaussian_blur_kernel":  req.gaussian_blur_kernel,
    }
    BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    applied = reload_binarize_params()
    logger.info("binarize_params обновлены вручную: %s", applied)
    return {"status": "ok", "applied_params": applied}


@app.get("/modes")
def get_modes():
    return {"modes": [{"id": m, "label": MODE_LABELS[m]} for m in ALL_MODES]}


@app.get("/ui", response_class=HTMLResponse)
def test_ui():
    html_file = BASE_DIR / "test_ui.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.get("/original/{image_path:path}")
def get_original_image(image_path: str):
    img_file = DATA_DIR / image_path
    if not img_file.exists() or img_file.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(404, "Изображение не найдено")
    return FileResponse(str(img_file))


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
