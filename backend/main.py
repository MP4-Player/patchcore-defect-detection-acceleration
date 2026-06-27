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

# Добавьте в начало файла после других импортов
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from datetime import datetime

import base64
import json
import logging
import math
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

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
    extract_defects_from_heatmap,
    _full_defect_analysis,
    get_binarize_params,
    get_or_create_pool,
    parallel_cpu_process,
    parallel_gpu_process,
    reload_binarize_params,
    sequential_process,
    shutdown_all_pools,
)
from defect_classification import (
    classify_defects as ml_classify_defects,
    classify_defects_batch as ml_classify_batch,
    compare_methods as ml_compare_methods,
    result_to_json_safe,
    batch_result_to_json_safe,
    DEFECT_TYPE_NAMES,
    FEATURE_NAMES,
)
from defect_embedding_classifier import (
    classify_by_embeddings,
    classify_single_by_embeddings,
    collect_batch_embeddings as emb_collect_batch,
    compute_2d_projection,
    hungarian_accuracy as emb_hungarian_accuracy,
    result_to_json_safe as emb_result_to_json_safe,
    _emb_process_worker,
    UMAP_AVAILABLE,
)
from research_grid_search import (
    run_grid_search as gs_run_grid,
    run_stability_check as gs_stability,
    run_stability_extended as gs_stability_ext,
    get_all_results as gs_get_results,
    clear_cache as gs_clear_cache,
    compute_area_stability_correlation as gs_area_corr,
    precompute_inference as gs_precompute,
    run_single_experiment as gs_single_exp,
    run_config_detailed as gs_config_detailed,
    aggregate_problem_pairs as gs_problem_pairs,
    diagnose_class_detectability as gs_diagnose_detectability,
    diagnose_recall_vs_fpr as gs_recall_fpr,
    run_full_grid_search as gs_run_full_grid,
    get_full_grid_results as gs_get_full_grid_results,
    clear_full_grid_cache as gs_clear_full_grid_cache,
    DEFECT_TYPE_NAMES as GS_DEFECT_NAMES,
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
from calibration import (
    batch_calibrate           as cal_batch,
    clear_calib_cache         as cal_clear_cache,
    get_auto_calib_status     as cal_auto_status,
    pick_reference_images     as cal_pick_refs,
    run_auto_calibrate_grid   as cal_auto_grid,
    CALIB_CLASSES             as CAL_CLASSES,
    compute_baseline_map      as cal_compute_baseline,
    save_baseline_map         as cal_save_baseline,
    load_baseline_map         as cal_load_baseline,
    BASELINE_PATH             as CAL_BASELINE_PATH,
)

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

# Version token for the calibration raw-heatmap cache — changes if model is reloaded.
_CAL_MB_VER: str = str(time.time())

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
    "parallel_batch_embed", # true batch: один forward ResNet-50 для всех фото
]

MODE_LABELS = {
    "sequential":            "Последовательный",
    "parallel_cpu":          "Параллельные дефекты (CPU)",
    "parallel_images":       "Параллельные изображения (CPU)",
    "parallel_gpu":          "Параллельные дефекты (GPU)",
    "parallel_images_gpu":   "Параллельные изображения (GPU)",
    "parallel_combined":     "Комбинированный CPU (фото+дефекты)",
    "parallel_combined_gpu": "Batch GPU (все фото за раз)",
    "parallel_batch_embed":  "Batch Embed (один forward ResNet-50)",
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
        "parallel_batch_embed",
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

            # ── Batch Embed: один forward ResNet-50 для всех изображений ─────
            elif req.mode == "parallel_batch_embed":
                t_wall = time.perf_counter()
                images_data = []
                for img_file in files:
                    rel = str(img_file.relative_to(DATA_DIR)).replace("\\", "/")
                    img = _read_image(img_file)
                    if img is None:
                        continue
                    images_data.append((img_file, rel, img))

                if not images_data:
                    yield json.dumps({"type": "error", "detail": "Нет изображений для обработки."}) + "\n"
                    return

                imgs_only = [d[2] for d in images_data]
                heatmaps_list = runner.infer_batch(imgs_only)

                # Последовательная обработка дефектов по готовым heatmap
                seq_total_ms = 0.0
                all_results = []
                for i, (img_file, rel, img) in enumerate(images_data):
                    hm = heatmaps_list[i]
                    elapsed, contours, defects = cpu_heatmap_pipeline(
                        hm, req.S_min, req.S_max, req.K, req.threshold
                    )
                    seq_total_ms += elapsed
                    all_results.append((rel, img, contours, defects, elapsed))

                wall_ms     = round((time.perf_counter() - t_wall) * 1_000, 2)
                per_img_ref = round(seq_total_ms / len(images_data), 3) if images_data else 0.0
                speedup     = round(per_img_ref * len(images_data) / wall_ms, 3) if wall_ms > 0 else None
                per_img_ms  = round(wall_ms / len(images_data), 3) if images_data else 0.0

                for rel, img, contours, defects, _ in all_results:
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
                        "ref_time_ms": per_img_ref, "speedup": speedup,
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
                per_image_times: List[float] = []

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
                        avg_per = batch_ms / len(images_data)
                        for i in range(len(images_data)):
                            total_found    += len(all_c[i])
                            total_filtered += len(all_d[i])
                            total_time_ms  += avg_per
                            per_image_times.append(round(avg_per, 2))
                            n_proc         += 1

                elif mode == "parallel_batch_embed":
                    imgs = []
                    for img_file in files:
                        img = _read_image(img_file)
                        if img is not None:
                            imgs.append(img)
                    if imgs:
                        hm_list = runner.infer_batch(imgs)
                        for hm in hm_list:
                            _, contours, defects = cpu_heatmap_pipeline(
                                hm, req.S_min, req.S_max, req.K, req.threshold
                            )
                            total_found    += len(contours)
                            total_filtered += len(defects)
                            n_proc         += 1
                        # wall time включает весь batch — делим равномерно
                        wall_so_far = (time.perf_counter() - t_wall) * 1_000
                        total_time_ms = wall_so_far
                        avg_per = wall_so_far / len(imgs) if imgs else 0.0
                        per_image_times = [round(avg_per, 2)] * len(imgs)

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
                        per_image_times.append(round(entry["time_ms"], 2))
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
                        per_image_times.append(round(entry["time_ms"], 2))
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
                    "per_image_times": per_image_times,
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
    html_file = BASE_DIR.parent / "frontend" / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.get("/original/{image_path:path}")
def get_original_image(image_path: str):
    img_file = DATA_DIR / image_path
    if not img_file.exists() or img_file.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(404, "Изображение не найдено")
    return FileResponse(str(img_file))


# ---------------------------------------------------------------------------
# ML-классификация дефектов
# ---------------------------------------------------------------------------

DEFECT_TYPE_TO_IDX: dict = {name: i for i, name in enumerate(DEFECT_TYPE_NAMES)}


def _get_gt_label(image_rel_path: str) -> Optional[int]:
    """Возвращает индекс GT-метки из названия папки (color=0, cut=1, ...) или None."""
    parts = image_rel_path.replace("\\", "/").split("/")
    if not parts:
        return None
    return DEFECT_TYPE_TO_IDX.get(parts[0])


def _default_classify_config(
    method: str = "kmeans",
    n_clusters: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 8,
    threshold: float = 0.6,
    s_min_percentile: float = 5.0,
    s_max_percentile: float = 95.0,
    score_only: bool = False,
) -> dict:
    return {
        "method":           method,
        "n_clusters":       n_clusters,
        "k_range":          [k_min, k_max],
        "threshold":        threshold,
        "s_min_percentile": s_min_percentile,
        "s_max_percentile": s_max_percentile,
        "score_only":       score_only,
    }


# ---------------------------------------------------------------------------
# Worker для ProcessPool (module-level — обязательно для Windows spawn)
# ---------------------------------------------------------------------------

def _classify_process_worker(args: tuple) -> dict:
    """
    Worker для ProcessPoolExecutor: classify_defects на одном изображении.
    Аргументы передаются через pickle → все numpy/dict/list.
    """
    from defect_classification import classify_defects as _cd
    (
        heatmap_bytes, heatmap_shape,
        binary_bytes, binary_shape,
        config, gt_label, fitted_state_bytes,
        image_path,
    ) = args
    hm = np.frombuffer(heatmap_bytes, dtype=np.float32).reshape(heatmap_shape)
    bm = np.frombuffer(binary_bytes,  dtype=np.uint8).reshape(binary_shape)
    fitted_state = None
    if fitted_state_bytes is not None:
        import pickle
        fitted_state = pickle.loads(fitted_state_bytes)
    r = _cd(hm, bm, config, gt_label=gt_label, fitted_state=fitted_state)
    safe = result_to_json_safe(r)
    safe["image_path"] = image_path
    return safe


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    image_path: str = Field(..., description="Путь относительно test/")
    method: Literal["kmeans", "gmm", "hierarchical", "prototype", "linear"] = "kmeans"
    n_clusters: Optional[int] = Field(default=None, ge=1, le=20)
    k_min: int = Field(default=2, ge=2)
    k_max: int = Field(default=8)
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    s_min_percentile: float = Field(default=5.0, ge=0.0, le=50.0)
    s_max_percentile: float = Field(default=95.0, ge=50.0, le=100.0)
    score_only: bool = Field(default=False)
    compare_all_methods: bool = Field(default=False)


class ClassifyBatchRequest(BaseModel):
    subfolder: str = Field(default="")
    method: Literal["kmeans", "gmm", "hierarchical", "prototype", "linear"] = "kmeans"
    n_clusters: Optional[int] = Field(default=None, ge=1, le=20)
    k_min: int = Field(default=2, ge=2)
    k_max: int = Field(default=8)
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    s_min_percentile: float = Field(default=5.0)
    s_max_percentile: float = Field(default=95.0)
    max_images: int = Field(default=50, ge=1, le=300)
    mode: Literal[
        "sequential", "parallel_cpu", "parallel_images", "parallel_gpu",
        "parallel_images_gpu", "parallel_combined",
        "parallel_combined_gpu", "parallel_batch_embed",
    ] = "sequential"
    workers: int = Field(default=4, ge=1, le=16)
    include_good: bool = Field(default=False)
    score_only: bool = Field(default=False)
    run_benchmark: bool = Field(default=False, description="Замерять время по всем 8 режимам")


# ---------------------------------------------------------------------------
# Endpoint: один файл
# ---------------------------------------------------------------------------

@app.post("/classify_defects")
def classify_defects_endpoint(req: ClassifyRequest):
    """
    ML-классификация дефектов одного изображения.

    Возвращает:
    - cluster_labels, features, silhouette, accuracy (если GT доступен)
    - Если compare_all_methods=True: результаты для KMeans, GMM и Hierarchical
    """
    img_file = DATA_DIR / req.image_path
    if not img_file.exists():
        raise HTTPException(404, f"Файл не найден: {req.image_path}")
    if img_file.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Неподдерживаемый формат.")

    image = _read_image(img_file)
    if image is None:
        raise HTTPException(400, "Не удалось декодировать изображение.")

    heatmap, anomaly_score = runner.infer_with_score(image)
    contours, binary = extract_defects_from_heatmap(heatmap, req.threshold)
    gt_label = _get_gt_label(req.image_path)

    config = _default_classify_config(
        method=req.method,
        n_clusters=req.n_clusters,
        k_min=req.k_min,
        k_max=req.k_max,
        threshold=req.threshold,
        s_min_percentile=req.s_min_percentile,
        s_max_percentile=req.s_max_percentile,
        score_only=req.score_only,
    )

    if req.compare_all_methods:
        comparison = ml_compare_methods(heatmap, binary, config, contours, gt_label)
        best   = comparison["best_method"]
        result = {
            "kmeans":       result_to_json_safe(comparison["kmeans"]),
            "gmm":          result_to_json_safe(comparison["gmm"]),
            "hierarchical": result_to_json_safe(comparison["hierarchical"]),
            "best_method":  best,
            "best_score":   comparison["best_score"],
        }
    else:
        r      = ml_classify_defects(heatmap, binary, config, contours, gt_label)
        result = result_to_json_safe(r)

    return {
        "image_path":    req.image_path,
        "gt_label":      gt_label,
        "gt_type":       DEFECT_TYPE_NAMES[gt_label] if gt_label is not None else None,
        "anomaly_score": round(anomaly_score, 4),
        "n_contours":    len(contours),
        "result":        result,
        "feature_names": FEATURE_NAMES,
        "defect_type_names": DEFECT_TYPE_NAMES,
    }


# ---------------------------------------------------------------------------
# Endpoint: батч (директория) + интеграция с 8 режимами
# ---------------------------------------------------------------------------

@app.post("/classify_batch")
def classify_batch_endpoint(req: ClassifyBatchRequest):
    """
    Batch ML-классификация дефектов для директории изображений.

    Поддерживает все 8 режимов параллельной обработки:
    - sequential, parallel_cpu (ProcessPool), parallel_images (ThreadPool),
    - parallel_gpu, parallel_images_gpu, parallel_combined,
    - parallel_combined_gpu (batch), parallel_batch_embed (batch)

    Если run_benchmark=True — запускает все 8 режимов и возвращает сравнение времён.
    """
    def _generate():
        try:
            search_root = DATA_DIR / req.subfolder if req.subfolder else DATA_DIR
            if not search_root.exists():
                yield json.dumps({"type": "error", "detail": f"Директория не найдена: {req.subfolder}"}) + "\n"
                return

            all_files: List[Path] = sorted(
                f for f in search_root.rglob("*")
                if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
            )
            if not req.include_good:
                all_files = [
                    f for f in all_files
                    if f.relative_to(DATA_DIR).parts[0] != "good"
                ]
            if not all_files:
                yield json.dumps({"type": "error", "detail": "Изображения не найдены."}) + "\n"
                return

            files = all_files[: req.max_images]
            config = _default_classify_config(
                method=req.method,
                n_clusters=req.n_clusters,
                k_min=req.k_min,
                k_max=req.k_max,
                threshold=req.threshold,
                s_min_percentile=req.s_min_percentile,
                s_max_percentile=req.s_max_percentile,
                score_only=req.score_only,
            )

            yield json.dumps({
                "type": "start",
                "total": len(files),
                "mode": req.mode,
                "method": req.method,
            }) + "\n"

            # ── Шаг 1: инференс PatchCore для всех изображений ─────────────
            t_infer = time.perf_counter()
            image_data: List[tuple] = []
            for img_file in files:
                rel = str(img_file.relative_to(DATA_DIR)).replace("\\", "/")
                img = _read_image(img_file)
                if img is None:
                    continue
                hm, score = runner.infer_with_score(img)
                conts, binary = extract_defects_from_heatmap(hm, req.threshold)
                gt = _get_gt_label(rel)
                image_data.append((rel, hm, binary, conts, gt, score))
            infer_ms = round((time.perf_counter() - t_infer) * 1000, 2)

            if not image_data:
                yield json.dumps({"type": "error", "detail": "Нет изображений для классификации."}) + "\n"
                return

            yield json.dumps({"type": "infer_done", "n_images": len(image_data), "infer_ms": infer_ms}) + "\n"

            # ── Шаг 2: классификация в выбранном режиме ─────────────────────
            t_class = time.perf_counter()
            per_image_results: List[dict] = []
            benchmark_times: Optional[dict] = None

            # ── Batch-режимы (7 и 8) → classify_defects_batch ──────────────
            if req.mode in ("parallel_combined_gpu", "parallel_batch_embed"):
                heatmaps = [d[1] for d in image_data]
                binaries  = [d[2] for d in image_data]
                conts_l   = [d[3] for d in image_data]
                gt_labels = [d[4] for d in image_data]
                paths     = [d[0] for d in image_data]

                batch_r = ml_classify_batch(
                    heatmaps, binaries, config,
                    contours_list=conts_l,
                    gt_labels=gt_labels,
                    image_paths=paths,
                )
                per_image_results = [result_to_json_safe(r) for r in batch_r["per_image"]]

                for r in per_image_results:
                    yield json.dumps({"type": "image", "result": r}) + "\n"

            # ── Thread-режимы (parallel_images, parallel_images_gpu, parallel_combined) ──
            elif req.mode in ("parallel_images", "parallel_images_gpu", "parallel_combined"):
                # Fit модели в главном потоке → передать во все воркеры (thread-safe)
                if len(image_data) > 1:
                    sample_hm  = [d[1] for d in image_data[:min(10, len(image_data))]]
                    sample_bm  = [d[2] for d in image_data[:min(10, len(image_data))]]
                    sample_con = [d[3] for d in image_data[:min(10, len(image_data))]]
                    prefitted  = ml_classify_batch(sample_hm, sample_bm, config, contours_list=sample_con)
                    fitted_state = (prefitted["fitted_scaler"], prefitted["fitted_model"])
                else:
                    fitted_state = None

                import threading
                lock = threading.Lock()

                def _thread_task(item):
                    rel, hm, binary, conts, gt, score = item
                    r = ml_classify_defects(hm, binary, config, conts, gt, fitted_state)
                    safe = result_to_json_safe(r)
                    safe["image_path"] = rel
                    with lock:
                        pass  # просто синхронизируем доступ если нужно
                    return safe

                from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac
                with _TPE(max_workers=req.workers) as ex:
                    futs = {ex.submit(_thread_task, item): item for item in image_data}
                    for fut in _ac(futs):
                        safe = fut.result()
                        per_image_results.append(safe)
                        yield json.dumps({"type": "image", "result": safe}) + "\n"

            # ── Process-режим (parallel_cpu) ─────────────────────────────────
            elif req.mode == "parallel_cpu":
                import pickle as _pk
                # Fit в главном процессе → сериализуем fitted_state для воркеров
                fitted_state_bytes = None
                if len(image_data) > 1:
                    sample_hm  = [d[1] for d in image_data[:min(8, len(image_data))]]
                    sample_bm  = [d[2] for d in image_data[:min(8, len(image_data))]]
                    sample_con = [d[3] for d in image_data[:min(8, len(image_data))]]
                    prefitted  = ml_classify_batch(sample_hm, sample_bm, config, contours_list=sample_con)
                    if prefitted["fitted_scaler"] is not None:
                        fitted_state_bytes = _pk.dumps(
                            (prefitted["fitted_scaler"], prefitted["fitted_model"])
                        )

                args_list = [
                    (
                        d[1].tobytes(), d[1].shape,
                        d[2].tobytes(), d[2].shape,
                        config, d[4], fitted_state_bytes, d[0],
                    )
                    for d in image_data
                ]
                pool = get_or_create_pool(req.workers)
                for safe in pool.map(_classify_process_worker, args_list):
                    per_image_results.append(safe)
                    yield json.dumps({"type": "image", "result": safe}) + "\n"

            # ── Sequential / parallel_gpu ────────────────────────────────────
            else:
                for rel, hm, binary, conts, gt, score in image_data:
                    r = ml_classify_defects(hm, binary, config, conts, gt)
                    safe = result_to_json_safe(r)
                    safe["image_path"] = rel
                    per_image_results.append(safe)
                    yield json.dumps({"type": "image", "result": safe}) + "\n"

            classify_ms = round((time.perf_counter() - t_class) * 1000, 2)

            # ── Шаг 3: Benchmark по всем 8 режимам (опционально) ──────────────
            if req.run_benchmark:
                benchmark_times = {}
                heatmaps = [d[1] for d in image_data]
                binaries  = [d[2] for d in image_data]
                conts_l   = [d[3] for d in image_data]
                gt_labels = [d[4] for d in image_data]

                for bmode in ALL_MODES:
                    tb = time.perf_counter()
                    try:
                        if bmode in ("parallel_combined_gpu", "parallel_batch_embed"):
                            ml_classify_batch(heatmaps, binaries, config, conts_l, gt_labels)
                        elif bmode in ("parallel_images", "parallel_images_gpu", "parallel_combined"):
                            prefitted = ml_classify_batch(
                                heatmaps[:min(5, len(heatmaps))],
                                binaries[:min(5, len(binaries))],
                                config,
                            )
                            fs = (prefitted["fitted_scaler"], prefitted["fitted_model"])
                            with ThreadPoolExecutor(max_workers=req.workers) as ex:
                                list(ex.map(
                                    lambda args: ml_classify_defects(*args),
                                    [(hm, bm, config, c, gt, fs)
                                     for hm, bm, c, gt in zip(heatmaps, binaries, conts_l, gt_labels)]
                                ))
                        else:
                            for hm, bm, c, gt in zip(heatmaps, binaries, conts_l, gt_labels):
                                ml_classify_defects(hm, bm, config, c, gt)
                    except Exception as bex:
                        logger.warning("Benchmark [%s]: %s", bmode, bex)
                    benchmark_times[bmode] = round((time.perf_counter() - tb) * 1000, 2)
                yield json.dumps({"type": "benchmark", "times_ms": benchmark_times}) + "\n"

            # ── Итоговая статистика ──────────────────────────────────────────
            n_with_results = len([r for r in per_image_results if r.get("n_valid", 0) > 0])
            all_acc = [r["accuracy_single"] for r in per_image_results if r.get("accuracy_single") is not None]
            all_sil = [r["silhouette"]      for r in per_image_results if r.get("silhouette")      is not None]
            gt_map  = DEFECT_TYPE_NAMES

            # Матрица ошибок: cluster → gt_type
            all_cluster_labels: List[int] = []
            all_gt_labels:      List[int] = []
            for r in per_image_results:
                gt = r.get("gt_label")
                if gt is not None:
                    all_cluster_labels.extend(r.get("cluster_labels", []))
                    all_gt_labels.extend([gt] * len(r.get("cluster_labels", [])))

            overall_acc: Optional[float] = None
            overall_mapping: dict = {}
            if all_cluster_labels and all_gt_labels:
                from defect_classification import hungarian_accuracy
                n_cl = req.n_clusters or max(all_cluster_labels, default=0) + 1
                overall_acc, overall_mapping = hungarian_accuracy(
                    np.array(all_cluster_labels),
                    np.array(all_gt_labels),
                    n_cl, len(DEFECT_TYPE_NAMES),
                )

            yield json.dumps({
                "type": "done",
                "mode": req.mode,
                "method": req.method,
                "n_images": len(image_data),
                "n_with_defects": n_with_results,
                "infer_ms": infer_ms,
                "classify_ms": classify_ms,
                "classify_per_image_ms": round(classify_ms / max(len(image_data), 1), 3),
                "mean_accuracy": round(float(np.mean(all_acc)), 4) if all_acc else None,
                "mean_silhouette": round(float(np.mean(all_sil)), 4) if all_sil else None,
                "overall_accuracy": round(overall_acc, 4) if overall_acc is not None else None,
                "overall_mapping": {str(k): v for k, v in overall_mapping.items()},
                "gt_type_names": DEFECT_TYPE_NAMES,
                "benchmark_times_ms": benchmark_times,
            }) + "\n"

        except Exception as e:
            logger.exception("classify_batch stream error")
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# Endpoint: benchmark по всем 8 режимам (только классификация, без инференса)
# ---------------------------------------------------------------------------

@app.post("/classify_benchmark")
def classify_benchmark_endpoint(req: ClassifyBatchRequest):
    """
    Замеряет время ML-классификации по всем 8 режимам.
    Инференс PatchCore выполняется один раз (не входит в замер).

    Returns:
        benchmark: {mode: {time_ms, speedup_vs_sequential, n_defects}}
    """
    search_root = DATA_DIR / req.subfolder if req.subfolder else DATA_DIR
    if not search_root.exists():
        raise HTTPException(404, f"Директория не найдена: {req.subfolder}")

    all_files: List[Path] = sorted(
        f for f in search_root.rglob("*")
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    )
    if not req.include_good:
        all_files = [f for f in all_files if f.relative_to(DATA_DIR).parts[0] != "good"]
    files = all_files[: req.max_images]

    config = _default_classify_config(
        method=req.method,
        n_clusters=req.n_clusters,
        k_min=req.k_min,
        k_max=req.k_max,
        threshold=req.threshold,
        s_min_percentile=req.s_min_percentile,
        s_max_percentile=req.s_max_percentile,
        score_only=req.score_only,
    )

    # Pre-инференс (один раз для всех)
    image_data: List[tuple] = []
    for img_file in files:
        rel = str(img_file.relative_to(DATA_DIR)).replace("\\", "/")
        img = _read_image(img_file)
        if img is None:
            continue
        hm, _ = runner.infer_with_score(img)
        conts, binary = extract_defects_from_heatmap(hm, req.threshold)
        gt = _get_gt_label(rel)
        image_data.append((rel, hm, binary, conts, gt))

    if not image_data:
        raise HTTPException(404, "Нет изображений для бенчмарка.")

    heatmaps = [d[1] for d in image_data]
    binaries  = [d[2] for d in image_data]
    conts_l   = [d[3] for d in image_data]
    gt_labels = [d[4] for d in image_data]

    benchmark: dict = {}
    seq_ms: Optional[float] = None

    for bmode in ALL_MODES:
        tb = time.perf_counter()
        n_defects_total = 0
        try:
            if bmode in ("parallel_combined_gpu", "parallel_batch_embed"):
                br = ml_classify_batch(heatmaps, binaries, config, conts_l, gt_labels)
                n_defects_total = br.get("n_total_defects", 0)

            elif bmode in ("parallel_images", "parallel_images_gpu", "parallel_combined"):
                prefitted = ml_classify_batch(
                    heatmaps[:min(5, len(heatmaps))],
                    binaries[:min(5, len(binaries))],
                    config,
                )
                fs = (prefitted["fitted_scaler"], prefitted["fitted_model"])
                with ThreadPoolExecutor(max_workers=req.workers) as ex:
                    results_b = list(ex.map(
                        lambda a: ml_classify_defects(a[0], a[1], config, a[2], a[3], fs),
                        [(hm, bm, c, gt) for hm, bm, c, gt in zip(heatmaps, binaries, conts_l, gt_labels)]
                    ))
                n_defects_total = sum(r.get("n_valid", 0) for r in results_b)

            elif bmode == "parallel_cpu":
                import pickle as _pk
                prefitted = ml_classify_batch(
                    heatmaps[:min(5, len(heatmaps))],
                    binaries[:min(5, len(binaries))],
                    config,
                )
                fs_bytes = _pk.dumps(
                    (prefitted["fitted_scaler"], prefitted["fitted_model"])
                ) if prefitted["fitted_scaler"] is not None else None

                args_list = [
                    (hm.tobytes(), hm.shape, bm.tobytes(), bm.shape, config, gt, fs_bytes, d[0])
                    for d, hm, bm, c, gt in zip(image_data, heatmaps, binaries, conts_l, gt_labels)
                ]
                pool = get_or_create_pool(req.workers)
                res_list = list(pool.map(_classify_process_worker, args_list))
                n_defects_total = sum(r.get("n_valid", 0) for r in res_list)

            else:  # sequential / parallel_gpu
                for hm, bm, c, gt in zip(heatmaps, binaries, conts_l, gt_labels):
                    r = ml_classify_defects(hm, bm, config, c, gt)
                    n_defects_total += r.get("n_valid", 0)

        except Exception as bex:
            logger.warning("Classify benchmark [%s]: %s", bmode, bex)

        elapsed_b = round((time.perf_counter() - tb) * 1000, 2)
        if bmode == "sequential":
            seq_ms = elapsed_b

        speedup = round(seq_ms / elapsed_b, 3) if seq_ms and elapsed_b > 0 and bmode != "sequential" else 1.0
        benchmark[bmode] = {
            "mode":              bmode,
            "label":             MODE_LABELS[bmode],
            "time_ms":           elapsed_b,
            "speedup_vs_sequential": speedup,
            "n_defects":         n_defects_total,
            "per_image_ms":      round(elapsed_b / len(image_data), 3) if image_data else 0,
        }

    return {
        "n_images":    len(image_data),
        "method":      req.method,
        "workers":     req.workers,
        "benchmark":   benchmark,
        "benchmark_list": list(benchmark.values()),
        "best_mode":   min(benchmark, key=lambda m: benchmark[m]["time_ms"]),
    }


# ---------------------------------------------------------------------------
# Endpoint: confusion matrix (исправленная версия)
# ---------------------------------------------------------------------------

@app.post("/confusion_matrix")
def compute_confusion_matrix(
    subfolder: str = "",
    threshold: float = 0.6,
    method: str = "kmeans",
    n_clusters: Optional[int] = None,
    max_images: int = 100,
):
    """
    Confusion matrix: ML-кластеры vs GT-типы дефектов.
    Использует Hungarian algorithm для сопоставления.
    """
    from sklearn.metrics import confusion_matrix as sk_cm
    from defect_classification import hungarian_accuracy

    search_root = DATA_DIR / subfolder if subfolder else DATA_DIR
    if not search_root.exists():
        raise HTTPException(404, f"Папка не найдена: {subfolder}")

    image_data: List[tuple] = []
    for folder in sorted(search_root.iterdir()):
        if not folder.is_dir():
            continue
        defect_type = folder.name
        if defect_type == "good":
            continue
        gt_idx = DEFECT_TYPE_TO_IDX.get(defect_type)
        if gt_idx is None:
            continue
        for img_file in sorted(folder.glob("*.*")):
            if img_file.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            image_data.append((img_file, defect_type, gt_idx))

    if not image_data:
        raise HTTPException(404, "Изображения с дефектами не найдены.")

    image_data = image_data[:max_images]
    config = _default_classify_config(method=method, n_clusters=n_clusters, threshold=threshold)

    heatmaps, binaries, conts_l, gt_labels, paths = [], [], [], [], []
    for img_file, defect_type, gt_idx in image_data:
        img = _read_image(img_file)
        if img is None:
            continue
        hm, _ = runner.infer_with_score(img)
        conts, binary = extract_defects_from_heatmap(hm, threshold)
        heatmaps.append(hm)
        binaries.append(binary)
        conts_l.append(conts)
        gt_labels.append(gt_idx)
        paths.append(str(img_file.relative_to(DATA_DIR)).replace("\\", "/"))

    if not heatmaps:
        raise HTTPException(404, "Нет изображений для оценки.")

    batch_r = ml_classify_batch(heatmaps, binaries, config, conts_l, gt_labels, paths)
    safe    = batch_result_to_json_safe(batch_r)

    # Строим confusion matrix cluster → gt_type
    all_cl, all_gt = [], []
    for r in batch_r["per_image"]:
        gt = r.get("gt_label")
        if gt is not None:
            all_cl.extend(r.get("cluster_labels", []))
            all_gt.extend([gt] * len(r.get("cluster_labels", [])))

    mapping = safe.get("overall_mapping", {})
    # Переводим cluster_label → predicted gt_type по mapping
    pred_gt = [mapping.get(str(cl), mapping.get(cl, cl)) for cl in all_cl]

    labels_used = list(range(len(DEFECT_TYPE_NAMES)))
    cm = sk_cm(all_gt, pred_gt, labels=labels_used)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)

    return {
        "labels":                    DEFECT_TYPE_NAMES,
        "confusion_matrix":          cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "overall_accuracy":          safe.get("overall_accuracy"),
        "silhouette_batch":          safe.get("silhouette_batch"),
        "method":                    method,
        "n_clusters":                safe.get("n_clusters"),
        "n_images":                  len(heatmaps),
        "n_defects":                 safe.get("n_total_defects"),
        "mapping":                   safe.get("overall_mapping", {}),
        "defect_type_names":         DEFECT_TYPE_NAMES,
    }


# ---------------------------------------------------------------------------
# Endpoint: embedding-based classification (новая архитектура)
# ---------------------------------------------------------------------------

class EmbeddingClassifyRequest(BaseModel):
    subfolder:      str   = Field(default="")
    method:         Literal["kmeans", "gmm", "hierarchical", "prototype", "linear"] = "kmeans"
    n_clusters:     Optional[int] = Field(default=None, ge=1, le=20)
    k_min:          int   = Field(default=2, ge=2)
    k_max:          int   = Field(default=8)
    pca_components: int   = Field(default=20, ge=2, le=100)
    umap_method:    Literal["umap", "tsne", "pca2"] = "umap"
    compute_umap:   bool  = Field(default=True)
    n_prototypes:   int   = Field(default=3, ge=1, le=20)
    test_size:      float = Field(default=0.5, ge=0.1, le=0.9)
    threshold:      float = Field(default=0.6, ge=0.0, le=1.0)
    S_min:          float = Field(default=30.0, ge=0)
    S_max:          float = Field(default=500.0, gt=0)
    max_images:     int   = Field(default=50, ge=1, le=300)
    workers:        int   = Field(default=4, ge=1, le=16)
    mode:           Literal[
        "sequential", "parallel_cpu", "parallel_images", "parallel_gpu",
        "parallel_images_gpu", "parallel_combined",
        "parallel_combined_gpu", "parallel_batch_embed",
    ] = "sequential"
    include_good:   bool  = Field(default=False)


@app.post("/classify_embeddings")
def classify_embeddings_endpoint(req: EmbeddingClassifyRequest):
    """
    Новая архитектура классификации дефектов на основе патч-эмбеддингов backbone.

    Поток:
      1. GPU: runner.infer_with_embeddings() → heatmap + [784, 1536] raw features
      2. CPU: extract_defect_embedding() per contour → усредн. дефект-вектор
      3. PCA(20) → KMeans/GMM/Hierarchical
      4. UMAP(2D) для визуализации
      5. Hungarian accuracy vs GT labels

    Поддерживает все 8 режимов параллельной обработки для шага классификации.
    GPU-инференс всегда последовательный (единственный GPU).
    """
    def _generate():
        try:
            search_root = DATA_DIR / req.subfolder if req.subfolder else DATA_DIR
            if not search_root.exists():
                yield json.dumps({"type": "error", "detail": f"Нет директории: {req.subfolder}"}) + "\n"
                return

            all_files: List[Path] = sorted(
                f for f in search_root.rglob("*")
                if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
            )
            if not req.include_good:
                all_files = [f for f in all_files
                             if f.relative_to(DATA_DIR).parts[0] != "good"]
            files = all_files[: req.max_images]

            if not files:
                yield json.dumps({"type": "error", "detail": "Нет изображений."}) + "\n"
                return

            yield json.dumps({
                "type": "start", "total": len(files),
                "mode": req.mode, "method": req.method,
                "umap_available": UMAP_AVAILABLE,
                "pca_components": req.pca_components,
            }) + "\n"

            # ── Шаг 1: GPU-инференс (всегда последовательный) ───────────────
            t_infer = time.perf_counter()
            image_data: List[dict] = []

            for img_file in files:
                rel = str(img_file.relative_to(DATA_DIR)).replace("\\", "/")
                img = _read_image(img_file)
                if img is None:
                    continue

                hm, score, patch_embs, sp_shape = runner.infer_with_embeddings(img)
                conts, binary = extract_defects_from_heatmap(hm, req.threshold)
                # Фильтрация по S_min / S_max через диаметр
                conts_filtered = []
                for c in conts:
                    area = float(cv2.contourArea(c))
                    diam = math.sqrt(4 * area / math.pi)
                    if req.S_min <= diam <= req.S_max:
                        conts_filtered.append(c)

                gt = _get_gt_label(rel)
                image_data.append({
                    "rel": rel, "img": img,
                    "hm": hm, "score": score,
                    "patch_embs": patch_embs,
                    "sp_shape":   sp_shape,
                    "orig_shape": (img.shape[0], img.shape[1]),
                    "contours":   conts_filtered,
                    "gt": gt,
                })

            infer_ms = round((time.perf_counter() - t_infer) * 1000, 2)
            yield json.dumps({"type": "infer_done", "n_images": len(image_data),
                              "infer_ms": infer_ms}) + "\n"

            if not image_data:
                yield json.dumps({"type": "error", "detail": "Нет изображений после фильтрации."}) + "\n"
                return

            config = {
                "method":         req.method,
                "n_clusters":     req.n_clusters,
                "k_range":        [req.k_min, req.k_max],
                "pca_components": req.pca_components,
                "umap_method":    req.umap_method,
                "n_prototypes":   req.n_prototypes,
                "test_size":      req.test_size,
            }

            # ── Шаг 2: Классификация по эмбеддингам ─────────────────────────
            t_cls = time.perf_counter()

            patch_embs_list = [d["patch_embs"] for d in image_data]
            spatial_shapes  = [d["sp_shape"]   for d in image_data]
            orig_shapes     = [d["orig_shape"]  for d in image_data]
            contours_list   = [d["contours"]    for d in image_data]
            gt_labels       = [d["gt"]          for d in image_data]
            paths           = [d["rel"]         for d in image_data]

            # ── Batch и Thread-режимы: classify_by_embeddings на всём батче ──
            if req.mode in ("parallel_combined_gpu", "parallel_batch_embed",
                            "parallel_images", "parallel_images_gpu",
                            "parallel_combined", "sequential",
                            "parallel_gpu"):
                result = classify_by_embeddings(
                    patch_embs_list, spatial_shapes, orig_shapes,
                    contours_list, config, gt_labels, paths,
                    compute_umap=req.compute_umap,
                )

            # ── ProcessPool: fit на подвыборке → параллельный predict ────────
            elif req.mode == "parallel_cpu":
                import pickle as _pk
                # Fit PCA + clusterer на первых min(10, N) изображениях
                n_fit = min(10, len(image_data))
                fit_result = classify_by_embeddings(
                    patch_embs_list[:n_fit], spatial_shapes[:n_fit],
                    orig_shapes[:n_fit], contours_list[:n_fit],
                    config, gt_labels[:n_fit], paths[:n_fit],
                    compute_umap=False,
                )
                pca_b       = _pk.dumps(fit_result["fitted_pca"])      if fit_result["fitted_pca"]      else None
                scaler_b    = _pk.dumps(fit_result["fitted_scaler"])   if fit_result["fitted_scaler"]   else None
                clusterer_b = _pk.dumps(fit_result["fitted_clusterer"])if fit_result["fitted_clusterer"]else None
                n_cl        = fit_result["n_clusters"]

                args_list = [
                    (
                        image_data[i]["patch_embs"].tobytes(),
                        image_data[i]["patch_embs"].shape,
                        image_data[i]["sp_shape"],
                        image_data[i]["orig_shape"],
                        [c.tolist() for c in image_data[i]["contours"]],
                        pca_b, scaler_b, clusterer_b,
                        req.method, n_cl,
                        image_data[i]["gt"], image_data[i]["rel"],
                    )
                    for i in range(len(image_data))
                ]
                pool = get_or_create_pool(req.workers)
                per_img_results = list(pool.map(_emb_process_worker, args_list))

                # Восстанавливаем result-структуру
                result = {**fit_result, "per_image": per_img_results,
                          "n_images": len(image_data)}
                # Пересчитываем глобальную accuracy по всем изображениям
                all_cl, all_gt = [], []
                for r_i in per_img_results:
                    gt_i = r_i.get("gt_label")
                    if gt_i is not None:
                        all_cl.extend(r_i.get("cluster_labels", []))
                        all_gt.extend([int(gt_i)] * len(r_i.get("cluster_labels", [])))
                if all_cl and all_gt:
                    acc_all, map_all = emb_hungarian_accuracy(
                        np.array(all_cl), np.array(all_gt),
                        n_cl, len(DEFECT_TYPE_NAMES)
                    )
                    result["overall_accuracy"] = acc_all
                    result["mapping"] = map_all

                # UMAP на полном батче
                if req.compute_umap and fit_result["fitted_pca"] is not None:
                    emb_all, meta_all = emb_collect_batch(
                        patch_embs_list, spatial_shapes, orig_shapes,
                        contours_list, gt_labels, paths,
                    )
                    if len(emb_all) > 1:
                        coords = compute_2d_projection(emb_all, fit_result["fitted_pca"],
                                                       req.umap_method)
                        if coords is not None:
                            all_labels_flat = []
                            for r_i in per_img_results:
                                all_labels_flat.extend(r_i.get("cluster_labels", []))
                            if len(all_labels_flat) == len(emb_all):
                                result["umap_points"] = [
                                    {
                                        "x": float(coords[i, 0]),
                                        "y": float(coords[i, 1]),
                                        "cluster":    int(all_labels_flat[i]),
                                        "gt_label":   meta_all[i]["gt_label"],
                                        "gt_name":    DEFECT_TYPE_NAMES[meta_all[i]["gt_label"]]
                                                      if meta_all[i]["gt_label"] is not None else None,
                                        "image_path": meta_all[i]["image_path"],
                                        "area":       meta_all[i]["area"],
                                    }
                                    for i in range(len(emb_all))
                                ]
            else:
                result = classify_by_embeddings(
                    patch_embs_list, spatial_shapes, orig_shapes,
                    contours_list, config, gt_labels, paths,
                    compute_umap=req.compute_umap,
                )

            cls_ms = round((time.perf_counter() - t_cls) * 1000, 2)

            # ── Стримим per-image результаты ─────────────────────────────────
            for per in result.get("per_image", []):
                yield json.dumps({"type": "image", "result": per}) + "\n"

            # ── Финальный summary ─────────────────────────────────────────────
            safe = emb_result_to_json_safe(result)
            safe.pop("per_image", None)
            yield json.dumps({
                "type":          "done",
                "infer_ms":      infer_ms,
                "classify_ms":   cls_ms,
                "total_ms":      round(infer_ms + cls_ms, 2),
                "mode":          req.mode,
                **safe,
            }) + "\n"

        except Exception as e:
            logger.exception("classify_embeddings error")
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# Comprehensive Benchmark: все методы × все 8 режимов
# ---------------------------------------------------------------------------

class BenchmarkComprehensiveRequest(BaseModel):
    subfolder:      str   = Field(default="")
    max_images:     int   = Field(default=20, ge=1, le=500)
    threshold:      float = Field(default=0.6, ge=0.0, le=1.0)
    S_min:          float = Field(default=30.0, ge=0)
    S_max:          float = Field(default=500.0, gt=0)
    n_clusters:     int   = Field(default=5, ge=2, le=20)
    pca_components: int   = Field(default=20, ge=2, le=100)
    n_prototypes:   int   = Field(default=3, ge=1, le=10)
    workers:        int   = Field(default=4, ge=1, le=16)
    include_good:   bool  = Field(default=False)
    methods:        Optional[List[str]] = Field(default=None,
        description="Список методов. None = все 5.")


_ALL_CLS_METHODS = ["kmeans", "gmm", "hierarchical", "prototype", "linear"]
_GEO_METHODS     = {"kmeans", "gmm", "hierarchical"}
_EMB_METHODS     = {"prototype", "linear"}


@app.post("/benchmark_comprehensive")
def benchmark_comprehensive(req: BenchmarkComprehensiveRequest):
    """
    Запускает все методы классификации × все 8 режимов параллельной обработки.

    Для каждой комбинации (метод, режим) измеряет только время КЛАССИФИКАЦИИ
    (без GPU-инференса — он вычисляется один раз заранее).

    Geometry методы (kmeans/gmm/hierarchical): извлечение признаков + кластеризация.
    Embedding методы (prototype/linear): PCA + predict (эмбеддинги уже предвычислены).

    Возвращает NDJSON-поток прогресса и финальную матрицу результатов.
    """
    def _generate():
        try:
            search_root = DATA_DIR / req.subfolder if req.subfolder else DATA_DIR
            if not search_root.exists():
                yield json.dumps({"type": "error", "detail": f"Нет директории: {req.subfolder}"}) + "\n"
                return

            all_files: List[Path] = sorted(
                f for f in search_root.rglob("*")
                if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
            )
            if not req.include_good:
                all_files = [f for f in all_files
                             if f.relative_to(DATA_DIR).parts[0] != "good"]
            files = all_files[: req.max_images]
            if not files:
                yield json.dumps({"type": "error", "detail": "Нет изображений."}) + "\n"
                return

            methods_to_run = req.methods or _ALL_CLS_METHODS
            n_combos = len(methods_to_run) * len(ALL_MODES)

            yield json.dumps({
                "type":    "start",
                "n_images": len(files),
                "methods": methods_to_run,
                "modes":   ALL_MODES,
                "n_combos": n_combos,
            }) + "\n"

            # ── Предвычисление: инференс GPU один раз ────────────────────────
            t_infer = time.perf_counter()
            image_data: List[dict] = []
            for img_file in files:
                rel = str(img_file.relative_to(DATA_DIR)).replace("\\", "/")
                img = _read_image(img_file)
                if img is None:
                    continue
                hm, score, patch_embs, sp_shape = runner.infer_with_embeddings(img)
                conts, binary = extract_defects_from_heatmap(hm, req.threshold)
                conts_filtered = [c for c in conts
                                  if req.S_min <= math.sqrt(4*float(cv2.contourArea(c))/math.pi) <= req.S_max]
                gt = _get_gt_label(rel)
                image_data.append({
                    "rel": rel, "img": img, "hm": hm, "binary": binary,
                    "patch_embs": patch_embs, "sp_shape": sp_shape,
                    "orig_shape": (img.shape[0], img.shape[1]),
                    "contours": conts_filtered, "gt": gt,
                })
            infer_ms = round((time.perf_counter() - t_infer) * 1000, 2)
            yield json.dumps({"type": "infer_done", "infer_ms": infer_ms,
                              "n_images": len(image_data)}) + "\n"

            if not image_data:
                yield json.dumps({"type": "error", "detail": "Нет изображений после фильтрации."}) + "\n"
                return

            # Подготовка данных для embedding методов (общая для всех режимов)
            patch_embs_list = [d["patch_embs"] for d in image_data]
            spatial_shapes  = [d["sp_shape"]   for d in image_data]
            orig_shapes     = [d["orig_shape"]  for d in image_data]
            contours_list   = [d["contours"]    for d in image_data]
            gt_labels       = [d["gt"]          for d in image_data]
            paths           = [d["rel"]         for d in image_data]
            heatmaps        = [d["hm"]          for d in image_data]
            binaries        = [d["binary"]      for d in image_data]

            # Geo данные: для каждого изображения — contours + anomaly_map
            geo_config_base = _default_classify_config(
                n_clusters=req.n_clusters, threshold=req.threshold,
                s_min_percentile=5.0, s_max_percentile=95.0,
            )

            emb_config_base = {
                "n_clusters":    req.n_clusters,
                "k_range":       [2, req.n_clusters + 2],
                "pca_components": req.pca_components,
                "n_prototypes":  req.n_prototypes,
                "test_size":     0.5,
                "umap_method":   "pca2",  # быстро для benchmark
            }

            # Справочник: режим sequential → время (для расчёта speedup)
            seq_times: Dict[str, float] = {}  # method → seq_time_ms
            matrix: Dict[str, Dict[str, float]] = {m: {} for m in methods_to_run}

            done_count = 0

            for method in methods_to_run:
                is_emb = method in _EMB_METHODS
                emb_cfg = {**emb_config_base, "method": method}
                geo_cfg = {**geo_config_base, "method": method if not is_emb else "kmeans"}

                for bmode in ALL_MODES:
                    tb = time.perf_counter()
                    n_defects_bm = 0

                    try:
                        if is_emb:
                            # ── Embedding методы ───────────────────────────
                            # Режим не меняет алгоритм существенно — используем batch
                            if bmode in ("parallel_combined_gpu", "parallel_batch_embed"):
                                r = classify_by_embeddings(
                                    patch_embs_list, spatial_shapes, orig_shapes,
                                    contours_list, emb_cfg, gt_labels, paths,
                                    compute_umap=False,
                                )
                            elif bmode in ("parallel_images", "parallel_images_gpu",
                                           "parallel_combined"):
                                # ThreadPool для per-image predict
                                from defect_embedding_classifier import (
                                    fit_pca_and_cluster as _fit_pac,
                                    collect_batch_embeddings as _cbe,
                                    predict_in_pca_space as _pips,
                                )
                                emb_all, _ = _cbe(patch_embs_list, spatial_shapes,
                                                  orig_shapes, contours_list, gt_labels)
                                from sklearn.decomposition import PCA as _PCA
                                from sklearn.preprocessing import StandardScaler as _SS
                                _pca = _PCA(n_components=min(req.pca_components,
                                                            max(2, len(emb_all)-1))).fit(emb_all)
                                _sc  = _SS().fit(_pca.transform(emb_all))
                                with ThreadPoolExecutor(max_workers=req.workers) as ex:
                                    list(ex.map(
                                        lambda a: classify_single_by_embeddings(*a),
                                        [(patch_embs_list[i], spatial_shapes[i], orig_shapes[i],
                                          contours_list[i], _pca, _sc, None, method,
                                          req.n_clusters, gt_labels[i], paths[i])
                                         for i in range(len(image_data))]
                                    ))
                            elif bmode == "parallel_cpu":
                                import pickle as _pk
                                r0 = classify_by_embeddings(
                                    patch_embs_list[:min(5, len(patch_embs_list))],
                                    spatial_shapes[:5], orig_shapes[:5],
                                    contours_list[:5], emb_cfg, gt_labels[:5],
                                    compute_umap=False,
                                )
                                pca_b = _pk.dumps(r0["fitted_pca"]) if r0["fitted_pca"] else None
                                sc_b  = _pk.dumps(r0["fitted_scaler"]) if r0["fitted_scaler"] else None
                                cl_b  = _pk.dumps(r0["fitted_clusterer"]) if r0["fitted_clusterer"] else None
                                args_bm = [
                                    (patch_embs_list[i].tobytes(), patch_embs_list[i].shape,
                                     spatial_shapes[i], orig_shapes[i],
                                     [c.tolist() for c in contours_list[i]],
                                     pca_b, sc_b, cl_b, method, req.n_clusters,
                                     gt_labels[i], paths[i])
                                    for i in range(len(image_data))
                                ]
                                pool = get_or_create_pool(req.workers)
                                list(pool.map(_emb_process_worker, args_bm))
                            else:
                                r = classify_by_embeddings(
                                    patch_embs_list, spatial_shapes, orig_shapes,
                                    contours_list, emb_cfg, gt_labels, paths,
                                    compute_umap=False,
                                )
                                n_defects_bm = r.get("n_defects", 0)

                        else:
                            # ── Geometry методы ────────────────────────────
                            if bmode in ("parallel_combined_gpu", "parallel_batch_embed"):
                                br = ml_classify_batch(heatmaps, binaries, geo_cfg,
                                                       gt_labels=gt_labels, image_paths=paths)
                                n_defects_bm = br.get("n_total_defects", 0)
                            elif bmode in ("parallel_images", "parallel_images_gpu",
                                           "parallel_combined"):
                                pre = ml_classify_batch(
                                    heatmaps[:min(5, len(heatmaps))],
                                    binaries[:min(5, len(binaries))],
                                    geo_cfg,
                                )
                                fs = (pre.get("fitted_scaler"), pre.get("fitted_model"))
                                with ThreadPoolExecutor(max_workers=req.workers) as ex:
                                    list(ex.map(
                                        lambda a: ml_classify_defects(*a),
                                        [(hm, bm, geo_cfg, c, gt, fs)
                                         for hm, bm, c, gt in
                                         zip(heatmaps, binaries, contours_list, gt_labels)]
                                    ))
                            elif bmode == "parallel_cpu":
                                import pickle as _pk
                                pre = ml_classify_batch(
                                    heatmaps[:min(5, len(heatmaps))],
                                    binaries[:min(5, len(binaries))], geo_cfg,
                                )
                                fs_b = _pk.dumps((pre.get("fitted_scaler"),
                                                  pre.get("fitted_model")))
                                args_g = [
                                    (hm.tobytes(), hm.shape, bm.tobytes(), bm.shape,
                                     geo_cfg, gt, fs_b, rel)
                                    for hm, bm, gt, rel in
                                    zip(heatmaps, binaries, gt_labels, paths)
                                ]
                                pool = get_or_create_pool(req.workers)
                                list(pool.map(_classify_process_worker, args_g))
                            else:
                                for hm, bm, c, gt in zip(heatmaps, binaries,
                                                          contours_list, gt_labels):
                                    r_i = ml_classify_defects(hm, bm, geo_cfg, c, gt)
                                    n_defects_bm += r_i.get("n_valid", 0)

                    except Exception as bex:
                        logger.warning("benchmark_comprehensive [%s][%s]: %s",
                                       method, bmode, bex)

                    elapsed_bm = round((time.perf_counter() - tb) * 1000, 2)
                    if bmode == "sequential":
                        seq_times[method] = elapsed_bm

                    speedup = round(seq_times.get(method, elapsed_bm) / max(elapsed_bm, 0.1), 3) \
                              if bmode != "sequential" else 1.0

                    matrix[method][bmode] = elapsed_bm
                    done_count += 1

                    yield json.dumps({
                        "type":      "result",
                        "method":    method,
                        "mode":      bmode,
                        "mode_label": MODE_LABELS[bmode],
                        "time_ms":   elapsed_bm,
                        "speedup":   speedup,
                        "n_defects": n_defects_bm,
                        "done":      done_count,
                        "total":     n_combos,
                    }) + "\n"

            yield json.dumps({
                "type":    "done",
                "matrix":  matrix,
                "methods": methods_to_run,
                "modes":   ALL_MODES,
                "mode_labels": {m: MODE_LABELS[m] for m in ALL_MODES},
                "n_images": len(image_data),
                "infer_ms": infer_ms,
            }) + "\n"

        except Exception as e:
            logger.exception("benchmark_comprehensive error")
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


@app.get("/classify_methods")
def get_classify_methods():
    """Возвращает доступные методы классификации и признаки."""
    return {
        "methods": [
            {"id": "kmeans",       "label": "K-Means",              "description": "Быстрый, хорошо масштабируется, предполагает сферические кластеры"},
            {"id": "gmm",          "label": "GMM (Gaussian Mixture)","description": "Мягкое присвоение, лучше при перекрытии кластеров"},
            {"id": "hierarchical", "label": "Hierarchical (Ward)",   "description": "Без предположения о форме, дендрограмма"},
            {"id": "prototype",    "label": "Prototype (few-shot)",  "description": "Косинусное сходство с прототипами классов"},
            {"id": "linear",       "label": "Linear (upper bound)",  "description": "Логистическая регрессия, теоретический потолок"},
            {"id": "knn",          "label": "KNN",                   "description": "K ближайших соседей в PCA-пространстве"},
            {"id": "rf",           "label": "Random Forest",         "description": "Ансамбль деревьев, feature importance"},
            {"id": "dt",           "label": "Decision Tree",         "description": "Интерпретируемое дерево решений"},
        ],
        "feature_names": FEATURE_NAMES,
        "defect_types":  DEFECT_TYPE_NAMES,
        "all_modes":     [{"id": m, "label": MODE_LABELS[m]} for m in ALL_MODES],
    }


# ---------------------------------------------------------------------------
# RESEARCH / JOINT OPTIMIZATION
# ---------------------------------------------------------------------------

class ResearchGridRequest(BaseModel):
    subfolder:    str   = Field(default="", description="Поддиректория или '' для всех")
    max_images:   int   = Field(default=30, ge=1, le=500)
    thresholds:   List[float] = Field(default=[0.70, 0.80, 0.85, 0.90, 0.92, 0.95])
    pca_values:   List[int]   = Field(default=[20, 35, 50, 75, 100])
    methods:      List[str]   = Field(default=["linear", "knn", "rf", "dt", "kmeans"])
    s_min:        float = Field(default=40.0, ge=0)
    s_max:        float = Field(default=1500.0, gt=0)
    k_param:      int   = Field(default=5, ge=1, le=21)
    n_neighbors:  int   = Field(default=5, ge=1, le=21)
    knn_metric:   str   = Field(default="euclidean")
    n_estimators: int   = Field(default=100, ge=10, le=500)
    max_depth:    Optional[int] = Field(default=None)
    random_state: int   = Field(default=42)
    use_cache:    bool  = Field(default=True)


class ResearchFullGridRequest(BaseModel):
    subfolder:    str  = Field(default="")
    max_images:   int  = Field(default=30, ge=1, le=500)
    # Detection grid — each list is a sweep axis
    blur_values:        List[int]   = Field(default=[0, 5, 7])
    pct_clip_values:    List[float] = Field(default=[90.0, 95.0, 99.0])
    vmin_values:        List[float] = Field(default=[0.0, 0.10, 0.20])
    gamma_values:       List[float] = Field(default=[1.0, 2.0, 3.0])
    bin_thr_values:     List[float] = Field(default=[15.0, 25.0, 40.0])
    smin_values:        List[int]   = Field(default=[10, 40])
    smax_values:        List[int]   = Field(default=[200_000, 600_000, 1_000_000])
    # Detection fixed params
    use_edge_mask:      bool  = Field(default=True)
    k_morph:            int   = Field(default=5, ge=1)
    use_baseline_sub:   bool  = Field(default=False)
    baseline_alpha:     float = Field(default=1.0, ge=0.0, le=3.0)
    use_comp_sup:       bool  = Field(default=False)
    suppression_factor: float = Field(default=0.3, ge=0.05, le=1.0)
    min_comp_area:      int   = Field(default=0, ge=0)
    # Classification grid
    pca_values:   List[int]  = Field(default=[20, 50, 100])
    methods:      List[str]  = Field(default=["linear", "knn", "rf"])
    n_neighbors:  int        = Field(default=5, ge=1, le=21)
    knn_metric:   str        = Field(default="euclidean")
    n_estimators: int        = Field(default=100, ge=10, le=500)
    max_depth:    Optional[int] = Field(default=None)
    random_state: int        = Field(default=42)
    use_cache:    bool       = Field(default=True)
    max_combinations: int    = Field(default=1000, ge=10, le=10000)


class ResearchStabilityRequest(BaseModel):
    subfolder:  str        = Field(default="")
    max_images: int        = Field(default=30, ge=1, le=200)
    configs:    List[Dict] = Field(...)
    n_repeats:  int        = Field(default=3, ge=2, le=10)


class ResearchSingleRequest(BaseModel):
    subfolder:    str   = Field(default="")
    max_images:   int   = Field(default=30, ge=1, le=500)
    threshold:    float = Field(default=0.85)
    pca_n:        int   = Field(default=35, ge=2, le=200)
    method:       str   = Field(default="linear")
    s_min:        float = Field(default=40.0, ge=0)
    s_max:        float = Field(default=1500.0, gt=0)
    k_param:      int   = Field(default=5, ge=1, le=21)
    n_neighbors:  int   = Field(default=5, ge=1, le=21)
    knn_metric:   str   = Field(default="euclidean")
    n_estimators: int   = Field(default=100, ge=10, le=500)
    max_depth:    Optional[int] = Field(default=None)
    random_state: int   = Field(default=42)


def _collect_images_for_research(subfolder: str, max_images: int) -> List[Path]:
    """Собирает изображения для research endpoints."""
    if subfolder:
        folder = DATA_DIR / subfolder
        files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in ALLOWED_EXTENSIONS)
    else:
        files = sorted(f for f in DATA_DIR.rglob("*") if f.suffix.lower() in ALLOWED_EXTENSIONS)
    return files[:max_images]


@app.post("/research/grid_search")
async def research_grid_search(req: ResearchGridRequest):
    """
    Streaming grid search: Threshold × PCA × Method.
    Возвращает NDJSON-поток с событиями прогресса и результатами.
    """
    async def _generate():
        try:
            image_files = _collect_images_for_research(req.subfolder, req.max_images)
            if not image_files:
                yield json.dumps({"type": "error", "detail": "Нет изображений"}) + "\n"
                return

            grid = {
                "thresholds":   req.thresholds,
                "pca_values":   req.pca_values,
                "methods":      req.methods,
                "s_min":        req.s_min,
                "s_max":        req.s_max,
                "k_param":      req.k_param,
                "n_neighbors":  req.n_neighbors,
                "knn_metric":   req.knn_metric,
                "n_estimators": req.n_estimators,
                "max_depth":    req.max_depth,
            }

            for event in gs_run_grid(
                runner, image_files, GT_DIR, grid,
                use_cache=req.use_cache,
                random_state=req.random_state,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"

        except Exception as e:
            logger.exception("research/grid_search error")
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


@app.get("/research/results")
def research_results():
    """Возвращает все кэшированные результаты grid search."""
    results = gs_get_results()
    return {"results": results, "n": len(results)}


@app.delete("/research/cache")
def research_clear_cache():
    """Очищает кэш grid search."""
    n = gs_clear_cache()
    return {"deleted": n, "message": f"Удалено {n} записей из кэша"}


@app.post("/research/full_grid")
async def research_full_grid(req: ResearchFullGridRequest):
    """
    Streaming full grid search: LC detection pipeline × embedding classification.
    Перебирает все LC-параметры детекции совместно с PCA/методом классификации.
    """
    async def _generate():
        try:
            image_files = _collect_images_for_research(req.subfolder, req.max_images)
            if not image_files:
                yield json.dumps({"type": "error", "detail": "Нет изображений"}) + "\n"
                return

            _baseline = cal_load_baseline(CAL_BASELINE_PATH) if req.use_baseline_sub else None

            for event in gs_run_full_grid(
                runner, image_files, GT_DIR,
                blur_values=req.blur_values,
                pct_clip_values=req.pct_clip_values,
                vmin_values=req.vmin_values,
                gamma_values=req.gamma_values,
                bin_thr_values=req.bin_thr_values,
                smin_values=req.smin_values,
                smax_values=req.smax_values,
                use_edge_mask=req.use_edge_mask,
                k_morph=req.k_morph,
                baseline_map=_baseline,
                baseline_alpha=req.baseline_alpha,
                use_baseline_sub=req.use_baseline_sub,
                use_comp_sup=req.use_comp_sup,
                suppression_factor=req.suppression_factor,
                min_comp_area=req.min_comp_area,
                pca_values=req.pca_values,
                methods=req.methods,
                n_neighbors=req.n_neighbors,
                knn_metric=req.knn_metric,
                n_estimators=req.n_estimators,
                max_depth=req.max_depth,
                random_state=req.random_state,
                use_cache=req.use_cache,
                max_combinations=req.max_combinations,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"

        except Exception as e:
            logger.exception("research/full_grid error")
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


@app.get("/research/full_grid/results")
def research_full_grid_results():
    """Возвращает все кэшированные результаты full grid search."""
    results = gs_get_full_grid_results()
    return {"results": results, "n": len(results)}


@app.delete("/research/full_grid/cache")
def research_full_grid_clear():
    """Очищает кэш full grid search."""
    n = gs_clear_full_grid_cache()
    return {"deleted": n, "message": f"Удалено {n} записей из full grid кэша"}


@app.post("/research/stability")
async def research_stability(req: ResearchStabilityRequest):
    """Проверяет стабильность топ-N конфигураций (3 повтора)."""
    image_files = _collect_images_for_research(req.subfolder, req.max_images)
    if not image_files:
        raise HTTPException(400, "Нет изображений")

    results = gs_stability(runner, image_files, GT_DIR, req.configs, n_repeats=req.n_repeats)
    return {"stability": results, "n_configs": len(results)}


@app.post("/research/single")
async def research_single(req: ResearchSingleRequest):
    """Запускает один эксперимент (одна точка сетки)."""
    image_files = _collect_images_for_research(req.subfolder, req.max_images)
    if not image_files:
        raise HTTPException(400, "Нет изображений")

    precomp = gs_precompute(runner, image_files, GT_DIR)
    result = gs_single_exp(
        precomp,
        threshold=req.threshold,
        pca_n=req.pca_n,
        method=req.method,
        s_min=req.s_min,
        s_max=req.s_max,
        k_param=req.k_param,
        n_neighbors=req.n_neighbors,
        knn_metric=req.knn_metric,
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        random_state=req.random_state,
        compute_umap=False,
    )
    # Удаляем не-JSON объекты если вдруг попали
    result.pop("fitted_pca", None)
    result.pop("fitted_scaler", None)
    result.pop("fitted_clusterer", None)
    return result


@app.post("/research/area_correlation")
async def research_area_correlation(req: ResearchSingleRequest):
    """Корреляция площадь дефекта vs точность классификации."""
    image_files = _collect_images_for_research(req.subfolder, req.max_images)
    if not image_files:
        raise HTTPException(400, "Нет изображений")

    result = gs_area_corr(
        runner, image_files, GT_DIR,
        threshold=req.threshold,
        pca_n=req.pca_n,
        method=req.method,
        s_min=req.s_min,
        s_max=req.s_max,
    )
    return result


@app.post("/research/analyze_config")
async def research_analyze_config(req: ResearchSingleRequest):
    """
    Детальный анализ одной конфигурации:
    confusion_matrix [5×5], per_class_accuracy, class_distribution, per_class_n_test.
    """
    image_files = _collect_images_for_research(req.subfolder, req.max_images)
    if not image_files:
        raise HTTPException(400, "Нет изображений")

    precomp = gs_precompute(runner, image_files, GT_DIR)
    result = gs_config_detailed(
        precomp,
        threshold=req.threshold,
        pca_n=req.pca_n,
        method=req.method,
        s_min=req.s_min,
        s_max=req.s_max,
        k_param=req.k_param,
        n_neighbors=req.n_neighbors,
        knn_metric=req.knn_metric,
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        random_state=req.random_state,
    )
    return result


class ResearchStabilityExtRequest(BaseModel):
    subfolder:  str        = Field(default="")
    max_images: int        = Field(default=30, ge=1, le=200)
    configs:    List[Dict] = Field(...)
    n_repeats:  int        = Field(default=10, ge=2, le=20)


@app.post("/research/stability_10")
async def research_stability_10(req: ResearchStabilityExtRequest):
    """
    Расширенная проверка стабильности: n_repeats запусков, confusion matrix per run.
    Если std > 10% — помечает результат как нестабильный.
    """
    image_files = _collect_images_for_research(req.subfolder, req.max_images)
    if not image_files:
        raise HTTPException(400, "Нет изображений")

    results = gs_stability_ext(
        runner, image_files, GT_DIR,
        configs=req.configs,
        n_repeats=req.n_repeats,
    )
    return {"stability": results, "n_configs": len(results)}


class ResearchProblemPairsRequest(BaseModel):
    results: List[Dict] = Field(...)
    top_n:   int        = Field(default=5, ge=1, le=15)


@app.post("/research/problem_pairs")
async def research_problem_pairs(req: ResearchProblemPairsRequest):
    """
    Агрегирует confusion matrices из переданных результатов.
    Возвращает топ-N проблемных пар классов.
    """
    pairs = gs_problem_pairs(req.results, top_n=req.top_n)
    return {
        "pairs": pairs,
        "class_names": GS_DEFECT_NAMES,
        "n_matrices": sum(1 for r in req.results if r.get("confusion_matrix")),
    }


# ---------------------------------------------------------------------------
# Диагностика детектируемости по типам дефектов
# ---------------------------------------------------------------------------

class ResearchDetectabilityRequest(BaseModel):
    subfolder:  str         = Field(default="")
    max_images: int         = Field(default=200, ge=1, le=500)
    thresholds: List[float] = Field(
        default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]
    )
    s_min:  float = Field(default=40.0,   ge=0.0)
    s_max:  float = Field(default=1500.0, ge=1.0)
    k_param: int  = Field(default=5,      ge=1, le=15)


class ResearchRecallFprRequest(BaseModel):
    target_class: str        = Field(default="thread")
    max_images:   int        = Field(default=200, ge=1, le=500)
    max_good:     int        = Field(default=60,  ge=1, le=200)
    thresholds:   List[float] = Field(
        default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )
    s_max_values: List[float] = Field(
        default=[1_000, 5_000, 10_000, 50_000, 100_000, 300_000, 600_000, 1_000_000]
    )
    s_min:   float = Field(default=5.0, ge=0.0)
    k_param: int   = Field(default=5, ge=1, le=15)


@app.post("/research/recall_fpr")
async def research_recall_fpr(req: ResearchRecallFprRequest):
    """
    ROC-like кривая: Recall(target_class) vs FPR(good) по сетке (threshold × S_max).

    Анализирует:
    - Baseline FPR на папке good/ при текущем S_max=1500
    - Shape stats (solidity, compactness) дефекта vs good контуров
    - Sweep S_max: как меняется recall и FPR при увеличении S_max
    """
    defect_files = _collect_images_for_research("", req.max_images)
    if not defect_files:
        raise HTTPException(400, "Нет изображений с дефектами")

    good_dir   = DATA_DIR / "good"
    good_files = sorted(
        p for p in good_dir.iterdir()
        if p.suffix.lower() in ALLOWED_EXTENSIONS
    )[:req.max_good] if good_dir.is_dir() else []
    if not good_files:
        raise HTTPException(400, "Папка good/ пуста или не найдена")

    result = gs_recall_fpr(
        runner       = runner,
        defect_files = defect_files,
        good_files   = good_files,
        target_class = req.target_class,
        thresholds   = req.thresholds,
        s_max_values = req.s_max_values,
        s_min        = req.s_min,
        k_param      = req.k_param,
    )
    return result


@app.post("/research/detectability")
async def research_detectability(req: ResearchDetectabilityRequest):
    """
    Детектируемость по типам дефектов.

    Для каждого класса и каждого threshold вычисляет:
    - % изображений с ≥1 обнаружением до S_min фильтра (raw)
    - % изображений с ≥1 обнаружением после S_min/S_max фильтра (filtered)
    - Статистики аномальных scores, площадей и аспект-отношений контуров

    Позволяет диагностировать причины нулевой детектируемости:
    тонкие контуры (area<S_min, perimeter>>S_min) vs низкие аномальные scores.
    """
    image_files = _collect_images_for_research(req.subfolder, req.max_images)
    if not image_files:
        raise HTTPException(400, "Нет изображений")

    result = gs_diagnose_detectability(
        runner=runner,
        image_files=image_files,
        thresholds=req.thresholds,
        s_min=req.s_min,
        s_max=req.s_max,
        k_param=req.k_param,
    )
    return result


# ---------------------------------------------------------------------------
# Live Heatmap Calibration endpoints
# ---------------------------------------------------------------------------

class HeatmapCalibrateRequest(BaseModel):
    vmin_floor:                float = Field(default=0.0,       ge=0.0,  le=0.5)
    gamma:                     float = Field(default=1.0,       ge=0.5,  le=6.0)
    binary_threshold:          float = Field(default=20.0,      ge=0.0,  le=100.0)
    s_min:                     int   = Field(default=40,        ge=0)
    s_max:                     int   = Field(default=1_500_000, ge=1)
    use_edge_mask:             bool  = Field(default=True)
    k_morph:                   int   = Field(default=5,         ge=0,    le=15)
    blur_kernel:               int   = Field(default=0,         ge=0,    le=15)
    percentile_clip:           float = Field(default=95.0,      ge=50.0, le=99.9)
    img_idx:                   int   = Field(default=0,         ge=0,    le=49)
    # Noise suppression (Stage A + Stage B)
    use_baseline_subtraction:  bool  = Field(default=False)
    baseline_alpha:            float = Field(default=1.0,  ge=0.0, le=3.0)
    use_component_suppression: bool  = Field(default=False)
    suppression_factor:        float = Field(default=0.3,  ge=0.05, le=1.0)
    min_component_area:        int   = Field(default=0,    ge=0)


class HeatmapAutoCalibRequest(BaseModel):
    n_calib:                   int         = Field(default=5,  ge=2, le=15)
    n_good:                    int         = Field(default=10, ge=5, le=30)
    vmin_values:               List[float] = Field(default=[0.0, 0.05, 0.10, 0.15, 0.20])
    gamma_values:              List[float] = Field(default=[1.0, 1.5, 2.0, 2.5, 3.0])
    thr_values:                List[float] = Field(default=[10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0])
    s_min_values:              List[int]   = Field(default=[5, 20, 40])
    s_max_values:              List[int]   = Field(default=[50_000, 200_000, 600_000, 1_000_000])
    blur_values:               List[int]   = Field(default=[0, 5, 7])
    pct_clip_values:           List[float] = Field(default=[90.0, 95.0, 99.0])
    use_edge_mask:             bool        = Field(default=True)
    k_morph:                   int         = Field(default=5, ge=0, le=15)
    # Phase 2: noise suppression search space
    baseline_alpha_values:     List[float] = Field(default=[0.5, 1.0, 1.5])
    suppression_factor_values: List[float] = Field(default=[0.2, 0.3, 0.5])
    min_comp_multipliers:      List[float] = Field(default=[2.0, 5.0, 10.0])


@app.post("/heatmap_calibrate")
def heatmap_calibrate(req: HeatmapCalibrateRequest):
    """
    Run calibrated visualization on 6 reference images (one per class).
    First call warms inference cache (~300ms/img × 6); subsequent calls are fast (~30ms total).
    """
    display_refs, _ = cal_pick_refs(
        DATA_DIR, n_calib=0, gt_dir=GT_DIR, img_idx=req.img_idx
    )
    if not display_refs:
        raise HTTPException(500, "Не найдены эталонные изображения в DATA_DIR")

    # Load baseline map if it exists (lightweight npy read; None → Stage A disabled)
    _baseline = cal_load_baseline(CAL_BASELINE_PATH) if req.use_baseline_subtraction else None

    results = cal_batch(
        runner                    = runner,
        ref_images                = display_refs,
        mb_ver                    = _CAL_MB_VER,
        vmin_floor                = req.vmin_floor,
        gamma                     = req.gamma,
        binary_threshold          = req.binary_threshold,
        s_min                     = req.s_min,
        s_max                     = req.s_max,
        use_edge_mask             = req.use_edge_mask,
        k_morph                   = req.k_morph,
        blur_kernel               = req.blur_kernel,
        percentile_clip           = req.percentile_clip,
        baseline_map              = _baseline,
        baseline_alpha            = req.baseline_alpha,
        use_baseline_subtraction  = req.use_baseline_subtraction,
        use_component_suppression = req.use_component_suppression,
        suppression_factor        = req.suppression_factor,
        min_component_area        = req.min_component_area,
    )

    # Summary
    fp_classes: List[str] = []
    missed_classes: List[str] = []
    correct_count = 0
    for cls_name, r in results.items():
        if "error" in r:
            continue
        verdict = r.get("verdict", "ok")
        if cls_name == "good":
            if verdict == "defect":
                fp_classes.append(cls_name)
            else:
                correct_count += 1
        else:
            if verdict == "defect":
                correct_count += 1
            else:
                missed_classes.append(cls_name)

    return {
        "results":        results,
        "correct_count":  correct_count,
        "total_classes":  len(results),
        "fp_classes":     fp_classes,
        "missed_classes": missed_classes,
        "has_fp":         len(fp_classes) > 0,
    }


@app.post("/heatmap_calibrate/baseline")
def heatmap_build_baseline(n_good: int = 10):
    """
    Build pixel-wise noise baseline (p75 of good-image raw heatmaps) and save to disk.
    Uses inference cache → no extra GPU work if images were already processed.
    Call this once before enabling use_baseline_subtraction in /heatmap_calibrate.
    """
    import numpy as _np
    _, calib_samples = cal_pick_refs(DATA_DIR, n_calib=max(n_good, 5), gt_dir=None, img_idx=0)
    good_paths = [s["path"] for s in calib_samples.get("good", [])]
    if len(good_paths) < 2:
        raise HTTPException(400, f"Недостаточно good-кадров: {len(good_paths)}, нужно ≥ 2")
    baseline = cal_compute_baseline(runner, good_paths, _CAL_MB_VER)
    cal_save_baseline(baseline, CAL_BASELINE_PATH)
    return {
        "status":   "ok",
        "n_images": len(good_paths),
        "shape":    list(baseline.shape),
        "min":      round(float(baseline.min()), 5),
        "max":      round(float(baseline.max()), 5),
        "mean":     round(float(baseline.mean()), 5),
        "p75":      round(float(_np.percentile(baseline, 75)), 5),
        "saved_to": str(CAL_BASELINE_PATH),
    }


@app.post("/heatmap_calibrate/auto")
def heatmap_auto_calibrate(req: HeatmapAutoCalibRequest):
    """
    Start two-phase grid-search auto-calibration in background.
    Phase 1: 7-parameter base grid (~2520 combos, noise suppression off).
    Phase 2: noise-suppression grid fixed to Phase 1 best params (~40 combos).
    Poll /heatmap_calibrate/auto/status for progress and result.
    """
    status = cal_auto_status()
    if status["running"]:
        return {"status": "already_running", "progress": status["progress"],
                "total": status["total"]}
    # Load baseline map for Phase 2 (None → baseline-subtraction options skipped in Phase 2)
    _baseline = cal_load_baseline(CAL_BASELINE_PATH)
    cal_auto_grid(
        runner                    = runner,
        data_dir                  = DATA_DIR,
        gt_dir                    = GT_DIR,
        mb_ver                    = _CAL_MB_VER,
        n_calib                   = req.n_calib,
        n_good                    = req.n_good,
        vmin_values               = req.vmin_values,
        gamma_values              = req.gamma_values,
        thr_values                = req.thr_values,
        s_min_values              = req.s_min_values,
        s_max_values              = req.s_max_values,
        blur_values               = req.blur_values,
        pct_clip_values           = req.pct_clip_values,
        use_edge_mask             = req.use_edge_mask,
        k_morph                   = req.k_morph,
        baseline_map              = _baseline,
        baseline_alpha_values     = req.baseline_alpha_values,
        suppression_factor_values = req.suppression_factor_values,
        min_comp_multipliers      = req.min_comp_multipliers,
    )
    return {
        "status":          "started",
        "baseline_loaded": _baseline is not None,
    }


@app.get("/heatmap_calibrate/auto/status")
def heatmap_auto_status():
    """Poll auto-calibration progress and result."""
    s = cal_auto_status()
    return {
        "running":     s["running"],
        "progress":    s["progress"],
        "total":       s["total"],
        "status":      s["status"],
        "result":      s["result"],
        "log":         s["log"],
        "best_params": s["best_params"],
        "best_score":  s["best_score"] if s["best_score"] != float("-inf") else None,
    }


@app.delete("/heatmap_calibrate/cache")
def heatmap_clear_cache():
    """Clear raw anomaly map cache (useful after reloading model)."""
    global _CAL_MB_VER
    _CAL_MB_VER = str(time.time())
    n = cal_clear_cache()
    return {"cleared": n, "new_mb_ver": _CAL_MB_VER}


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
