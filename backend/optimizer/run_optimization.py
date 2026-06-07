"""
CLI-скрипт оптимизации параметров бинаризации PatchCore heatmap.

Использование:
    python run_optimization.py
    python run_optimization.py --subfolder cut_damage --max_images 50 --workers 8
    python run_optimization.py --no_viz --save_dir results/
    python run_optimization.py --device cpu

Что делает:
    1. Загружает изображения из data/.../test/
    2. Прогоняет PatchCore inference → heatmaps
    3. Запускает grid_search (2016 комбинаций × N изображений)
    4. Сохраняет best_params.json → optimizer/best_params.json
    5. Выводит топ-10 в терминал
    6. Показывает / сохраняет 4 визуализации
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# backend/ в sys.path (для запуска из любой директории)
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from optimizer.parameter_optimizer import (
    DEFAULT_PARAM_GRID,
    find_best_params,
    grid_search,
    save_best_params,
    top_n_results,
)
from metrics_evaluator import find_gt_mask
from patchcore_runner import PatchCoreRunner

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimize PatchCore binarization parameters via grid search"
    )
    p.add_argument("--subfolder",   default="",
                   help="Subdirectory inside test/ ('' = all)")
    p.add_argument("--max_images",  type=int, default=50,
                   help="Max images to use (default: 50)")
    p.add_argument("--workers",     type=int, default=4,
                   help="ThreadPool workers for grid search (default: 4)")
    p.add_argument("--device",      default="cuda", choices=["cuda", "cpu"],
                   help="PatchCore device (default: cuda)")
    p.add_argument("--no_viz",      action="store_true",
                   help="Skip visualization")
    p.add_argument("--save_dir",    default="",
                   help="Save plots here instead of showing ('' = show)")
    p.add_argument("--model_path",  default="",
                   help="Path to patchcore.pt (auto-detected if empty)")
    return p.parse_args()


def _read_image(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ── Пути ──────────────────────────────────────────────────────────────
    base_dir   = BACKEND_DIR.parent / "data" / "carpet" / "carpet"
    data_dir   = base_dir / "test"
    gt_dir     = base_dir / "ground_truth"
    model_path = (
        Path(args.model_path) if args.model_path
        else BACKEND_DIR / "models" / "patchcore.pt"
    )

    if not model_path.exists():
        log.error("Модель не найдена: %s", model_path)
        sys.exit(1)

    search_root = data_dir / args.subfolder if args.subfolder else data_dir
    if not search_root.exists():
        log.error("Поддиректория не найдена: %s", search_root)
        sys.exit(1)

    # ── Список файлов ──────────────────────────────────────────────────────
    all_files = sorted(
        f for f in search_root.rglob("*")
        if f.is_file() and f.suffix.lower() in ALLOWED_EXT
    )[: args.max_images]

    if not all_files:
        log.error("Изображения не найдены в %s", search_root)
        sys.exit(1)

    log.info("Файлов для обработки: %d", len(all_files))

    # ── Инициализация PatchCore ────────────────────────────────────────────
    log.info("Загрузка PatchCore (%s): %s", args.device, model_path)
    runner = PatchCoreRunner(str(model_path), device=args.device)

    # ── Inference → heatmaps + GT маски ───────────────────────────────────
    log.info("Inference: генерируем heatmaps...")
    heatmaps, gt_masks = [], []
    skipped_gt = 0

    for img_file in all_files:
        img = _read_image(img_file)
        if img is None:
            log.warning("Пропуск (нечитаемо): %s", img_file.name)
            continue
        hm  = runner.infer(img)
        rel = str(img_file.relative_to(data_dir)).replace("\\", "/")
        gt  = find_gt_mask(rel, gt_dir)
        heatmaps.append(hm)
        gt_masks.append(gt)
        if gt is None:
            skipped_gt += 1

    n_with_gt = len(heatmaps) - skipped_gt
    log.info(
        "Inference завершён: %d изображений, %d с GT-маской, %d без GT",
        len(heatmaps), n_with_gt, skipped_gt,
    )

    if not heatmaps:
        log.error("Ни одного изображения не обработано")
        sys.exit(1)

    # ── Grid Search ────────────────────────────────────────────────────────
    n_combos = (
        len(DEFAULT_PARAM_GRID["threshold"])
        * len(DEFAULT_PARAM_GRID["min_area"])
        * len(DEFAULT_PARAM_GRID["morphology_kernel"])
        * len(DEFAULT_PARAM_GRID["morphology_iterations"])
        * len(DEFAULT_PARAM_GRID["gaussian_blur_kernel"])
    )
    log.info(
        "Grid search: %d комбинаций × %d изображений, workers=%d",
        n_combos, len(heatmaps), args.workers,
    )

    elapsed_ms, results = grid_search(
        heatmaps, gt_masks,
        param_grid=DEFAULT_PARAM_GRID,
        workers=args.workers,
    )
    log.info("Grid search завершён: %.0f мс (%.1f с)", elapsed_ms, elapsed_ms / 1000)

    # ── Лучшие параметры ──────────────────────────────────────────────────
    best = find_best_params(results)
    p = best["params"];  m = best["metrics"]
    log.info(
        "Лучшие параметры:\n"
        "  threshold=%s  min_area=%s  morphology_kernel=%s\n"
        "  morphology_iterations=%s  gaussian_blur_kernel=%s\n"
        "  loss=%.6f  precision=%.4f  recall=%.4f  f1=%.4f",
        p["threshold"], p["min_area"], p["morphology_kernel"],
        p["morphology_iterations"], p["gaussian_blur_kernel"],
        best["loss"], m["precision"], m["recall"], m["f1"],
    )

    # ── Терминальная таблица топ-10 ────────────────────────────────────────
    print(
        f"\n{'#':<4} {'Loss':>8} {'Prec':>7} {'Rec':>7} {'F1':>7} "
        f"{'Thr':>5} {'MinA':>6} {'MK':>4} {'MI':>4} {'Blur':>5}"
    )
    print("-" * 60)
    for i, r in enumerate(top_n_results(results, 10), 1):
        rp = r["params"];  rm = r["metrics"]
        print(
            f"{i:<4} {r['loss']:>8.4f} {rm['precision']:>7.3f} {rm['recall']:>7.3f} "
            f"{rm['f1']:>7.3f} {rp['threshold']:>5.2f} {int(rp['min_area']):>6} "
            f"{rp['morphology_kernel']:>4} {rp['morphology_iterations']:>4} "
            f"{rp['gaussian_blur_kernel']:>5}"
        )

    # ── Сохранение best_params.json ────────────────────────────────────────
    saved = save_best_params(best)
    log.info("Параметры сохранены: %s", saved)

    # ── Визуализации ──────────────────────────────────────────────────────
    if args.no_viz:
        log.info("Визуализации пропущены (--no_viz)")
        return

    try:
        from optimizer.visualizer import (
            plot_fp_fn_tradeoff,
            plot_loss_distribution,
            plot_params_heatmap,
            plot_threshold_vs_loss,
        )
    except ImportError as e:
        log.warning("Визуализации недоступны: %s", e)
        return

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    show = save_dir is None

    def _path(name: str) -> Optional[Path]:
        return save_dir / name if save_dir else None

    log.info("Генерация графиков...")

    plot_loss_distribution(results, save_path=_path("loss_distribution.png"), show=show)
    plot_threshold_vs_loss(results, save_path=_path("threshold_vs_loss.png"), show=show)
    plot_fp_fn_tradeoff(results,    save_path=_path("fp_fn_tradeoff.png"),    show=show)
    plot_params_heatmap(
        results, x_param="threshold", y_param="min_area",
        save_path=_path("params_heatmap.png"), show=show,
    )

    if save_dir:
        log.info("Графики сохранены в: %s", save_dir)


if __name__ == "__main__":
    main()
