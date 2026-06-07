"""
Визуализация результатов оптимизации параметров бинаризации.

Функции:
    show_single_result       — 5 панелей: original, heatmap, binary, GT, overlay
    show_top_results         — сводная таблица топ-N лучших и N худших
    plot_threshold_vs_loss   — loss vs threshold (mean + min)
    plot_fp_fn_tradeoff      — scatter FPR × FNR, цвет = loss
    plot_params_heatmap      — 2D тепловая карта двух параметров → loss
    plot_loss_distribution   — гистограмма распределения loss по всем комбинациям
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

from .parameter_optimizer import apply_parameters


def _require_mpl() -> None:
    if not _MPL:
        raise ImportError("matplotlib не установлен. Выполните: pip install matplotlib")


# ---------------------------------------------------------------------------
# show_single_result
# ---------------------------------------------------------------------------

def show_single_result(
    image: np.ndarray,
    heatmap: np.ndarray,
    params: Dict,
    gt_mask: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    5-панельная визуализация одного результата:
      original | heatmap | binary mask | GT mask | overlay (red=pred, green=GT)
    """
    _require_mpl()

    binary = apply_parameters(heatmap, params)
    h, w   = heatmap.shape

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(
        title or (
            f"thr={params.get('threshold','?')}  min_area={params.get('min_area','?')}  "
            f"mk={params.get('morphology_kernel','?')}  mi={params.get('morphology_iterations','?')}  "
            f"blur={params.get('gaussian_blur_kernel','?')}"
        ),
        fontsize=10,
    )

    # 1. Original
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
    if img_rgb.shape[:2] != (h, w):
        img_rgb = cv2.resize(img_rgb, (w, h))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Оригинал")
    axes[0].axis("off")

    # 2. Heatmap
    im = axes[1].imshow(heatmap, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Binary
    axes[2].imshow(binary, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Бинарная маска")
    axes[2].axis("off")

    # 4. GT
    if gt_mask is not None:
        gt_show = (gt_mask > 0).astype(np.uint8) * 255
        if gt_show.shape != (h, w):
            gt_show = cv2.resize(gt_show, (w, h), interpolation=cv2.INTER_NEAREST)
        axes[3].imshow(gt_show, cmap="gray", vmin=0, vmax=255)
        axes[3].set_title("GT маска")
    else:
        axes[3].imshow(np.zeros((h, w), dtype=np.uint8), cmap="gray")
        axes[3].set_title("GT маска (нет)")
    axes[3].axis("off")

    # 5. Overlay
    overlay = img_rgb.copy().astype(np.float32)
    pred_bool = binary > 0
    overlay[pred_bool] = overlay[pred_bool] * 0.4 + np.array([255, 0, 0]) * 0.6
    if gt_mask is not None:
        gt_bool = (gt_mask > 0)
        if gt_bool.shape != (h, w):
            gt_bool = cv2.resize(gt_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay[gt_bool] = overlay[gt_bool] * 0.4 + np.array([0, 255, 0]) * 0.6
    axes[4].imshow(overlay.clip(0, 255).astype(np.uint8))
    axes[4].set_title("Overlay (кр=пред, зел=GT)")
    axes[4].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# show_top_results
# ---------------------------------------------------------------------------

def show_top_results(
    results: List[Dict],
    n: int = 10,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Сводная таблица топ-N лучших и N худших результатов.
    """
    _require_mpl()

    best  = results[:n]
    worst = results[-n:][::-1]

    cols = ["#", "Loss", "Prec", "Rec", "F1", "Thr", "MinArea", "MK", "MI", "Blur"]

    def _rows(items: List[Dict], prefix: str) -> List[List[str]]:
        rows = []
        for i, r in enumerate(items, 1):
            p = r["params"];  m = r["metrics"]
            rows.append([
                f"{prefix}{i}",
                f"{r['loss']:.4f}",
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['f1']:.3f}",
                str(p["threshold"]),
                str(int(p["min_area"])),
                str(p["morphology_kernel"]),
                str(p["morphology_iterations"]),
                str(p["gaussian_blur_kernel"]),
            ])
        return rows

    fig, axes = plt.subplots(1, 2, figsize=(22, n * 0.5 + 1.5))
    fig.suptitle(f"Топ-{n} / Худшие-{n} результатов оптимизации", fontsize=13)

    for ax, rows, color, lbl in [
        (axes[0], _rows(best,  "✓"),  "#d4edda", "Лучшие"),
        (axes[1], _rows(worst, "✗"),  "#f8d7da", "Худшие"),
    ]:
        ax.axis("off")
        tbl = ax.table(
            cellText=rows,
            colLabels=cols,
            cellLoc="center",
            loc="center",
            cellColours=[[color] * 10] * n,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.4)
        ax.set_title(lbl, fontsize=11, pad=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_threshold_vs_loss
# ---------------------------------------------------------------------------

def plot_threshold_vs_loss(
    results: List[Dict],
    fixed_params: Optional[Dict] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Зависимость loss от порога бинаризации.
    fixed_params: фиксируем остальные параметры (ключ → значение).
    """
    _require_mpl()

    by_thr: Dict[float, List[float]] = defaultdict(list)
    for r in results:
        p = r["params"]
        if fixed_params and any(p.get(k) != v for k, v in fixed_params.items() if k != "threshold"):
            continue
        by_thr[p["threshold"]].append(r["loss"])

    thresholds  = sorted(by_thr)
    mean_losses = [float(np.mean(by_thr[t])) for t in thresholds]
    min_losses  = [float(np.min(by_thr[t]))  for t in thresholds]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, mean_losses, "o-",  color="#2196F3", linewidth=2, label="Mean loss")
    ax.plot(thresholds, min_losses,  "s--", color="#4CAF50", linewidth=1.5, label="Min loss")

    best_idx = int(np.argmin(mean_losses))
    ax.annotate(
        f"best thr={thresholds[best_idx]:.2f}\nloss={mean_losses[best_idx]:.4f}",
        xy=(thresholds[best_idx], mean_losses[best_idx]),
        xytext=(thresholds[best_idx] + 0.04, mean_losses[best_idx] + 0.002),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Loss  (FP×10 + FN×5) / total", fontsize=12)
    ax.set_title("Зависимость loss от threshold бинаризации", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_fp_fn_tradeoff
# ---------------------------------------------------------------------------

def plot_fp_fn_tradeoff(
    results: List[Dict],
    n_points: int = 500,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Scatter-plot FPR × FNR. Цвет = loss (зелёный = хорошо, красный = плохо).
    """
    _require_mpl()

    subset = results[:n_points]
    fprs   = [r["metrics"]["fpr"]  for r in subset]
    fnrs   = [r["metrics"]["fnr"]  for r in subset]
    losses = [r["loss"]            for r in subset]

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(fprs, fnrs, c=losses, cmap="RdYlGn_r",
                    s=20, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Loss")

    best = results[0]
    ax.scatter(
        [best["metrics"]["fpr"]], [best["metrics"]["fnr"]],
        c="blue", s=120, zorder=5, marker="*",
        label=f"Лучший (loss={best['loss']:.4f})",
    )
    ax.legend(fontsize=9)
    ax.set_xlabel("FPR (False Positive Rate)", fontsize=12)
    ax.set_ylabel("FNR (False Negative Rate)", fontsize=12)
    ax.set_title(f"FP/FN Tradeoff (топ-{n_points} результатов)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_params_heatmap
# ---------------------------------------------------------------------------

def plot_params_heatmap(
    results: List[Dict],
    x_param: str = "threshold",
    y_param: str = "min_area",
    agg: str = "min",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    2D тепловая карта x_param × y_param → агрегированный loss.
    agg: 'min' | 'mean'
    """
    _require_mpl()

    grid: Dict[Tuple, List[float]] = defaultdict(list)
    for r in results:
        key = (r["params"][x_param], r["params"][y_param])
        grid[key].append(r["loss"])

    xs = sorted({k[0] for k in grid})
    ys = sorted({k[1] for k in grid})
    xi_map = {v: i for i, v in enumerate(xs)}
    yi_map = {v: i for i, v in enumerate(ys)}

    matrix = np.zeros((len(ys), len(xs)))
    for (xv, yv), losses in grid.items():
        matrix[yi_map[yv], xi_map[xv]] = np.min(losses) if agg == "min" else np.mean(losses)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=matrix.min(), vmax=matrix.max())
    plt.colorbar(im, ax=ax, label=f"{agg.capitalize()} Loss")

    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(v) for v in xs], rotation=45, ha="right")
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([str(v) for v in ys])
    ax.set_xlabel(x_param, fontsize=12)
    ax.set_ylabel(y_param, fontsize=12)
    ax.set_title(f"{agg.capitalize()} loss: {x_param} × {y_param}", fontsize=13)

    mid = (matrix.max() + matrix.min()) / 2
    for i in range(len(ys)):
        for j in range(len(xs)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=7, color="white" if v > mid else "black")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_loss_distribution
# ---------------------------------------------------------------------------

def plot_loss_distribution(
    results: List[Dict],
    bins: int = 50,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Гистограмма распределения loss по всем комбинациям параметров.
    Зелёный ≤ медиана, жёлтый ≤ 75-й перцентиль, красный > 75-го.
    """
    _require_mpl()

    losses     = [r["loss"] for r in results]
    best_loss  = losses[0]
    median     = float(np.median(losses))
    p75        = float(np.percentile(losses, 75))

    fig, ax = plt.subplots(figsize=(9, 5))
    n, bin_edges, patches = ax.hist(losses, bins=bins,
                                    color="#2196F3", edgecolor="white", alpha=0.8)

    for patch, left in zip(patches, bin_edges[:-1]):
        if left <= median:
            patch.set_facecolor("#4CAF50")
        elif left <= p75:
            patch.set_facecolor("#FFC107")
        else:
            patch.set_facecolor("#F44336")

    ax.axvline(best_loss, color="blue",   linestyle="--", linewidth=1.5,
               label=f"Лучший: {best_loss:.4f}")
    ax.axvline(median,    color="orange", linestyle=":",  linewidth=1.5,
               label=f"Медиана: {median:.4f}")

    ax.set_xlabel("Loss  (FP×10 + FN×5) / total", fontsize=12)
    ax.set_ylabel("Число комбинаций", fontsize=12)
    ax.set_title(f"Распределение loss: {len(losses)} комбинаций", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
