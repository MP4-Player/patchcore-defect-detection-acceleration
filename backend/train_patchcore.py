# -*- coding: utf-8 -*-
"""
Обучение PatchCore — построение Memory Bank из нормальных изображений.

ИСПРАВЛЕНИЯ:
1. L2-нормализация признаков
2. Калибровка norm_max на основе mean + 2*std
3. Сохранение пространственной структуры
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Трансформация с ImageNet-нормализацией
# ---------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

OUTPUT_PATH = Path(__file__).parent / "models" / "patchcore.pt"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Feature Extractor с L2-нормализацией
# ---------------------------------------------------------------------------

class ResNetFeatureExtractor(torch.nn.Module):
    """
    ResNet-50 с хуками на layer2[-1] и layer3[-1].
    Возвращает L2-нормализованные признаки [784, feature_dim].
    """

    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.features = []

        def hook(module, inp, out):
            self.features.append(out)

        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def forward(self, x):
        self.features = []
        with torch.no_grad():
            self.model(x)

        # padding=1 сохраняет пространственный размер (28→28)
        avg = torch.nn.AvgPool2d(3, stride=1, padding=1)
        fmap_size = self.features[0].shape[-2]  # 28 для 224×224

        resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
        resized = [resize(avg(f)) for f in self.features]

        patch = torch.cat(resized, 1)           # [1, C2+C3, 28, 28]
        patch = patch.reshape(patch.shape[1], -1).T  # [784, C2+C3]

        # L2-нормализация (стандарт PatchCore)
        patch = patch / patch.norm(dim=1, keepdim=True).clamp(min=1e-8)

        return patch


# ---------------------------------------------------------------------------
# Coreset sampling — greedy k-center
# ---------------------------------------------------------------------------

def coreset_sampling(features: torch.Tensor, num_samples: int, device: torch.device):
    """
    Greedy k-center: каждый следующий элемент — самый далёкий от уже выбранных.
    """
    if num_samples >= features.shape[0]:
        return features

    features = features.to(device)
    n_patches = features.shape[0]

    if n_patches > 30_000:
        intermediate = min(30_000, n_patches)
        print(f"  Bank large ({n_patches}). Presample: {n_patches} -> {intermediate}")
        random_idx = torch.randperm(n_patches, device=device)[:intermediate]
        features = features[random_idx]
        n_patches = intermediate

        if num_samples > 10_000:
            print(f"  Цель ({num_samples}) велика — используем случайную выборку")
            return features[:num_samples]

    first_idx = torch.randint(0, n_patches, (1,), device=device).item()
    selected = [first_idx]
    min_dists = torch.full((n_patches,), float("inf"), device=device)

    log_every = max(1, (num_samples - 1) // 10)

    for i in range(num_samples - 1):
        last = features[selected[-1]]
        dists = torch.sum((features - last) ** 2, dim=1)
        min_dists = torch.minimum(min_dists, dists)

        next_idx = torch.argmax(min_dists).item()
        selected.append(next_idx)
        min_dists[next_idx] = 0

        if (i + 1) % log_every == 0 or i == num_samples - 2:
            pct = (i + 1) / (num_samples - 1) * 100
            print(f"  Coreset: {i+1}/{num_samples-1} ({pct:.0f}%)")

    return features[torch.tensor(selected, device=device)]


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def train(train_dir: Path, output_path: Path, coreset_ratio: float, no_coreset: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # Собираем список изображений
    images = sorted(p for p in train_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"Ошибка: изображения не найдены в {train_dir}")
        sys.exit(1)
    print(f"Найдено {len(images)} обучающих изображений")

    # Загружаем backbone
    print("Загрузка ResNet-50...")
    backbone = ResNetFeatureExtractor().to(device)
    backbone.eval()

    # Извлекаем признаки
    print("Извлечение признаков...")
    parts = []
    for img_path in tqdm(images, desc="Изображения"):
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = backbone(tensor)   # [784, feature_dim] (уже L2-нормализован)
        parts.append(feats.cpu())

    bank = torch.cat(parts, dim=0)   # [N*784, feature_dim]
    print(f"Банк признаков: {bank.shape}  ({bank.nbytes/1e6:.1f} МБ)")

    # Проверка L2-нормы
    norms = torch.norm(bank, dim=1)
    print(f"L2-норма признаков: min={norms.min().item():.4f}, "
          f"max={norms.max().item():.4f}, mean={norms.mean().item():.4f}")

    # Coreset sampling
    if no_coreset:
        coreset = bank
        print("Coreset sampling отключён — сохраняем весь банк")
    else:
        n_samples = max(100, int(len(bank) * coreset_ratio))
        print(f"Coreset sampling: {len(bank)} -> {n_samples} точек ({coreset_ratio*100:.1f}%)")
        coreset = coreset_sampling(bank, n_samples, device)

    print(f"Корсет: {coreset.shape}  ({coreset.nbytes/1e6:.1f} МБ)")

    # Калибровка нормализации: вычисляем anomaly_score = mean + 2*std
    # для каждого обучающего изображения и берём 99-й перцентиль
    print("Вычисление калибровочной статистики...")
    coreset_dev = coreset.to(device)
    calib_scores = []

    for img_path in tqdm(images, desc="Калибровка"):
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = backbone(tensor)                    # [784, C] L2-нормализован
        dists = torch.cdist(feats, coreset_dev, p=2.0)  # [784, M]
        min_d, _ = torch.min(dists, dim=1)              # [784]
        # Используем mean + 2*std вместо max
        score = float((min_d.mean() + 2 * min_d.std()).cpu())
        calib_scores.append(score)

    norm_max = float(np.percentile(calib_scores, 99))
    print(f"Калибровка: p99 anomaly_score = {norm_max:.4f}")
    print(f"Диагностика: min score={min(calib_scores):.4f}, "
          f"max score={max(calib_scores):.4f}, mean={np.mean(calib_scores):.4f}")

    # Сохраняем модель
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "memory_bank": coreset.cpu(),
        "fmap_size": 28,
        "input_size": 224,
        "backbone": "resnet50",
        "n_train_images": len(images),
        "norm_max": norm_max,
        "calibration_method": "mean_plus_2std_percentile_99",
    }
    torch.save(bundle, output_path)
    print(f"\nМодель сохранена -> {output_path}")

    # Smoke-test
    _smoke_test(backbone, coreset.to(device), images[0], device, norm_max)


def _smoke_test(backbone, coreset, img_path, device, norm_max):
    """Проверяем что инференс работает на одном изображении."""
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = backbone(tensor)  # [784, C] L2-нормализован

    dists = torch.cdist(feats, coreset, p=2.0)      # [784, M]
    dist_score, _ = torch.min(dists, dim=1)         # [784]

    anomaly_score = float((dist_score.mean() + 2 * dist_score.std()).cpu())
    normalized = anomaly_score / norm_max if norm_max > 0 else 1.0

    print(f"\nSmoke-test OK:")
    print(f"  anomaly score (mean+2*std): {anomaly_score:.4f}")
    print(f"  norm_max (p99 from training): {norm_max:.4f}")
    print(f"  normalized score: {normalized:.4f} ({'аномалия' if normalized > 1.0 else 'норма'})")
    print(f"  heatmap shape: 28x28")
    print(f"\nГотово. Запускайте сервер: python main.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Обучение PatchCore — построение Memory Bank",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python train_patchcore.py --train_dir ../data/carpet/carpet/train/good
  python train_patchcore.py --train_dir ../data/carpet/carpet/train/good --coreset_ratio 0.1
  python train_patchcore.py --train_dir ../data/carpet/carpet/train/good --no_coreset
        """
    )
    p.add_argument("--train_dir", type=Path, required=True,
                   help="Папка с нормальными (бездефектными) изображениями")
    p.add_argument("--output", type=Path, default=OUTPUT_PATH,
                   help=f"Куда сохранить модель (по умолчанию: {OUTPUT_PATH})")
    p.add_argument("--coreset_ratio", type=float, default=0.1,
                   help="Доля точек в корсете от всего банка (по умолчанию 0.1 = 10%%)")
    p.add_argument("--no_coreset", action="store_true",
                   help="Отключить coreset sampling — сохранить весь банк")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.train_dir.exists():
        print(f"Ошибка: папка не найдена: {args.train_dir}")
        sys.exit(1)
    train(
        train_dir=args.train_dir,
        output_path=args.output,
        coreset_ratio=args.coreset_ratio,
        no_coreset=args.no_coreset,
    )