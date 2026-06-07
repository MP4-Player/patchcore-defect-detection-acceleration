# -*- coding: utf-8 -*-
"""
Интерактивный инструмент для построения Memory Bank для алгоритма PatchCore.

Описание:
    Данный скрипт позволяет интерактивно выбирать области изображений (патчи),
    которые будут использоваться для построения базы "нормальных" признаков.
    Memory Bank - это набор признаков из областей без дефектов, с которым
    сравниваются признаки при детекции аномалий.

Использование:
    python memory_bank_builder.py --image_dir <путь_к_папке_с_изображениями> --output memory_bank.pt

Управление:
    ЛКМ (левый клик)  - переключить выделение ячейки
    A                 - выделить все ячейки на текущем изображении
    C                 - очистить выделение на текущем изображении
    N / Пробел        - следующее изображение (добавить выбранные патчи в memory bank)
    S                 - сохранить memory_bank и выйти
    Q / Esc           - выйти без сохранения
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from torchvision.transforms import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from typing import Tuple, List

# Константы для визуализации
GRID = 28                    # Размер сетки (28x28 патчей)
SCALE = 3                    # Масштаб для отображения
DISPLAY_SIZE = 224 * SCALE   # Размер окна отображения (672x672)
CELL_SIZE = DISPLAY_SIZE // GRID  # Размер одной ячейки в пикселях (24px)

# Трансформация изображений для нейросети
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменение размера до 224x224
    transforms.ToTensor()            # Преобразование в тензор
])


def coreset_sampling(features: torch.Tensor, num_samples: int, device: torch.device) -> Tuple[torch.Tensor, List[int]]:
    """
    Coreset sampling using greedy k-center algorithm.
    Selects most representative patches from feature set.

    Args:
        features: feature tensor [N, D] where N is number of patches
        num_samples: number of samples to select
        device: torch device

    Returns:
        selected_features: tensor of selected features [num_samples, D]
        selected_indices: list of selected indices
    """
    if num_samples >= features.shape[0]:
        # If requested samples >= available patches, return all
        return features, list(range(features.shape[0]))

    features = features.to(device)
    n_patches = features.shape[0]

    # For very large datasets, use hybrid approach:
    # 1. Random sampling to reduce dataset size
    # 2. Greedy k-center on reduced dataset
    if n_patches > 30000:
        # Use hybrid approach: random pre-sampling + greedy
        intermediate_samples = min(30000, n_patches)
        print(f"Large dataset detected ({n_patches} patches). Using hybrid sampling:")
        print(f"  Step 1: Random pre-sampling {n_patches} -> {intermediate_samples} patches")

        # Random pre-sampling
        random_indices = torch.randperm(n_patches, device=device)[:intermediate_samples]
        features = features[random_indices]

        # If target is still large, just return random sample
        if num_samples > 10000:
            print(f"  Step 2: Target too large ({num_samples}), using random sampling")
            final_indices = random_indices[:num_samples]
            return features[:num_samples], final_indices.tolist()

        print(f"  Step 2: Greedy k-center on reduced dataset -> {num_samples} patches")
        # Continue with greedy on reduced dataset
        n_patches = intermediate_samples
        base_indices = random_indices
    else:
        base_indices = None

    # Initialize: select random starting patch for better diversity
    first_idx = torch.randint(0, n_patches, (1,), device=device).item()
    selected_indices = [first_idx]

    # Compute distances to selected patches
    # min_distances[i] = minimum distance from patch i to any selected patch
    min_distances = torch.full((n_patches,), float('inf'), device=device)

    # Progress logging
    log_every = max(1, (num_samples - 1) // 10)  # Log 10 times during process

    print(f"Starting greedy k-center coreset sampling...")

    for iteration in range(num_samples - 1):
        # Update minimum distances to selected patches
        last_selected = features[selected_indices[-1]]  # [D]

        # Compute distances from all patches to the last selected patch
        # More efficient: compute squared distances without sqrt
        distances = torch.sum((features - last_selected) ** 2, dim=1)  # [N]

        # Update minimum distances
        min_distances = torch.minimum(min_distances, distances)

        # Select patch with maximum minimum distance (farthest from all selected)
        next_idx = torch.argmax(min_distances).item()
        selected_indices.append(next_idx)

        # Set distance to zero for selected patch to avoid re-selection
        min_distances[next_idx] = 0

        # Progress logging
        if (iteration + 1) % log_every == 0 or iteration == num_samples - 2:
            progress = ((iteration + 1) / (num_samples - 1)) * 100
            print(f"  Coreset sampling progress: {iteration + 1}/{num_samples - 1} ({progress:.1f}%)")

    selected_features = features[selected_indices]

    # Map back to original indices if we did pre-sampling
    if base_indices is not None:
        selected_indices = [base_indices[idx].item() for idx in selected_indices]

    print(f"Coreset sampling completed: selected {len(selected_indices)} patches")
    return selected_features, selected_indices


class ResNetFeatureExtractor(torch.nn.Module):
    """
    Экстрактор признаков на основе предобученной ResNet-50.

    Извлекает признаки из промежуточных слоев layer2 и layer3,
    которые содержат информацию о локальных паттернах изображения.
    """
    def __init__(self):
        super().__init__()
        # Загрузка предобученной ResNet-50
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()  # Режим инференса (без обучения)

        # Заморозка весов (не обучаем модель)
        for p in self.model.parameters():
            p.requires_grad = False

        self.features = []

        # Регистрация хуков для извлечения признаков из промежуточных слоев
        def hook(module, inp, out):
            """Функция-хук для сохранения выходов слоев"""
            self.features.append(out)

        # Извлекаем признаки из последних блоков layer2 и layer3
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def forward(self, x):
        """
        Прямой проход через сеть с извлечением признаков.

        Args:
            x: входной тензор изображения [B, C, H, W]

        Returns:
            patch: тензор признаков размером [N_patches, feature_dim]
                   где N_patches = 28*28 = 784
        """
        self.features = []

        # Прямой проход через ResNet
        with torch.no_grad():
            self.model(x)

        # Применение Average Pooling для сглаживания
        avg = torch.nn.AvgPool2d(3, stride=1)
        fmap_size = self.features[0].shape[-2]  # Размер feature map

        # Приведение всех feature map к одному размеру
        resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
        resized = [resize(avg(f)) for f in self.features]

        # Конкатенация признаков из разных слоев
        patch = torch.cat(resized, 1)

        # Преобразование в формат [N_patches, feature_dim]
        # где каждый патч представлен вектором признаков
        patch = patch.reshape(patch.shape[1], -1).T

        return patch


class GridSelector:
    """
    Класс для интерактивного выбора патчей на изображении.

    Отображает изображение с наложенной сеткой 28x28,
    позволяя выбирать отдельные ячейки для включения в memory bank.
    """
    def __init__(self, img_bgr):
        """
        Инициализация селектора.

        Args:
            img_bgr: изображение в формате BGR (OpenCV)
        """
        # Масштабирование изображения для отображения
        self.base = cv2.resize(img_bgr, (DISPLAY_SIZE, DISPLAY_SIZE))

        # Массив выделенных ячеек (True = выбрана, False = не выбрана)
        self.selected = np.zeros((GRID, GRID), dtype=bool)

    def toggle(self, mx, my):
        """
        Переключить состояние ячейки по координатам мыши.

        Args:
            mx, my: координаты мыши в пикселях
        """
        # Преобразование координат мыши в индексы ячейки
        col, row = mx // CELL_SIZE, my // CELL_SIZE

        # Проверка границ и переключение состояния
        if 0 <= row < GRID and 0 <= col < GRID:
            self.selected[row, col] = not self.selected[row, col]

    def render(self):
        """
        Отрисовка изображения с визуализацией выделенных ячеек.

        Returns:
            vis: визуализированное изображение с сеткой и выделениями
        """
        vis = self.base.copy()
        overlay = vis.copy()

        # Отрисовка выделенных ячеек зеленым цветом
        for r in range(GRID):
            for c in range(GRID):
                if self.selected[r, c]:
                    # Вычисление координат ячейки
                    y1, y2 = r * CELL_SIZE, (r + 1) * CELL_SIZE
                    x1, x2 = c * CELL_SIZE, (c + 1) * CELL_SIZE
                    # Рисование прямоугольника
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)

        # Наложение полупрозрачного слоя с выделениями
        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        # Отрисовка сетки
        for i in range(1, GRID):
            pos = i * CELL_SIZE
            cv2.line(vis, (pos, 0), (pos, DISPLAY_SIZE), (80, 80, 80), 1)
            cv2.line(vis, (0, pos), (DISPLAY_SIZE, pos), (80, 80, 80), 1)

        # Отображение счетчика выбранных ячеек
        cnt = int(self.selected.sum())
        cv2.putText(vis, f"Выбрано: {cnt}/{GRID*GRID}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return vis

    def get_indices(self):
        """
        Получить линейные индексы выбранных ячеек.

        Returns:
            indices: массив индексов выбранных патчей (от 0 до 783)
        """
        rows, cols = np.where(self.selected)
        return rows * GRID + cols


def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Интерактивное построение Memory Bank')
    parser.add_argument("--image_dir", required=True, help="Папка с изображениями")
    parser.add_argument("--output", default="memory_bank.pt", help="Имя файла для сохранения memory bank")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Устройство для вычислений (cuda/cpu)")
    parser.add_argument("--ext", default="png", help="Расширение файлов изображений (png, jpg, bmp)")
    parser.add_argument("--coreset_percentage", type=float, default=0.1,
                        help="Процент патчей для coreset sampling (0.0-1.0). По умолчанию 0.1 (10%). Используйте 1.0 чтобы отключить.")
    args = parser.parse_args()

    # Инициализация устройства (GPU или CPU)
    device = torch.device(args.device)

    # Создание экстрактора признаков
    backbone = ResNetFeatureExtractor().to(device)
    backbone.eval()

    # Поиск изображений в указанной папке
    image_paths = sorted(Path(args.image_dir).glob(f"*.{args.ext}"))
    if not image_paths:
        # Если не найдено с указанным расширением, ищем все поддерживаемые форматы
        image_paths = sorted(Path(args.image_dir).glob("*.*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]

    if not image_paths:
        print(f"Изображения не найдены в {args.image_dir}")
        return

    print(f"Найдено {len(image_paths)} изображений")
    print("Управление: ЛКМ=выбрать | A=все | C=очистить | N/Пробел=далее | S=сохранить | Q/Esc=выход")

    # Список для хранения частей memory bank (признаки из всех изображений)
    memory_bank_parts = []

    # Создание окна для отображения
    win = "Memory Bank Builder"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    selector = None

    def on_mouse(event, x, y, flags, param):
        """Обработчик событий мыши"""
        if event == cv2.EVENT_LBUTTONDOWN and selector is not None:
            selector.toggle(x, y)
            cv2.imshow(win, selector.render())

    cv2.setMouseCallback(win, on_mouse)

    # Основной цикл обработки изображений
    idx = 0
    while idx < len(image_paths):
        path = image_paths[idx]

        # Загрузка изображения
        pil_img = Image.open(path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Создание селектора для текущего изображения
        selector = GridSelector(img_bgr)

        # Отображение изображения с информацией
        title_img = selector.render()
        cv2.putText(title_img, f"[{idx+1}/{len(image_paths)}] {path.name}",
                    (10, DISPLAY_SIZE - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, title_img)

        # Цикл обработки клавиатурных команд для текущего изображения
        while True:
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:  # Q или Esc - выход без сохранения
                cv2.destroyAllWindows()
                print("Выход без сохранения.")
                return

            elif key == ord('s'):  # S - сохранить и выйти
                # Добавить текущее выделение перед сохранением
                if selector.selected.any():
                    indices = selector.get_indices()
                    tensor = transform(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = backbone(tensor)
                    memory_bank_parts.append(features[indices].cpu())
                    print(f"  Добавлено {len(indices)} патчей из {path.name}")
                goto_save = True
                break

            elif key in (ord('n'), ord(' ')):  # N или Пробел - следующее изображение
                if selector.selected.any():
                    # Извлечение индексов выбранных патчей
                    indices = selector.get_indices()

                    # Преобразование изображения в тензор
                    tensor = transform(pil_img).unsqueeze(0).to(device)

                    # Извлечение признаков
                    with torch.no_grad():
                        features = backbone(tensor)

                    # Добавление выбранных признаков в memory bank
                    memory_bank_parts.append(features[indices].cpu())
                    print(f"  Добавлено {len(indices)} патчей из {path.name}")
                else:
                    print(f"  Пропущено {path.name} (ничего не выбрано)")
                idx += 1
                goto_save = False
                break

            elif key == ord('a'):  # A - выделить все ячейки
                selector.selected[:] = True
                vis = selector.render()
                cv2.putText(vis, f"[{idx+1}/{len(image_paths)}] {path.name}",
                            (10, DISPLAY_SIZE - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(win, vis)

            elif key == ord('c'):  # C - очистить выделение
                selector.selected[:] = False
                vis = selector.render()
                cv2.putText(vis, f"[{idx+1}/{len(image_paths)}] {path.name}",
                            (10, DISPLAY_SIZE - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(win, vis)
            else:
                # Обновление отображения (для отклика на события мыши)
                vis = selector.render()
                cv2.putText(vis, f"[{idx+1}/{len(image_paths)}] {path.name}",
                            (10, DISPLAY_SIZE - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(win, vis)

        if goto_save:
            break

    cv2.destroyAllWindows()

    # Сохранение memory bank
    if memory_bank_parts:
        # Объединение всех частей в один тензор
        memory_bank = torch.cat(memory_bank_parts, dim=0)

        # Применение coreset sampling если требуется
        original_size = memory_bank.shape[0]
        if args.coreset_percentage < 1.0:
            num_samples = max(1, int(original_size * args.coreset_percentage))
            print(f"\nПрименение coreset sampling: {original_size} -> {num_samples} патчей ({args.coreset_percentage*100:.1f}%)")

            memory_bank, selected_indices = coreset_sampling(memory_bank, num_samples, device)
            print(f"Coreset sampling завершен: выбрано {len(selected_indices)} наиболее репрезентативных патчей")
        else:
            print(f"\nCoreset sampling отключен (percentage={args.coreset_percentage})")

        # Сохранение в файл
        torch.save(memory_bank, args.output)
        print(f"Memory bank сохранен: размер {memory_bank.shape} -> {args.output}")
    else:
        print("Патчи не выбраны, ничего не сохранено.")


if __name__ == "__main__":
    main()
