# -*- coding: utf-8 -*-
"""
Классификация дефектов для датасета ковра.

Типы дефектов из папок:
- color (color) - дефект цвета, обычно овальный/неправильной формы
- cut (cut) - порез, вытянутый, узкий
- hole (hole) - дырка, маленькая круглая
- metal_contamination (metal) - металлическая заноза, маленькая угловатая
- thread (thread) - нитка, очень вытянутая

Цвета (BGR) для визуализации:
- color    → зелёный     (0, 200, 0)
- cut      → красный     (0, 0, 220)
- hole     → фиолетовый  (200, 0, 200)
- metal    → оранжевый   (200, 100, 0)
- thread   → жёлтый      (0, 200, 200)
"""

import math
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Типы дефектов (соответствуют папкам в датасете)
# ---------------------------------------------------------------------------

class DefectType:
    COLOR = "color"                 # дефект цвета
    CUT = "cut"                     # порез
    HOLE = "hole"                   # дырка
    METAL = "metal_contamination"   # металлическая заноза
    THREAD = "thread"               # нитка
    UNKNOWN = "unknown"


# Цвета для визуализации (BGR)
DEFECT_COLORS = {
    DefectType.COLOR:   (0,   200, 0),     # зелёный
    DefectType.CUT:     (0,   0,   220),   # красный
    DefectType.HOLE:    (200, 0,   200),   # фиолетовый
    DefectType.METAL:   (200, 100, 0),     # оранжевый
    DefectType.THREAD:  (0,   200, 200),   # жёлтый
    DefectType.UNKNOWN: (150, 150, 150),   # серый
}

# Русские названия
DEFECT_NAMES = {
    DefectType.COLOR:   "Цвет",
    DefectType.CUT:     "Порез",
    DefectType.HOLE:    "Дырка",
    DefectType.METAL:   "Металл",
    DefectType.THREAD:  "Нитка",
    DefectType.UNKNOWN: "Неизвестный",
}

# Английские названия (для метрик)
DEFECT_NAMES_EN = {
    DefectType.COLOR:   "color",
    DefectType.CUT:     "cut",
    DefectType.HOLE:    "hole",
    DefectType.METAL:   "metal",
    DefectType.THREAD:  "thread",
    DefectType.UNKNOWN: "unknown",
}


def get_defect_color(defect_type: str) -> Tuple[int, int, int]:
    """Возвращает цвет для типа дефекта в формате BGR."""
    return DEFECT_COLORS.get(defect_type, (150, 150, 150))


def get_defect_name(defect_type: str, use_english: bool = False) -> str:
    """Возвращает название типа дефекта."""
    if use_english:
        return DEFECT_NAMES_EN.get(defect_type, "unknown")
    return DEFECT_NAMES.get(defect_type, "Неизвестный")


# ---------------------------------------------------------------------------
# Геометрические характеристики
# ---------------------------------------------------------------------------

class DefectGeometry:
    """Геометрические характеристики дефекта."""
    
    def __init__(self, contour: np.ndarray):
        self.contour = contour
        self._compute_all()
    
    def _compute_all(self):
        # Базовые
        self.area = float(cv2.contourArea(self.contour))
        self.perimeter = float(cv2.arcLength(self.contour, True))
        self.x, self.y, self.w, self.h = cv2.boundingRect(self.contour)
        self.diameter = math.sqrt(4.0 * self.area / math.pi) if self.area > 0 else 0.0
        
        # Оси эллипса
        self.major_axis = self.diameter
        self.minor_axis = self.diameter
        self.aspect_ratio = 1.0
        
        if len(self.contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(self.contour)
                (_, _), (ma, mi), _ = ellipse
                self.major_axis = max(ma, mi)
                self.minor_axis = min(ma, mi)
                self.aspect_ratio = self.major_axis / max(self.minor_axis, 1)
            except cv2.error:
                pass
        
        # Округлость
        if self.perimeter > 0:
            self.circularity = (4 * math.pi * self.area) / (self.perimeter ** 2)
            self.circularity = min(1.0, self.circularity)
        else:
            self.circularity = 0.0
        
        # Выпуклая оболочка
        hull = cv2.convexHull(self.contour)
        hull_area = float(cv2.contourArea(hull))
        self.solidity = self.area / hull_area if hull_area > 0 else 0.0
        
        # Прямоугольность
        rect = cv2.minAreaRect(self.contour)
        rect_area = rect[1][0] * rect[1][1]
        self.rectangularity = self.area / rect_area if rect_area > 0 else 0.0
        self.rectangularity = min(1.0, self.rectangularity)
        
        # Угловатость
        self.angularity = 1.0 - self.circularity
        
        # Количество вершин
        epsilon = 0.02 * self.perimeter
        approx = cv2.approxPolyDP(self.contour, epsilon, True)
        self.num_vertices = len(approx)
        
        # Центр масс
        M = cv2.moments(self.contour)
        if M["m00"] != 0:
            self.centroid_x = int(M["m10"] / M["m00"])
            self.centroid_y = int(M["m01"] / M["m00"])
        else:
            self.centroid_x = self.x + self.w // 2
            self.centroid_y = self.y + self.h // 2


# ---------------------------------------------------------------------------
# Классификатор дефектов для датасета ковра
# ---------------------------------------------------------------------------

class DefectClassifier:
    """
    Классифицирует дефекты для датасета ковра.
    
    Характеристики типов дефектов:
    - color:   овальный/неправильный, средний/крупный, circularity 0.4-0.7
    - cut:     вытянутый, aspect_ratio 2.5-5, узкий
    - hole:    маленький, круглый, circularity > 0.6, area < 100
    - metal:   маленький, угловатый, circularity < 0.4, area < 150
    - thread:  очень вытянутый, aspect_ratio > 5, тонкий
    """
    
    # Пороги для классификации (настраиваются под ваш датасет)
    # Color (дефект цвета)
    COLOR_MIN_AREA = 80
    COLOR_MIN_CIRC = 0.35
    COLOR_MAX_CIRC = 0.75
    
    # Cut (порез)
    CUT_MIN_AR = 2.5
    CUT_MAX_AR = 5.0
    CUT_MAX_CIRC = 0.55
    
    # Hole (дырка)
    HOLE_MAX_AREA = 120
    HOLE_MIN_CIRC = 0.55
    
    # Metal (металлическая заноза)
    METAL_MAX_AREA = 180
    METAL_MAX_CIRC = 0.45
    METAL_MIN_ANG = 0.55
    
    # Thread (нитка)
    THREAD_MIN_AR = 4.5
    THREAD_MAX_CIRC = 0.4
    
    def classify(self, geometry: DefectGeometry, size_class: int, K: int) -> Tuple[str, float, str]:
        """
        Определяет тип дефекта.
        
        Returns:
            (тип_дефекта, уверенность, размерная_категория)
        """
        # Размерная категория
        if K <= 1:
            size_name = "средний"
        elif size_class == 0:
            size_name = "малый"
        elif size_class >= K - 1:
            size_name = "крупный"
        else:
            size_name = "средний"
        
        scores = {}
        ar = geometry.aspect_ratio
        circ = geometry.circularity
        area = geometry.area
        ang = geometry.angularity
        
        # 1. Нитка (thread) - очень вытянутая
        if ar >= self.THREAD_MIN_AR and circ <= self.THREAD_MAX_CIRC:
            ar_score = min(1.0, (ar - 4) / 10)
            confidence = ar_score * 0.9
            scores[DefectType.THREAD] = min(0.9, confidence)
        
        # 2. Дырка (hole) - маленькая, круглая
        if area <= self.HOLE_MAX_AREA and circ >= self.HOLE_MIN_CIRC:
            area_score = 1.0 - (area / self.HOLE_MAX_AREA)
            circ_score = circ
            confidence = (area_score * 0.5 + circ_score * 0.5) * 0.85
            scores[DefectType.HOLE] = min(0.9, confidence)
        
        # 3. Заноза (metal) - маленькая, угловатая
        if area <= self.METAL_MAX_AREA and circ <= self.METAL_MAX_CIRC and ang >= self.METAL_MIN_ANG:
            area_score = 1.0 - (area / self.METAL_MAX_AREA)
            ang_score = ang
            confidence = (area_score * 0.5 + ang_score * 0.5) * 0.85
            scores[DefectType.METAL] = min(0.85, confidence)
        
        # 4. Порез (cut) - вытянутый, узкий
        if self.CUT_MIN_AR <= ar <= self.CUT_MAX_AR and circ <= self.CUT_MAX_CIRC:
            ar_score = 1.0 - abs(ar - 3.5) / 3.0
            confidence = ar_score * 0.8
            scores[DefectType.CUT] = min(0.85, confidence)
        
        # 5. Цвет (color) - овальный, неправильный
        if area >= self.COLOR_MIN_AREA and self.COLOR_MIN_CIRC <= circ <= self.COLOR_MAX_CIRC:
            area_score = min(1.0, area / 400)
            circ_score = 1.0 - abs(circ - 0.55) / 0.4
            confidence = (area_score * 0.6 + circ_score * 0.4) * 0.85
            if area > 300:
                confidence += 0.1
            scores[DefectType.COLOR] = min(0.9, confidence)
        
        # Выбираем лучший
        if scores:
            best_type = max(scores, key=scores.get)
            return best_type, scores[best_type], size_name
        
        # Если ни один не подошёл, определяем по форме
        if ar > 3.0:
            return DefectType.CUT, 0.5, size_name
        elif circ > 0.5:
            return DefectType.COLOR, 0.4, size_name
        else:
            return DefectType.UNKNOWN, 0.3, size_name


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def classify_defect_full(
    contour: np.ndarray,
    size_class: int,
    K: int,
    area: float,
    diameter: float,
) -> Tuple[str, float, str, Dict]:
    """Полная классификация дефекта."""
    geometry = DefectGeometry(contour)
    classifier = DefectClassifier()
    
    defect_type, confidence, size_name = classifier.classify(geometry, size_class, K)
    
    details = {
        "area": round(area, 2),
        "diameter": round(diameter, 2),
        "aspect_ratio": round(geometry.aspect_ratio, 3),
        "circularity": round(geometry.circularity, 4),
        "rectangularity": round(geometry.rectangularity, 4),
        "solidity": round(geometry.solidity, 4),
        "num_vertices": geometry.num_vertices,
    }
    
    return defect_type, confidence, size_name, details


def get_defect_description(defect_type: str, confidence: float, size_name: str) -> Dict:
    """Возвращает описание дефекта."""
    descriptions = {
        DefectType.COLOR: {
            "ru": f"Цвет ({size_name})",
            "description": "Дефект цвета, овальная или неправильная форма",
            "severity": "средняя",
        },
        DefectType.CUT: {
            "ru": f"Порез ({size_name})",
            "description": "Линейный разрез материала",
            "severity": "высокая",
        },
        DefectType.HOLE: {
            "ru": f"Дырка ({size_name})",
            "description": "Маленькое круглое отверстие",
            "severity": "средняя",
        },
        DefectType.METAL: {
            "ru": f"Металл ({size_name})",
            "description": "Металлическая частица или заноза",
            "severity": "высокая",
        },
        DefectType.THREAD: {
            "ru": f"Нитка ({size_name})",
            "description": "Длинная тонкая нить",
            "severity": "низкая",
        },
        DefectType.UNKNOWN: {
            "ru": f"Неизвестный ({size_name})",
            "description": "Дефект неопределённого типа",
            "severity": "неизвестна",
        },
    }
    
    info = descriptions.get(defect_type, descriptions[DefectType.UNKNOWN])
    info["confidence"] = round(confidence, 2)
    info["type"] = defect_type
    info["size"] = size_name
    
    return info