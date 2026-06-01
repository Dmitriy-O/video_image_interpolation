"""
Модуль витягування ознак з відеокадрів.
Включає статистичні, текстурні та temporal ознаки.

Внутрішні назви колонок — англійські (стабільні для коду).
Для UI використовуйте get_feature_display_map().
"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Dict


FEATURE_DISPLAY_MAP = {
    'frame_idx': 'Номер кадру',
    'mean_b': 'mean_b',
    'mean_g': 'mean_g',
    'mean_r': 'mean_r',
    'std_b': 'std_b',
    'std_g': 'std_g',
    'std_r': 'std_r',
    'brightness': 'Яскравість',
    'contrast': 'Контраст',
    'texture_laplacian': 'Текстура (Laplacian)',
    'motion_mean': 'Рух (середнє)',
    'motion_std': 'Рух (std)',
}


def get_feature_display_map() -> Dict[str, str]:
    """Повертає мапу англійських ключів → українські назви для UI."""
    return FEATURE_DISPLAY_MAP.copy()


def extract_frame_features(frames: List[np.ndarray], use_motion: bool = True) -> pd.DataFrame:
    """
    Витягує розширені ознаки з кадрів відео.
    
    Ознаки:
    - Статистики кольору (BGR)
    - Яскравість та контраст
    - Текстура (енергія Лапласіана)
    - Ознаки руху (різниця між кадрами)
    
    Колонки (англійські, стабільні):
    frame_idx, mean_b, mean_g, mean_r, std_b, std_g, std_r,
    brightness, contrast, texture_laplacian, motion_mean, motion_std
    """
    features = []
    prev_gray = None

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(frame)

        # Текстурна ознака через Laplacian (з лекцій)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_energy = float(np.mean(np.abs(laplacian)))

        feat: Dict = {
            'frame_idx': i,
            'mean_b': float(np.mean(b)),
            'mean_g': float(np.mean(g)),
            'mean_r': float(np.mean(r)),
            'std_b': float(np.std(b)),
            'std_g': float(np.std(g)),
            'std_r': float(np.std(r)),
            'brightness': float(np.mean(gray)),
            'contrast': float(np.std(gray)),
            'texture_laplacian': texture_energy,
        }

        if use_motion and prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            feat['motion_mean'] = float(np.mean(diff))
            feat['motion_std'] = float(np.std(diff))
        else:
            feat['motion_mean'] = 0.0
            feat['motion_std'] = 0.0

        features.append(feat)
        prev_gray = gray

    return pd.DataFrame(features)


def discretize_features(df: pd.DataFrame, n_bins: int = 5, feature_cols: List[str] | None = None) -> pd.DataFrame:
    """
    Дискретизує числові ознаки в категорії (0..n-1) для KModes.

    feature_cols: optional subset; за замовчанням — всі крім frame_idx.
    Повертає копію з дискретизованими колонками (int).
    """
    df_disc = df.copy()
    if feature_cols is None:
        numeric_cols = [col for col in df.columns if col != 'frame_idx']
    else:
        numeric_cols = [c for c in feature_cols if c in df.columns and c != 'frame_idx']

    for col in numeric_cols:
        try:
            # qcut з duplicates='drop' може дати менше бінів, ніж n_bins
            df_disc[col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            df_disc[col] = pd.cut(df[col], bins=n_bins, labels=False, include_lowest=True)

        # На випадок NaN після binning
        df_disc[col] = df_disc[col].fillna(0).astype(int)

    return df_disc