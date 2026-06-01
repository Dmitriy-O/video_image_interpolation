"""
Базові unit-тести для нової коректної логіки оцінки інтерполяції та кластеризації.
Запуск: pytest tests/ -q
"""

import numpy as np
import pandas as pd

from src.feature_extraction import extract_frame_features, discretize_features
from src.clustering import run_ward_clustering, compare_clusterings, compute_cluster_profiles
from src.interpolation import (
    compute_temporal_interp_errors,
    simulate_adaptive_policies,
    calculate_psnr,
    simple_linear_interpolation,
)


def make_moving_frames(n_frames=40, h=32, w=48, slow_speed=1, fast_speed=5):
    """Синтетичне відео: повільна фаза + швидка фаза (горизонтальний бар)."""
    frames = []
    for t in range(n_frames):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        speed = slow_speed if t < n_frames // 2 else fast_speed
        pos = (t * speed) % (w - 6)
        f[8:24, pos:pos+6, :] = 200
        frames.append(f)
    labels = np.array([0] * (n_frames // 2) + [1] * (n_frames // 2))
    return frames, labels


def test_temporal_errors_detects_motion_difficulty():
    """Ключова перевірка тези проєкту: швидкий рух → нижча PSNR при лінійній інтерполяції."""
    frames, labels = make_moving_frames(50, slow_speed=1, fast_speed=6)
    detailed, summary = compute_temporal_interp_errors(
        frames, labels, sample_step=1, random_state=123
    )
    assert not summary.empty
    # Кластер 1 (швидкий) повинен мати помітно нижчу середню PSNR
    psnr_slow = summary[summary['Кластер'] == 0]['Середній PSNR (temporal)'].values[0]
    psnr_fast = summary[summary['Кластер'] == 1]['Середній PSNR (temporal)'].values[0]
    assert psnr_fast < psnr_slow - 2.0, f"Expected fast cluster worse: {psnr_fast} vs {psnr_slow}"


def test_adaptive_hold_can_improve_or_preserve_on_fast():
    """Перевірка, що simulate повертає структуру і не падає; delta залежить від контенту."""
    frames, labels = make_moving_frames(36, slow_speed=1, fast_speed=4)
    policy = {0: 'linear', 1: 'hold'}
    res = simulate_adaptive_policies(frames, labels, policy, sample_step=2, random_state=42)
    assert 'overall_uniform' in res
    assert 'delta' in res
    assert isinstance(res['per_cluster'], pd.DataFrame)
    assert res['n_samples'] > 5


def test_compare_clusterings_runs():
    frames, _ = make_moving_frames(30)
    feats = extract_frame_features(frames, use_motion=True)
    w = run_ward_clustering(feats, 3)
    # Для KModes спочатку дискретизуємо
    disc = discretize_features(feats, 4)
    k = run_ward_clustering(feats, 3)  # fallback, щоб не вимагати kmodes в тесті
    cmp = compare_clusterings(w, k)
    assert 'adjusted_rand_index' in cmp
    assert -1.0 <= cmp['adjusted_rand_index'] <= 1.0


def test_profiles_and_display_map():
    frames, labels = make_moving_frames(20)
    feats = extract_frame_features(frames)
    prof = compute_cluster_profiles(feats, labels)
    assert 'cluster' in prof.columns and 'size' in prof.columns
    from src.feature_extraction import get_feature_display_map
    m = get_feature_display_map()
    assert 'frame_idx' in m and m['frame_idx'] == 'Номер кадру'


def test_psnr_and_blend_basic():
    f1 = np.full((10, 10, 3), 100, dtype=np.uint8)
    f2 = np.full((10, 10, 3), 200, dtype=np.uint8)
    blended = simple_linear_interpolation(f1, f2, 0.5)
    assert blended[0, 0, 0] == 150
    p = calculate_psnr(f1, f1)
    assert p > 99.0
