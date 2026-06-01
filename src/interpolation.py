"""
Модуль інтерполяції та оцінки якості.
"""

import cv2
import numpy as np
import pandas as pd


def simple_linear_interpolation(frame1: np.ndarray, frame2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def evaluate_interpolation_on_clusters(frames, labels, n_samples_per_cluster=5, random_state: int = 42):
    """Стара (не рекомендована) оцінка — використовуйте compute_temporal... для наукової коректності."""
    rng = np.random.default_rng(random_state)
    results = []
    for cluster in np.unique(labels):
        idxs = np.where(labels == cluster)[0]
        if len(idxs) < 2:
            continue

        psnr_values = []
        n_pairs = min(n_samples_per_cluster, len(idxs) // 2)
        if n_pairs > 0:
            # Використовуємо Generator для відтворюваності
            chosen = rng.choice(idxs, size=2 * n_pairs, replace=False)
            for p in range(n_pairs):
                pair = chosen[2*p : 2*p+2]
                interp = simple_linear_interpolation(frames[pair[0]], frames[pair[1]], alpha=0.5)
                psnr_values.append(calculate_psnr(interp, frames[pair[1]]))

        results.append({
            'Кластер': int(cluster),
            'Середній PSNR': round(np.mean(psnr_values), 2) if psnr_values else 0.0,
            'Розмір кластера': len(idxs),
            'Прикладів оцінено': len(psnr_values)
        })
    return results


# ====================== НОВА КОРЕКТНА ЛОГІКА (Phase 1) ======================

def _blend_pair(f1: np.ndarray, f2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return simple_linear_interpolation(f1, f2, alpha)


def compute_temporal_interp_errors(
    frames: list[np.ndarray],
    labels: np.ndarray,
    *,
    sample_step: int = 2,
    max_triples: int | None = None,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Коректна оцінка якості лінійної інтерполяції на часовій структурі.

    Для кожного i (з кроком) беремо кадри i-1 та i+1, інтерполюємо середній,
    порівнюємо з реальним frames[i]. Прив'язуємо помилку до кластеру кадру i.

    Повертає:
      - detailed: DataFrame з позиціями, cluster, psnr_uniform тощо
      - summary: агреговані метрики по кластерах (mean, std, n, complexity proxy)
    """
    import pandas as _pd  # локальний, щоб не ламати верхній імпорт у деяких контекстах

    n = len(frames)
    if n < 3:
        empty = _pd.DataFrame(columns=['position', 'cluster', 'psnr_uniform'])
        return empty, empty

    rng = np.random.default_rng(random_state)
    records = []

    positions = list(range(1, n - 1, max(1, sample_step)))
    if max_triples is not None and len(positions) > max_triples:
        positions = list(rng.choice(positions, size=max_triples, replace=False))
        positions.sort()

    for i in positions:
        left = frames[i - 1]
        right = frames[i + 1]
        gt = frames[i]
        est = _blend_pair(left, right, 0.5)
        psnr = calculate_psnr(est, gt)
        cl = int(labels[i])
        records.append({
            'position': i,
            'cluster': cl,
            'psnr_uniform': round(psnr, 2),
        })

    if not records:
        empty = _pd.DataFrame(columns=['position', 'cluster', 'psnr_uniform'])
        return empty, empty

    detailed = _pd.DataFrame(records)

    # Агрегація + proxy складності (середній розмір кластера як грубий індикатор)
    cluster_sizes = detailed.groupby('cluster').size().to_dict()
    summary_rows = []
    for cl in sorted(detailed['cluster'].unique()):
        sub = detailed[detailed['cluster'] == cl]
        size = cluster_sizes.get(cl, len(sub))
        summary_rows.append({
            'Кластер': cl,
            'Середній PSNR (temporal)': round(sub['psnr_uniform'].mean(), 2),
            'Std PSNR': round(sub['psnr_uniform'].std(), 2) if len(sub) > 1 else 0.0,
            'К-сть зразків': len(sub),
            'Розмір кластера': size,
        })

    summary = _pd.DataFrame(summary_rows)
    return detailed, summary


def simulate_adaptive_policies(
    frames: list[np.ndarray],
    labels: np.ndarray,
    policy_map: dict[int, str],
    *,
    sample_step: int = 2,
    max_triples: int | None = None,
    random_state: int = 42
) -> dict:
    """
    Симулює адаптивну інтерполяцію: для кожного кластеру застосовується своя стратегія.

    Підтримувані політики:
      - 'linear'     : звичайний 0.5 blend (базова)
      - 'hold'       : повторювати попередній кадр (консервативно для високого руху)
      - 'biased'     : blend з alpha=0.35 (зсув до попереднього)

    Повертає словник з метриками, таблицями та прикладами для візуалізації.
    """
    n = len(frames)
    if n < 3:
        return {'overall_uniform': 0.0, 'overall_adaptive': 0.0, 'delta': 0.0,
                'per_cluster': pd.DataFrame(), 'examples': []}

    rng = np.random.default_rng(random_state)
    uniform_psnrs = []
    adaptive_psnrs = []
    per_cluster_stats = {}
    examples = []

    positions = list(range(1, n - 1, max(1, sample_step)))
    if max_triples is not None and len(positions) > max_triples:
        positions = list(rng.choice(positions, size=max_triples, replace=False))
        positions.sort()

    for i in positions:
        cl = int(labels[i])
        policy = policy_map.get(cl, 'linear')

        left = frames[i - 1]
        right = frames[i + 1]
        gt = frames[i]

        # uniform baseline
        est_u = _blend_pair(left, right, 0.5)
        psnr_u = calculate_psnr(est_u, gt)

        # adaptive
        if policy == 'hold':
            est_a = left.copy()
        elif policy == 'biased':
            est_a = _blend_pair(left, right, 0.35)
        else:  # linear or unknown
            est_a = _blend_pair(left, right, 0.5)

        psnr_a = calculate_psnr(est_a, gt)

        uniform_psnrs.append(psnr_u)
        adaptive_psnrs.append(psnr_a)

        if cl not in per_cluster_stats:
            per_cluster_stats[cl] = {'u': [], 'a': []}
        per_cluster_stats[cl]['u'].append(psnr_u)
        per_cluster_stats[cl]['a'].append(psnr_a)

        # Збираємо приклади для найгірших позицій (пізніше фільтруємо в app)
        if len(examples) < 6:  # обмежуємо
            examples.append({
                'position': i,
                'cluster': cl,
                'policy': policy,
                'left': left,
                'gt': gt,
                'uniform': est_u,
                'adaptive': est_a,
                'psnr_uniform': round(psnr_u, 2),
                'psnr_adaptive': round(psnr_a, 2),
            })

    overall_u = float(np.mean(uniform_psnrs)) if uniform_psnrs else 0.0
    overall_a = float(np.mean(adaptive_psnrs)) if adaptive_psnrs else 0.0
    delta = round(overall_a - overall_u, 2)

    # per cluster table
    rows = []
    for cl, d in per_cluster_stats.items():
        mu = float(np.mean(d['u']))
        ma = float(np.mean(d['a']))
        rows.append({
            'Кластер': cl,
            'Uniform PSNR': round(mu, 2),
            'Adaptive PSNR': round(ma, 2),
            'Δ (підйом)': round(ma - mu, 2),
            'Політика': policy_map.get(cl, 'linear')
        })
    per_cluster_df = pd.DataFrame(rows)

    # Приклади — відсортуємо за найбільшим локальним підйомом
    examples.sort(key=lambda e: e['psnr_adaptive'] - e['psnr_uniform'], reverse=True)

    return {
        'overall_uniform': round(overall_u, 2),
        'overall_adaptive': round(overall_a, 2),
        'delta': delta,
        'per_cluster': per_cluster_df,
        'examples': examples[:4],  # топ-4 для UI
        'n_samples': len(uniform_psnrs)
    }


def recommend_policies_from_profiles(
    profiles_df: pd.DataFrame,
    *,
    high_motion_threshold: float = 0.6,
    high_texture_threshold: float = 0.6,
    motion_col: str = 'motion_mean_mean',
    texture_col: str = 'texture_laplacian_mean'
) -> dict[int, str]:
    """
    Автоматично рекомендує політики на основі профілів кластерів.
    Кластери з високим рухом АБО високою текстурою отримують 'hold' (консервативно).
    Інші — 'linear'.
    Повертає policy_map {cluster_id: policy}.
    """
    policy_map = {}
    if profiles_df.empty:
        return policy_map

    # Нормалізуємо за max для грубої "відносної складності"
    motion_vals = profiles_df.get(motion_col, pd.Series([0]*len(profiles_df)))
    texture_vals = profiles_df.get(texture_col, pd.Series([0]*len(profiles_df)))

    if motion_vals.max() > 0:
        motion_norm = motion_vals / motion_vals.max()
    else:
        motion_norm = pd.Series([0]*len(profiles_df))

    if texture_vals.max() > 0:
        texture_norm = texture_vals / texture_vals.max()
    else:
        texture_norm = pd.Series([0]*len(profiles_df))

    for _, row in profiles_df.iterrows():
        cl = int(row['cluster'])
        m = float(motion_norm.iloc[profiles_df.index.get_loc(row.name)]) if len(motion_norm) > 0 else 0
        t = float(texture_norm.iloc[profiles_df.index.get_loc(row.name)]) if len(texture_norm) > 0 else 0

        if m >= high_motion_threshold or t >= high_texture_threshold:
            policy_map[cl] = 'hold'
        else:
            policy_map[cl] = 'linear'

    return policy_map


def create_adaptive_reconstruction_demo(
    frames: list[np.ndarray],
    labels: np.ndarray,
    policy_map: dict[int, str],
    start_idx: int = 0,
    length: int = 12
) -> list[np.ndarray]:
    """
    Створює коротку послідовність "реконструкції" для демо (не повна інтерполяція відео).
    Використовується для простої візуалізації адаптивної стратегії.
    Повертає список кадрів (можна зберегти як gif/mp4 окремо).
    """
    n = len(frames)
    out = []
    for i in range(start_idx, min(start_idx + length, n)):
        if i == 0 or i == n-1:
            out.append(frames[i])
            continue
        cl = int(labels[i])
        pol = policy_map.get(cl, 'linear')
        if pol == 'hold':
            out.append(frames[i-1].copy())
        elif pol == 'biased':
            out.append(_blend_pair(frames[i-1], frames[i+1], 0.35))
        else:
            out.append(_blend_pair(frames[i-1], frames[i+1], 0.5))
    return out

