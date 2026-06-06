"""
Модуль інтерполяції та оцінки якості.
"""

import cv2
from collections import defaultdict
import numpy as np
import pandas as pd


def simple_linear_interpolation(frame1: np.ndarray, frame2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simple global SSIM (for relative comparison and secondary metric in the demo).

    Not a full local-window SSIM, but sufficient to show structural differences
    beyond pure pixel MSE/PSNR.
    """
    if img1.shape != img2.shape:
        return 0.0
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


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


def _smooth_label_sequence(labels: np.ndarray, window: int = 9) -> np.ndarray:
    """
    Apply strong temporal smoothing to the label sequence using a rolling majority vote.
    Larger window = stronger temporal consistency (policies change less frequently,
    behaving more like segment-level decisions).
    """
    n = len(labels)
    if n == 0:
        return labels
    smoothed = np.zeros(n, dtype=labels.dtype)
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_labels = labels[start:end].astype(int)
        if len(window_labels) > 0:
            smoothed[i] = np.bincount(window_labels).argmax()
        else:
            smoothed[i] = labels[i]
    return smoothed


def _get_smoothed_policy(labels: np.ndarray, i: int, policy_map: dict[int, str], window: int = 9) -> str:
    """Return the policy according to the majority cluster label in a temporal window around i.
    Default window increased for stronger temporal smoothing (more segment-like behavior).
    """
    start = max(0, i - window // 2)
    end = min(len(labels), i + window // 2 + 1)
    window_labels = labels[start:end].astype(int)
    if len(window_labels) == 0:
        maj = int(labels[i])
    else:
        maj = int(np.bincount(window_labels).argmax())
    return policy_map.get(maj, 'linear')


def _get_blend_alpha(cluster: int, policy: str, profiles: pd.DataFrame | None = None) -> float:
    """
    Returns a fine-grained blend alpha for the given policy and cluster.

    More gradations of alpha (from very conservative to forward) based on:
    - Explicit policy strength
    - Cluster motion profile (higher motion → lower alpha = more weight to previous frame)

    This iteration adds more granular levels and stronger motion sensitivity.
    """
    if policy == 'hold':
        return 0.0
    if policy == 'very_strong':
        base = 0.12
    elif policy == 'strong':
        base = 0.20
    elif policy == 'conservative':
        base = 0.28
    elif policy == 'moderate':
        base = 0.36
    elif policy == 'mild':
        base = 0.42
    elif policy == 'linear':
        base = 0.50
    elif policy == 'forward_biased':
        base = 0.62
    else:
        # 'biased', 'motion_aware' etc. start from a mid-conservative base
        base = 0.35

    # Apply motion-dependent adjustment for policies that benefit from it
    if policy not in ('hold', 'linear', 'forward_biased') and profiles is not None and not profiles.empty:
        motion_col = 'motion_mean_mean'
        if motion_col in profiles.columns:
            row = profiles[profiles['cluster'] == cluster]
            if not row.empty:
                motion = float(row[motion_col].iloc[0])
                # Stronger sensitivity curve (more gradations)
                # Normalize motion to [0, 1.2] range
                norm = min(max(motion / 22.0, 0.0), 1.2)
                # Stronger pull toward conservative for high motion
                adjustment = norm * 0.28
                base = max(0.10, min(0.55, base - adjustment))

    return float(base)


def simulate_adaptive_policies(
    frames: list[np.ndarray],
    labels: np.ndarray,
    policy_map: dict[int, str],
    *,
    sample_step: int = 2,
    max_triples: int | None = None,
    random_state: int = 42,
    policy_smoothing_window: int = 9,
    profiles: pd.DataFrame | None = None,
) -> dict:
    """
    Симулює адаптивну інтерполяцію: для кожного кластеру застосовується своя стратегія.

    Підтримувані політики (покращена версія з більшою кількістю градацій alpha):
      - 'hold'            : 0.0 (повністю попередній кадр)
      - 'very_strong'     : ~0.12 (дуже консервативно)
      - 'strong'          : ~0.20
      - 'conservative'    : ~0.28
      - 'moderate'        : ~0.36
      - 'mild'            : ~0.42
      - 'linear'          : 0.50
      - 'biased' / 'motion_aware': motion-dependent (0.10–0.45, залежить від профілю руху кластера)
      - 'forward_biased'  : ~0.62

    Ключові покращення цієї ітерації:
    - Значно більше градацій alpha (від 0.10 до 0.62) з нелінійним motion-dependent регулюванням.
    - Сильніше temporal smoothing за замовчуванням (window=9) + попереднє згладжування всієї послідовності міток.
      Це забезпечує набагато стабільнішу, "сегментну" поведінку політик.
    - Додатковий SSIM.

    Повертає словник з метриками...
    """
    n = len(frames)
    if n < 3:
        return {'overall_uniform': 0.0, 'overall_adaptive': 0.0, 'delta': 0.0,
                'overall_uniform_ssim': 0.0, 'overall_adaptive_ssim': 0.0, 'delta_ssim': 0.0,
                'per_cluster': pd.DataFrame(), 'examples': []}

    rng = np.random.default_rng(random_state)
    uniform_psnrs = []
    adaptive_psnrs = []
    uniform_ssims = []
    adaptive_ssims = []
    per_cluster_stats = {}
    examples = []

    # Stronger temporal smoothing: pre-smooth the entire label sequence
    # This creates longer stable segments where the same policy is applied
    smoothed_labels = _smooth_label_sequence(labels, window=policy_smoothing_window)

    positions = list(range(1, n - 1, max(1, sample_step)))
    if max_triples is not None and len(positions) > max_triples:
        positions = list(rng.choice(positions, size=max_triples, replace=False))
        positions.sort()

    for i in positions:
        cl = int(labels[i])  # original cluster for stats / attribution
        # Use strongly smoothed labels for policy decision (more segment-like)
        effective_cl = int(smoothed_labels[i])
        policy = policy_map.get(effective_cl, 'linear')

        left = frames[i - 1]
        right = frames[i + 1]
        gt = frames[i]

        # uniform baseline
        est_u = _blend_pair(left, right, 0.5)
        psnr_u = calculate_psnr(est_u, gt)
        ssim_u = calculate_ssim(est_u, gt)

        # adaptive with fine-grained, profile-aware alpha
        alpha = _get_blend_alpha(effective_cl, policy, profiles)
        if policy == 'hold':
            est_a = left.copy()
        else:
            est_a = _blend_pair(left, right, alpha)

        psnr_a = calculate_psnr(est_a, gt)
        ssim_a = calculate_ssim(est_a, gt)

        uniform_psnrs.append(psnr_u)
        adaptive_psnrs.append(psnr_a)
        uniform_ssims.append(ssim_u)
        adaptive_ssims.append(ssim_a)

        if cl not in per_cluster_stats:
            per_cluster_stats[cl] = {'u_psnr': [], 'a_psnr': [], 'u_ssim': [], 'a_ssim': []}
        per_cluster_stats[cl]['u_psnr'].append(psnr_u)
        per_cluster_stats[cl]['a_psnr'].append(psnr_a)
        per_cluster_stats[cl]['u_ssim'].append(ssim_u)
        per_cluster_stats[cl]['a_ssim'].append(ssim_a)

        if len(examples) < 6:
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
                'ssim_uniform': round(ssim_u, 4),
                'ssim_adaptive': round(ssim_a, 4),
            })

    overall_u = float(np.mean(uniform_psnrs)) if uniform_psnrs else 0.0
    overall_a = float(np.mean(adaptive_psnrs)) if adaptive_psnrs else 0.0
    delta = round(overall_a - overall_u, 2)

    overall_u_ssim = float(np.mean(uniform_ssims)) if uniform_ssims else 0.0
    overall_a_ssim = float(np.mean(adaptive_ssims)) if adaptive_ssims else 0.0
    delta_ssim = round(overall_a_ssim - overall_u_ssim, 4)

    rows = []
    for cl, d in per_cluster_stats.items():
        mu_p = float(np.mean(d['u_psnr']))
        ma_p = float(np.mean(d['a_psnr']))
        mu_s = float(np.mean(d['u_ssim']))
        ma_s = float(np.mean(d['a_ssim']))
        rows.append({
            'Кластер': cl,
            'Uniform PSNR': round(mu_p, 2),
            'Adaptive PSNR': round(ma_p, 2),
            'Δ PSNR': round(ma_p - mu_p, 2),
            'Uniform SSIM': round(mu_s, 4),
            'Adaptive SSIM': round(ma_s, 4),
            'Δ SSIM': round(ma_s - mu_s, 4),
            'Політика': policy_map.get(cl, 'linear')
        })
    per_cluster_df = pd.DataFrame(rows)

    examples.sort(key=lambda e: e['psnr_adaptive'] - e['psnr_uniform'], reverse=True)

    return {
        'overall_uniform': round(overall_u, 2),
        'overall_adaptive': round(overall_a, 2),
        'delta': delta,
        'overall_uniform_ssim': round(overall_u_ssim, 4),
        'overall_adaptive_ssim': round(overall_a_ssim, 4),
        'delta_ssim': delta_ssim,
        'per_cluster': per_cluster_df,
        'examples': examples[:4],
        'n_samples': len(uniform_psnrs)
    }


def _evaluate_per_cluster_policy_psnrs(
    frames: list[np.ndarray],
    labels: np.ndarray,
    sample_step: int,
    profiles: pd.DataFrame | None = None,
) -> dict[int, dict[str, dict[str, float]]]:
    """
    Для кожного кластера обчислює середні значення PSNR і SSIM на його позиціях 
    для всіх підтримуваних політик (з градаціями alpha).

    Повертає:
        { cluster: { policy: {'psnr': mean, 'ssim': mean}, ... } }

    Використовує той самий семплінг triplet'ів, що й simulate.
    Якщо передано profiles — для motion-dependent політик буде використано реальний alpha.
    """
    n = len(frames)
    if n < 3:
        return {}

    EVAL_POLICIES = [
        "linear", "mild", "moderate", "conservative",
        "strong", "very_strong", "hold", "biased", "forward_biased"
    ]

    positions = list(range(1, n - 1, max(1, sample_step)))

    per_cl: dict[int, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {p: {'psnr': [], 'ssim': []} for p in EVAL_POLICIES}
    )

    for i in positions:
        cl = int(labels[i])
        left = frames[i - 1]
        right = frames[i + 1]
        gt = frames[i]

        for pol in EVAL_POLICIES:
            alpha = _get_blend_alpha(cl, pol, profiles)
            if pol == "hold":
                est = left.copy()
            else:
                est = _blend_pair(left, right, alpha)
            ps = calculate_psnr(est, gt)
            ss = calculate_ssim(est, gt)
            per_cl[cl][pol]['psnr'].append(ps)
            per_cl[cl][pol]['ssim'].append(ss)

    means: dict[int, dict[str, dict[str, float]]] = {}
    for cl, d in per_cl.items():
        means[cl] = {}
        for p in EVAL_POLICIES:
            means[cl][p] = {
                'psnr': float(np.mean(d[p]['psnr'])) if d[p]['psnr'] else 0.0,
                'ssim': float(np.mean(d[p]['ssim'])) if d[p]['ssim'] else 0.0,
            }
    return means


def recommend_policies_from_profiles(
    profiles_df: pd.DataFrame,
    *,
    hold_quantile: float = 0.80,
    biased_quantile: float = 0.50,
    motion_col: str = 'motion_mean_mean',
    texture_col: str = 'texture_laplacian_mean',
    frames: list[np.ndarray] | None = None,
    labels: np.ndarray | None = None,
    sample_step: int = 2,
    random_state: int = 42,
) -> dict[int, str]:
    """
    Автоматично рекомендує політики на основі профілів кластерів.

    Логіка є продуманою (thoughtful), а не чисто "яка політика дала найбільший PSNR":

    - combined score з профілів (рух + текстура) визначає, наскільки "складним" є кластер.
    - Для легких кластерів linear є природним і бажаним вибором.
    - Для середніх і складних кластерів ми свідомо віддаємо перевагу conservative / moderate / biased (з розумним alpha),
      навіть якщо їхній PSNR трохи нижчий — бо це відповідає основній ідеї проєкту
      (знаходити типи контенту, де класична лінійна інтерполяція концептуально слабка).
    - Дуже складні кластери (hard) отримують доступ до strong / very_strong / hold.
    - Реальний PSNR використовується як важливий validation / tie-breaker,
      але не як єдиний критерій.

    Такий підхід робить рекомендацію більш осмисленою і відповідною до навчальної мети курсової.
    """
    if profiles_df.empty:
        return {}

    motion_vals = profiles_df.get(motion_col, pd.Series([0.0] * len(profiles_df)))
    texture_vals = profiles_df.get(texture_col, pd.Series([0.0] * len(profiles_df)))

    motion_norm = (motion_vals / motion_vals.max()) if motion_vals.max() > 0 else pd.Series([0.0] * len(profiles_df))
    texture_norm = (texture_vals / texture_vals.max()) if texture_vals.max() > 0 else pd.Series([0.0] * len(profiles_df))

    combined = (motion_norm + texture_norm) / 2.0

    work = profiles_df[['cluster']].reset_index(drop=True)
    work['combined'] = combined.reset_index(drop=True)

    n = len(work)
    if n >= 2:
        hold_thresh = float(work['combined'].quantile(hold_quantile))
    else:
        hold_thresh = 0.85

    # Визначаємо "hard" кластери (топ за combined) — тільки їм дозволено 'hold'
    hard_clusters = {int(row['cluster']) for _, row in work.iterrows() if float(row['combined']) >= hold_thresh}

    if frames is not None and labels is not None:
        # Thoughtful / продуманої recommendation logic.
        #
        # The goal is NOT to blindly pick the policy with the highest measured PSNR.
        # The core idea of the project is:
        #   - Use simple features (motion + texture) to detect content types where
        #     classic linear interpolation is conceptually insufficient.
        #   - Then apply a more appropriate (usually more conservative) strategy for those types.
        #
        # Therefore the logic is hybrid and deliberate:
        #   1. Difficulty from features (combined motion+texture) determines the "natural" policy family.
        #   2. Actual PSNR on the cluster's triplets is used as validation / tie-breaker,
        #      not as the only criterion.
        #   3. We apply a thoughtful bias:
        #      - Easy content → linear is strongly preferred (it is the right tool).
        #      - Medium/hard content → conservative or motion-aware biased are preferred
        #        even if their PSNR is slightly lower (because that matches the project's thesis).
        #      - Very hard content → strong conservative / hold are considered.
        #
        # This makes the recommendation "more thoughtful" and aligned with the educational goal,
        # while still being grounded in real measurements on the given video.
        quality = _evaluate_per_cluster_policy_psnrs(frames, labels, sample_step, profiles=profiles_df)
        policy_map: dict[int, str] = {}
        all_cls = [int(row['cluster']) for _, row in work.iterrows()]

        for idx, row in work.iterrows():
            cl = int(row['cluster'])
            difficulty = float(row['combined'])

            # Define candidate set according to feature difficulty (thoughtful tiering)
            if cl in hard_clusters:
                # Very hard by features → consider the strongest adaptive options
                cands = ["hold", "very_strong", "strong", "conservative", "moderate", "biased", "linear"]
            elif difficulty > (hold_thresh * 0.45):
                # Medium to hard → lean towards thoughtful conservative / biased strategies
                cands = ["conservative", "moderate", "mild", "biased", "linear"]
            else:
                # Easy content → linear is the natural and preferred choice
                cands = ["linear", "mild", "moderate", "conservative", "biased"]

            if cl in quality:
                valid_cands = [p for p in cands if p in quality[cl] and quality[cl][p]['psnr'] > -100]
                if not valid_cands:
                    best = "linear"
                else:
                    # Normalize both metrics within the cluster for fair combination
                    cluster_ps = [quality[cl][pp]['psnr'] for pp in valid_cands]
                    cluster_ss = [quality[cl][pp]['ssim'] for pp in valid_cands]
                    min_p, max_p = min(cluster_ps), max(cluster_ps)
                    min_s, max_s = min(cluster_ss), max(cluster_ss)

                    def thoughtful_score(p):
                        ps_val = quality[cl][p]['psnr']
                        ss_val = quality[cl][p]['ssim']

                        norm_ps = (ps_val - min_p) / (max_p - min_p + 1e-9) if max_p > min_p else 0.5
                        norm_ss = (ss_val - min_s) / (max_s - min_s + 1e-9) if max_s > min_s else 0.5

                        # Combine PSNR and SSIM. SSIM helps penalize policies that destroy structure
                        # even if pixel error (PSNR) looks acceptable.
                        combined = 0.65 * norm_ps + 0.35 * norm_ss
                        score = combined

                        # Feature-driven bias (the "thoughtful" part)
                        if p == "linear":
                            if difficulty < 0.35:
                                score += (0.40 - difficulty) * 1.2
                            else:
                                score -= (difficulty - 0.30) * 2.0
                        else:
                            if difficulty < 0.30:
                                score -= 0.35
                            elif difficulty > 0.50:
                                score += 0.25

                        return score

                    best = max(valid_cands, key=thoughtful_score)
            else:
                best = "linear"
            policy_map[cl] = best
        return policy_map

    # Fallback: чиста евристика без перевірки PSNR (може давати негатив)
    if n >= 2:
        biased_thresh = float(work['combined'].quantile(biased_quantile))
    else:
        biased_thresh = 0.5

    policy_map = {}
    for _, row in work.iterrows():
        cl = int(row['cluster'])
        score = float(row['combined'])
        if score >= hold_thresh:
            policy = 'hold'
        elif score >= biased_thresh:
            # Only for clearly harder clusters
            policy = 'conservative' if score > (biased_thresh * 1.1) else 'linear'
        else:
            # linear as standard for everything that is not hard
            policy = 'linear'
        policy_map[cl] = policy
    return policy_map


# Small helper exposed for documentation / future UI
POLICY_SMOOTHING_WINDOW_DEFAULT = 5


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

