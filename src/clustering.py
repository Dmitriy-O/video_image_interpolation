"""
Модуль model-based кластеризації.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from kmodes.kmodes import KModes


def run_ward_clustering(features: pd.DataFrame, n_clusters: int, feature_cols: list[str] | None = None, random_state: int | None = None) -> np.ndarray:
    """
    Ward Hierarchical Clustering (model-based).
    
    Note: Ward linkage is deterministic, so random_state is accepted only for API consistency
    and is ignored.
    """
    if feature_cols is None:
        cols_to_drop = ['frame_idx']
    else:
        cols_to_drop = [c for c in features.columns if c not in feature_cols]
    X = features.drop(columns=[c for c in cols_to_drop if c in features.columns]).values
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean')
    return model.fit_predict(X)


def run_kmodes_clustering(features_disc: pd.DataFrame, n_clusters: int, feature_cols: list[str] | None = None, random_state: int = 42) -> np.ndarray:
    """KModes — категориальна model-based кластеризація (аналог COOLCAT)."""
    if feature_cols is None:
        cols_to_drop = ['frame_idx']
    else:
        cols_to_drop = [c for c in features_disc.columns if c not in feature_cols]
    X = features_disc.drop(columns=[c for c in cols_to_drop if c in features_disc.columns]).values.astype(int)
    km = KModes(n_clusters=n_clusters, init='Huang', n_init=5, random_state=random_state)
    return km.fit_predict(X)


def get_cluster_stats(labels: np.ndarray) -> pd.DataFrame:
    unique, counts = np.unique(labels, return_counts=True)
    return pd.DataFrame({
        'Кластер': unique,
        'Кількість кадрів': counts,
        'Частка (%)': np.round(counts / len(labels) * 100, 1)
    })
    
# Додаємо синонім для сумісності з app.py
get_cluster_statistics = get_cluster_stats


def compare_clusterings(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """
    Порівняння двох розбиттів (Ward vs KModes).
    Повертає ARI, NMI та % точної згоди.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    if len(labels_a) != len(labels_b):
        raise ValueError("Labels must have same length")
    ari = float(adjusted_rand_score(labels_a, labels_b))
    nmi = float(normalized_mutual_info_score(labels_a, labels_b))
    agreement = float(np.mean(labels_a == labels_b) * 100)
    return {
        'adjusted_rand_index': round(ari, 3),
        'normalized_mutual_info': round(nmi, 3),
        'exact_agreement_percent': round(agreement, 1)
    }


def compute_cluster_profiles(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Профілі кластерів: середні значення ознак + розмір.
    Допомагає інтерпретувати, чому кадри потрапили в кластер (високий рух / текстура?).
    """
    if feature_cols is None:
        feature_cols = [c for c in feature_df.columns if c != 'frame_idx']

    df = feature_df.copy()
    df['cluster'] = labels
    profiles = []
    for cl in sorted(np.unique(labels)):
        sub = df[df['cluster'] == cl]
        row = {'cluster': int(cl), 'size': len(sub)}
        for col in feature_cols:
            if col in sub:
                row[f'{col}_mean'] = round(float(sub[col].mean()), 3)
                row[f'{col}_std'] = round(float(sub[col].std()), 3)
        profiles.append(row)
    return pd.DataFrame(profiles)


def recommend_policies_from_features(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    hold_quantile: float = 0.80,
    biased_quantile: float = 0.50,
) -> dict[int, str]:
    """
    Зручна обгортка: будує профілі і рекомендує політики.

    Використовує комбінований скор (рух + текстура одночасно).
    'hold' — тільки топ ~20-25% (quantile 0.80 за замовчуванням) найскладніших кластерів.
    Інші: 'biased' (вище медіани) або 'linear'.

    Це узгоджено з recommend_policies_from_profiles (виправлення проблеми,
    коли рекомендація давала гірший результат за baseline).
    """
    prof = compute_cluster_profiles(feature_df, labels)
    if prof.empty:
        return {}

    # Знаходимо колонки з середніми значеннями
    motion_col = next((c for c in prof.columns if 'motion_mean' in c and '_mean' in c), None)
    texture_col = next((c for c in prof.columns if 'texture_laplacian' in c and '_mean' in c), None)

    if motion_col is None or texture_col is None:
        # fallback: все linear
        return {int(row['cluster']): 'linear' for _, row in prof.iterrows()}

    # Нормалізуємо
    m = prof[motion_col]
    t = prof[texture_col]
    motion_norm = (m / m.max()) if m.max() > 0 else pd.Series([0.0] * len(prof))
    texture_norm = (t / t.max()) if t.max() > 0 else pd.Series([0.0] * len(prof))

    combined = (motion_norm + texture_norm) / 2.0
    work = prof[['cluster']].reset_index(drop=True)
    work['combined'] = combined.reset_index(drop=True)

    n = len(work)
    if n >= 2:
        hold_thresh = float(work['combined'].quantile(hold_quantile))
        biased_thresh = float(work['combined'].quantile(biased_quantile))
    else:
        hold_thresh = 0.85
        biased_thresh = 0.5

    policy_map: dict[int, str] = {}
    for _, row in work.iterrows():
        cl = int(row['cluster'])
        score = float(row['combined'])
        if score >= hold_thresh:
            policy = 'hold'
        elif score >= biased_thresh:
            policy = 'biased'
        else:
            policy = 'linear'
        policy_map[cl] = policy

    return policy_map
