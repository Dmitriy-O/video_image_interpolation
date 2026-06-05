# Архітектура та дизайн проєкту

**Адаптивна інтерполяція відео на основі model-based кластеризації**

Це документ описує структуру, компоненти, потоки даних та ключові дизайнерські рішення проєкту у стилі UML-діаграм (з використанням Mermaid для рендеру в GitHub / VS Code / Markdown viewers).

---

## 1. Огляд проєкту

Проєкт реалізує **контент-адаптивну** систему інтерполяції кадрів відео. 

Основна ідея:
- Класична лінійна інтерполяція погано працює на складному контенті (швидкий рух, текстура, зміни яскравості).
- Застосовується **model-based кластеризація** (Ward + KModes) для автоматичного виявлення типів контенту.
- Для кожного кластера обирається своя **політика інтерполяції** (`linear`, `hold`, `biased`).

Оцінка якості — **тільки на часовій структурі** (temporal triplets `i-1, i, i+1`), а не на випадкових парах.

---

## 2. Структура директорій

```
video_Interpolation_course_w/
├── app.py                      # Точка входу. Gradio UI (весь інтерфейс)
├── requirements.txt
├── README.md
├── pack_for_share.py           # Утиліта для створення легкого ZIP-архіву
│
├── src/                        # Бізнес-логіка (ядро)
│   ├── feature_extraction.py   # Витягування ознак + дискретизація
│   ├── clustering.py           # Ward, KModes, порівняння, профілі кластерів
│   ├── interpolation.py        # Temporal PSNR + симуляція адаптивних політик
│   └── visualization.py        # Побудова графіків (Plotly + matplotlib)
│
├── tests/
│   └── test_core_logic.py      # Unit-тести ключової логіки
│
└── docs/
    └── architecture.md         # Цей документ
```

---

## 3. Компонентна діаграма (Component Diagram)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        direction TB
        GradioUI["app.py<br/>Gradio Blocks<br/>• Параметри<br/>• Вкладки (Кластеризація / Профілі / Якість / Стратегії)<br/>• State management"]
    end

    subgraph Core["Core Domain Layer (src/)"]
        direction LR
        FE["feature_extraction.py<br/>• extract_frame_features()<br/>• discretize_features()"]
        CL["clustering.py<br/>• run_ward_clustering()<br/>• run_kmodes_clustering()<br/>• compute_cluster_profiles()<br/>• compare_clusterings()"]
        IP["interpolation.py<br/>• compute_temporal_interp_errors()<br/>• simulate_adaptive_policies()<br/>• recommend_policies_from_profiles()"]
        VIS["visualization.py<br/>• plot_*<br/>• get_cluster_representatives()"]
    end

    subgraph Infrastructure["Infrastructure"]
        OpenCV["OpenCV (cv2)"]
        SK["scikit-learn<br/>AgglomerativeClustering"]
        KM["kmodes.KModes"]
        Pandas["pandas + numpy"]
        Plotly["Plotly Express / Graph Objects"]
    end

    GradioUI -->|викликає| FE
    GradioUI -->|викликає| CL
    GradioUI -->|викликає| IP
    GradioUI -->|викликає| VIS

    FE --> OpenCV
    FE --> Pandas
    CL --> SK
    CL --> KM
    CL --> Pandas
    IP --> OpenCV
    IP --> Pandas
    VIS --> Plotly
    VIS --> Pandas
```

---

## 4. Діаграма потоків даних (Data Flow)

```mermaid
flowchart TD
    A[Завантаження відео<br/>app.py: _load_frames] --> B[Субдискретизація]
    B --> C[Витягування ознак<br/>extract_frame_features]
    
    C --> D1[Безперервні ознаки<br/>для Ward]
    C --> D2[Дискретизація<br/>discretize_features → для KModes]
    
    D1 --> E1[run_ward_clustering]
    D2 --> E2[run_kmodes_clustering]
    
    E1 --> F[compute_cluster_profiles]
    E1 --> G[compare_clusterings<br/>ARI + NMI + Exact Agreement]
    
    F --> H[Temporal Evaluation<br/>compute_temporal_interp_errors<br/>на triplets i-1,i,i+1]
    
    H --> I[Візуалізації + Галерея репрезентативних кадрів]
    H --> J[Початкові політики: linear для всіх кластерів]
    
    J --> K[Інтерактивне редагування політик<br/>вкладка «Стратегії»]
    K --> L[simulate_adaptive_policies]
    L --> M[Порівняння Δ PSNR<br/>Uniform vs Adaptive]
    
    M --> N[Експорт результатів у JSON]
```

---

## 5. Sequence Diagram — Основний пайплайн аналізу

```mermaid
sequenceDiagram
    participant User
    participant Gradio as Gradio UI (app.py)
    participant FE as feature_extraction
    participant CL as clustering
    participant IP as interpolation
    participant VIS as visualization

    User->>Gradio: Завантажує відео + параметри (k, bins, seed, ...)
    Gradio->>Gradio: _load_frames() → список BGR кадрів
    Gradio->>FE: extract_frame_features(frames, use_motion)
    FE-->>Gradio: feature_df (12+ колонок)

    Gradio->>FE: discretize_features(...)
    FE-->>Gradio: feature_disc

    Gradio->>CL: run_ward_clustering(feature_df)
    CL-->>Gradio: ward_labels

    Gradio->>CL: run_kmodes_clustering(feature_disc)
    CL-->>Gradio: kmodes_labels

    Gradio->>CL: compute_cluster_profiles(...)
    CL-->>Gradio: profiles

    Gradio->>CL: compare_clusterings(ward, kmodes)
    CL-->>Gradio: {ARI, NMI, agreement}

    Gradio->>IP: compute_temporal_interp_errors(frames, ward_labels, sample_step)
    IP-->>Gradio: (detailed, temporal_summary)

    Gradio->>VIS: plot_cluster_timeline_plotly(...)
    Gradio->>VIS: plot_temporal_psnr(...)
    Gradio->>VIS: get_cluster_representatives(...)

    Gradio-->>User: Результати + вкладки + таблиця політик
```

---

## 6. Діаграма класів / модулів (Class Diagram style)

Оскільки проєкт написаний у функціональному стилі (не важкий ООП), діаграма показує основні функції як операції модулів.

```mermaid
classDiagram
    class feature_extraction {
        +extract_frame_features(frames, use_motion) pd.DataFrame
        +discretize_features(df, n_bins) pd.DataFrame
        +get_feature_display_map() Dict
    }

    class clustering {
        +run_ward_clustering(features, n_clusters, random_state) ndarray
        +run_kmodes_clustering(features_disc, n_clusters, random_state) ndarray
        +get_cluster_statistics(labels) pd.DataFrame
        +compare_clusterings(labels_a, labels_b) dict
        +compute_cluster_profiles(feature_df, labels) pd.DataFrame
        +recommend_policies_from_features(...) dict
    }

    class interpolation {
        +simple_linear_interpolation(f1, f2, alpha)
        +calculate_psnr(img1, img2) float
        +compute_temporal_interp_errors(frames, labels, sample_step, ...) tuple[DataFrame, DataFrame]
        +simulate_adaptive_policies(frames, labels, policy_map, ...) dict
        +recommend_policies_from_profiles(profiles_df) dict[int,str]
    }

    class visualization {
        +plot_cluster_timeline_plotly(labels, title) Figure
        +plot_temporal_psnr(summary_df) Figure
        +plot_policy_comparison(per_cluster_df) Figure
        +get_cluster_representatives(frames, labels) dict
    }

    class app {
        +process_video(...) → tuple (багато outputs)
        +recommend_policies_ui(ctx)
        +run_simulation(policy_df, ctx, ...)
        +build_interface() gr.Blocks
    }

    app --> feature_extraction
    app --> clustering
    app --> interpolation
    app --> visualization
```

---

## 7. Ключові структури даних

| Структура | Де використовується | Опис |
|-----------|---------------------|------|
| `frames: list[np.ndarray]` | app, interpolation, visualization | Список BGR кадрів (після ресайзу) |
| `feature_df: pd.DataFrame` | feature_extraction → clustering | Колонки: `frame_idx`, `mean_*`, `std_*`, `brightness`, `contrast`, `texture_laplacian`, `motion_*` |
| `feature_disc: pd.DataFrame` | discretize_features | Ті самі колонки, але значення — категорії 0..n_bins-1 (int) |
| `labels: np.ndarray[int]` | Всюди | Мітки кластерів (0..k-1) для кожного кадру |
| `profiles: pd.DataFrame` | clustering → app | `cluster`, `size`, `{feature}_mean`, `{feature}_std` |
| `temporal_summary: pd.DataFrame` | interpolation | `Кластер`, `Середній PSNR (temporal)`, `Std PSNR`, ... |
| `policy_map: dict[int, str]` | interpolation | `{cluster_id: "linear" \| "hold" \| "biased"}` |
| `ctx (Gradio State)` | app.py | Зберігає `frames`, `ward_labels`, `profiles`, `sample_step`, `random_seed` між кліками |

---

## 8. Підтримувані політики інтерполяції

```mermaid
flowchart LR
    Linear["linear<br/>α = 0.5 (стандарт)"] 
    Hold["hold<br/>повторити попередній кадр<br/>(для високого руху)"]
    Biased["biased<br/>α = 0.35 (зсув до left)"]
    
    Policy["Політика кластера"] --> Linear
    Policy --> Hold
    Policy --> Biased
```

- **linear** — базова лінійна інтерполяція.
- **hold** — консервативна стратегія при високому русі/текстурі (уникає артефактів).
- **biased** — проміжний варіант.

---

## 9. Дизайнерські рішення та принципи

1. **Два незалежні кластеризації** (Ward на неперервних ознаках + KModes на дискретних) — для крос-валідації.
2. **Temporal-only evaluation** — оцінка тільки на реальних послідовних triplet'ах (науково коректніше).
3. **Інтерпретованість** — профілі кластерів + прості правила рекомендацій замість чорного ящика.
4. **Gradio State** — дозволяє зберігати важкі дані (кадри) між різними вкладками без повторного завантаження відео.
5. **Поділ на чисті функції** — кожен модуль `src/` легко тестується (див. `tests/test_core_logic.py`).
6. **Мінімалізм** — немає глибокого навчання, тільки класичні алгоритми курсу.

---

## 10. Як розширювати проєкт (рекомендації)

- Додати нові ознаки → `feature_extraction.py`
- Додати новий алгоритм кластеризації → `clustering.py` + оновити `compare_clusterings`
- Додати нову політику → `interpolation.py` (`simulate_adaptive_policies`) + UI
- Покращити візуалізацію → `visualization.py`
- Додати нові тести → `tests/test_core_logic.py`

---

## 11. Запуск тестів

```bash
pytest tests/ -q
```

---

*Документ згенеровано для курсового проєкту. Оновлюйте діаграми при зміні архітектури.*
