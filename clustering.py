import numpy as np


# ---- Ізотонічна нормалізація ----
# Формула 11: ділимо кожен елемент на суму свого стовпця
# Формула 12: w_i = сума нормованого рядка
def isotonic_normalize(matrix):
    col_sums = matrix.sum(axis=0)
    norm = matrix / col_sums
    w = norm.sum(axis=1)
    return norm, col_sums, w


# ---- Ізоморфна нормалізація ----
# Формула 14: ділимо кожен рядок на його евклідову норму
def isomorphic_normalize(matrix):
    row_norms = np.sqrt((matrix ** 2).sum(axis=1, keepdims=True))
    return matrix / row_norms


# ---- Матриця відстаней (ізотонічна) ----
# Формула 13: d_ij = |w_i - w_j|
def calc_distances_isotonic(w):
    n = len(w)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i][j] = abs(w[i] - w[j])
    return D


# ---- Матриця відстаней (ізоморфна) ----
# Формула 15: d_ij = евклідова відстань між нормованими рядками
def calc_distances_isomorphic(norm):
    n = norm.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i][j] = np.sqrt(np.sum((norm[i] - norm[j]) ** 2))
    return D


# ---- Критичний радіус (метод куль) ----
# Формула 16: для кожного об'єкта знаходимо найближчого сусіда,
# потім беремо максимум з цих мінімальних відстаней
def calc_critical_radius(D):
    n = D.shape[0]
    min_dists = []
    for i in range(n):
        row_without_self = [D[i][j] for j in range(n) if j != i]
        min_dists.append(min(row_without_self))
    r = max(min_dists)
    return r, min_dists


# ---- Формування кластерів методом куль ----
# Якщо d_ij <= r — об'єкти в одному кластері
def form_clusters(D, r, labels):
    n = D.shape[0]
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [labels[i]]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and D[i][j] <= r:
                cluster.append(labels[j])
                visited[j] = True
        clusters.append(cluster)

    return clusters


# ---- Побудова дендриту ----
# Алгоритм мінімального остовного дерева
def build_dendrite(D, labels):
    n = D.shape[0]
    in_tree = [False] * n
    in_tree[0] = True
    edges = []

    for _ in range(n - 1):
        best_dist = float('inf')
        best_i, best_j = 0, 0
        for i in range(n):
            if in_tree[i]:
                for j in range(n):
                    if not in_tree[j] and D[i][j] < best_dist:
                        best_dist = D[i][j]
                        best_i, best_j = i, j
        in_tree[best_j] = True
        edges.append((labels[best_i], labels[best_j], best_dist))

    return edges


# ---- Критична відстань дендриту ----
# Формула 18: D_crit = середнє + стандартне відхилення
def calc_critical_distance(edges):
    dists = [d for _, _, d in edges]
    return float(np.mean(dists) + np.std(dists))


# ---- Розрив дендриту ----
# Видаляємо ребра де відстань > D_crit, отримуємо фінальні кластери
def cut_dendrite(edges, d_crit):
    all_nodes = set()
    for u, v, _ in edges:
        all_nodes.add(u)
        all_nodes.add(v)

    parent = {node: node for node in all_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for u, v, d in edges:
        if d <= d_crit:
            union(u, v)

    clusters = {}
    for node in all_nodes:
        root = find(node)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(node)

    return list(clusters.values())


# ---- Запуск повного аналізу ----
def run_analysis(matrix, labels):
    matrix = np.array(matrix, dtype=float)

    # Ізотонічна
    norm_iso, col_sums, w = isotonic_normalize(matrix)
    D_iso = calc_distances_isotonic(w)
    r_iso, min_dists_iso = calc_critical_radius(D_iso)
    clusters_iso = form_clusters(D_iso, r_iso, labels)
    edges_iso = build_dendrite(D_iso, labels)
    d_crit_iso = calc_critical_distance(edges_iso)
    final_iso = cut_dendrite(edges_iso, d_crit_iso)

    # Ізоморфна
    norm_ism = isomorphic_normalize(matrix)
    D_ism = calc_distances_isomorphic(norm_ism)
    r_ism, min_dists_ism = calc_critical_radius(D_ism)
    clusters_ism = form_clusters(D_ism, r_ism, labels)
    edges_ism = build_dendrite(D_ism, labels)
    d_crit_ism = calc_critical_distance(edges_ism)
    final_ism = cut_dendrite(edges_ism, d_crit_ism)

    return {
        "labels": labels,
        "isotonic": {
            "col_sums": col_sums.tolist(),
            "norm": norm_iso.tolist(),
            "w": w.tolist(),
            "distances": D_iso.tolist(),
            "min_dists": min_dists_iso,
            "r": round(r_iso, 6),
            "clusters_balls": clusters_iso,
            "edges": [(u, v, round(d, 6)) for u, v, d in edges_iso],
            "d_crit": round(d_crit_iso, 6),
            "clusters_final": final_iso,
        },
        "isomorphic": {
            "norm": norm_ism.tolist(),
            "distances": D_ism.tolist(),
            "min_dists": min_dists_ism,
            "r": round(r_ism, 6),
            "clusters_balls": clusters_ism,
            "edges": [(u, v, round(d, 6)) for u, v, d in edges_ism],
            "d_crit": round(d_crit_ism, 6),
            "clusters_final": final_ism,
        },
    }