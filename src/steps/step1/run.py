import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from src.steps.step1.assigned_leaf_to_point import prepare_bin_data, build_rtree, assign_bin_to_points
from src.steps.step1.quadtree_builder import QuadTree, QuadTreeConfig

def construct_quadtree(cfg, gene_df, all_gene_ids, markers, type_system):
    import time
    start_time = time.time()
    config = QuadTreeConfig(
        limit_depth=cfg.qtree_config.limit_depth, 
        assign_thresh=cfg.qtree_config.assign_thresh, 
        assign_thresh_lower=cfg.qtree_config.assign_thresh_lower, 
        sim_thresh=cfg.qtree_config.sim_thresh, 
        neighbor_weight=cfg.qtree_config.neighbor_weight, 
        score_margin=cfg.qtree_config.score_margin
    )


    quadtree = QuadTree(
        width=cfg.base_config.width,
        height=cfg.base_config.height,
        data=gene_df.copy(),
        all_gene_ids=all_gene_ids,
        markers=markers,
        valid_cluster_ids=type_system.valid_type_ids,
        background_id=type_system.background_id,
        unknown_id=type_system.unknown_id,
        num_clusters=type_system.C,
        config=config,
        sim_method=cfg.qtree_config.sim_method,
    )
        
    quadtree.generate_quadtree()
    end_time = time.time()
    
    print(f"QuadTree construction time: {end_time - start_time} seconds")
    leaves_df = quadtree.collect_leaves()
    return quadtree, leaves_df

def assign_point_to_nodes(leaves_df: pd.DataFrame, gene_df: pd.DataFrame, unique_celltypes_list: list):
    leaves_df['leaf_id'] = range(len(leaves_df))
    qtree_df = leaves_df.copy()
    gene_df["point_id"] = range(len(gene_df))
    gene_df.reset_index(inplace=True)   
    
    qtree_df = leaves_df.copy()
    bin_data = prepare_bin_data(qtree_df=qtree_df, unique_celltypes=unique_celltypes_list)
    rtree_idx = build_rtree(bin_data)

    import time
    start_time = time.time()
    input_df = gene_df.copy()
    gene_df_with_bins = assign_bin_to_points(input_df, bin_data, rtree_idx)
    end_time = time.time()
    print(f"Assign bin to points (fast) time: {end_time - start_time} seconds")
    gene_df_with_bins.info()
    return gene_df_with_bins


def soft_weights(raw, tau):
    eps =1e-12
    z = raw / (tau + eps)
    z = z - z.max()      # 安定化
    w = np.exp(z)
    return w / (w.sum() + eps)

def compute_celltype_weights_from_quadtree(
    points_df: pd.DataFrame,
    leaves_df: pd.DataFrame,
    n_celltypes: int,
    unknown_label: int = -1,
    background_label: int = -2,
    boundary_threshold_px: float = 10.0,
    neighbor_score_threshold: float = 0.2,
    soft_k: int = 3,
    alpha: float = 2.0,
    beta: float = 1.5,
    min_weight: float = 1e-3,
):
    N = len(points_df)
    T = n_celltypes
    rows, cols, data = [], [], []

    # ==== 1. leaf情報を辞書化 ====
    leaf_dict = {
        row.leaf_id: {
            "bbox": (row.x0, row.y0, row.x0 + row.w, row.y0 + row.h),
            "label": int(row.qtree_celltype),
            "scores": np.array([row[f"fv_{t}"] for t in range(T)]),
            "old_scores": soft_weights(
                raw=np.array([row[f"fv_{t}"] for t in range(T)]),
                tau=np.log1p(row.n_gene)
            ),
            "entropy": row.entropy,
        }
        for _, row in leaves_df.iterrows()
    }

    leaf_ids = list(leaf_dict.keys())
    leaf_centers = np.array([
        [(v["bbox"][0] + v["bbox"][2]) / 2, (v["bbox"][1] + v["bbox"][3]) / 2]
        for v in leaf_dict.values()
    ])

    # ==== 2. KDTree で候補隣接を探索 ====
    tree = cKDTree(leaf_centers)
    dists, neighbor_idx = tree.query(leaf_centers, k=12)

    # ==== 3. 隣接関係を構築 ====
    eps = 1e-6
    adjacency = {lid: {"left": [], "right": [], "top": [], "bottom": []} for lid in leaf_ids}
    for i, lid in enumerate(leaf_ids):
        x0, y0, x1, y1 = leaf_dict[lid]["bbox"]
        for ni in neighbor_idx[i]:
            nb_id = leaf_ids[ni]
            if nb_id == lid:
                continue
            bx0, by0, bx1, by1 = leaf_dict[nb_id]["bbox"]

            x_overlap = not (x1 <= bx0 or bx1 <= x0)
            y_overlap = not (y1 <= by0 or by1 <= y0)
            if abs(x1 - bx0) < eps and y_overlap:
                adjacency[lid]["right"].append(nb_id)
            elif abs(bx1 - x0) < eps and y_overlap:
                adjacency[lid]["left"].append(nb_id)
            elif abs(y1 - by0) < eps and x_overlap:
                adjacency[lid]["bottom"].append(nb_id)
            elif abs(by1 - y0) < eps and x_overlap:
                adjacency[lid]["top"].append(nb_id)

    # ==== 4. 発現点ごとの重み計算 ====
    for idx, p in points_df.iterrows():
        leaf_id = int(p.leaf_id)
        
        if leaf_id == -1 or leaf_id not in leaf_dict:
            continue
        leaf = leaf_dict[leaf_id]
        
        label = leaf["label"]
        scores = leaf["scores"]
        weights = np.zeros(T, dtype=float)
        x, y = p.local_pixel_x, p.local_pixel_y
        x0, y0, x1, y1 = leaf["bbox"]

        # 背景スキップ
        if label == background_label:
            continue

        dist_edge = {
            "left": x - x0,
            "right": x1 - x,
            "top": y - y0,
            "bottom": y1 - y,
        }
        nearest_side = min(dist_edge, key=dist_edge.get)
        dist_min = dist_edge[nearest_side]

        # case A: Unknown leaf
        if label == unknown_label:
            top_idx = np.argsort(scores)[-soft_k:]
            soft_scores = scores[top_idx] ** alpha
            soft_scores /= np.sum(soft_scores) + 1e-12
            weights[top_idx] = soft_scores

            for nb_id in adjacency[leaf_id][nearest_side]:
                nb_leaf = leaf_dict[nb_id]
                nb_label = nb_leaf["label"]
                if nb_label not in (unknown_label, background_label):
                    nb_scores = nb_leaf["scores"]
                    if np.max(nb_scores) > neighbor_score_threshold:
                        t = np.argmax(nb_scores)
                        weights[t] += beta * np.max(nb_scores)
            weights /= np.sum(weights) + 1e-12

        # case B: 確定 leaf
        else:
            t_self = int(label)
            if dist_min >= boundary_threshold_px:
                weights[t_self] = 1.0
            else:
                nb_ids = adjacency[leaf_id][nearest_side]
                nb_labels = [leaf_dict[nid]["label"] for nid in nb_ids]
                if all(lbl == label for lbl in nb_labels):
                    weights[t_self] = 1.0
                else:
                    neighbor_types = [
                        int(lbl)
                        for lbl in nb_labels
                        if lbl not in (unknown_label, background_label, label)
                    ]
                    ids_all = [t_self] + neighbor_types
                    scores_all = [scores[t_self]] + [scores[t] for t in neighbor_types]
                    s = np.array(scores_all) ** alpha
                    s /= np.sum(s) + 1e-12
                    weights[ids_all] = s

        # 疎行列構築
        nz = np.where(weights > min_weight)[0]
        weights /= np.sum(weights) + 1e-12
        for t in nz:
            rows.append(idx)
            cols.append(t)
            data.append(weights[t])

    weights_sparse = csr_matrix((data, (rows, cols)), shape=(N, T))
    return weights_sparse

# 拡張性の都合上、この書き方をする
def run_step1(cfg, gene_df, gene_registry, type_system, unique_celltypes_list, markers):
    tmp = gene_df.copy() # 元々の gene_df を変更しないようにコピー

    all_gene_ids = gene_df["gene"].map(
        gene_registry.gene_to_gid
    ).to_numpy(dtype=np.int32)
    
    quadtree, leaves_df = construct_quadtree(
        cfg, 
        gene_df, # gene_df_marker.copy(), 
        all_gene_ids, 
        markers, 
        type_system
    ) 
    
    gene_df_with_node = assign_point_to_nodes(leaves_df, tmp, unique_celltypes_list)
    points_df = gene_df_with_node.copy()

    weights_sparse = compute_celltype_weights_from_quadtree(
        points_df=points_df,
        leaves_df=leaves_df,
        n_celltypes=type_system.C,
        unknown_label=type_system.unknown_id,
        background_label=type_system.background_id,
        boundary_threshold_px=cfg.weight_calculation_config.boundary_threshold_px,
        neighbor_score_threshold=cfg.weight_calculation_config.neighbor_score_threshold,
        soft_k=cfg.weight_calculation_config.soft_k,
        alpha=cfg.weight_calculation_config.alpha, # α↑で支配的細胞種を強調，β↑で隣接葉影響を強める
        beta=cfg.weight_calculation_config.beta,
        min_weight=cfg.weight_calculation_config.min_weight,    
    )

    print(weights_sparse.shape)  # (N_points, n_celltypes)
    print(weights_sparse.nnz, "non-zero entries")
    return quadtree, leaves_df, gene_df_with_node, weights_sparse