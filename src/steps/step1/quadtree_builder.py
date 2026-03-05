from typing import Optional
import numpy as np
import pandas as pd

def softmax(x, tau=1.0):
    x = x / tau
    x = x - np.max(x)
    expx = np.exp(x)
    return expx / expx.sum()

# ----------------------------
# Data structures
# ----------------------------
class Node:
    def __init__(self, x0: float, y0: float, w: float, h: float, depth: int = 0,
                 parent: Optional["Node"] = None, 
                 data_idx: Optional[np.ndarray] = None, 
                 all_gene_ids: Optional[np.ndarray] = None):
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h
        
        self.depth = depth
        self.parent = parent
        self.children: Optional[list[Node]] = None  # order: tl, tr, bl, br
        
        self.data_idx = data_idx if data_idx is not None else np.array([], dtype=np.int32)
        self.feature_vector: Optional[np.ndarray] = None
        self.entropy: Optional[float] = None
        self.celltype: Optional[int] = None
        self.all_gene_ids = all_gene_ids
        self.gene_ids, self.gene_counts = self._build_gene_stats()

    def _build_gene_stats(self) -> tuple[np.ndarray, np.ndarray]:
        if self.data_idx.size == 0 or self.all_gene_ids is None:
            empty_ids = np.empty(0, dtype=np.int32)
            empty_counts = np.empty(0, dtype=np.int32)
            return empty_ids, empty_counts

        genes = self.all_gene_ids[self.data_idx]
        gene_ids, counts = np.unique(genes, return_counts=True)
        return gene_ids.astype(np.int32), counts.astype(np.int32)



class QuadTreeConfig:
    def __init__(self,
                 limit_depth: int = 7,
                 neighbor_weight: float = 0.7,
                 sim_thresh: float = 0.9,
                 assign_thresh: float = 0.9,
                 assign_thresh_lower: float = -1,
                 bkg_thresh: float = 0.001,
                 param: float = 0.0,
                 score_margin: float = 0.1,
                 entropy_thresh: Optional[float] = None):
        self.limit_depth = limit_depth
        self.neighbor_weight = neighbor_weight
        self.sim_thresh = sim_thresh
        self.assign_thresh = assign_thresh
        self.assign_thresh_lower = -1 # 不使用
        self.bkg_thresh = bkg_thresh
        self.param = param
        self.score_margin = score_margin  # margin between top2 scores to consider split
        self.entropy_thresh = entropy_thresh  # optional ΔH threshold for split decision

class QuadTree:
    def __init__(
        self, 
        width: float,  # pixel
        height: float, # pixel
        data: pd.DataFrame, # ST data , num_points x ['local_pixel_x', 'local_pixel_y', 'gene']
        all_gene_ids: np.ndarray,  # shape (num_points,)
        markers: dict[int, dict],           # cluster_id -> marker dict
        valid_cluster_ids: list[int],       # Unknown / Background を除く
        background_id: int,
        unknown_id: int,
        num_clusters: int,
        config: QuadTreeConfig = QuadTreeConfig(),
        sim_method: str = "cosine",
    ):
        # Store raw dataframe and create numpy views once
        self.data_df = data.reset_index(drop=True)
        self.N = len(self.data_df) # number of points
        # Expect columns 'local_pixel_x', 'local_pixel_y', 'gene' in data_df
        self.xs = self.data_df['local_pixel_x'].values if 'local_pixel_x' in self.data_df.columns else np.zeros(self.N)
        self.ys = self.data_df['local_pixel_y'].values if 'local_pixel_y' in self.data_df.columns else np.zeros(self.N)

        self.all_gene_ids = all_gene_ids  # shape = (N_total,)
        self.markers = markers  # precomputed marker indices per cell type
        
        self.valid_cluster_ids = valid_cluster_ids
        self.unknown_id = unknown_id
        self.background_id = background_id  
        
        self.C = num_clusters # Number of valid clusters (excluding unknown/background)
        
        self.config = config
        self.sim_method = sim_method

        # Root node holds all indices initially
        root_idx = np.arange(self.N, dtype=np.int32)
        self.all_gene_ids = all_gene_ids  # shape = (N_total,)
        self.root = Node(0.0, 0.0, width, height, depth=0, parent=None, data_idx=root_idx, all_gene_ids=all_gene_ids)

        # Precompute root feature & entropy
        self.root.feature_vector = self.calc_feature_vector(self.root)
        self.root.entropy = self.calc_entropy(self.root)
        


    def calc_feature_vector(self, node: Node) -> np.ndarray:
        """
        Return: np.ndarray, shape (C,)
        """
        if node.gene_ids is None or node.gene_ids.size == 0:
            return np.zeros(self.C, dtype=float)

        gene_ids = node.gene_ids          # int array (mapped gene index)
        counts   = node.gene_counts       # same shape

        # rank-like score
        scores_g = np.log1p(counts + 1.0)
        total_score = scores_g.sum() + 1e-12
        
        lam = 1.0

        fv = np.zeros(self.C, dtype=float)
        for i, cid in enumerate(self.valid_cluster_ids):
            mk = self.markers[cid]
            pos_score = 0.0
            neg_score = 0.0


            pos_gids = mk["pos_gids"]
            pos_weights = mk.get("pos_weights", 1.0)
            if pos_gids.size:                        
                mask = np.isin(gene_ids, pos_gids)
                pos_score = (scores_g[mask]).sum()
                # np.sqrt(scores_g[mask]).sum() # scores_g[mask].sum() / n_detected # * (np.count_nonzero(mask) / pos_gids.size) # / total

            neg_gids = mk["neg_gids"]
            if neg_gids.size:
                mask = np.isin(gene_ids, neg_gids)
                neg_score =  (scores_g[mask]).sum()  # lam * np.sqrt(scores_g[mask]).sum() # lam  *  scores_g[mask].sum() / n_detected # * (np.count_nonzero(mask) / neg_gids.size)# / total
            
            fv[i] =( pos_score / total_score + 1e-12 )#(pos_score + neg_score + 1e-12))# **2
        

        # L1正規化
        s = fv.sum()
        if s > 0:
            fv = fv / s    


        return fv


    def calc_feature_vector_tmp(self, node: Node) -> np.ndarray:
        """
        Return: np.ndarray, shape (C,)
        """
        if node.gene_ids is None or node.gene_ids.size == 0:
            return np.zeros(self.C, dtype=float)

        gene_ids = node.gene_ids          # int array (mapped gene index)
        counts   = node.gene_counts       # same shape

        # rank-like score
        scores_g = np.log1p(counts)
        
        lam = 0.75

        fv = np.zeros(self.C, dtype=float)
        for i, cid in enumerate(self.valid_cluster_ids):
            mk = self.markers[cid]

            pos_gids = mk["pos_gids"]
            pos_weights = mk.get("pos_weights", 1.0)
            if pos_gids.size:
                mask = np.isin(gene_ids, pos_gids)
                fv[i] += pos_weights * scores_g[mask].sum()

            neg_gids = mk["neg_gids"]
            neg_weights = mk.get("neg_weights", 1.0)
            if neg_gids.size:
                mask = np.isin(gene_ids, neg_gids)
                fv[i] -= lam  *  neg_weights * scores_g[mask].sum()

        return fv


    def calc_entropy(self, node: Node) -> float:
        if node.feature_vector is None:
            node.feature_vector = self.calc_feature_vector(node)
        p = node.feature_vector

        mask = p > 0
        if not np.any(mask):
            return 0.0
        p_pos = p[mask]

        log_eps = 1e-12
        entropy = -np.sum(p_pos * np.log2(p_pos + log_eps))

        max_ent = np.log2(self.C) if self.C > 1 else 1.0
        return float(entropy / max_ent)


    def divide_node(self, node: Node) -> list[Node]:
        w2 = node.w / 2.0
        h2 = node.h / 2.0
        mid_x = node.x0 + w2
        mid_y = node.y0 + h2

        idx = node.data_idx
        if idx is None or idx.size == 0:
            children = [
                Node(node.x0, node.y0, w2, h2, node.depth + 1, node, data_idx=np.array([], dtype=np.int32), all_gene_ids=self.all_gene_ids),
                Node(node.x0 + w2, node.y0, w2, h2, node.depth + 1, node, data_idx=np.array([], dtype=np.int32), all_gene_ids=self.all_gene_ids),
                Node(node.x0, node.y0 + h2, w2, h2, node.depth + 1, node, data_idx=np.array([], dtype=np.int32), all_gene_ids=self.all_gene_ids),
                Node(node.x0 + w2, node.y0 + h2, w2, h2, node.depth + 1, node, data_idx=np.array([], dtype=np.int32), all_gene_ids=self.all_gene_ids),
            ]
            return children

        xs_local = self.xs[idx]
        ys_local = self.ys[idx]

        mask_tl = (xs_local < mid_x) & (ys_local < mid_y)
        mask_tr = (xs_local >= mid_x) & (ys_local < mid_y)
        mask_bl = (xs_local < mid_x) & (ys_local >= mid_y)
        mask_br = (xs_local >= mid_x) & (ys_local >= mid_y)

        idx_tl = idx[mask_tl]
        idx_tr = idx[mask_tr]
        idx_bl = idx[mask_bl]
        idx_br = idx[mask_br]

        children = [
            Node(node.x0, node.y0, w2, h2, node.depth + 1, node, data_idx=idx_tl, all_gene_ids=self.all_gene_ids),
            Node(node.x0 + w2, node.y0, w2, h2, node.depth + 1, node, data_idx=idx_tr, all_gene_ids=self.all_gene_ids),
            Node(node.x0, node.y0 + h2, w2, h2, node.depth + 1, node, data_idx=idx_bl, all_gene_ids=self.all_gene_ids),
            Node(node.x0 + w2, node.y0 + h2, w2, h2, node.depth + 1, node, data_idx=idx_br, all_gene_ids=self.all_gene_ids),
        ]
        return children


    def calc_similarity(self, P: np.ndarray, Qs: np.ndarray) -> np.ndarray:
        if P is None or np.linalg.norm(P) == 0:
            return np.zeros(Qs.shape[0], dtype=float)
        pnorm = np.linalg.norm(P)
        Pn = P / (pnorm + 1e-12)
        Qn = Qs / (np.linalg.norm(Qs, axis=1, keepdims=True) + 1e-12)
        if self.sim_method == 'cosine':
            sims = np.dot(Qn, Pn)  # shape (num_children,)
            return sims
        elif self.sim_method == 'euclidean':
            dists = np.linalg.norm(Qs - P, axis=1)
            sims = 1 - dists / np.sqrt(2)
            return sims
        elif self.sim_method == 'manhattan':
            d = np.sum(np.abs(Qs - P), axis=1)
            sims = 1 - d / 2.0
            return sims
        else:
            return np.zeros(Qs.shape[0], dtype=float)


    def apply_sibling_bias(self, children: list[Node]):
        if children is None or len(children) == 0:
            return
        vectors = [c.feature_vector if c.feature_vector is not None else np.zeros(self.C) for c in children]
        mat = np.vstack(vectors)  # (4, C)
        avg = np.mean(mat, axis=0)

        s = avg.sum()
        if s > 0:
            avg = avg / s

        w = self.config.neighbor_weight
        for i, c in enumerate(children):
            if c.feature_vector is None:
                c.feature_vector = avg.copy()
            else:
                vec = (1.0 - w) * c.feature_vector + w * avg
                s2 = vec.sum()
                c.feature_vector = vec / (s2 if s2 > 0 else 1.0)


    def assign_celltype(self, node: Node) -> int:
        # ---- background check ----
        area = max(node.w * node.h, 1e-12)
        n_points = node.data_idx.size if node.data_idx is not None else 0
        density = n_points / area

        if n_points == 0 or density <= self.config.bkg_thresh:
            return self.background_id

        # ---- feature vector ----
        if node.feature_vector is None:
            node.feature_vector = self.calc_feature_vector(node)

        fv = node.feature_vector
        if fv is None or fv.sum() == 0:
            return self.unknown_id

        # fvについて最大成分を取る細胞種への割り当てを考える
        top_score = np.max(node.feature_vector) 
        second_score = np.partition(node.feature_vector, -2)[-2]
        margin = (top_score - second_score) #/ (abs(top_score) + abs(second_score) + 1e-12)

        if top_score < self.config.assign_thresh and node.depth >= self.config.limit_depth: # 閾値未満ならUnknownに割り当て（領域サイズが下限に到達した時の処理としてこれが要る）
            return self.unknown_id
        else:
            return self.valid_cluster_ids[np.argmax(node.feature_vector)]

    def _ratio_to_margin(self) -> float:
        r = self.config.score_margin
        if r <= 1.0:
            raise ValueError("decisive_ratio must be > 1.0")
        return (r - 1.0) / (r + 1.0)
    
    
    def check_division(self, node: Node) -> bool:
        # 分割続行時にTrue、分割終了時にFalseを返す
        if node.feature_vector is None:
            node.feature_vector = self.calc_feature_vector(node)
            node.entropy = self.calc_entropy(node)

        if np.linalg.norm(node.feature_vector) == 0:
            return False

        top_score = np.max(node.feature_vector) 

        if node.children is None:
            self._prepare_children(node)

        Qs = np.vstack([child.feature_vector for child in node.children])
        P = node.feature_vector
        
        sims = self.calc_similarity(P, Qs)  
        is_high_similarity = np.all(sims >= self.config.sim_thresh) # コサイン類似度は -1~1 なのでユーザが直接指定する
        
        second_score = np.partition(node.feature_vector, -2)[-2]
        margin = (top_score - second_score) #/ (abs(top_score) + abs(second_score) + 1e-12)

        rel_margin = (top_score - second_score) / (top_score + 1e-12)
        is_decisive_score = rel_margin > self.config.score_margin # 例: 0.2

        # is_decisive_score = margin > self.config.score_margin # margin > margin_thresh
        
        if is_high_similarity and is_decisive_score:
            return False        
        return True
 


    def _prepare_children(self, node: Node):
        """子供の作成と特徴ベクトル計算を共通化する補助メソッド"""
        node.children = self.divide_node(node)
        for child in node.children:
            child.feature_vector = self.calc_feature_vector(child)
            child.entropy = self.calc_entropy(child)
        self.apply_sibling_bias(node.children)
    
    def generate_quadtree(self):
        stack = [self.root]
        
        while stack:
            node = stack.pop()

            if node.depth >= self.config.limit_depth:
                node.celltype = self.assign_celltype(node)
                node.children = None
                continue

            if self.check_division(node):
                if node.children:
                    stack.extend(node.children[::-1])
                continue

            node.celltype = self.assign_celltype(node)
            
            if node.celltype == self.unknown_id and node.children:
                stack.extend(node.children[::-1])
            else:
                node.children = None

    def collect_leaves(self) -> pd.DataFrame:
        """Return a DataFrame summarizing leaf nodes (bounds, n_points, entropy, assigned type)."""
        leaves = []
        stack = [self.root]
        
        while stack:
            node = stack.pop()
            if node.children is None:
                leaves.append({
                    'x0': int(node.x0), 'y0': int(node.y0), 'w': int(node.w), 'h': int(node.h),
                    'leaf_id': f"{int(node.x0)}_{int(node.y0)}_{int(node.w)}_{int(node.h)}",
                    'depth': node.depth, 'n_gene': int(node.data_idx.size),
                    'entropy': float(node.entropy) if node.entropy is not None else None,
                    'qtree_celltype': node.celltype if node.celltype is not None else self.unknown_id,
                    'feature_vector': node.feature_vector.tolist() if node.feature_vector is not None else None,
                    **{f'fv_{i}': float(node.feature_vector[i]) if node.feature_vector is not None else 0.0 for i in range(self.C)}
                })
            else:
                stack.extend(node.children)
        return pd.DataFrame(leaves)