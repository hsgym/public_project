from sklearn.neighbors import KDTree
from scipy.sparse import coo_matrix, csr_matrix, diags
import numpy as np
import pandas as pd


def precompute_bins(input_df, width, height, grid_size):
    input_df = input_df.copy()
    
    # binの最大インデックスを計算
    max_bin_x = (width // grid_size) - 1
    max_bin_y = (height // grid_size) - 1

    # bin座標を計算
    bx = (input_df["local_pixel_x"] // grid_size).astype(np.int32)
    by = (input_df["local_pixel_y"] // grid_size).astype(np.int32)

    # 【修正の核心】範囲外の値を、有効な最小値(0)〜最大値に収める (Clipping)
    # これにより、行を削除することなく「一番端のbin」に所属させます
    input_df["bin_x"] = np.clip(bx, 0, max_bin_x)
    input_df["bin_y"] = np.clip(by, 0, max_bin_y)

    # valid_mask によるフィルタリング（行の削除）は行わない
    return input_df

def precompute_bins_old(input_df, width, height, grid_size):
    input_df = input_df.copy()
    input_df["bin_x"] = (input_df["local_pixel_x"] // grid_size).astype(np.int32)
    input_df["bin_y"] = (input_df["local_pixel_y"] // grid_size).astype(np.int32)

    # 不正なbinの除外（marginでズレた点など）
    valid_mask = (
        (input_df["bin_x"] >= 0) & (input_df["bin_x"] < width // grid_size) &
        (input_df["bin_y"] >= 0) & (input_df["bin_y"] < height // grid_size)
    )
    input_df = input_df[valid_mask].copy()
    filtered_df = input_df.reset_index(drop=True)

    return filtered_df, valid_mask

# 重みベースで celltype 用の点抽出
def extract_points_for_celltype(cid, input_df, weights_sparse, weight_threshold):
    col = weights_sparse.getcol(cid).tocoo()
    active_rows = col.row[col.data >= weight_threshold]

    if active_rows.size == 0:
        return None
    
    return input_df.iloc[active_rows].copy() # input_df.loc[active_rows].copy()


# gene の低頻度フィルタ
def filter_genes(df, min_ratio=0.01, min_count=5):
    genes = df["gene"].values
    unique, counts = np.unique(genes, return_counts=True)
    threshold = max(min_count, int(len(df) * min_ratio))
    
    keep_genes = unique[counts >= threshold]
    df2 = df[df["gene"].isin(keep_genes)]
    
    if df2.empty:
        return None, None
    
    keep_genes = df2["gene"].unique().tolist()
    gene_to_idx = {g: i for i, g in enumerate(keep_genes)}
    return df2, (keep_genes, gene_to_idx)


# binned count matrix (Nbins × G) を sparse で構築
def build_binned_matrix(df, width, height, bin_size, gene_to_idx):
    binned_x = width // bin_size
    binned_y = height // bin_size
    Nbins = binned_x * binned_y
    
    bin_ids = df["bin_y"].values * binned_x + df["bin_x"].values
    gene_ids = df["gene"].map(gene_to_idx).values

    mat = coo_matrix(
        (np.ones(len(df), dtype=np.float32), (bin_ids, gene_ids)),
        shape=(Nbins, len(gene_to_idx))
    ).tocsr()
    return mat, (binned_x, binned_y, Nbins)


# 9近傍 adjacency matrix W
def build_adjacency_matrix(binned_x, binned_y):
    Nbins = binned_x * binned_y
    coords = np.array([[i, j] for i in range(binned_y) for j in range(binned_x)])

    kdt = KDTree(coords)
    _, neigh = kdt.query(coords, k=9)  # 自分 + 8近傍

    I = np.repeat(np.arange(Nbins), 8)
    J = neigh[:, 1:].reshape(-1)

    W = csr_matrix((np.ones_like(I), (I, J)), shape=(Nbins, Nbins))

    # row-standardize
    W = diags(1 / np.array(W.sum(axis=1)).flatten()) @ W
    return W, coords


# Global Moran's I（簡易）
def compute_global_moran(mat, W):
    X = mat.toarray().astype(np.float32)
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    WX = W @ Xz
    numerator = (Xz * WX).sum(axis=0)
    denom = (Xz * Xz).sum(axis=0)

    moran_I = numerator / (denom + 1e-6)
    return Xz, WX, moran_I


# Local Moran + HH 抽出
def compute_local_hh(Xz, WX, gene_ids, coords, compactness_threshold):
    results = []

    for g in gene_ids:
        x = Xz[:, g]
        Wx = WX[:, g]
        local_I = x * Wx
        
        q = np.zeros_like(local_I, dtype=np.int32) 
        q[(x > 0) & (Wx > 0)] = 4 # HH 
        hh_idx = np.where(q == 4)[0]

        if len(hh_idx) < 3:
            continue

        c = coords[hh_idx]
        center = c.mean(axis=0)
        comp = np.mean(np.linalg.norm(c - center, axis=1))

        if comp > compactness_threshold:
            continue

        results.append((g, comp))
    return results



def select_svg(
    cid: int, input_df: pd.DataFrame, width:int, height:int, bin_size:int=16,
    min_ratio=0.01, min_count=5,
    compactness_threshold=np.inf, top_k:int=20, weights_sparse=None,
    weight_threshold:float=0.1, global_moran_threshold:float = 0.05
):

    df = extract_points_for_celltype(cid, input_df, weights_sparse, weight_threshold)
    if df is None:
        return []

    df, gene_info = filter_genes(df, min_ratio=min_ratio, min_count=min_count)
    if df is None:
        return []
    keep_genes, gene_to_idx = gene_info
    G = len(keep_genes)


    mat, (binned_x, binned_y, Nbins) = build_binned_matrix(df, width, height, bin_size, gene_to_idx)
    W, coords = build_adjacency_matrix(binned_x, binned_y)

    Xz, WX, moran_I = compute_global_moran(mat, W)
    candidate_genes = np.where(moran_I > global_moran_threshold)[0]
    if len(candidate_genes) == 0:
        return []

    local_res = compute_local_hh(Xz, WX, candidate_genes, coords, compactness_threshold)
    if not local_res:
        return []
    
    scored = []
    for g, comp in local_res:
        score = moran_I[g] / (comp + 1e-6)
        scored.append((keep_genes[g], moran_I[g], comp, score))

    scored.sort(key=lambda x: x[3], reverse=True)
    return [s[0] for s in scored[:top_k]]