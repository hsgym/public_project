
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scanpy as sc

from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from src.preprocess.color_generator import get_celltype_color_dict

# 距離行列の計算
def compute_linkage_matrix(ref_df: pd.DataFrame) -> np.ndarray:
    
    """
    Args:
    - ref_df (pd.DataFrame) : 参照データのデータフレーム gene x celltype(or cluster)
    Returns:
    - nd.ndarray : linkage matrix
    """
    corr_matrix = ref_df.corr(method="spearman")
    distance_matrix = 1 - corr_matrix
    return linkage(squareform(distance_matrix, checks=False), method="average")


def map_celltypes_to_clusters(linkage_matrix: np.ndarray, ref_df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Args:
    - linkage_matrix (np.ndarray) : linkage matrix
    - ref_df (pd.DataFrame) : 参照データのデータフレーム gene x celltype(or cluster)
    - threshold (float) : クラスタリングの閾値
    Returns:
    - dict : {cluster_name: [celltype1, celltype2, ...]}
    """
    
    cluster_labels = fcluster(linkage_matrix, t=threshold, criterion="distance")
    cluster_labels = [f"cluster_{label}" for label in cluster_labels]

    cluster_df = pd.DataFrame({"Celltype": ref_df.columns, "Cluster": cluster_labels})

    # クラスタごとに細胞種のリストを作成
    cluster_dict = {
        cluster: group_df["Celltype"].tolist()
        for cluster, group_df in cluster_df.groupby("Cluster")
    }

    print("\nCluster assignments:")
    for cluster, group_df in cluster_df.groupby("Cluster"):
        celltypes = group_df["Celltype"].tolist()

    for cluster, celltypes in cluster_dict.items():
        print(f"{cluster}: {', '.join(celltypes)}")

    return cluster_df, cluster_dict

def assign_cluster_to_ref_df(ref_df_com: pd.DataFrame, celltype_col: str, cluster_celltype_dict: dict) -> pd.DataFrame:
    cluster_col = "cluster_name"
    for idx, row in ref_df_com.iterrows():
        celltype = row[celltype_col]
        
        # cluster_celltype_dict から celltype が含まれる cluster_id を探す
        for cluster_id, celltypes in cluster_celltype_dict.items():
            if celltype in celltypes:
                ref_df_com.at[idx, cluster_col] = cluster_id
                break 
    return ref_df_com, cluster_col


def visualize_dendrogram(linkage_matrix: np.ndarray, labels: list[str], ax=None) -> None:
    """
    Args:
    - linkage_matrix (np.ndarray) : linkage matrix
    - labels (list[str]) : ラベルのリスト
    - figsize (tuple[int, int]) : 図のサイズ
    Returns:
    - None : デンドログラムを表示する
    """    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,4 ))
    else:
        # axが渡された場合は、そのaxが属するfigureを取得
        fig = ax.get_figure()
    
    dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_font_size=10.5, leaf_rotation=90)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Celltypes")
    plt.ylabel("Distance")
    
    plt.tight_layout()
    return fig, ax



def load_color_info(config, save_dir, unique_celltypes_list):
    tissue_type = config.base_config.tissue_type 
    file_name = f"celltype_color_dict_{tissue_type}_category{len(unique_celltypes_list)}.json"
    json_path = save_dir / file_name
    print(json_path)
    celltype_color_dict = get_celltype_color_dict(json_path, unique_celltypes_list)
    return celltype_color_dict

def plot_color_palette(color_map, ax = None):
    names = list(color_map.keys())
    colors = list(color_map.values())
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, len(names) * 0.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, len(names))
    
    for i, name in enumerate(names):
        # 色付きの四角を描画
        ax.add_patch(plt.Rectangle((0, i-0.4), 1.5, 0.8, facecolor=colors[i], edgecolor='black'))
        # 名称をテキストで表示
        ax.text(2, i, name, va='center', fontsize=12)
    
    ax.axis('off')
    return fig, ax