import matplotlib.pyplot as plt
import numpy as np
import math
from src.steps.step1.quadtree_viewer import draw_quadtree

def visualize_step1(cfg, quadtree, leaves_df, gene_df_with_node, weights_sparse, type_system):
    fig_qtree, ax_qtree = draw_quadtree(
        quadtree,
        leaves_df,
        type_system,
        title=None,
        is_alpha_entropy=False,
        show_legend = False
    )

    fig_weights, axes_weights = draw_weights_scatter_all(
        gene_df_with_node,
        weights_sparse,
        type_system,
        type_system.unique_celltypes_list,
        type_system.valid_type_ids,
        cfg
    )
    
    return fig_qtree, ax_qtree, fig_weights, axes_weights


def create_subplot_grid(
    n_items: int,
    *,
    n_cols: int = 4,
    panel_size: float = 6.0,
    sharex: bool = False,
    sharey: bool = False,
):
    n_rows = math.ceil(n_items / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * panel_size, n_rows * panel_size),
        sharex=sharex,
        sharey=sharey,
    )

    axes = axes.flatten()

    return fig, axes


def draw_weights_scatter_for_celltype(gene_df_with_bins, weights_sparse, type_registry, cfg, target_cid, ax):
    # input_df = gene_df_with_bins.copy()
    weight_threshold = 0.01 # 可視化する最小重み（ここを調整）

    # 疎行列：1回だけ取得
    col = weights_sparse.getcol(target_cid).tocoo()
    mask = col.data >= weight_threshold

    # col = weights_sparse.getcol(target_cid).tocoo()
    # active_rows = col.row[col.data >= weight_threshold]
    
    # 抽出された行がない場合は何も描画せずに終了(実際に業を取得する前に判定)
    if not np.any(mask):
        # ax.invert_yaxis()  # 画像座標系に合わせて
        ax.set_xlim(0, cfg.base_config.width)
        ax.set_ylim(cfg.base_config.height, 0)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    active_rows = col.row[mask]
    
    # 点の数が多いと遅くなるので固定値にする
    n_points = active_rows.size
    w_vals = col.data[mask]
    if n_points >= 10000 or w_vals.size <= 1:
        alpha_vals = np.ones_like(w_vals)
        point_size = 20
    else:
        # 各点の重みを透明度に使う
        wmin = w_vals.min()
        alpha_vals = (w_vals - wmin) / (w_vals.max() - wmin + 1e-8)
        point_size = 10


    color = (
        type_registry.type_id_to_color(target_cid)
        if target_cid in type_registry.type_ids
        else "gray"
    )

    # 座標を NumPy で直接取得
    xs_all = gene_df_with_bins["local_pixel_x"].to_numpy()
    ys_all = gene_df_with_bins["local_pixel_y"].to_numpy()
    xs = xs_all[active_rows]
    ys = ys_all[active_rows]

    # 抽出した行だけを用いて散布図描画
    ax.scatter(
        xs,
        ys,
        s= point_size, 
        color=color,
        alpha=alpha_vals,
        edgecolors="none",
    )
    ax.set_xlim(0, cfg.base_config.width)
    ax.set_ylim(cfg.base_config.height, 0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    return



def draw_weights_scatter_all(
    gene_df_with_bins,
    weights_sparse,
    type_registry,
    unique_celltypes_list,
    valid_ids,
    cfg
):
    fig, axes = create_subplot_grid(
        len(unique_celltypes_list),
        n_cols=4,
        panel_size=6, 
    )

    for ax, target_cid in zip(axes, valid_ids):
        draw_weights_scatter_for_celltype(gene_df_with_bins, weights_sparse, type_registry, cfg, target_cid=target_cid, ax=ax)

    # 余った axes を消す
    for ax in axes[len(valid_ids):]:
        ax.axis("off")
    return fig, axes


def visualize_weights_scatter(
    gene_df_with_bins,
    weights_sparse,
    type_registry,
    unique_celltypes_list,
    valid_ids,
    cfg,
):
    fig, axes = draw_weights_scatter_all(
        gene_df_with_bins,
        weights_sparse,
        type_registry,
        unique_celltypes_list,
        valid_ids,
        cfg,
    )
    plt.tight_layout()
    plt.show()
    
    