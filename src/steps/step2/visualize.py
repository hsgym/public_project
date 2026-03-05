import math
import matplotlib.pyplot as plt
from src.steps.step2.VisualizeCluster import ClusterVisualizer

def draw_cluster_points(visualizer, *, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    visualizer.visualize_points(ax=ax, title=title)
    return fig, ax

def draw_cluster_hulls(visualizer, *, ax=None, show_legend=True, title=None, type_system=None,  max_percentile = 99):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    visualizer.visualize_polygons(
        ax=ax,
        title=title,
        is_legend=show_legend,
        type_system=type_system,
        max_percentile = max_percentile
    )
    return fig, ax


def prepare_step2_visualization_data(result_df, valid_ids):
    df = result_df[result_df["result_celltype"].isin(valid_ids)].copy()
    print(f"Number of visualized points: {len(df)}")
    print(f"Number of clusters: {len(df['cluster_id'].unique())}")  
    return df



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


def draw_gmm_initial_for_celltype(histograms, mu, type_system, cfg, target_cid, ax):
    ax.imshow(
        histograms, origin="lower",
        extent=(0, cfg.base_config.width, 0, cfg.base_config.height),
        cmap="viridis"
    )
    if mu is not None:
        ax.scatter(mu[:, 0], mu[:, 1], c="red", s=50)
    ax.set_title(f"{type_system.type_id_to_celltypes(target_cid) if not type_system.type_id_to_celltypes(target_cid) is None else 'Unknown'} (ID: {target_cid})")
    ax.invert_yaxis()  # 画像座標系に合わせて
    ax.set_xlim(0, cfg.base_config.width)
    ax.set_ylim(cfg.base_config.height, 0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    return   

def draw_gmm_initial_all(
    histogam_dict,
    mu_dict,
    type_system,
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
        if target_cid not in histogam_dict or mu_dict[target_cid] is None:
            # 何も描画しないようにしたい。
            ax.axis("off")
            continue
        
        h = histogam_dict[target_cid]
        mu = mu_dict[target_cid]
        draw_gmm_initial_for_celltype(h, mu, type_system, cfg, target_cid=target_cid, ax=ax)

    # 余った axes を消す
    for ax in axes[len(valid_ids):]:
        ax.axis("off")
    return fig, axes
    
    
    
def visualize_gmm_initial(histogram_dict, mu_dict, type_system, unique_celltypes_list, valid_ids, cfg):
    fig, axes = draw_gmm_initial_all(histogram_dict, mu_dict, type_system, unique_celltypes_list, valid_ids, cfg)
    plt.tight_layout()
    plt.show()



def visualize_step2(result_df, gmm_mu_init_dict, histogram_dict, valid_ids, type_system, cfg):
    fig_initial, axes_initial = draw_gmm_initial_all(histogram_dict, gmm_mu_init_dict, type_system, type_system.unique_celltypes_list, valid_ids, cfg)

    tmp_df = prepare_step2_visualization_data(result_df, valid_ids)
   
    visualizer = ClusterVisualizer(
        data=tmp_df,
        x_col="local_pixel_x",
        y_col="local_pixel_y",
        cluster_id_col="cluster_id",
        celltype_col="result_celltype",
        width=cfg.base_config.width,
        height=cfg.base_config.height,
        unique_celltypes=valid_ids,
        random_seed=0
    )

    fig_cluster, axes_cluster = plt.subplots(1, 2, figsize=(13, 6))
    draw_cluster_points(visualizer, ax=axes_cluster[0], title="Clustering Points")
    draw_cluster_hulls(visualizer, ax=axes_cluster[1], title="Convex Hull Polygons", type_system=type_system)
    fig_cluster.tight_layout()
    
    return fig_initial, axes_initial, fig_cluster, axes_cluster

