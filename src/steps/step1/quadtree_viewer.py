import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PatchCollection

def plot_quadtree_from_df_old(ax, leaves_df, type_system, is_alpha_entropy:bool=True):
    """葉ノードDataFrameから四分木を高速に可視化"""
    rectangles = []
    colors = []
    alphas = []

    for _, row in leaves_df.iterrows():
        rect = Rectangle((row["x0"], row["y0"]), row["w"], row["h"])
        rectangles.append(rect)

        color = type_system.type_id_to_color(row["qtree_celltype"]) if row["qtree_celltype"] in type_system.type_ids else "white"
        colors.append(color)
    
    if is_alpha_entropy:
        alphas = 1.0 - leaves_df["entropy"].fillna(0).clip(0.1) 
    else:
        alphas = [1.0] * len(leaves_df)
    collection = PatchCollection(
        rectangles, facecolor=colors, edgecolor="black", linewidths=0.1, alpha=alphas
    )
    ax.add_collection(collection)


def plot_quadtree_from_df(ax, leaves_df, type_system, is_alpha_entropy=False):
    xs = leaves_df["x0"].to_numpy()
    ys = leaves_df["y0"].to_numpy()
    ws = leaves_df["w"].to_numpy()
    hs = leaves_df["h"].to_numpy()

    rectangles = [
        Rectangle((x, y), w, h)
        for x, y, w, h in zip(xs, ys, ws, hs)
    ]

    color_map = {
        tid: type_system.type_id_to_color(tid)
        for tid in type_system.type_ids
    }
    colors = (
        leaves_df["qtree_celltype"]
        .map(color_map)
        .fillna("white")
        .to_numpy()
    )

    if is_alpha_entropy:
        alphas = (
            1.0 - leaves_df["entropy"]
            .fillna(0.0)
            .clip(lower=0.1)
            .to_numpy()
        )
        from matplotlib.colors import to_rgba
        facecolors = [to_rgba(c, a) for c, a in zip(colors, alphas)]
    else:
        facecolors = colors

    collection = PatchCollection(
        rectangles,
        facecolor=facecolors,
        edgecolor="black",
        linewidths=0.1
    )
    ax.add_collection(collection)


def add_legend(ax: plt.Axes, leaves_df: pd.DataFrame, type_system, is_one_to_one_mapping: bool=True):
    legend_items = []
    used_keys = set(leaves_df["qtree_celltype"].unique())
    
    for key in used_keys:
        # row = type_registry[type_registry["type_id"] == int(key)]
        label = type_system.type_id_to_celltypes(int(key)) if int(key) in type_system.type_ids else "Unknown"
        color = type_system.type_id_to_color(int(key)) if int(key) in type_system.type_ids else "white"
        
        legend_items.append(Patch(facecolor=color, label=label, linewidth=0.05, edgecolor="black"))
    
    ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1.01, 0.95), title="celltype")


def visualize_quadtree_fast(quadtree, leaves_df, type_system, is_alpha_entropy: bool=True, title="", save_name=None):
    # === Step 1: 葉ノード情報の抽出 ===
    # leaves_df = quadtree.collect_leaves()

    # === Step 2: プロット領域設定 ===
    fig, ax = plt.subplots(figsize=(16, 16), dpi=300)
    ax.set_xlim(0, quadtree.root.w)
    ax.set_ylim(quadtree.root.h, 0)
    ax.set_aspect('equal', 'box')

    # === Step 3: 描画 ===
    plot_quadtree_from_df(ax, leaves_df, type_system, is_alpha_entropy=is_alpha_entropy)
    add_legend(ax, leaves_df, type_system)

    # === Step 4: 保存 or 表示 ===
    ax.axis("off")
    if save_name is not None:
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        # 凡例をオフする
        ax.legend_.remove() 
        fig.tight_layout()
        fig.savefig(f"{save_name}.svg", bbox_inches='tight', pad_inches=0)
        fig.savefig(f"{save_name}.pdf", bbox_inches='tight', pad_inches=0)
        print(f"Saved quadtree visualization to {save_name}.svg and {save_name}.pdf")
        plt.show()
    else:
        ax.set_title(title)
        fig.tight_layout()
        plt.show()

def save_legend(leaves_df: pd.DataFrame, type_system, save_name: str):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    legend_items = []
    used_keys = set(leaves_df["qtree_celltype"].unique())
    
    for key in used_keys:
        label = type_system.type_id_to_celltypes(int(key)) if int(key) in type_system.type_ids else "Unknown"
        color = type_system.type_id_to_color(int(key)) if int(key) in type_system.type_ids else "white"
        legend_items.append(Patch(facecolor=color, label=label, linewidth=0.05, edgecolor="black"))
    
    ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1.01, 0.95), title="celltype")

    # 保存
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{save_name}.svg", bbox_inches='tight', pad_inches=0)
    fig.savefig(f"{save_name}.pdf", bbox_inches='tight', pad_inches=0)
    print(f"Saved legend to {save_name}.svg and {save_name}.pdf")
    plt.show()
    


def draw_quadtree(
    quadtree,
    leaves_df: pd.DataFrame,
    type_system,
    *,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
    title: str | None = None,
    is_alpha_entropy: bool = True,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 16))
    else:
        fig = ax.figure

    ax.set_xlim(0, quadtree.root.w)
    ax.set_ylim(quadtree.root.h, 0)
    ax.set_aspect("equal", "box")

    plot_quadtree_from_df(ax, leaves_df, type_system, is_alpha_entropy=is_alpha_entropy)

    if show_legend:
        add_legend(ax, leaves_df, type_system)

    ax.axis("off")

    if title is not None:
        ax.set_title(title)

    return fig, ax
