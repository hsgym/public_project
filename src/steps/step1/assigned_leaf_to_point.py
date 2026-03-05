import pandas as pd
from rtree import index


def prepare_bin_data(qtree_df: pd.DataFrame, unique_celltypes: list[str] = None) -> pd.DataFrame:

    df = qtree_df[["leaf_id", "x0", "y0", "w", "h", "qtree_celltype"]].copy()
    df["x_right"] = df["x0"] + df["w"]
    df["y_bottom"] = df["y0"] + df["h"]

    # unique_celltypes の推定（なければ qtree_df から自動抽出）
    if unique_celltypes is None:
        excluded_cols = ["leaf_id", "x0", "y0", "w", "h", "x_right", "y_bottom", "qtree_celltype"]
        score_cols = [c for c in qtree_df.columns if c not in excluded_cols]
        unique_celltypes = score_cols

    return df


def build_rtree(bin_df: pd.DataFrame) -> index.Index:

    idx = index.Index()
    for _, row in bin_df.iterrows():
        idx.insert(
            int(row["leaf_id"]),
            (row["x0"], row["y0"], row["x_right"], row["y_bottom"])
        )
    return idx

def assign_bin_to_points(points_df, bin_df, rtree_idx):
    # leaf_id -> (x0,xr,y0,yr, qtree_celltype)
    bin_lookup = {
        int(r["leaf_id"]): (r["x0"], r["x_right"], r["y0"], r["y_bottom"], r["qtree_celltype"])
        for _, r in bin_df.iterrows()
    }

    leafs = []
    cats = []
    for p in points_df.itertuples(index=False):
        x = getattr(p, "local_pixel_x")
        y = getattr(p, "local_pixel_y")
        hits = list(rtree_idx.intersection((x, y, x, y)))
        found_leaf = -1
        found_cat = "Unknown"
        for h in hits:
            x0, xr, y0, yr, cat = bin_lookup[h]
            if x0 <= x <= xr and y0 <= y <= yr:
                found_leaf = h
                found_cat = cat
                break
        leafs.append(found_leaf)
        cats.append(found_cat)

    res = points_df.copy()
    res["leaf_id"] = leafs
    res["qtree_celltype"] = cats
    return res