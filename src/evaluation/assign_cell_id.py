import rtree.index as index
from shapely.geometry import Point

from shapely.wkt import loads as load_wkt
from shapely.geometry import Point
from rtree import index
import pandas as pd
import numpy as np
from tqdm import tqdm

import pandas as pd
from rtree import index
from shapely.geometry import Point


# 実質 merfish (Cellpose) の結果を付与する用
def assign_cell_ids_from_matrix(gene_df, cropped_img_mem, background_id=-1):
    """
    遺伝子発現点データに対して、画像マトリックスから直接セルIDを割り当てる関数
    
    Args:
        gene_df: 遺伝子発現点データ（local_pixel_x, local_pixel_yを含む）
        cropped_img_mem: セルIDマトリックス（0=背景、1以上=セルID）
        
    Returns:
        gene_df_with_cells: セルIDが付与された遺伝子発現点データ
    """
    
    # 結果用のDataFrameをコピー
    gene_df_with_cells = gene_df.copy()
    
    # 画像の境界を取得
    img_height, img_width = cropped_img_mem.shape
    
    # 各遺伝子発現点に対してセルIDを割り当て
    assigned_cell_ids = []
    valid_count = 0
    background_count = 0
    out_of_bounds_count = 0
    
    for idx, row in gene_df.iterrows():
        x = int(row['local_pixel_x'])
        y = int(row['local_pixel_y'])
        
        # 座標が画像境界内にあるかチェック
        if 0 <= x < img_width and 0 <= y < img_height:
            # マトリックスから値を取得
            cell_id = cropped_img_mem[y, x]  # 注意: [y, x]の順序
            
            if cell_id == 0:
                # 背景の場合はNoneを割り当て
                assigned_cell_ids.append(None)
                background_count += 1
            else:
                # セルIDを割り当て
                assigned_cell_ids.append(int(cell_id))
                valid_count += 1
        else:
            # 境界外の場合はNoneを割り当て
            assigned_cell_ids.append(None)
            out_of_bounds_count += 1
    
    # 結果をDataFrameに追加 (int 型)
    gene_df_with_cells['cell_id'] = assigned_cell_ids
    # none -1でを埋める
    gene_df_with_cells['cell_id'] = gene_df_with_cells['cell_id'].fillna(background_id).astype(int)

    
    return gene_df_with_cells

# Xeniumの結果を付与する要用途
def assign_cell_ids_to_gene_points(gene_df, cell_df, background_id):
    # 結果用のDataFrame
    gene_df_with_cells = gene_df.copy()
    gene_coords = gene_df_with_cells[['local_pixel_x', 'local_pixel_y']].values.astype(np.float64)
    cell_ids = np.full(len(gene_df_with_cells), background_id, dtype=object)
    
    # 準備
    cell_spatial_index = index.Index()
    cell_polygons = []
    cell_id_mapping = {}
    valid_cell_count = 0
    
    # --- 1. インデックス作成フェーズ ---
    for idx, row in tqdm(cell_df.iterrows(), total=len(cell_df), desc="Indexing cells"):
        wkt_str = row.get('Geometry_local') or row.get('geometry')
        
        if pd.isna(wkt_str):
            continue
            
        try:
            # 文字列(WKT)をShapelyオブジェクトに変換
            if isinstance(wkt_str, str):
                geom = load_wkt(wkt_str)
            else:
                geom = wkt_str
            
            if geom and geom.is_valid:
                # 空間インデックスに登録
                cell_spatial_index.insert(valid_cell_count, geom.bounds)
                cell_polygons.append(geom)
                
                # cell_idの取得（'cell_id'カラムを優先）
                cid = row['cell_id'] if 'cell_id' in row else idx
                cell_id_mapping[valid_cell_count] = cid
                
                valid_cell_count += 1
        except Exception as e:
            # 変換失敗などはスキップ
            continue

    print(f"Successfully indexed {valid_cell_count} cells.")
    if valid_cell_count == 0:
        return gene_df_with_cells

    # --- 2. 判定フェーズ (高速化のため Prepared Geometry を使用) ---
    from shapely.prepared import prep
    prepared_polygons = [prep(g) for g in cell_polygons]

    # バッチ処理
    batch_size = 50000
    for start_idx in tqdm(range(0, len(gene_coords), batch_size), desc="Processing genes"):
        end_idx = min(start_idx + batch_size, len(gene_coords))
        
        for i in range(start_idx, end_idx):
            x, y = gene_coords[i]
            
            # 候補の絞り込み
            candidates = list(cell_spatial_index.intersection((x, y, x, y)))
            
            if candidates:
                p = Point(x, y)
                for c_idx in candidates:
                    # prepared geometryで高速判定
                    if prepared_polygons[c_idx].contains(p):
                        cell_ids[i] = cell_id_mapping[c_idx]
                        break
    
    gene_df_with_cells['cell_id'] = cell_ids
    return gene_df_with_cells