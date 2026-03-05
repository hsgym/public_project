import pandas as pd

def build_type_registry(cluster_celltype_dict):
    rows = []
    sorted_keys = sorted(cluster_celltype_dict.keys(),
                         key=lambda k: int(k.split('_')[1]))  # cluster_1,2,3順に並べる

    for new_id, ckey in enumerate(sorted_keys):
        rows.append({
            "type_id": new_id,
            "label": ckey,  # 内部クラスタラベル
            # 長さが１の時は文字列、複数の時はリストで格納
            "celltypes": cluster_celltype_dict[ckey] if len(cluster_celltype_dict[ckey]) > 1 else cluster_celltype_dict[ckey][0],  # 実バイオ名称リスト
#            "celltypes": cluster_celltype_dict[ckey],  # 実バイオ名称リスト # 
            "color": None  # 後で付与
        })

    # Unknown / Background を追加
    rows.append({"type_id": len(rows), "label": "Unknown", "celltypes": "Unknown", "color": [1.0, 1.0, 1.0]})
    rows.append({"type_id": len(rows), "label": "Background", "celltypes": "Background", "color": [0.0, 0.0, 0.0]})

    return pd.DataFrame(rows)

def append_colors_to_registry(type_registry: pd.DataFrame, celltype_color_dict: dict) -> pd.DataFrame:
    color_list = []
    for _, row in type_registry.iterrows():
        celltypes = row["celltypes"]
        # 最初の細胞種の色を使う（複数ある場合）
        color = celltype_color_dict.get(celltypes[0] if isinstance(celltypes, list) else celltypes, "gray")  # デフォルトは灰色
        color_list.append(color)
    type_registry["color"] = color_list
    return type_registry


class TypeSystem:
    def __init__(self, type_registry: pd.DataFrame):
        self.df = type_registry.reset_index(drop=True)

        # --- 基本情報 ---
        self.type_ids = self.df["type_id"].astype(int).tolist()

        self.labels = dict(zip(self.df.type_id, self.df.label))
        self.celltypes = dict(zip(self.df.type_id, self.df.celltypes))
        self.colors = dict(zip(self.df.type_id, self.df.color))
        self.unique_celltypes_list = sorted(set(
            ct for ct in self.df.celltypes
            if isinstance(ct, str) and ct not in ("Unknown", "Background")
        ))

        # --- special types ---
        self.unknown_id = int(
            self.df[self.df.label == "Unknown"].type_id.values[0]
        )
        self.background_id = int(
            self.df[self.df.label == "Background"].type_id.values[0]
        )

        # --- valid cluster ids ---
        self.valid_type_ids = [
            tid for tid in self.type_ids
            if tid not in (self.unknown_id, self.background_id)
        ]

        self.C = len(self.valid_type_ids)
        
    # str -> int への変換用
    def celltype_to_type_id(self, celltype: str) -> int:
        for tid, ct in self.celltypes.items():
            if isinstance(ct, list):
                if celltype in ct:
                    return tid
            else:
                if celltype == ct:
                    return tid
        return -1  # 見つからない場合
    
    # 確認用辞書の作成： key:str, value:color
    def to_dict(self) -> dict:
        type_dict = {}
        for tid in self.type_ids:
            label = str(self.celltypes[tid])
            color = self.colors[tid]
            type_dict[label] = color
        return type_dict
    
    def label_to_type_id(self, label: str) -> int:
        for tid, lbl in self.labels.items():
            if label == lbl:
                return tid
        return -1  # 見つからない場合
    
    def type_id_to_color(self, type_id: int):
        return self.colors.get(type_id, None)
    
    def type_id_to_celltypes(self, type_id: int):
        return self.celltypes.get(type_id, None)

    def is_valid(self, type_id: int) -> bool:
        return type_id in self.valid_type_ids

    def is_special(self, type_id: int) -> bool:
        return type_id in (self.unknown_id, self.background_id)


"""
外側でこんな感じで使う。

from codes.type_registry_builder import build_type_registry, append_colors_to_registry

type_registry = build_type_registry(cluster_celltype_dict)
type_registry = append_colors_to_registry(type_registry, celltype_color_dict)

num_celltypes = len(unique_celltypes_list)
# 0 ~ num_celltypes - 1 が実細胞種ID
unknown_id = num_celltypes
background_id = num_celltypes + 1

valid_ids = [] # type_regisry の type_id でループを回すときに使う
for idx, row in type_registry.iterrows():
    if row["label"] == "Unknown" or row["label"] == "Background":
        continue
    else:
        valid_ids.append(row["type_id"]) 
"""