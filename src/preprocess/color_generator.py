# 描画用辞書の作成

"""
## output 
- クラスタ名と描画に使う情報の辞書 : `cluster_visual_dict`
  - key : クラスタ名
  - value : 細胞種名のリスト、描画に使う色
"""


import json
import os
import random
import seaborn as sns
from collections import defaultdict


"""
このプログラムの目的：
{"cluster_name" : "celltypes": ["celltype_A", ...], "color": (r, g, b)} の辞書を作成する
クラスタ名->細胞種かつ色

処理：
0. クラスタ名と細胞種{"cluster_name" : ["celltype_A", ...] } の辞書は与えられる
1. 細胞種名と色{"celltype_name" : (r,g,b) }の辞書の準備
  - .json ファイルが既に存在する場合⇒読み出し
  - .json ファイルが存在しない場合⇒新規作成 & 保存
2. 細胞種名と色の辞書に情報を追加
  - "Unknown" と "Background" の色を追加（存在しない場合）
3. クラスタ名->細胞種の辞書と細胞種名->色の辞書とを照らし合わせて、目的の辞書を作成
  - クラスタ名と細胞種名が一対一対応の場合⇒クラスタ名->細胞種名->色の形から目的の辞書の形にする
  - クラスタ名と細胞種名が一対多対応の場合⇒クラスタ名に対応する色を新規で作成し、目的に辞書の形にする
  - ※"Unknown" と "Background" の場合はクラスタ名 = 細胞種名とする


設計：
どのような関数を作るか？
- jsonファイルの読み込み `read_json()`
  - param: json_path
  - return: 読み込んだ内容
- jsonファイルに保存する `write_json()`
  - param: json_path, 書き込む内容
  - return: None
- カラーパレットを新規で作成する `create_palette()`
  - param: keyになりうるもの（celltypeの一覧 or cluster の一覧）、ファイルに保存するかのオプション、パレットのテーマhuslをデフォルトに
  - return: 辞書
- 細胞種名->色の辞書を作成or読み出し `get_celltype_color_dict()`
  - param: json_path, celltypeの一覧
  - return: celltype->colorの辞書
- 目的の辞書を作成 `get_cluster_visual_dict()`
  - param: cluster->celltypeの辞書、celltype->colorの辞書、一対一対応か否かを示すフラグ
  - return: cluster->celltype,colorの辞書
"""


def read_json(json_path: str) -> dict:
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    return data

def write_json(json_path: str, data: dict):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    return

def create_palette(keys: list, palette: str = "hls", save_path: str = None) -> dict:
    palette = sns.color_palette(palette, len(keys))
    
    # keys.sort() # keyの要素を昇順にソートする
    keys_shuffled = random.sample(keys, len(keys))
    color_dict = {key: color for key, color in zip(keys_shuffled, palette)} 
    print(color_dict) # 確認用⇒後にコメントアウト予定
    
    if save_path is not None:
        write_json(save_path, color_dict)
    
    return color_dict


def get_celltype_color_dict(json_path: str, celltypes: list) -> dict:
    if os.path.exists(json_path):
        celltype_color_dict = read_json(json_path)
    else:
        celltype_color_dict = create_palette(celltypes, save_path=json_path)
        
    celltype_color_dict.setdefault("Unknown", "white")  # 追加
    celltype_color_dict.setdefault("Background", "black")
    return celltype_color_dict


def get_cluster_visual_dict(cluster_celltype_dict: dict, celltype_color_dict: dict, is_mapping: bool) -> dict:
    cluster_visual_dict = {}
    
    if not is_mapping: # 一対一対応のとき
       celltype_color_dict_replace = create_palette(cluster_celltype_dict.keys()) # cluster -> colorの取得
    
    # Unknown と Background は予め作ってしまう（エラー回避用）
    cluster_visual_dict = {
        "Unknown": {"celltypes": ["Unknown"], "color": "white"}, 
        "Background": {"celltypes": ["Background"], "color": "black"}
    }

    for cluster in cluster_celltype_dict.keys():
        celltype_tmp = cluster_celltype_dict[cluster]
        
        if celltype_tmp == ["Unknown"] or celltype_tmp == ["Background"]:
            continue
        
        if is_mapping:
            color_tmp = celltype_color_dict[celltype_tmp[0]]
        else:
            color_tmp = celltype_color_dict_replace[cluster]

        cluster_visual_dict[cluster] = {
            "celltypes": celltype_tmp,
            "color": color_tmp
        }
    
    print(cluster_visual_dict) # 確認用⇒後にコメントアウト予定
    return cluster_visual_dict
            

