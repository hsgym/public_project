# src/io/yaml.py

"""
YAMLの読み書きユーティリティ
- load_yaml: YAMLファイルを辞書として読み込む
- save_yaml: 辞書をYAMLファイルとして保存する
"""

from pathlib import Path
from typing import Union
import yaml
import numpy as np

def load_yaml(path: Union[str, Path]) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"YAML is empty: {path}")

    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be dict: {path}")

    return data


def to_builtin_type(obj):
    if isinstance(obj, dict):
        return {k: to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_builtin_type(v) for v in obj]
    elif hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        # NumPyの整数型をPython標準のintへ
        return int(obj)
    
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        # NumPyの浮動小数点型をPython標準のfloatへ
        return float(obj)
    
    elif isinstance(obj, np.ndarray):
        # NumPy配列をリストへ変換し、さらに中身を処理
        return to_builtin_type(obj.tolist())
    
    elif isinstance(obj, (np.bool_)):
        # NumPyのbool型をPython標準のboolへ
        return bool(obj)
    
    # すでに標準型(str, int, float, bool, None)の場合はそのまま
    return obj


def save_yaml(data: dict, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_data = to_builtin_type(data)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            safe_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    
    print("Saved dict to:", path)
