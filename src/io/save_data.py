import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union

def save_dataframe(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        df.to_parquet(path)
    elif path.suffix == ".pkl":
        df.to_pickle(path)
    else:
        raise ValueError(path)
    print("Saved dataframe to:", path)

def save_markdown(data: str, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)
    
    print("Saved comments to:", path)

def save_figure(fig, path: Union[str, Path], dpi:int=300, bbox_inches="tight", pad_inches=0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )
    
    print("Saved figure to:", path)


