from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from src.io.save_data import save_dataframe, save_figure, save_markdown
from src.configs.parameter_schemas import SaveConfig
from src.io.treat_yaml import load_yaml, save_yaml, to_builtin_type
from src.configs.parameter_converter import dict_to_config, config_to_dict

def save_step1_outputs(run_dir: Path, *, cfg: SaveConfig, ts_data, fig_qtree, fig_weights, gene_df_node, metamesseage):
    if not cfg.save_config.enabled or cfg.save_config.phase != "step1":
        print("Not save")
        return
    timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S")
    save_dir = run_dir / f"{timestamp}_step1_outputs"
    
    save_yaml(metamesseage, save_dir / "metadata.yaml")
    save_yaml(config_to_dict(cfg.qtree_config), save_dir / "config_qtree.yaml")
    save_yaml(config_to_dict(cfg.weight_calculation_config), save_dir / "config_weight_calc.yaml")
    save_figure(fig_qtree, save_dir / "results" / "figures" / "quadtree.svg", bbox_inches='tight')
    save_figure(fig_qtree, save_dir / "results" / "figures" / "quadtree.png", bbox_inches='tight')
    save_figure(fig_qtree, save_dir / "results" / "figures" / "quadtree.pdf", bbox_inches='tight'   )
    
    save_figure(fig_weights, save_dir / "results" / "figures" / "weights_sparse.svg", bbox_inches='tight')
    save_figure(fig_weights, save_dir / "results" / "figures" / "weights_sparse.pdf", bbox_inches='tight')
    save_figure(fig_weights, save_dir / "results" / "figures" / "weights_sparse.png", bbox_inches='tight')
    save_dataframe(gene_df_node, save_dir / "cache" / "dataframes" / "gene_df_with_node.pkl")
    
    ts_data["ts_celltype_estimation"] = timestamp # 上書き更新する
    save_yaml(ts_data, run_dir / "ts_data.yaml")
    cfg.save_config.enabled = False

    print("Step 1 outputs have been saved to:", save_dir)

def save_step2_outputs(run_dir: Path, *, ts_data, cfg: SaveConfig, fig_initial, fig_cluster, result_df, metamesseage):
    if not cfg.save_config.enabled or cfg.save_config.phase != "step2":
        print("Not save")
        return
    
    timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S")
    save_dir = run_dir / f"{timestamp}_step2_outputs"
    save_yaml(metamesseage, save_dir / "metadata.yaml")
    
    save_figure(fig_initial, save_dir / "results" / "figures" / "initial_value.svg", bbox_inches='tight')
    save_figure(fig_cluster, save_dir / "results" / "figures" / "convex_hull.svg", bbox_inches='tight')
    save_figure(fig_initial, save_dir / "results" / "figures" / "initial_value.pdf", bbox_inches='tight')
    save_figure(fig_cluster, save_dir / "results" / "figures" / "convex_hull.png", bbox_inches='tight')
    save_figure(fig_initial, save_dir / "results" / "figures" / "initial_value.png", bbox_inches='tight')
    save_figure(fig_cluster, save_dir / "results" / "figures" / "convex_hull.pdf", bbox_inches='tight')
    save_dataframe(result_df, save_dir / "cache" / "dataframes" / "result_df.pkl")

    ts_data["ts_cell_estimation"] = timestamp # 上書き更新する
    save_yaml(ts_data, run_dir / "ts_data.yaml")
    cfg.save_config.enabled = False

    print("Step 2 outputs have been saved to:", save_dir)

def save_eval_outputs(run_dir: Path, *, cfg: SaveConfig, ts_data, evaluation_data, metamesseage):
    if not cfg.save_config.enabled or cfg.save_config.phase != "eval":
        print("Not save")
        return
    
    timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S")
    save_dir = run_dir / f"{timestamp}_eval_outputs"

    ts_data["ts_evaluation"] = timestamp # 上書き更新する
    save_yaml(ts_data, run_dir / "ts_data.yaml")
    
    tmp = to_builtin_type(evaluation_data)
    save_yaml(tmp, save_dir /  "metrics.yaml")
    save_markdown(metamesseage, save_dir /  "notes.md")
    save_yaml(config_to_dict(cfg), save_dir / "used_config.yaml")
    cfg.save_config.enabled = False
    
    print("Metrics have been saved to:", save_dir)