# src/configs/converters.py
from dataclasses import asdict
from .parameter_schemas import (
    ExperimentConfig,
    BaseConfig,
    StDataLoadConfig,
    ScDataLoadConfig,
    MarkerSelectionConfig,
    QtreeConfig,
    WeightCalculationConfig,
    PriorEstimationConfig,
    ModelConfig,
    SaveConfig,
)

def dict_to_config(d: dict) -> ExperimentConfig:
    return ExperimentConfig(
        base_config=BaseConfig(**d.get("base_config", {})),
        st_data_load_config=StDataLoadConfig(**d["st_data_load_config"]),
        sc_data_load_config=ScDataLoadConfig(**d["sc_data_load_config"]),
        marker_selection_config=MarkerSelectionConfig(**d["marker_selection_config"]),
        qtree_config=QtreeConfig(**d["qtree_config"]),
        weight_calculation_config=WeightCalculationConfig(**d["weight_calculation_config"]),
        prior_estimation_config=PriorEstimationConfig(**d.get("prior_estimation_config", {})),
        model_config=ModelConfig(**d["model_config"]),
        save_config=SaveConfig(**d.get("save_config", {})),
    )

def config_to_dict(cfg: ExperimentConfig) -> dict:
    return asdict(cfg)
