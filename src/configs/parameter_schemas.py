# src/configs/parameter_schemas.py

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BaseConfig:
    measurement_type: str # merscopre, xenium, simulation, etc.
    tissue_type: str      # ileum, breast_cancer, lung_cancer, etc.
    width: int
    height: int

@dataclass
class StDataLoadConfig:
    fov: Optional[str] = None
    z: Optional[float] = None
    input_dir: str = None  # Must be specified
    output_dir: Optional[str] = None

@dataclass
class ScDataLoadConfig:
    preprocessed_type: str = None # どんな処理をするかに応じたフラグ
    ref_path: str = None # 参照データへのパス
    
@dataclass
class MarkerSelectionConfig:
    number_of_markers: int = 10 # Number of markers to select
   
    
@dataclass
class QtreeConfig:
    limit_depth: int = 5
    assign_thresh: float = 0.8
    assign_thresh_lower: float = 0.5
    sim_thresh: float = 0.5
    neighbor_weight: float = 0.5
    score_margin: float = 0.5
    sim_method: str = "cosine"

@dataclass # 各点の重み計算に使う
class WeightCalculationConfig:
    boundary_threshold_px: float = 8.0
    neighbor_score_threshold: float = 0.2
    soft_k: int = 2
    alpha: float = 1.5  # α↑で支配的細胞種を強調，β↑で隣接葉影響を強める
    beta: float = 1.0
    min_weight: float = 0.01    


@dataclass # モデルの初期値推定に使う
class PriorEstimationConfig:
    # Peak検出用のパラメータ（デフォルト値なし）
    gmm_sigma_list: list[float]    # GMMのσ候補リスト
    grid_size: int = 8 # (px) 集約するグリッドのサイズ
    weight_threshold: float = 0.01 # 点抽出に使用する最小重み
    # SVG 検出用のパラメータ
    gene_filter_min_ratio: float = 0.01 # gene の低頻度フィルタの最小比率
    gene_filter_min_count: int = 5    # gene の低頻度フィルタの最小出現数
    global_moran_threshold: float = 0.05 # グローバルMoran's I閾値
    number_of_svgs: int = 20  # 抽出するSVGの数
    # Peak検出用のパラメータ
    distance_threshold: float = 100.0  # px; 分布のμ間の最低距離＝細胞の最小半径
    size_for_minimum_filter: int = 10  # 近傍の範囲(px)
    
@dataclass
class ModelConfig:
    max_iter: int = 20
    batch_size: int = 200000
    reg_covar: float = 0.000001
    smoothing_alpha: float = 0.01
    entropy_beta: float = 0.1
    search_radius: float = 100.0 # E-stepで考慮する近傍距離(px)
    distance_threshold: float = 75.0 # 最大半径
    # スコア計算に使うやつ
    lambda_weight: float = 1.0
    lambda_dist: float = 1.0
    scale_T: float = 1.0
    # その他
    weight_threshold: float = 0.01 # 点抽出に使用する最小重み
    n_jobs: int = -1 # 並列処理数 (-1で全コア使用)
    def __post_init__(self):
        self.reg_covar = float(self.reg_covar)
        self.smoothing_alpha = float(self.smoothing_alpha)
   
    
@dataclass
class SaveConfig:
    enabled: bool = False
    phase: str | None = None   # "step1" | "step2"
    run_name: str | None = None


@dataclass
class ExperimentConfig:
    base_config: BaseConfig = field(default_factory=BaseConfig)
    st_data_load_config: StDataLoadConfig = field(default_factory=StDataLoadConfig)
    sc_data_load_config: ScDataLoadConfig = field(default_factory=ScDataLoadConfig)
    marker_selection_config: MarkerSelectionConfig = field(default_factory=MarkerSelectionConfig)
    qtree_config: QtreeConfig = field(default_factory=QtreeConfig)
    weight_calculation_config: WeightCalculationConfig = field(default_factory=WeightCalculationConfig)
    prior_estimation_config: PriorEstimationConfig = field(default_factory=PriorEstimationConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    save_config: SaveConfig = field(default_factory=SaveConfig)
    
    