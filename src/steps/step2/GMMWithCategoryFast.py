import numpy as np
import numpy.typing as npt
import math
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp, softmax
from sklearn.neighbors import KDTree

from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from scipy.sparse import csr_matrix

# type: ignore[reportAttributeAccessIssue]

class GMMWithCategoryFast:
    """
    Gaussian-Categorical mixture EM with:
      - log-domain computations
      - batch processing to limit memory
      - Dirichlet smoothing for phi updates
      - vectorized full-cluster label prediction
    """
    def __init__(self, n_components: int, n_categories: int,
                 max_iter: int = 20, batch_size: int = 5000,
                 reg_covar: float = 1e-6, smoothing_alpha: float = 1e-2, random_state: int = 0):
        
        self.K = n_components
        self.M = n_categories
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.reg_covar = reg_covar
        self.smoothing_alpha = smoothing_alpha
        self.random_state = random_state

        # params
        self.weights_: npt.NDArray[np.float64] | None = None     # (K,)
        self.means_: npt.NDArray[np.float64] | None  = None      # (K,D)
        self.covariances_ : npt.NDArray[np.float64] | None = None# (K,D,D)
        self.phi_ : npt.NDArray[np.float64] | None = None        # (K,M)

        # constants
        self._log_2pi: npt.NDArray[np.float64] | None  = None

    def _init_params(self, X, C):
        N, D = X.shape
        gmm = GaussianMixture(n_components=self.K, covariance_type='full',
                              random_state=self.random_state, init_params='k-means++',
                              max_iter=max(10, self.max_iter))
        gmm.fit(X)

        self.weights_ = gmm.weights_.astype(np.float64)
        self.means_ = gmm.means_.astype(np.float64)
        covs = gmm.covariances_.astype(np.float64)
        for k in range(self.K):
            covs[k] += self.reg_covar * np.eye(D, dtype=np.float64)
        self.covariances_ = covs

        rng = np.random.RandomState(self.random_state)
        
        # Dirichlet分布から初期化： $\phi_k \sim Dir(\alpha, \alpha, ..., \alpha)$、 $\alpha=1$ として一様分布に近い形にする
        self.phi_ = rng.dirichlet(np.ones(self.M, dtype=np.float64), size=self.K).astype(np.float64)

        self._log_2pi = D * math.log(2 * math.pi)

    def _precompute_cov_info(self):
        K = self.K
        D = self.means_.shape[1] 
        self.inv_covs_ = np.empty((K, D, D), dtype=np.float64)
        self.logdets_ = np.empty(K, dtype=np.float64)
        for k in range(K):
            cov = self.covariances_[k] + self.reg_covar * np.eye(D, dtype=np.float64)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                cov += 1e-6 * np.eye(D, dtype=np.float64)
                sign, logdet = np.linalg.slogdet(cov)
            self.logdets_[k] = logdet
            self.inv_covs_[k] = np.linalg.inv(cov)

    def fit(self, X, C):
        X = np.asarray(X, dtype=np.float64)
        C = np.asarray(C, dtype=np.int32)
        N, D = X.shape
        K, M = self.K, self.M

        self._init_params(X, C)
        self._precompute_cov_info()

        eps = 1e-12
        log_phi = np.log(np.maximum(self.phi_, eps))

        for it in range(self.max_iter):
            Nk = np.zeros(K, dtype=np.float64)
            means_num = np.zeros((K, D), dtype=np.float64)
            cov_num = np.zeros((K, D, D), dtype=np.float64)
            phi_counts = np.zeros((K, M), dtype=np.float64)

            # process in batches
            for start in range(0, N, self.batch_size):
                end = min(N, start + self.batch_size)
                xb = X[start:end]
                cb = C[start:end]
                B = xb.shape[0]

                # vectorized Mahalanobis distances
                diff = xb[:, None, :] - self.means_[None, :, :]  # (B, K, D)
                m = np.einsum('bkd,kde,bke->bk', diff, self.inv_covs_, diff)
                log_px = -0.5 * (self._log_2pi + self.logdets_[None, :] + m)   

                log_pi = np.log(np.maximum(self.weights_, 1e-300))[None, :]  # (1, K)  $\pi_k$ を対数を取る
                log_phi_k = log_phi[:, cb].T  # (B, K) $\phi_{k,c_i}$ を対数を取る
                log_num = log_pi + log_px + log_phi_k # いわゆる責任度の分子部分 (B, K)

                # responsibilities
                log_norm = logsumexp(log_num, axis=1, keepdims=True) # log_numについて、各行の和の対数を計算 (B, 1)（これが分母になる）
                log_resp = log_num - log_norm # log-domainでの責任度 (B, K)、割り算の代わりに引き算
                resp = np.exp(log_resp, dtype=np.float64) # 対数を元の値に戻す

                # accumulate
                Nk += resp.sum(axis=0)
                means_num += resp.T @ xb
                # covariance numerator
                for k in range(K):
                    diff_k = xb - self.means_[k]
                    cov_num[k] += (resp[:, k][:, None] * diff_k).T @ diff_k
                # phi counts
                for k in range(K):
                    phi_counts[k] += np.bincount(cb, weights=resp[:, k], minlength=M)

            # M-step
            Nk_safe = np.maximum(Nk, 1e-8)
            self.weights_ = Nk_safe / Nk_safe.sum()
            self.means_ = (means_num.T / Nk_safe).T
            for k in range(K):
                cov_k = cov_num[k] / Nk_safe[k]
                cov_k += self.reg_covar * np.eye(D, dtype=np.float64)
                self.covariances_[k] = cov_k 
            self.phi_ = (phi_counts + self.smoothing_alpha)
            self.phi_ = (self.phi_.T / self.phi_.sum(axis=1)).T
            self._precompute_cov_info()
            log_phi = np.log(np.maximum(self.phi_, eps)) 
            
    def predict_labels(self, X, C):
        """
        Vectorized full-cluster prediction.
        """
        X = np.asarray(X, dtype=np.float64)
        C = np.asarray(C, dtype=np.int32)
        N, D = X.shape
        K, M = self.K, self.M
        eps = 1e-12 
        log_phi = np.log(np.maximum(self.phi_, eps))

        labels = np.empty(N, dtype=np.int32)
        scores = np.empty(N, dtype=np.float64)

        for start in range(0, N, self.batch_size):
            end = min(N, start + self.batch_size)
            xb = X[start:end]
            cb = C[start:end]
            B = xb.shape[0]

            diff = xb[:, None, :] - self.means_[None, :, :]  # (B, K, D)
            m = np.einsum('bkd,kde,bke->bk', diff, self.inv_covs_, diff)
            log_px = -0.5 * (self._log_2pi + self.logdets_[None, :] + m)

            log_pi = np.log(np.maximum(self.weights_, 1e-300))[None, :]
            log_phi_k = log_phi[:, cb].T
            log_num = log_pi + log_px + log_phi_k # (B, K)

            labels[start:end] = np.argmax(log_num, axis=1)
            scores[start:end] = self._compute_scores(log_num)  # 拡張ポイント
        return labels, scores
    
    def _compute_scores(self, log_num):
        return np.max(log_num, axis=1)


class GMMWithCategoryWeighted_v2(GMMWithCategoryFast):
    def __init__(self, n_components: int, n_categories: int,
                 max_iter: int = 20, batch_size: int = 200000,
                 reg_covar: float = 1e-6, smoothing_alpha: float = 1e-2, 
                 random_state: int = 0, entropy_beta: float = 0.1, means_init=None, sigma_init = None, search_radius=100.0):
        super().__init__(n_components, n_categories, max_iter, batch_size, reg_covar, smoothing_alpha, random_state)
        self.entropy_beta = entropy_beta  # スコア補正強度
        self.means_init = means_init
        self.sigma_init = sigma_init
        self.search_radius = search_radius


    # 初期化
    def _init_params(self, X, C):
        # gene 出現頻度に基づいて初期 φ を設定
        gene_counts = np.bincount(C, minlength=self.M) + 1e-2
        gene_probs = gene_counts / gene_counts.sum()
        self.phi_ = np.tile(gene_probs, (self.K, 1))
        
        N, D = X.shape
        if (self.means_init is not None) and (self.sigma_init is not None):
            inv_variance = 1 / (self.sigma_init ** 2)
            precisions_init = (np.eye(2)[np.newaxis, :, :] * inv_variance[:, np.newaxis, np.newaxis])
            gmm = GaussianMixture(n_components=self.K, covariance_type='full',
                              random_state=self.random_state, init_params='k-means++',
                              max_iter=max(10, self.max_iter), means_init=self.means_init, precisions_init=precisions_init, n_init=1)
        else:
            gmm = GaussianMixture(n_components=self.K, covariance_type='full',
                              random_state=self.random_state, init_params='k-means++',
                              max_iter=max(10, self.max_iter), n_init=1)
        gmm.fit(X)

        self.weights_ = gmm.weights_.astype(np.float64)
        self.means_ = gmm.means_.astype(np.float64)
        covs = gmm.covariances_.astype(np.float64)
        for k in range(self.K):
            covs[k] += self.reg_covar * np.eye(D, dtype=np.float64)
        self.covariances_ = covs

        rng = np.random.RandomState(self.random_state)
        
        # Dirichlet分布から初期化： $\phi_k \sim Dir(\alpha, \alpha, ..., \alpha)$、 $\alpha=1$ として一様分布に近い形にする
        self.phi_ = rng.dirichlet(np.ones(self.M, dtype=np.float64), size=self.K).astype(np.float64)

        self._log_2pi = D * math.log(2 * math.pi)


    def _precompute_cov_info(self):
        K, D, _ = self.covariances_.shape
        self.inv_covs_ = np.linalg.inv(self.covariances_)
        self.logdets_ = np.array([np.log(np.linalg.det(self.covariances_[k]))
                                  for k in range(K)])
        self._log_2pi = D * math.log(2 * math.pi)


    def _sparse_estep_batch(self, xb, cb, log_phi, tree):

        B, D = xb.shape
        K = self.K

        # 初期化（すべて -∞ = ほぼ0 の責務に）
        log_resp = np.full((B, K), -np.inf, dtype=np.float64)

        # 近い成分を KD-tree で探索
        neighbor_ids = tree.query_radius(xb, r=self.search_radius)

        log_pi = np.log(np.maximum(self.weights_, 1e-300))

        for i in range(B):
            neigh = neighbor_ids[i]

            if len(neigh) == 0:
                # 近いものがない → 最近傍1個
                _, ind = tree.query(xb[i:i+1], k=1)
                neigh = ind[0]

            Xi = xb[i]

            diff = Xi - self.means_[neigh]   # (K_near, D)
            m = np.einsum("kd,kde,ke->k", diff,
                          self.inv_covs_[neigh], diff)

            log_px = -0.5 * (self._log_2pi + self.logdets_[neigh] + m)

            # φ (カテゴリ) を加える
            log_phi_neigh = log_phi[neigh, cb[i]]

            log_num = log_pi[neigh] + log_px + log_phi_neigh

            log_resp[i, neigh] = log_num

        # 正規化（数値安定）
        log_norm = logsumexp(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - log_norm)

        return resp


    # ====================================================
    #  メインの EM
    # ====================================================
    def fit(self, X, C):
        X = np.asarray(X, dtype=np.float64)
        C = np.asarray(C, dtype=np.int32)
        N, D = X.shape
        K, M = self.K, self.M

        # 初期化（既存コード）
        self._init_params(X, C)
        self._precompute_cov_info()

        eps = 1e-12

        for it in range(self.max_iter):

            # ★ E-step 前に KD-tree を構築
            tree = KDTree(self.means_)     # (K, D) の中心に対して

            Nk = np.zeros(K)
            means_num = np.zeros((K, D))
            cov_num = np.zeros((K, D, D))
            phi_counts = np.zeros((K, M))

            log_phi = np.log(np.maximum(self.phi_, eps))

            # ---------------------------
            # E-step（各バッチで sparse 計算）
            # ---------------------------
            for start in range(0, N, self.batch_size):
                end = min(N, start + self.batch_size)
                xb = X[start:end]
                cb = C[start:end]

                resp = self._sparse_estep_batch(xb, cb, log_phi, tree)

                # ---- 集約 ----
                Nk += resp.sum(axis=0)
                means_num += resp.T @ xb

                for k in range(K):
                    w = resp[:, k]
                    if w.sum() == 0:
                        continue
                    diff_k = xb - self.means_[k]
                    cov_num[k] += (w[:, None] * diff_k).T @ diff_k
                    phi_counts[k] += np.bincount(cb, weights=w, minlength=M)

            # ---------------------------
            # M-step（既存コードと同じ）
            # ---------------------------
            Nk_safe = np.maximum(Nk, 1e-8)
            self.weights_ = Nk_safe / Nk_safe.sum()
            pi_min = np.maximum(5 / Nk_safe.sum(), 1e-4)
            self.weights_ = np.maximum(self.weights_, pi_min)   # 下限拘束
            self.weights_ = self.weights_ / self.weights_.sum()  
            
            self.means_ = (means_num.T / Nk_safe).T

            for k in range(K):
                cov = cov_num[k] / Nk_safe[k] + self.reg_covar * np.eye(D)
                self.covariances_[k] = cov

            # φ 更新
            self.phi_ = phi_counts + self.smoothing_alpha * Nk_safe.sum() / K
            self.phi_ = (self.phi_.T / self.phi_.sum(axis=1)).T

            self._precompute_cov_info()

        return self

    # =============================
    # 予測 & スコア計算
    # =============================
    def predict_labels(self, X, C, sample_weight=None,
                    lambda_weight=1.0, lambda_dist=1.0,
                    scale_T=None, distance_threshold=75.0):

        X = np.asarray(X, dtype=np.float64)
        C = np.asarray(C, dtype=np.int32)
        N, D = X.shape
        K, M = self.K, self.M

        eps = 1e-12
        log_phi = np.log(np.maximum(self.phi_, eps))
        labels = np.full(N, -1, dtype=np.int32)
        scores = np.full(N, -np.inf, dtype=np.float64)

        # type weight（この type の事前確率）
        if sample_weight is None:
            type_weights = np.ones(N, dtype=np.float64)
        else:
            type_weights = np.asarray(sample_weight, dtype=np.float64).ravel()

        # 空間スケール（共分散 trace の平均など）
        if scale_T is None:
            scale_T = np.mean([np.trace(cov) for cov in self.covariances_]) + 1e-12

        # 事前計算
        sigma2_max = np.mean([
            np.max(np.linalg.eigvalsh(cov))
            for cov in self.covariances_
        ])

        T2 = (distance_threshold ** 2) / (sigma2_max + 1e-12)

        # 推奨
        lambda_dist = 0.5


        for start in range(0, N, self.batch_size):
            end = min(N, start + self.batch_size)
            xb = X[start:end]
            cb = C[start:end]
            wb = type_weights[start:end]
            B = xb.shape[0]

            # diff[b,k,d] = x_b - mu_k
            diff = xb[:, None, :] - self.means_[None, :, :]  # (B,K,D)

            # Mahalanobis距離の対数項
            m = np.einsum('bkd,kde,bke->bk', diff, self.inv_covs_, diff)

            # log p(x|k)
            log_px = -0.5 * (self._log_2pi + self.logdets_[None, :] + m)

            # log π_k（混合重み）
            log_pi = np.log(np.maximum(self.weights_, 1e-300))[None, :]

            # log φ(k,c)
            log_phi_k = log_phi[:, cb].T

            # log_num[b,k] = log p(k,x,c) （同時対数確率）
            log_num = log_pi + log_px + log_phi_k  # (B,K)
            
            # クラスタ割り当て用スコア計算（全クラスタ対象）
            penalty = lambda_dist * np.minimum(m, T2)
            cluster_score = log_num - penalty
            labels[start:end] = np.argmax(cluster_score, axis=1) 
            
            # 細胞種割り当て用スコア計算関数（単一の値のはず）
            celltype_score = self._compute_celltype_score(cluster_score, wb, lambda_w=lambda_weight, lambda_d=lambda_dist)
            scores[start:end] = celltype_score
        return labels, scores
            
    
    def _compute_celltype_score(self, cluster_score, type_weights, lambda_w=1, lambda_d=1):

        # log(w) + log(sum(exp(log_num_valid))) を計算したい。

        eps = 1e-12 # 数値安定化用

        # log(w)
        lw = lambda_w * np.log(type_weights + eps)
        
        # log(sum(exp(log_num_valid)))
        lse = logsumexp(cluster_score, axis=1)
        
        score = lw + lse
        return score 

    def _compute_score(self, log_num_valid, cluster_mask_valid, type_weight_valid,
                        labels_valid, lambda_w=0.5, lambda_d=0, scale_T=1.0):

        # log(w) + log(sum(exp(log_num_valid))) を計算したい。

        eps = 1e-12 # 数値安定化用
        B = len(labels_valid)
        
        # log(w)
        lw = lambda_w * np.log(type_weight_valid + eps)
        
        # log(sum(exp(log_num_valid)))
        log_num_masked = np.where(cluster_mask_valid, log_num_valid, -np.inf)
        lse = lambda_d * logsumexp(log_num_masked, axis=1)
        
        score = lw + lse
        return score
        
        
def process_celltype(celltype_name, input_subset, n_clusters, weights_sparse, weight_threshold=0.01,
            max_iter=None,
            batch_size=None,
            reg_covar=None,
            smoothing_alpha=None,
            entropy_beta=None,
            expected_cluster_coords=None,
            sigma_init=None,
            search_radius=None,
            distance_threshold=None,
            background_id=None, 
            lambda_weight=None,
            lambda_dist=None,
            scale_T=None
    ):
    try:
        if n_clusters <= 0:
            return None
        celltype_id = celltype_name

        # ==== まずcelltypeでフィルタ ====
        col = weights_sparse.getcol(celltype_id).tocoo()
        active_rows = col.row[col.data >= weight_threshold]
        if active_rows.size == 0:
            return None

        gene_df_celltype = input_subset.loc[active_rows].copy()
        if gene_df_celltype.empty:
            return None

        # ==== celltypeサブセットに合わせて sample_weight を抽出 ====
        idx = gene_df_celltype.index.values
        sample_weight = weights_sparse[idx, celltype_id].toarray().ravel()
        gene_df_celltype["sample_weight"] = sample_weight
        X = gene_df_celltype[["local_pixel_x", "local_pixel_y"]].values

        # ==== Rare gene 処理（ここでも sample_weight を同期） ====
        gene_counts = gene_df_celltype["gene"].value_counts()
        threshold_count = max(1, int(0.005 * len(gene_df_celltype)))
        rare_genes = gene_counts[gene_counts < threshold_count].index
        gene_df_celltype.loc[gene_df_celltype["gene"].isin(rare_genes), "gene"] = "other"
        unique_genes = sorted(gene_df_celltype["gene"].unique())
        gene_to_int = {g:i for i,g in enumerate(unique_genes)}
        C = np.array([gene_to_int[g] for g in gene_df_celltype["gene"]])
        n_categories = len(unique_genes)
        sample_weight_2d = gene_df_celltype["sample_weight"].values# .reshape(-1, 1)

        # ==== モデルの定義 ====
        model = GMMWithCategoryWeighted_v2(
            n_components=n_clusters,
            n_categories=n_categories,
            max_iter=max_iter,
            batch_size=batch_size,
            reg_covar=reg_covar,
            smoothing_alpha=smoothing_alpha,
            entropy_beta=entropy_beta,
            means_init=expected_cluster_coords,
            sigma_init=sigma_init,
            search_radius=search_radius
        )
        model.fit(X, C)
        
        # ==== 予測と結果格納 ====
        cluster_labels, cluster_scores = model.predict_labels(X, C, 
                        sample_weight=sample_weight_2d, distance_threshold=distance_threshold,
                        lambda_weight=lambda_weight, lambda_dist=lambda_dist,
                        scale_T= scale_T)
        
        mask = cluster_scores > -np.inf
        if not np.any(mask):
            return None

        gene_df_celltype = gene_df_celltype.loc[mask].copy()
        gene_df_celltype["cluster_id"] = [
            f"{celltype_name}_{cid}"
            for cid in cluster_labels[mask]
        ]
        gene_df_celltype["cluster_score"] = cluster_scores[mask]
        gene_df_celltype["result_celltype"] = celltype_name

        return gene_df_celltype[["unique_id", "cluster_id", "cluster_score", "result_celltype"]]


    except Exception as e:
        print(f"Error processing {celltype_name}: {e}")
        return None




# 使用例
if __name__ == "__main__":
    import pandas as pd

    # ダミーデータの作成
    np.random.seed(0)
    N = 1000 # データポイント数
    D = 2    # 特徴量の次元（二次元点群）
    M = 5    # カテゴリ数
    K = 10   # クラスタ数

    # ランダムなデータポイント
    X = np.random.randn(N, D)

    # ランダムなカテゴリ（0からM-1までの整数）
    C = np.random.randint(0, M, size=N).tolist()

    # モデルの初期化と学習
    model = GMMWithCategoryFast(n_components=K, n_categories=M, max_iter=20, random_state=42)
    model.fit(X, C)

    # クラスタラベルの予測
    labels, scores = model.predict_labels(X, C)
    print("Predicted labels:", labels)
    print("Scores:", scores)
    
    
    
