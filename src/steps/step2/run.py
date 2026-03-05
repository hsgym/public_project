import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_laplace, minimum_filter

from src.steps.step2.GMMWithCategoryFast import process_celltype
from src.steps.step2.select_svg import precompute_bins, select_svg


def run_svg_selection(cfg, gene_df_with_nodes,weights_sparse, valid_ids, type_system):
    # 発現点データに最小 bin を割り当てる
    grid_size = cfg.prior_estimation_config.grid_size # (px) グリッドの一辺のサイズ（集約処理はあるので最も小さいものを設定する想定）
    input_df = gene_df_with_nodes.copy()

    gene_df_with_bins = precompute_bins(input_df = input_df, width=cfg.base_config.width, height=cfg.base_config.height, grid_size= grid_size)

    input_df = gene_df_with_bins.copy()
    svgs_of_celltypes = {
        celltype: select_svg(
            celltype,
            input_df,
            width=cfg.base_config.width, 
            height=cfg.base_config.height,
            bin_size=grid_size,
            min_ratio=cfg.prior_estimation_config.gene_filter_min_ratio,
            min_count=cfg.prior_estimation_config.gene_filter_min_count,
            weights_sparse=weights_sparse,
            weight_threshold=cfg.prior_estimation_config.weight_threshold,
            global_moran_threshold = cfg.prior_estimation_config.global_moran_threshold,
            top_k = cfg.prior_estimation_config.number_of_svgs
        )
        for celltype in valid_ids
    }

    return svgs_of_celltypes, gene_df_with_bins



def run_gmm_initial(cfg, gene_df_with_bins, weights_sparse, valid_ids, svgs_of_celltypes):

    weight_threshold = cfg.prior_estimation_config.weight_threshold # 処理に使用する最小重み
    grid_size = cfg.prior_estimation_config.grid_size
    coords = gene_df_with_bins[['local_pixel_x','local_pixel_y']].values  
    coords = np.asarray(coords, dtype=np.float64)

    # 設定できる重み
    sigma_list = cfg.prior_estimation_config.gmm_sigma_list    
    distance_threshold = cfg.prior_estimation_config.distance_threshold


    gmm_components_dict = {}
    gmm_mu_init_dict = {}
    gmm_sigma_init_dict = {}
    histogram_dict = {}

    for cid in valid_ids:
        #初期化
        gmm_components_dict[cid] = 0
        gmm_mu_init_dict[cid] = None
        gmm_sigma_init_dict[cid] = None
        histogram_dict[cid] = None
        
        # 重み閾値で点を抽出
        w_col = weights_sparse[:,cid].toarray().ravel()
        idx_k = np.where(w_col > weight_threshold)[0]
        if len(idx_k) == 0:
            continue
        Xk = coords[idx_k]

        # SVG遺伝子の取得
        svgs = svgs_of_celltypes.get(cid, [])
        if len(svgs) == 0:
            # Xk はそのまま利用
            pass
        else:
            # SVG遺伝子に基づくフィルタリング
            gene_mask = gene_df_with_bins.iloc[idx_k]["gene"].isin(svgs).values
            Xk = Xk[gene_mask]
            if len(Xk) == 0:
                # SVGに該当する点がない場合はスキップ
                gmm_components_dict[cid] = 0
                gmm_mu_init_dict[cid] = None
                gmm_sigma_init_dict[cid] = None
                continue


        # 2Dヒストグラムによるグリッド化 
        H_all_gene, xedges, yedges = np.histogram2d(
            Xk[:,0], Xk[:,1], bins=[cfg.base_config.width // grid_size, cfg.base_config.height // grid_size], 
            range=[[0, cfg.base_config.width], [0, cfg.base_config.height]]
        )
        
        # 値を 0~1 で正規化
        H_normalized = H_all_gene / H_all_gene.sum() + 1e-12
        H_normalized = H_normalized.T # 画像と揃える

        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2
        
        # マルチスケールLoGを実施する
        peak_candidates = []
        for sigma in sigma_list:
            # H_smooth = gaussian_filter(H_normalized, sigma=sigma, mode='reflect')
            LoG = gaussian_laplace(H_normalized, sigma=sigma) * (sigma ** 2)
            # threshold = np.percentile(LoG, 5)  # 自動スケールは局所最大に変えてもOK
            # peak_indices = np.argwhere(LoG < threshold)

            # ローカル極小値（LoGが負で極小値を示す箇所）を見つける
            # LoGの負の極値は、ブロブの中心を示す候補
            local_minima = (LoG == minimum_filter(LoG, size=cfg.prior_estimation_config.size_for_minimum_filter)) # size は近傍の範囲

            # LoGの値が十分に低い（負に大きい）点のみを候補とする
            threshold = np.percentile(LoG, 5) # 応答が強い点を選ぶ（元のロジックを維持しつつ極値検出に近づける）
            strong_responses = (LoG < threshold)

            # ローカル極小値であり、かつ応答が強い点
            peak_indices = np.argwhere(local_minima & strong_responses)
            
            for py, px in peak_indices:
                val = LoG[py, px] # 応答の強さ (負の値。絶対値が大きいほど強い)
                real_x = x_centers[px]
                real_y = y_centers[py]
                peak_candidates.append([real_x, real_y, sigma, val])   
        
        peak_candidates = np.array(peak_candidates)


        if len(peak_candidates) > 0:
            # 応答が強い順 (小さい順=負に大きい順) にソート
            peak_candidates = peak_candidates[np.argsort(peak_candidates[:, 3])] 
            
            final_mu = []
            final_sigma = []
            
            while len(peak_candidates) > 0:
                best = peak_candidates[0]
                final_mu.append(best[:2])     # x, y
                final_sigma.append(best[2])   # sigma
                
                # この点に近いものを候補から削除
                dists = np.linalg.norm(peak_candidates[:, :2] - best[:2], axis=1)
                keep_mask = dists > distance_threshold
                peak_candidates = peak_candidates[keep_mask]
                
            mu_init = np.array(final_mu)
            sigma_init = np.array(final_sigma)
            n_cells_estimate = len(mu_init)
        else:
            mu_init = None
            sigma_init = None
            n_cells_estimate = max(1, len(Xk)//20)
        
        gmm_components_dict[cid] = n_cells_estimate
        gmm_mu_init_dict[cid] = mu_init   
        gmm_sigma_init_dict[cid] = sigma_init
        histogram_dict[cid] = H_normalized
        

    for cid in valid_ids:
        print(f"Celltype ID: {cid}, Estimated components: {gmm_components_dict[cid]}")
    print("total cell numbers:", sum(gmm_components_dict[cid] for cid in valid_ids))
    return gmm_components_dict, gmm_mu_init_dict, gmm_sigma_init_dict, histogram_dict



def run_cell_region_prediction(cfg, gene_df_with_bins, weights_sparse, valid_ids, gmm_components_dict, gmm_mu_init_dict, gmm_sigma_init_dict):
    input_df = gene_df_with_bins[
        ["local_pixel_x", "local_pixel_y", "gene", "qtree_celltype"]
    ].copy()
    input_df = input_df.reset_index().rename(columns={"index": "unique_id"})

    # --------------------------------
    # 並列処理で各細胞種を学習
    # --------------------------------
    results = Parallel(n_jobs=cfg.model_config.n_jobs)(
        delayed(process_celltype)(
            cid,
            input_df,
            gmm_components_dict[cid],
            weights_sparse,
            weight_threshold=cfg.model_config.weight_threshold,
            max_iter=cfg.model_config.max_iter,
            batch_size=cfg.model_config.batch_size,
            reg_covar=cfg.model_config.reg_covar,
            smoothing_alpha=cfg.model_config.smoothing_alpha,
            entropy_beta=cfg.model_config.entropy_beta,
            expected_cluster_coords=gmm_mu_init_dict.get(cid, None),
            sigma_init = gmm_sigma_init_dict.get(cid,None),
            search_radius=cfg.model_config.search_radius,
            distance_threshold=cfg.model_config.distance_threshold,
            background_id = len(valid_ids) -1,
            lambda_weight=cfg.model_config.lambda_weight,
            lambda_dist=cfg.model_config.lambda_dist,
            scale_T=cfg.model_config.scale_T
        )
        for cid in valid_ids)
    return results, input_df

def integrate_results(input_df, results, background_id, merge_radius=60, max_cluster_size=400, min_cluster_size=10):
    results = [r for r in results if r is not None]
    result_df = pd.concat(results, ignore_index=True)

    df = input_df.merge(result_df, on="unique_id", how="left")

    df["cluster_id"] = df["cluster_id"].fillna(background_id)
    df["cluster_score"] = df["cluster_score"].fillna(-np.inf)
    df["result_celltype"] = df["result_celltype"].fillna(background_id).astype(int)

    df = df.sort_values(["unique_id", "cluster_score"], ascending=[True, False],)

    best = (df.drop_duplicates("unique_id", keep="first").reset_index(drop=True))

    cluster_sizes = (
        best[best["cluster_id"] != background_id].groupby(["result_celltype", "cluster_id"]).size().rename("cluster_size").reset_index()
    )

    best = best.merge(cluster_sizes, on=["result_celltype", "cluster_id"], how="left")
    best["cluster_size"] = best["cluster_size"].fillna(0).astype(int)

    # merge_radius 以内にあり、かつサイズ制限内のクラスタを統合
    cluster_info = (
        best[best["cluster_id"] != background_id].groupby(["result_celltype", "cluster_id"])
        .agg(size=("unique_id", "count"), cx=("local_pixel_x", "mean"), cy=("local_pixel_y", "mean")).reset_index()
    )

    merge_map = {}

    for celltype, df_ct in cluster_info.groupby("result_celltype"):
        df_ct = df_ct.sort_values("size").reset_index(drop=True)

        centers = df_ct[["cx", "cy"]].to_numpy()
        tree = KDTree(centers)

        for row in df_ct.itertuples():
            k = row.cluster_id
            if k in merge_map:
                continue

            idxs = tree.query_radius([[row.cx, row.cy]],r=merge_radius)[0]

            for j in idxs:
                other = df_ct.iloc[j]
                l = other["cluster_id"]
                if l == k or l in merge_map:
                    continue
                if row.size + other["size"] > max_cluster_size:
                    continue
                merge_map[k] = l
                break

    if merge_map:
        compressed = {}
        for k in merge_map:
            v = k
            while v in merge_map:
                v = merge_map[v]
            compressed[k] = v
        merge_map = compressed
        best["cluster_id"] = best["cluster_id"].replace(merge_map)

    # 小さいクラスタ検出
    small_mask = ((best["cluster_id"] != background_id) &(best["cluster_size"] < min_cluster_size))
    small_ids = best.loc[small_mask, "unique_id"]

    df_ranked = df.copy()
    df_ranked["rank"] = (df_ranked.groupby("unique_id")["cluster_score"].rank(method="first", ascending=False))
    df_ranked = df_ranked.merge(cluster_sizes,on=["result_celltype", "cluster_id"],how="left")
    df_ranked["cluster_size"] = df_ranked["cluster_size"].fillna(0)

    valid = ((df_ranked["unique_id"].isin(small_ids)) &(df_ranked["rank"] > 1) 
             &(df_ranked["cluster_id"] != background_id) &(df_ranked["cluster_size"] >= min_cluster_size))

    reassigned = (df_ranked[valid].sort_values(["unique_id", "rank"]).drop_duplicates("unique_id"))
    fallback = (
        df_ranked[df_ranked["unique_id"].isin(small_ids) &(df_ranked["rank"] == 1)]
        .assign(result_celltype=background_id,cluster_id=background_id,cluster_score=-np.inf,cluster_size=0,)
    )
    reassigned_df = (pd.concat([reassigned, fallback]).drop_duplicates("unique_id", keep="first"))

    best = best[~best["unique_id"].isin(small_ids)]
    best = pd.concat([best, reassigned_df], ignore_index=True)
    best = best.sort_values("unique_id").reset_index(drop=True)
    return best

def run_step2(cfg, gene_df_with_nodes, weights_sparse, type_system):
    valid_ids = type_system.valid_type_ids
    svgs_of_celltypes, gene_df_with_bins =  run_svg_selection(cfg, gene_df_with_nodes, weights_sparse, valid_ids, type_system)
    gmm_components_dict, gmm_mu_init_dict, gmm_sigma_init_dict, histogram_dict =run_gmm_initial(cfg, gene_df_with_bins, weights_sparse, valid_ids, svgs_of_celltypes)
    results, gene_df_with_bins = run_cell_region_prediction(cfg, gene_df_with_bins, weights_sparse, valid_ids, gmm_components_dict, gmm_mu_init_dict, gmm_sigma_init_dict)
    result_df = integrate_results(gene_df_with_bins, results, background_id=type_system.background_id, merge_radius=10, min_cluster_size=20)
    return result_df, gmm_mu_init_dict, histogram_dict