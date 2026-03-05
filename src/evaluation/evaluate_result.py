# import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def evaluate_performance(evaluatioin_df, type_system, config):
    """
    config: {'background_id': 10, 'unknown_id': 9, 'compare_bg_id': -1} などの設定
    """
    res = {}  # 結果格納用
    
    # --- 共通定義 ---
    bg_ids = [config['background_id'], config['unknown_id']]
    comp_bg_id = config.get('compare_bg_id', -1)
    proposed_cell_id = config.get('proposed_cell_id', 'cluster_id')
    compared_cell_id = config.get('compared_cell_id', 'cell_id')
    proposed_celltype = config.get('proposed_celltype', 'result_celltype')
    compared_celltype = config.get('compared_celltype', 'true_celltype')
    
    # 1. Basic Stats (提案手法のみ)
    res['total_points'] = len(evaluatioin_df)
    res['num_genes'] = evaluatioin_df['gene'].nunique()
    
    tmp_p = evaluatioin_df[~evaluatioin_df[proposed_cell_id].isin(bg_ids)]
    res['num_proposed_cells'] = tmp_p[proposed_cell_id].nunique()
    res['num_proposed_celltypes'] = tmp_p[proposed_celltype].nunique()
    total = len(evaluatioin_df)
    inside_p = len(tmp_p)
    res['proposed_coverage'] = round(inside_p / total, 4) if total > 0 else 0
    res['proposed_coverage_formula'] = f"{inside_p} / {total}"

    # 2. Comparison Basic (比較対象がある場合)
    if compared_cell_id in evaluatioin_df.columns:
        tmp_c = evaluatioin_df[evaluatioin_df[compared_cell_id] != comp_bg_id]
        res['num_compared_cells'] = tmp_c[compared_cell_id].nunique()
        if compared_celltype in evaluatioin_df.columns:
            res["num_compared_celltypes"] = tmp_c[compared_celltype].nunique()
        total = len(evaluatioin_df)
        inside_c = len(tmp_c)
        res['compared_coverage'] = round(inside_c / total, 4) if total > 0 else 0
        res['compared_coverage_formula'] = f"{inside_c} / {total}"
    
    
    # 3. Clustering Agreement (両方の細胞IDが存在し、背景でない点のみ)
    if compared_cell_id in evaluatioin_df.columns:
        # 両方が背景でない点を抽出
        mask = (~evaluatioin_df[proposed_cell_id].isin(bg_ids)) & (evaluatioin_df[compared_cell_id] != comp_bg_id)
        tmp_filtered = evaluatioin_df[mask]
        
        if len(tmp_filtered) > 0:
            res['ARI'] = round(adjusted_rand_score(tmp_filtered[compared_cell_id], tmp_filtered[proposed_cell_id]), 4)
            res['NMI'] = round(normalized_mutual_info_score(tmp_filtered[compared_cell_id], tmp_filtered[proposed_cell_id]), 4)
        else:
            res['ARI'], res['NMI'] = 0,0

    # 4. Celltype Accuracy (true_celltypeが存在する場合)
    if compared_celltype in evaluatioin_df.columns:
        # 背景やunknownを除外した精度を計算する場合 (推奨)
        labeled_mask = (~evaluatioin_df[compared_celltype].isin(bg_ids + [comp_bg_id]))
        labeled_df = evaluatioin_df[labeled_mask]
        
        if len(labeled_df) > 0:
            res['overall_accuracy'] = round((labeled_df[compared_celltype] == labeled_df[proposed_celltype]).mean(), 4)
            num_correct = (labeled_df[compared_celltype] == labeled_df[proposed_celltype]).sum()
            res['overall_accuracy_formula'] = f"{num_correct} / {len(labeled_df)}"
            
            # 細胞種ごとの内訳（可読性のための辞書）
            res['celltype_breakdown'] = {}
            res['celltype_breakdown_formula'] = {}
            for tid in labeled_df[compared_celltype].unique():
                c_df = labeled_df[labeled_df[compared_celltype] == tid]
                acc = round((c_df[compared_celltype] == c_df[proposed_celltype]).mean(), 4)
                
                tcelltype = type_system.type_id_to_celltypes(tid)
                res['celltype_breakdown'][tcelltype] = acc
                num_c = len(c_df)
                num_correct_c = (c_df[compared_celltype] == c_df[proposed_celltype]).sum()
                res['celltype_breakdown_formula'][tcelltype] = f"{num_correct_c} / {num_c}"
        
    return res


def print_evaluation_summary(res, type_system):
    """
    evaluate_performance の戻り値(res)を綺麗に表示する
    """
    # 細胞種IDから名前を引くための辞書
    # type_id_to_name = type_system.type_id_to_celltypes
    
    print(f"{' PERFORMANCE EVALUATION REPORT ':=^50}")

    # Section 1: Basic Dataset Info
    print(f"\n[1] Basic Statistics")
    print(f"  - Total Expression Points : {res.get('total_points', 0):,}")
    print(f"  - Number of Genes         : {res.get('num_genes', 0)}")

    # Section 2: Coverage (Proposed vs Compared)
    print(f"\n[2] Cell Recovery & Coverage")
    print(f"  - Proposed Method:")
    print(f"    - Cells found  : {res.get('num_proposed_cells', 0)}")
    print(f"    - Celltypes    : {res.get('num_proposed_celltypes', 0)}")
    print(f"    - Coverage     : {res.get('proposed_coverage', 0):.4f}  ({res.get('proposed_coverage_formula', '')})")
    
    if 'num_compared_cells' in res:
        print(f"  - Compared Method:")
        print(f"    - Cells found  : {res.get('num_compared_cells', 0)}")
        if 'num_compared_celltypes' in res:
            print(f"    - Celltypes    : {res.get('num_compared_celltypes', 0)}")
        print(f"    - Coverage     : {res.get('compared_coverage', 0):.4f}  ({res.get('compared_coverage_formula', '')})")

    # Section 3: Clustering Agreement
    if 'ARI' in res:
        print(f"\n[3] Clustering Agreement (Filtering Background)")
        print(f"  - ARI : {res.get('ARI'):.4f}")
        print(f"  - NMI : {res.get('NMI'):.4f}")

    # Section 4: Accuracy & Breakdown
    if 'overall_accuracy' in res:
        print(f"\n[4] Celltype Identification Accuracy")
        print(f"  - Overall Accuracy : {res.get('overall_accuracy'):.4f}  ({res.get('overall_accuracy_formula', '')})")
        
        print(f"  - Breakdown by Celltype:")
        # 正答率が高い順に表示
        breakdown = res.get('celltype_breakdown', {})
        formulas = res.get('celltype_breakdown_formula', {})
        sorted_types = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        
        for tid, acc in sorted_types:
            name = tid # ype_system.type_id_to_celltypes(tid)
            formula = formulas.get(tid, "")
            print(f"    - {name} :   {acc:<10.4f} ({formula})")

# 使用例
if __name__ == "__main__":
    from src.evaluation.evaluate_result import evaluate_performance, print_evaluation_summary
    from src.data.type_system import TypeSystem

    # Example usage
    evaluatioin_df = gene_df_with_cells.copy() # 名称的に分かりやすくするため

    evaluation_dict = evaluate_performance(
        evaluatioin_df=evaluatioin_df,
        type_registry=type_registry,
        config={
            'background_id': type_system.background_id,
            'unknown_id': type_system.unknown_id,
            'compare_bg_id': -1,
            'proposed_cell_id': "cluster_id",
            'compared_cell_id': "cell_id",
            'proposed_celltype': "result_celltype",
            'compared_celltype': "true_celltype"
        })

    print_evaluation_summary(evaluation_dict, type_registry)
    # evaluation_dict