class GeneRegistry:
    def __init__(self, sc_genes, st_genes):
        all_genes = sorted(set(sc_genes) | set(st_genes)) # scデータ or stデータのユニークな遺伝子リスト

        self.gid_to_gene = list(all_genes) # list
        self.gene_to_gid = {g: i for i, g in enumerate(all_genes)} # dict[str, int]

        self.stage_genes = {}
        self.stage_index = {}
        
    # 5.2 common genes の定義
    def define_common(self, genes: list[str]):
        gids = {self.gene_to_gid[g] for g in genes}
        self.stage_genes["common"] = gids
        self.stage_index["common"] = {
            gid: i for i, gid in enumerate(sorted(gids))
        }

    # 5.3 filtered genes
    def define_filtered(self, gids: set[int]):
        self.stage_genes["filtered"] = gids
        self.stage_index["filtered"] = {
            gid: i for i, gid in enumerate(sorted(gids))
        }

    # 5.4 marker genes
    def define_marker(self, marker_gids: set[int]):
        self.stage_genes["marker"] = marker_gids
        self.stage_index["marker"] = {
            gid: i for i, gid in enumerate(sorted(marker_gids))
        }