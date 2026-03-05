import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import random
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

class ClusterVisualizer:
    """
    クラスタリング結果を可視化するためのクラス
    """
    def __init__(self, data: pd.DataFrame, x_col: str, y_col: str, cluster_id_col: str, celltype_col: str, width: int, height: int, unique_celltypes:list[str], random_seed:int=0):
        self.data = data
        self.x_col = x_col
        self.y_col = y_col
        self.cluster_id_col = cluster_id_col
        self.celltype_col = celltype_col
        self.width = width
        self.height = height
        self.unique_celltypes = unique_celltypes
        self.random_seed = random_seed

    def _generate_colors(self, items):
        random.seed(self.random_seed)
        return {item: [random.random(), random.random(), random.random()] for item in items}

    def visualize_points(self, ax=None, size=1, alpha=0.5, title='Clustering Results'):
        plot_data = self.data[self.data[self.celltype_col].isin(self.unique_celltypes)].copy()
        if plot_data.empty:
            print("No data found for the specified 'unique_celltypes'.")
            return

        unique_clusters = plot_data[self.cluster_id_col].unique()

        cluster_to_color = self._generate_colors(unique_clusters)
        point_colors = [cluster_to_color[cid] for cid in plot_data[self.cluster_id_col]]

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 12))
        
        ax.scatter(plot_data[self.x_col], plot_data[self.y_col], c=point_colors, s=size, alpha=alpha)
        ax.set_title(title)
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()



    def visualize_polygons(
        self, ax=None, title="Convex Hull Polygons", is_legend=True, type_system=None, max_percentile = 99
    ):
        plot_data = self.data[self.data[self.celltype_col].isin(self.unique_celltypes)]
        if plot_data.empty:
            return

        if type_system is None:
            type_system = self._generate_colors(plot_data[self.celltype_col].unique())

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 12))

        polygons = []
        facecolors = []

        for cluster_id, cluster_data in plot_data.groupby(self.cluster_id_col):
            if len(cluster_data) < 3:
                continue

            celltype = int(cluster_data[self.celltype_col].iloc[0])
            points = cluster_data[[self.x_col, self.y_col]].to_numpy()

            # 重心
            center = points.mean(axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            threshold = np.percentile(distances, max_percentile)
            points_filtered = points[distances <= threshold]

            if len(points_filtered) < 3:
                continue

            hull = ConvexHull(points_filtered)
            polygon_coords = points_filtered[hull.vertices]

            polygons.append(MplPolygon(polygon_coords, closed=True))
            facecolors.append(type_system.type_id_to_color(celltype))

        if polygons:
            collection = PatchCollection(
                polygons,
                facecolor=facecolors,
                edgecolor="none",
                linewidths=0,
                alpha=0.9,
            )
            ax.add_collection(collection)

        ax.set_title(title)
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")

        if is_legend:
            handles = [
                mpatches.Patch(
                    color=color,
                    label=type_system.type_id_to_celltypes(celltype) or "Unknown",
                )
                for celltype, color in type_system.colors.items()
                if celltype in plot_data[self.celltype_col].unique()
            ]
            ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left")

        plt.tight_layout()



    def visualize_polygons_old(self, ax=None, title='Convex Hull Polygons', is_legend=True, type_system=None):
        plot_data = self.data[self.data[self.celltype_col].isin(self.unique_celltypes)].copy()
        if plot_data.empty:
            print("No data found for the specified 'unique_celltypes'.")
            return

        unique_clusters = plot_data[self.cluster_id_col].unique()
        unique_celltypes_in_data = plot_data[self.celltype_col].unique()

        if type_system is None:
            type_system = self._generate_colors(unique_celltypes_in_data)

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 12))
        
        for cluster_id in unique_clusters:
            cluster_data = plot_data[plot_data[self.cluster_id_col] == cluster_id]
            if len(cluster_data) < 3:
                continue
            
            celltype = cluster_data[self.celltype_col].iloc[0]
            cluster_points = cluster_data[[self.x_col, self.y_col]].values

            try:
                hull = ConvexHull(cluster_points)
                polygon_coords = cluster_points[hull.vertices]
                color = type_system.type_id_to_color(int(celltype)) # gistry[type_registry["type_id"] == int(celltype)]["color"].iloc[0]  # .get(celltype, '#CCCCCC')
                polygon = MplPolygon(polygon_coords, closed=True, facecolor=color, edgecolor=None, linewidth=0.5, alpha=0.9)
                ax.add_patch(polygon)
            except Exception as e:
                continue

        ax.set_title(title)
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')

        if is_legend:
            legend_handles = [mpatches.Patch(color=color, label=type_system.type_id_to_celltypes(celltype) if not type_system.type_id_to_celltypes(celltype) is None else 'Unknown') for celltype, color in type_system.colors.items() if celltype in unique_celltypes_in_data]
            ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10.5)

        plt.tight_layout()
    

