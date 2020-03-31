import statistics

import numpy as np
import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns


class AggregateGraph:
    def __init__(self, drama):
        self.drama = drama
        self.base_centrality = self.get_eigenvector_centrality(self.drama.export_graph())

    @staticmethod
    def get_eigenvector_centrality(graph, mean=True) -> dict:
        if mean:
            return statistics.mean(nx.eigenvector_centrality(graph).values())
        return nx.eigenvector_centrality(graph)

    def get_vitalities(self) -> dict:
        vitalities = dict()

        for character in self.drama.character_map.keys():
            _graph = self.drama.export_graph().copy()
            _graph.remove_node(character)

            vitalities[character] = self.base_centrality - self.get_eigenvector_centrality(_graph)

        return vitalities

    def plot_vitalities(self):
        pass


class TemporalGraph:
    def __init__(self, drama, e=1):
        self.drama = drama
        self.e = e

        self.supra_adjacency_matrix = self.build_supra_matrix()

    @staticmethod
    def get_eigen_centrality(graph, mean=False, normalize=True) -> dict:
        centralities = nx.eigenvector_centrality(graph)

        if normalize:
            N = len(centralities)
            centralities = {key: (value / (N - 1) * (N - 2)) for key, value in centralities.items()}

        if mean:
            return statistics.mean(centralities.values())
        return centralities

    def get_scenes_centralities(self) -> pd.DataFrame:
        scenes_df = pd.DataFrame(0, index=np.arange(1, len(self.drama.scenes) + 1),
                                 columns=self.drama.character_map.keys())

        for number, scene in enumerate(self.drama.scenes, 1):
            centrality = self.get_eigen_centrality(scene.export_graph(), mean=False, normalize=True)

            for character, centrality in centrality.items():
                scenes_df.loc[number, character] = centrality

        scenes_df[scenes_df <= 0.00001] = 0
        return scenes_df

    def plot_eigen_centralities(self, size=(10, 7)):
        eigen_centralities_df = self.get_scenes_centralities()[::-1]

        f, ax = plt.subplots(figsize=size)
        ax = sns.heatmap(eigen_centralities_df, annot=True, vmin=0, vmax=1, cmap="YlOrRd")
        ax.set(ylabel='scenes')
        ax.yaxis.label.set_size(20)

        plt.show()

    def build_supra_matrix(self):
        scenes = self.drama.scenes
        nodes = len(self.drama.character_map.keys())
        time_steps = len(scenes)

        identity_matrix = np.eye(nodes, nodes)

        supra_matrix = pd.DataFrame(np.nan, index=list(range(time_steps)), columns=list(range(time_steps)))

        for i in range(supra_matrix.shape[0]):
            for j in range(supra_matrix.shape[0]):
                if i == j:
                    supra_matrix.loc[i, j] = scenes[i]
                elif i == j + 1 or j == i + 1:
                    supra_matrix.loc[i, j] = "Identity"
                else:
                    supra_matrix.loc[i, j] = "Zero"

        return supra_matrix

    def freeman_index(self, aggregate=False):
        if aggregate:
            graph = self.drama.export_graph()
            self.get_eigen_centrality.centrality = nx.eigenvector_centrality(graph)

    def plot_freeman_index(self, aggregate=False):
        pass
