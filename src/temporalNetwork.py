import statistics

import numpy as np
import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns


class AggregateGraph:
    """Representation of a multi layer drama graph as aggregate graph

    Arguments:
        drama (Drama): Drama as instance of a Drama object.
    """
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
    """Representation of a multi layer drama graph as multi layer temporal graph as proposed by Sandra D. Prado et al.
    in 'Temporal Network Analysis of Literary Texts' (2016).

    Arguments:
        drama (Drama): Drama as instance of a Drama object.

    """

    def __init__(self, drama):
        self.drama = drama

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

    def build_supra_matrix(self, skip_character=""):
        supra_matrices = list()

        scenes = self.drama.scenes
        characters = len(self.drama.character_map.values())
        time_steps = len(scenes)

        if skip_character:
            try:
                skip_index = self.drama.character_map[skip_character]
            except KeyError as e:
                raise e

            characters -= 1

        for timestep, scene in enumerate(self.drama.scenes):
            scene_matrix = scene.adjacency_matrix
            if skip_character:
                scene_matrix = np.delete(scene_matrix, skip_index, 0)
                scene_matrix = np.delete(scene_matrix, skip_index, 1)

            time_step_matrices = [np.zeros((characters, characters)) for i in range(time_steps)]

            if timestep == 0:
                time_step_matrices[timestep] = scene_matrix
                if timestep < len(scenes):
                    time_step_matrices[timestep + 1] = np.eye(characters)
            if timestep == len(scenes) - 1:
                time_step_matrices[timestep] = scene_matrix
                if timestep > 0:
                    time_step_matrices[timestep - 1] = np.eye(characters)
            else:
                time_step_matrices[timestep] = scene_matrix
                time_step_matrices[timestep + 1] = np.eye(characters)
                time_step_matrices[timestep - 1] = np.eye(characters)
            supra_matrices.append(np.hstack(time_step_matrices))

        supra_matrix = np.vstack(supra_matrices)
        return supra_matrix

    @staticmethod
    def get_eigenvector_centralities(graph: nx.Graph, max_iter=10000) -> dict:
        return nx.eigenvector_centrality(graph, max_iter=max_iter)

    def get_freeman_index(self, graph: nx.Graph) -> int:
        eigenvector_centralities = self.get_eigenvector_centralities(graph)
        N = len(graph.nodes())

        c_max = max(eigenvector_centralities.values())

        counter = sum([c_max - c for c in eigenvector_centralities.values()])
        denominator = (N - 1) * (N - 2)
        normalisation = np.sqrt(N * (N - 1))

        return (counter / denominator) * normalisation

    @staticmethod
    def get_graph(matrix: np.ndarray) -> nx.Graph:
        return nx.from_numpy_matrix(matrix)

    def get_vitalities(self) -> Tuple[int, dict]:
        baseline = self.get_freeman_index(self.get_graph(self.build_supra_matrix()))
        vitalities = dict()

        for character in self.drama.character_map.keys():
            _graph = self.get_graph(self.build_supra_matrix(skip_character=character))

            vitalities[character] = baseline - self.get_freeman_index(_graph)

        return baseline, vitalities
