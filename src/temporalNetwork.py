from typing import Tuple

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
        self.graph = drama.export_graph()

    @staticmethod
    def get_eigenvector_centralities(graph: nx.Graph, max_iter=10000) -> dict:
        return nx.eigenvector_centrality(graph, max_iter=max_iter)

    def get_freeman_index(self, graph: nx.Graph) -> int:
        eigenvector_centralities = self.get_eigenvector_centralities(graph)
        N = len(graph.nodes())

        c_max = max(eigenvector_centralities.values())

        counter = sum([c_max - c for c in eigenvector_centralities.values()])
        denominator = (len(self.drama.character_map.keys()) - 1) * (len(self.drama.character_map.keys()) - 2)
        normalisation = np.sqrt(N * (N - 1))

        return (counter / denominator) * normalisation

    def get_vitalities(self) -> Tuple[int, dict]:
        baseline = self.get_freeman_index(self.graph)
        vitalities = dict()

        for character in self.drama.character_map.keys():
            _graph = self.graph.copy()
            _graph.remove_node(character)

            vitalities[character] = baseline - self.get_freeman_index(_graph)

        return baseline, vitalities

    def plot_vitalities(self):
        """TODO"""
        baseline, vitalities = self.get_vitalities()
        _df = pd.DataFrame(vitalities, index=[0])
        return _df


class TemporalGraph:
    """Representation of a multi layer drama graph as multi layer temporal graph as proposed by Sandra D. Prado et al.
    in 'Temporal Network Analysis of Literary Texts' (2016).

    Arguments:
        drama (Drama): Drama as instance of a Drama object.

    """

    def __init__(self, drama):
        self.drama = drama

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
    def get_eigenvector_centralities(graph: nx.Graph, normalize=False, max_iter=10000) -> dict:
        if normalize:
            N = len(graph.nodes())
            return {k: v/(N-1)*(N-2) for k, v in nx.eigenvector_centrality(graph, max_iter=max_iter).items()}
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

    def get_scenes_centralities(self) -> pd.DataFrame:
        scenes_df = pd.DataFrame(0, index=np.arange(1, len(self.drama.scenes) + 1),
                                 columns=self.drama.character_map.keys())

        for number, scene in enumerate(self.drama.scenes, 1):
            centrality = self.get_eigenvector_centralities(scene.export_graph(), normalize=True)

            for character, centrality in centrality.items():
                scenes_df.loc[number, character] = centrality

        scenes_df[scenes_df <= 0.00001] = 0
        return scenes_df

    def plot_scenes_centrality(self, size=(10, 7)):
        scenes_centrality = self.get_scenes_centralities()[::-1]

        f, ax = plt.subplots(figsize=size)
        ax = sns.heatmap(scenes_centrality, annot=True, vmin=0, vmax=1, cmap="YlOrRd")
        ax.set(ylabel='scenes')
        ax.yaxis.label.set_size(20)

        plt.show()
