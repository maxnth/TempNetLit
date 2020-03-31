from typing import Tuple

import numpy as np
import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns


class AggregateGraph:
    """Representation of a multi layer drama graph as aggregate graph

    Args:
        drama (Drama): Drama as instance of a Drama object.

    Attributes:
        graph (nx.Graph): NetworkX graph representation of the dramas adjacency matrix. Needed for centrality
                          calculation.
    """

    def __init__(self, drama):
        self.drama = drama
        self.graph = drama.export_graph()

    @staticmethod
    def get_eigenvector_centralities(graph: nx.Graph, max_iter=10000) -> dict:
        """Calculates the eigenvector centralities of all nodes in the given graph.
        Using the implementation provided by NetworkX.

        Args:
            graph (nx.Graph): NetworkX graph representation of the dramas adjacency matrix. Needed for centrality
                  calculation.
            max_iter (int): Maximum number of iterations in power method.
        """
        return nx.eigenvector_centrality(graph, max_iter=max_iter)

    def get_freeman_index(self, graph: nx.Graph) -> int:
        """Calculates the freeman index for a given graph, using the approach proposed by proposed by Sandra D. Prado
           et al. in 'Temporal Network Analysis of Literary Texts' (2016), page 8.

        Args:
            graph (nx.Graph): NetworkX graph representation of the dramas adjacency matrix. Needed for centrality
                              calculation.

        Returns:
            int: normalized freeman index
        """
        eigenvector_centralities = self.get_eigenvector_centralities(graph)
        N = len(graph.nodes())

        c_max = max(eigenvector_centralities.values())

        counter = sum([c_max - c for c in eigenvector_centralities.values()])
        denominator = (len(self.drama.character_map.keys()) - 1) * (len(self.drama.character_map.keys()) - 2)
        normalisation = np.sqrt(N * (N - 1))

        return (counter / denominator) * normalisation

    def get_vitalities(self) -> Tuple[int, dict]:
        """Calculates the vitality for each character in the drama.
        The vitality for a character is calculated by removing the character from the drama graph and getting the
        deviation of this graphs freeman index from the base freeman index.
        This implementation follows the approach described by Sandra D. Prado et al. in
        'Temporal Network Analysis of Literary Texts' (2016), page 6f.

        Returns:
            Tuple[int, dict]: Baseline freeman index and a dictionary with the deviation from the baseline for each
                              character in the drama.
        """
        baseline = self.get_freeman_index(self.graph)
        vitalities = dict()

        # Build a seperate graph for each character by removing the character node and all connected edges from the
        # graph
        for character in self.drama.character_map.keys():
            _graph = self.graph.copy()
            _graph.remove_node(character)

            vitalities[character] = baseline - self.get_freeman_index(_graph)

        return baseline, vitalities

    def plot_vitality(self, size=(20, 10)):
        """Plots the vitality of all characters in the drama with regards to the baseline.

        Args:
            size (Tuple[int, int]): Figure size for plotting.
        """
        baseline, vitalities = self.get_vitalities()

        # Prepare data for plotting
        data = ({k: v + baseline for k, v in vitalities.items()}.items())
        x, y = zip(*data)

        # Set plot options
        fig, ax = plt.subplots(figsize=size)
        ax.plot(x, y)
        ax.axhline(y=baseline, xmin=0.0, xmax=1.0, color='r')
        plt.xticks(x, x, rotation='vertical')

        plt.show()


class TemporalGraph:
    """Representation of a drama as multi timestep-layer graph as proposed by Sandra D. Prado et al. in
       'Temporal Network Analysis of Literary Texts' (2016).
       Each timestep (in this implementation represented by a scene (or act when no scenes exist) in the drama) gets
       represented by an adjacency matrix which represents the communication between characters.
       The whole drama is represented as a subra-adjacency matrix M which contains the timestep adjacency matrices
       in its diagonal. Neighbouring identity matrices are used to keep connection between timesteps.

    Args:
        drama (Drama): Drama as instance of a Drama object as defined the datamodel.py.
    """

    def __init__(self, drama):
        self.drama = drama

    def build_supra_matrix(self, skip_character=""):
        """Builds and fills a supra-adjacency matrix as defined in Sandra D. Prado et al. in
           'Temporal Network Analysis of Literary Texts' (2016), p. 5f.

        Args:
            skip_character (str): Which character in the drama should get removed from the supra matrix calculation.
                                  Necessary for vitality calculations.
        """
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

        # Generate a row of matrices representing one timestep each
        for timestep, scene in enumerate(self.drama.scenes):
            scene_matrix = scene.adjacency_matrix
            if skip_character:
                scene_matrix = np.delete(scene_matrix, skip_index, 0)
                scene_matrix = np.delete(scene_matrix, skip_index, 1)

            # Initiliaze timestep matrix list with zero matrices
            time_step_matrices = [np.zeros((characters, characters)) for i in range(time_steps)]

            # Insert scene matrices at the according position to stay in the diagonal of the supra-adjacency matrix and
            # insert neighbouring identity matrices
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

        # Vertically stacking the timestep matrices lists to get a supra-adjacency matrix representing all timesteps of
        # the drama.
        supra_matrix = np.vstack(supra_matrices)
        return supra_matrix

    @staticmethod
    def get_eigenvector_centralities(graph: nx.Graph, normalize=False, max_iter=10000) -> dict:
        """Calculates the eigenvector centralities of all nodes in the given graph.
        Using the implementation provided by NetworkX.

        Args:
            graph (nx.Graph): NetworkX graph representation of the dramas adjacency matrix. Needed for centrality
                  calculation.
            normalize (bool): Whether to normalize the eigenvector centralities directly.
            max_iter (int): Maximum number of iterations in power method.
        """
        if normalize:
            N = len(graph.nodes())
            return {k: v / (N - 1) * (N - 2) for k, v in nx.eigenvector_centrality(graph, max_iter=max_iter).items()}
        return nx.eigenvector_centrality(graph, max_iter=max_iter)

    def get_freeman_index(self, graph: nx.Graph) -> int:
        """Calculates the freeman index for a given graph, using the approach proposed by proposed by Sandra D. Prado
           et al. in 'Temporal Network Analysis of Literary Texts' (2016), page 8.

        Args:
            graph (nx.Graph): NetworkX graph representation of the dramas adjacency matrix. Needed for centrality
                              calculation.

        Returns:
            int: normalized freeman index
        """
        eigenvector_centralities = self.get_eigenvector_centralities(graph)
        N = len(graph.nodes())

        c_max = max(eigenvector_centralities.values())

        counter = sum([c_max - c for c in eigenvector_centralities.values()])
        denominator = (N - 1) * (N - 2)
        normalisation = np.sqrt(N * (N - 1))

        return (counter / denominator) * normalisation

    @staticmethod
    def get_graph(matrix: np.ndarray) -> nx.Graph:
        """Converts and returns aggregate adjacency matrix as networkx Graph object

        Args:
            matrix (np.ndarray): adjacency matrix which should get converted

        Returns:
            nx.Graph: Graph representation of the adjacency matrix
        """
        return nx.from_numpy_matrix(matrix)

    def get_vitalities(self) -> Tuple[int, dict]:
        """Calculates the vitality for each character in the drama.
        The vitality for a character is calculated by removing the character from the drama graph and getting the
        deviation of this graphs freeman index from the base freeman index.
        This implementation follows the approach described by Sandra D. Prado et al. in
        'Temporal Network Analysis of Literary Texts' (2016), page 6f.

        Returns:
            Tuple[int, dict]: Baseline freeman index and a dictionary with the deviation from the baseline for each
                              character in the drama.
        """
        baseline = self.get_freeman_index(self.get_graph(self.build_supra_matrix()))
        vitalities = dict()

        for character in self.drama.character_map.keys():
            _graph = self.get_graph(self.build_supra_matrix(skip_character=character))

            vitalities[character] = baseline - self.get_freeman_index(_graph)

        return baseline, vitalities

    def plot_vitality(self, size=(20, 10)):
        """Plots the vitality of all characters in the drama with regards to the baseline.

        Args:
            size (Tuple[int, int]): Figure size for plotting.
        """
        baseline, vitalities = self.get_vitalities()

        # Prepare data for plotting
        data = ({k: v + baseline for k, v in vitalities.items()}.items())
        x, y = zip(*data)

        # Set plot options
        fig, ax = plt.subplots(figsize=size)
        ax.plot(x, y)
        ax.axhline(y=baseline, xmin=0.0, xmax=1.0, color='r')
        plt.xticks(x, x, rotation='vertical')

        plt.show()

    def get_scenes_centralities(self) -> pd.DataFrame:
        """Get normalized eigenvector centralities (without calculting freeman index) for each scene/timestep of the
           drama.

        Returns:
            pd.DataFrame: Dataframe containing the centralities for each character for each scene,
        """
        scenes_df = pd.DataFrame(0, index=np.arange(1, len(self.drama.scenes) + 1),
                                 columns=self.drama.character_map.keys())

        for number, scene in enumerate(self.drama.scenes, 1):
            centrality = self.get_eigenvector_centralities(scene.export_graph(), normalize=True)

            for character, centrality in centrality.items():
                scenes_df.loc[number, character] = centrality

        scenes_df[scenes_df <= 0.00001] = 0
        return scenes_df

    def plot_scenes_centrality(self, size=(10, 7)):
        """Plots the normalized eigenvector centralities for each scene/timestep iof the drama.

        Args:
            size (Tuple[int, int]): Figure size for plotting.
        """
        scenes_centrality = self.get_scenes_centralities()[::-1]

        f, ax = plt.subplots(figsize=size)
        ax = sns.heatmap(scenes_centrality, annot=True, vmin=0, vmax=1, cmap="YlOrRd")
        ax.set(ylabel='scenes')
        ax.yaxis.label.set_size(20)

        plt.show()
