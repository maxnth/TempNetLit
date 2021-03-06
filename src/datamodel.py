import itertools
from typing import Dict, List, Union
from pathlib import Path

from lxml import etree
import numpy as np

import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

ns_dict = {"tei": "http://www.tei-c.org/ns/1.0"}


class Scene:
    """Representation of a scene in a drama for temporal network analysis.

    Args:
        drama_scene (etree.Element): lxml etree element containing all TEI encoded XML of the corresponding scene.
        character_map (Dict[str, int]): Dictionary with names of characters which appear as speakrs in the drama as key
                                        and a unique integer as value. The latter is used for mapping the indices of
                                        the calculated matrices to the characters.
        base_adjacency_matrix (np.ndarray): Base adjacency matrix of all characters for the drama (filled with zeroes).
    """

    def __init__(self, drama_scene: etree.Element, character_map: Dict[str, int],
                 base_adjacency_matrix: np.ndarray, binary_weights):
        self.tree = drama_scene
        self.character_map = character_map
        self.adjacency_matrix = base_adjacency_matrix.copy()
        self.binary_weights = binary_weights

        self.fill_adjacency_matrix()

    def fill_adjacency_matrix(self):
        """Fills adjacency matrix based on the communication between speaking characters in a scene.

        Note:
            Communication between characters is assumed when the speaking acts of two characters are in direct
            neighborhood alas

        """
        speech_acts = [speech_act.split("#")[1:] for speech_act in
                       self.tree.xpath("./tei:sp/@who", namespaces=ns_dict)]

        if len(speech_acts) > 1:
            edges = [(current, prev) for current, prev in zip(speech_acts, speech_acts[1:])]

            # Checks whether speech acts with multiple speaker exist and processes such edges
            cleaned_edges = list()
            for edge in edges:
                if len(edge[0]) > 1 and len(edge[1]) > 1:
                    multiple_speakers_one = [speaker.strip() for speaker in edge[0]]
                    multiple_speakers_two = [speaker.strip() for speaker in edge[1]]
                    for edge in list(itertools.combinations(multiple_speakers_one, 2)):
                        cleaned_edges.append(edge)
                    for edge in list(itertools.combinations(multiple_speakers_two, 2)):
                        cleaned_edges.append(edge)
                    for speaker_one in multiple_speakers_one:
                        for speaker_two in multiple_speakers_two:
                            cleaned_edges.append((speaker_one, speaker_two))
                elif len(edge[0]) > 1:
                    multiple_speakers = [speaker.strip() for speaker in edge[0]]
                    for edge in list(itertools.combinations(multiple_speakers, 2)):
                        cleaned_edges.append(edge)
                    for speaker in multiple_speakers:
                        cleaned_edges.append(("".join(edge[1]).strip(), speaker))
                elif len(edge[1]) > 2:
                    multiple_speakers = [speaker.strip() for speaker in edge[1]]
                    for edge in list(itertools.combinations(multiple_speakers, 2)):
                        cleaned_edges.append(edge)
                    for speaker in multiple_speakers:
                        cleaned_edges.append(("".join(edge[0]).strip(), speaker))
                else:
                    cleaned_edges.append(("".join(edge[0]).strip(), "".join(edge[1]).strip()))

            edges = cleaned_edges

            for edge in edges:
                first_node = self.character_map[edge[0]]
                second_node = self.character_map[edge[1]]
                self.adjacency_matrix[first_node][second_node] += 1
                self.adjacency_matrix[second_node][first_node] += 1
            np.fill_diagonal(self.adjacency_matrix, 0)

            # Binarization of character communications. All communication counts are set to zero when > 0
            if self.binary_weights:
                self.adjacency_matrix = np.where(self.adjacency_matrix > 0.0, 1, 0)

    def export_graph(self) -> nx.graph:
        graph = nx.from_numpy_matrix(self.adjacency_matrix)
        graph = nx.relabel_nodes(graph, {v: k for (k, v) in self.character_map.items()}, copy=False)
        return graph

    def plot_graph(self, size=(10, 10), node_scaling=500, width_scaling=1, heatmap=True):
        graph = self.export_graph()
        degree = nx.degree(graph)

        node_sizes = [d * node_scaling for d in dict(degree).values()]
        widths = [graph[u][v]['weight'] * width_scaling for u, v in graph.edges()]

        fig = plt.figure(figsize=size)

        if heatmap:
            nx.draw(graph, with_labels=True, node_size=node_sizes,
                    width=widths, cmap=plt.get_cmap('viridis'), node_color=node_sizes,
                    font_color="red", font_weight="bold")
        else:
            nx.draw(graph, with_labels=True, node_size=node_sizes,
                    width=widths, font_weight="bold")

        fig.set_facecolor("#686868")

    def plot(self, size=(10, 10)):
        plt.clf()
        fig, ax = plt.subplots(figsize=size)
        heat_map = sns.heatmap(self.adjacency_matrix, xticklabels=self.character_map.keys(),
                               yticklabels=self.character_map.keys(), annot=True, ax=ax)

        plt.title("Amount of interactions between to characters in this scene.")
        plt.ylabel("Character name")

        plt.show()


class Drama:
    """Representation of a drama for temporal network analysis.

    Args:
        drama (Union[str, Path]): Path to a file containing a TEI encoded drama

    Attribute:
        tree (etree.Element): lxml etree element containing the root tree representation of the TEI encoded XML
        title (str): title of the drama based on the TEI metadata annotation
        character_map (dict): Dictionary with names of characters which appear as speakrs in the drama as key and a
                            unique integer as value. The latter is used for mapping the indices of the calculated
                            matrices to the characters.
        base_adjacency_matrix (np.ndarray): Base adjacency matrix of all characters for the drama (filled with zeroes).
        scenes (List[Scene]): List of all scenes in the drama, represented by instances of the Scene class.
                              If a drama doesn't contain any scenes, acts are used as substitute.
        aggregate_adjacency_matrix (np.ndarray): An adjacancy matrix representing the aggregated frequency of
                                                 communication between characters.
    """

    def __init__(self, drama: Union[str, Path], binary_weights=True):
        self.tree = self.get_root_tree(drama)
        self.binary_weights = binary_weights

        self.title = self.get_title()
        self.character_map = self.get_character_mapping()
        self.base_adjacency_matrix = self.build_base_adjacency_matrix()
        self.scenes = self.get_scenes()
        self.aggregate_adjacency_matrix = self.build_aggregate_adjacency_matrix()

    @staticmethod
    def get_root_tree(path: Union[str, Path]) -> etree.Element:
        return etree.parse(path).getroot() if isinstance(path, str) else etree.parse(str(path)).getroot()

    def get_title(self) -> str:
        main_title = " ".join(self.tree.xpath('//tei:title[@type="main"]/text()', namespaces=ns_dict))
        sub_title = " ".join(self.tree.xpath('//tei:title[@type="sub"]/text()', namespaces=ns_dict))
        return f"{main_title}. {sub_title}"

    def get_character_mapping(self) -> Dict[str, int]:
        characters = set(list(itertools.chain(*[sp.replace("#", "").split(" ") for sp in
                                                self.tree.xpath("//tei:sp/@who", namespaces=ns_dict)])))
        return {character: num for (num, character) in enumerate(characters)}

    def build_base_adjacency_matrix(self) -> np.ndarray:
        return np.zeros((len(self.character_map), len(self.character_map)))

    def get_scenes(self) -> List[Scene]:
        scenes = self.tree.xpath('.//tei:div[@type="scene"]', namespaces=ns_dict)

        if len(scenes) == 0:
            scenes = self.tree.xpath('.//tei:div[@type="act"]', namespaces=ns_dict)

        assert len(scenes) > 0, "Drama contains neither acts nor scenes. Can't get processed."

        return [Scene(elem, self.character_map, self.base_adjacency_matrix, self.binary_weights) for elem in scenes]

    def build_aggregate_adjacency_matrix(self) -> np.ndarray:
        scene_matrices = [scene.adjacency_matrix for scene in self.scenes]
        return sum(scene_matrices)

    def export_graph(self) -> nx.graph:
        graph = nx.from_numpy_matrix(self.aggregate_adjacency_matrix)
        graph = nx.relabel_nodes(graph, {v: k for (k, v) in self.character_map.items()}, copy=False)
        return graph

    def plot_graph(self, size=(10, 10), node_scaling=500, width_scaling=1, heatmap=True):
        graph = self.export_graph()
        degree = nx.degree(graph)

        node_sizes = [d * node_scaling for d in dict(degree).values()]
        widths = [graph[u][v]['weight'] * width_scaling for u, v in graph.edges()]

        fig = plt.figure(figsize=size)

        if heatmap:
            nx.draw(graph, with_labels=True, node_size=node_sizes,
                    width=widths, cmap=plt.get_cmap('viridis'), node_color=node_sizes,
                    font_color="black", font_weight="bold")
        else:
            nx.draw(graph, with_labels=True, node_size=node_sizes,
                    width=widths, font_weight="bold")

        fig.set_facecolor("#686868")

    def plot(self, size=(10, 10)):
        plt.clf()
        fig, ax = plt.subplots(figsize=size)
        heat_map = sns.heatmap(self.aggregate_adjacency_matrix, xticklabels=self.character_map.keys(),
                               yticklabels=self.character_map.keys(), annot=True, ax=ax)

        plt.title("Amount of interactions between to characters.")
        plt.suptitle(f"'{self.title}'")
        plt.ylabel("Character name")

        plt.show()
