from typing import Dict, Union
from pathlib import Path

from lxml import etree
import numpy as np

ns_dict = {"tei": "http://www.tei-c.org/ns/1.0"}


class Drama:
    def __init__(self, drama: Union[str, Path]):
        self.tree = self.get_root_tree(drama)

        self.character_map = self.get_character_mapping()
        self.base_adjacency_matrix = self.build_base_adjacency_matrix()
        self.scenes = self.get_scenes()
        self.aggregate_adjacency_matrix = self.build_aggregate_adjacency_matrix()

    def get_root_tree(self, path: Union[str, Path]):
        return etree.parse(path).getroot() if isinstance(path, str) else etree.parse(str(path)).getroot()

    def get_character_mapping(self):
        characters = {sp.replace("#", "") for sp in self.tree.xpath("//tei:sp/@who", namespaces=ns_dict)}
        return {character: num for (num, character) in enumerate(characters)}

    def build_base_adjacency_matrix(self):
        return np.zeros((len(self.character_map), len(self.character_map)))

    def get_scenes(self):
        scenes = self.tree.xpath('.//tei:div[@type="scene"]', namespaces=ns_dict)

        if len(scenes) == 0:
            scenes = self.tree.xpath('.//tei:div[@type="act"]', namespaces=ns_dict)

        assert len(scenes) > 0, "Drama contains neither acts nor scenes."

        return [Scene(elem, self.character_map, self.base_adjacency_matrix) for elem in scenes]

    def build_aggregate_adjacency_matrix(self):
        scene_matrices = [scene.adjacency_matrix for scene in self.scenes]
        return sum(scene_matrices)


class Scene:
    def __init__(self, drama_scene: etree.Element, character_map: Dict[str, int],
                 base_adjacency_matrix: np.ndarray):
        self.tree = drama_scene
        self.character_map = character_map
        self.adjacency_matrix = base_adjacency_matrix.copy()

        self.fill_adjacency_matrix()

    def fill_adjacency_matrix(self):
        speech_acts = [speech_act.replace("#", "") for speech_act in
                       self.tree.xpath("./tei:sp/@who", namespaces=ns_dict)]

        if len(speech_acts) > 1:
            edges = [(current, prev) for current, prev in zip(speech_acts, speech_acts[1:])]

            for edge in edges:
                first_node = self.character_map[edge[0]]
                second_node = self.character_map[edge[1]]
                self.adjacency_matrix[first_node][second_node] += 1
                self.adjacency_matrix[second_node][first_node] += 1
