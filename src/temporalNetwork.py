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
    def __init__(self, drama):
        self.drama = drama
        self.supra_adjacency_matrix = self.build_supra_matrix()

    def build_supra_matrix(self):
        return linalg.block_diag(*list([scene_matrix.adjacency_matrix for scene_matrix in self.drama.scenes]))
