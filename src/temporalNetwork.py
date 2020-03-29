from scipy import linalg


class TemporalGraph:
    def __init__(self, drama):
        self.drama = drama
        self.supra_adjacency_matrix = self.build_supra_matrix()

    def build_supra_matrix(self):
        return linalg.block_diag(*list([scene_matrix.adjacency_matrix for scene_matrix in self.drama.scenes]))
