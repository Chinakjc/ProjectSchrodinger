import Geometry
import numpy as np


class Domain:
    def __init__(self, geo: dict, space_step: float):
        self.geometry = Geometry.Geometry(geo)
        self.space_step = space_step

    def get_discretized_points_by_label(self, label: str):
        range_ = self.geometry.get_range_by_label(label)
        if range_:
            return np.arange(range_[0], range_[1] + self.space_step, self.space_step)
        return []

    def get_discretized_points(self, n: int):
        range_ = self.geometry.get_range(n)
        if range_:
            return np.arange(range_[0], range_[1] + self.space_step, self.space_step)
        return []

    def get_dim(self):
        return self.geometry.get_dim()

    def get_sub_domain(self, index: tuple):
        dico = self.geometry.get_sub_geo(index).infos
        return Domain(dico, self.space_step)

    def __str__(self):
        return f"Geometry: {self.geometry}, Space Step: {self.space_step}"
