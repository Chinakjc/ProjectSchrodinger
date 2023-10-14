class Geometry:
    def __init__(self, infos: dict):
        self.infos = infos

    def get_range_by_label(self, label: str):
        return self.infos.get(label, None)

    def get_all_labels(self):
        return list(self.infos.keys())

    def get_range(self, n: int):
        label = self.get_all_labels()[n]
        return self.infos.get(label, None)

    def get_dim(self):
        return len(self.infos.keys())

    def get_sub_geo(self, index: tuple):
        keys = self.get_all_labels()
        dico = {keys[n]: self.infos[keys[n]] for n in index}
        return Geometry(dico)

    def __str__(self):
        return str(self.infos)