import numpy as np
import Domain


class Quadrature:
    @classmethod
    def rectangle_method_1D(cls, seg: np.arange, step: float, func):
        vals = func(seg)
        integral = np.sum(vals) * step
        return integral

    @classmethod
    def rectangle_method_1D_N(cls, step: float, vals: np.array):
        integral = np.sum(vals) * step
        return integral

    @classmethod
    def rectangle_method(cls, geo: dict, space_step: float, func):
        domain = Domain.Domain(geo, space_step)
        dim = domain.get_dim()
        seg = domain.get_discretized_points(0)
        if dim == 1:
            return Quadrature.rectangle_method_1D(seg, space_step, func)

        integral = 0.0
        sub_index = tuple(range(1, dim))
        sub_domain = domain.get_sub_domain(sub_index)
        sub_dico = sub_domain.geometry.infos
        sub_mono = (sub_domain.get_dim() == 1)

        for x in seg:
            def fx(s: list):
                if sub_mono:
                    return func([x] + [s])
                return func([x] + s)

            sq = Quadrature.rectangle_method(sub_dico, space_step, fx)
            integral += sq

        return integral * space_step

    @classmethod
    def rectangle_method_N(cls, geo: dict, space_step: float, vals: np.array):
        domain = Domain.Domain(geo, space_step)
        dim = domain.get_dim()

        if dim == 1:
            return Quadrature.rectangle_method_1D_N(space_step, vals)

        integral = 0.0
        sub_index = tuple(range(1, dim))
        sub_domain = domain.get_sub_domain(sub_index)
        sub_dico = sub_domain.geometry.infos

        for sub_vals in vals:
            integral += Quadrature.rectangle_method_N(sub_dico, space_step, sub_vals)

        return integral * space_step
