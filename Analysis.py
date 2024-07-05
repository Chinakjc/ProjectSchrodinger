import numpy as np
from matplotlib import pyplot as plt

import Schema


class ErrorList:
    """ This class is used to store the errors and the space data of the error. """

    def __init__(self, errors_data: np.array, space_data):
        self.errors_data = errors_data
        self.space_data = space_data

    def get_errors(self):
        return self.errors_data

    def get_space(self):
        return self.space_data

    def get_label(self):
        return "error"


class ExactErrorList(ErrorList):
    """ This class is used to store the exact errors and the space data of the error."""

    def __init__(self, errors_data: np.array, space_data):
        super().__init__(errors_data, space_data)

    def get_label(self):
        return "exact error"


class EstimatedErrorList(ErrorList):
    """ This class is used to store the estimated errors and the space data of the error."""

    def __init__(self, errors_data: np.array, space_data, unit):
        super().__init__(errors_data, space_data)
        self.unit = unit

    def get_label(self):
        return "unit of error"


class ConvergenceAnalyzer:
    """ This class is used to analyze and visualize the convergence of the error."""

    @classmethod
    def polynomial_convergence_visualizer(cls, error_list: ErrorList, label="Nx"):
        """ This method is used to visualize the polynomial convergence of the error."""
        # We can see the convergence curve is a straight line in the log-log plot if we have the polynomial convergence.
        errors = error_list.get_errors()
        Ns = error_list.get_space()
        r_e = errors[-1] / errors[-2]
        r_n = Ns[-1] / Ns[-2]

        k_p = np.log2(r_e) / np.log2(r_n)
        b_p = np.log2(errors[-1]) - k_p * np.log2(Ns[-1])
        y = np.exp2(b_p) * np.power(Ns, k_p)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel(f"log2 of {label}", fontsize=15)
        ax.set_ylabel(f"log2 of {error_list.get_label()}", fontsize=15)
        ax.scatter(Ns, errors, marker="X", label=error_list.get_label())
        ax.plot(Ns, y, linestyle='--', color='green', label=f" y = 2^b * n^k with k = {k_p:.3f} and b = {b_p:.3f}")
        print("ordre = ", -k_p)
        ax.legend()
        plt.show()

    @classmethod
    def full_precision_convergence_visualizer(cls, exact_error_list: ExactErrorList, label="Nx"):
        """ This method is used to visualize the full precision convergence of the error."""
        if not isinstance(exact_error_list, ExactErrorList):
            print("This method is only used for exact errors !")
            return
        errors = exact_error_list.get_errors()
        Ns = exact_error_list.get_space()
        e_c = np.mean(np.log2(errors)[len(errors) // 2:])

        if e_c <= -35:
            print("The schema has converged to the exact solution within a certain range, and any residual errors "
                  "beyond this point are solely attributable to machine precision limitations.")
        else:
            print("The schema has not yet converged to the exact solution.")

        y = np.exp2(-35) * np.ones(errors.size)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel(f"log2 of {label}", fontsize=15)
        ax.set_ylabel(f"log2 of {exact_error_list.get_label()}", fontsize=15)
        ax.scatter(Ns, errors, marker="X", label=exact_error_list.get_label())
        ax.plot(Ns, y, linestyle='--', color='green', label=f" y = 2^-35")
        ax.legend()
        plt.show()

    @classmethod
    def exponential_convergence_visualizer(cls, error_list: ErrorList, label="Nx"):
        """ This method is used to visualize the exponential convergence of the error."""
        # We can see the convergence curve is a straight line in the {log*[-log]}-log plot if we have the exponential
        # convergence.
        errors = error_list.get_errors()
        Ns = error_list.get_space()
        r_e = np.log2(errors[-1]) / np.log2(errors[-2])
        r_n = Ns[-1] / Ns[-2]

        k_e = np.log2(r_e) / np.log2(r_n)
        b_e = np.log2(-np.log2(errors[-1])) - k_e * np.log2(Ns[-1])
        c_e = -2 ** b_e

        y = np.exp2(c_e * np.power(Ns, k_e))
        fig, ax = plt.subplots(figsize=(8,6))
        #ax.set_xlim(Ns[0], Ns[-1])
        #ax.set_ylim(np.min(errors), np.max(errors))
        ax.set_xscale("log", base=2)
        ax.set_yscale("functionlog", base=2, functions=(lambda x: np.log2(-np.log2(x)), lambda x: np.exp2(-np.exp2(x))))
        ax.set_xlabel(f"log2 of {label}", fontsize=15)
        ax.set_ylabel(f"log2 of -log2 of {error_list.get_label()}", fontsize=15)
        ax.scatter(Ns, errors, marker="X", label="Unit of error")
        ax.plot(Ns, y, linestyle='--', color='green', label=f" 2^( - c * x^k ) with c = {-c_e:.3f} and k = {k_e:.3f}")
        print("error = O(2^(-c*x^k)) with c = ", -c_e, " and k = ", k_e)
        ax.legend()
        plt.show()


class ExactErrorAnalyzer:

    @classmethod
    def exact_value(cls, exact_func, t, x):
        """ This method is used to get the discretized value of the exact solution. """
        return np.array([exact_func(t, xt) for xt in x])

    @classmethod
    def distance(cls, u, v, Nt, Nx):
        """ This method is used to compute the distance between two discretized functions. """
        duv = u - v
        return np.linalg.norm([np.linalg.norm(duv[:, t], 2) for t in range(Nt)], np.inf) / np.sqrt(Nx)

    @classmethod
    def exact_errors_for_space(cls, psi0_func=lambda x: np.sin(x), V_func=lambda x, t: 0,
                               psi_exact_fun=lambda t, x: np.exp(-1j / 2.0 * t) * np.sin(x),
                               T=5, L=6 * np.pi,
                               Nt=600, N0=2 ** 3, n_iteration=8):
        """ This method is used to compute the exact errors for different space steps. """
        errors = list()
        Nxs = [N0 * 2 ** i for i in range(n_iteration)]
        t = np.linspace(0, T, Nt)
        for Nx in Nxs:
            x = np.linspace(-L / 2.0, L / 2.0, Nx)[:-1]
            psi_exact = ExactErrorAnalyzer.exact_value(exact_func=psi_exact_fun, t=t, x=x)
            print("Nx = ", Nx)
            psi_h = Schema.dynamics(psi0_fun=psi0_func, V_fun=V_func, L=L, T=T, Nx=Nx, Nt=Nt)
            error = ExactErrorAnalyzer.distance(u=psi_exact, v=psi_h, Nt=Nt, Nx=Nx)
            errors.append(error)

        res = ExactErrorList(np.array(errors), Nxs)
        return res

    @classmethod
    def exact_errors_for_time(cls, psi0_func=lambda x: np.sin(x), V_func=lambda x, t: 0,
                               psi_exact_fun=lambda t, x: np.exp(-1j / 2.0 * t) * np.sin(x),
                               T=1, L=6 * np.pi,
                               Nx=2**12, N0=30, n_iteration=8):
        """ This method is used to compute the exact errors for different time steps. """
        errors = list()
        Nts = [N0 * 2 ** i for i in range(n_iteration)]
        x = np.linspace(-L / 2.0, L / 2.0, Nx)[:-1]
        for Nt in Nts:
            t = np.linspace(0, T, Nt)
            psi_exact = ExactErrorAnalyzer.exact_value(exact_func=psi_exact_fun, t=t, x=x)
            print("Nt = ", Nt)
            psi_h = Schema.dynamics(psi0_fun=psi0_func, V_fun=V_func, L=L, T=T, Nx=Nx, Nt=Nt)
            error = ExactErrorAnalyzer.distance(u=psi_exact, v=psi_h, Nt=Nt, Nx=Nx)
            errors.append(error)

        res = ExactErrorList(np.array(errors), Nts)
        return res


class EstimatedErrorAnalyzer:

    @classmethod
    def estimated_errors_for_space(cls, psi0_func=lambda x: np.exp(-x ** 2), V_func=lambda x, t: 0,
                                   T=2, L=10,
                                   Nt=120, N0=2 ** 2, n_iteration=12,
                                   unit_error= np.exp2(-20)):
        """ This method is used to compute the estimated errors for different space steps. """
        # The Nx must be a power of 2 for simplifying the reconstruction of the estimated error.

        Nxs = [N0 * 2 ** i for i in range(n_iteration )]
        errors = list()
        # The real error is unknown, so we use the unit error as the first error.
        # i.e. the real error1 = O(unit_error)
        error1 = unit_error
        errors.append(error1)

        Nx_1 = Nxs[0]
        print("Nx = ", Nx_1)
        psi_1 = Schema.dynamics(psi0_fun=psi0_func, V_fun=V_func, L=L, T=T, Nx=Nx_1, Nt=Nt)
        x_1 = np.linspace(-L / 2.0, L / 2.0, Nx_1)[:-1]


        Nx_2 = Nxs[1]
        print("Nx = ", Nx_2)
        psi_2 = Schema.dynamics(psi0_fun=psi0_func, V_fun=V_func, L=L, T=T, Nx=Nx_2, Nt=Nt)
        x_2 = np.linspace(-L / 2.0, L / 2.0, Nx_2)[:-1]

        Nx_3 = Nxs[2]
        print("Nx = ", Nx_3)
        psi_3 = Schema.dynamics(psi0_fun=psi0_func, V_fun=V_func, L=L, T=T, Nx=Nx_3, Nt=Nt)
        x_3 = np.linspace(-L / 2.0, L / 2.0, Nx_3)[:-1]


        d1 = ExactErrorAnalyzer.distance(psi_2, np.transpose([np.interp(x_2, x_1, psi_1[:, ti]) for ti in range(Nt)]), Nt=Nt,
                                         Nx=Nx_2-1)
        d2 = ExactErrorAnalyzer.distance(psi_3, np.transpose([np.interp(x_3, x_2, psi_2[:, ti]) for ti in range(Nt)]), Nt=Nt,
                                         Nx=Nx_3-1)

        k = (np.log2(d2 / d1)) / (np.log2(Nx_2 / Nx_1))
        print("k_p = ", k)
        c1 = error1 / (Nx_1 ** k)
        error2 = (Nx_2 ** k) * c1
        c2 = error2 / (Nx_2 ** k)
        error3 = (Nx_3 ** k) * c2
        errors.append(error2)
        errors.append(error3)
        psi_1 = psi_2
        psi_2 = psi_3
        Nx_1 = Nx_2
        Nx_2 = Nx_3
        error1 = error2
        error2 = error3
        x_1 = x_2
        x_2 = x_3

        for Nx_3 in Nxs[3:]:
            print("Nx = ", Nx_3)
            x_3 = np.linspace(-L / 2.0, L / 2.0, Nx_3)[:-1]


            psi_3 = Schema.dynamics(psi0_fun=psi0_func, V_fun=V_func, L=L, T=T, Nx=Nx_3, Nt=Nt)
            d1 = ExactErrorAnalyzer.distance(psi_2,
                                             np.transpose([np.interp(x_2, x_1, psi_1[:, ti]) for ti in range(Nt)]),
                                             Nt=Nt,
                                             Nx=Nx_2-1)
            d2 = ExactErrorAnalyzer.distance(psi_3,
                                             np.transpose([np.interp(x_3, x_2, psi_2[:, ti]) for ti in range(Nt)]),
                                             Nt=Nt,
                                             Nx=Nx_3-1)

            # reconstruction of the estimated error by approximating the error as a local polynomial ordre function.
            # i.e. error = c_p * x^k_p for each interval [N_p, N_{p+1}].
            k = (np.log2(d2 / d1)) / (np.log2(Nx_2 / Nx_1))
            print("k_p = ", k)
            #c1 = error1 / (Nx_1 ** k)
            #error2 = (Nx_2 ** k) * c1
            c2 = error2 / (Nx_2 ** k)
            error3 = (Nx_3 ** k) * c2
            errors.append(error3)
            psi_1 = psi_2
            psi_2 = psi_3
            Nt_1 = Nx_2
            Nt_2 = Nx_3
            error1 = error2
            error2 = error3
            x_1 = x_2
            x_2 = x_3

        res = EstimatedErrorList(np.array(errors), Nxs, unit_error)
        return res

    @classmethod
    def estimated_errors_for_time(cls, psi0_func=lambda x: np.exp(-x ** 2), V_func=lambda x, t: 0,
                                   T=1, L=10,
                                   Nx=2**8, N0=15, n_iteration=8,
                                   unit_error= np.exp2(-10)):
        """ This method is used to compute the estimated errors for different time steps. """
        # The Nt must be a power of 2 for simplifying the reconstruction of the estimated error.
        Nts = [N0 * 2 ** i for i in range(n_iteration)]
        errors = list()
        # The real error is unknown, so we use the unit error as the first error.
        # i.e. the real error1 = O(unit_error)
        error1 = unit_error
        errors.append(error1)

        Nt_1 = Nts[0]
        print("Nt = ", Nt_1)
        psi_1 = Schema.dynamics(psi0_fun=psi0_func, V_fun=V_func, L=L, T=T, Nx=Nx, Nt=Nt_1)
        t_1 = np.linspace(-L / 2.0, L / 2.0, Nt_1)
        Nt_2 = Nts[1]
        print("Nt = ", Nt_2)
        psi_2 = Schema.dynamics(psi0_fun=psi0_func, V_fun=V_func, L=L, T=T, Nx=Nx, Nt=Nt_2)
        t_2 = np.linspace(-L / 2.0, L / 2.0, Nt_2)
        Nt_3 = Nts[2]
        print("Nt = ", Nt_3)
        psi_3 = Schema.dynamics(psi0_fun=psi0_func, V_fun=V_func, L=L, T=T, Nx=Nx, Nt=Nt_3)
        t_3 = np.linspace(-L / 2.0, L / 2.0, Nt_3)

        d1 = ExactErrorAnalyzer.distance(psi_2,[np.interp(t_2, t_1, psi_1[xi, :]) for xi in range(Nx-1)],Nt=Nt_2,Nx=Nx-1)
        d2 = ExactErrorAnalyzer.distance(psi_3,[np.interp(t_3, t_2, psi_2[xi, :]) for xi in range(Nx-1)],Nt=Nt_3,Nx=Nx-1)

        k = (np.log2(d2/d1)) / (np.log2(Nt_2/Nt_1))
        print("k_p = ", k)
        c1 = error1 / (Nt_1 ** k)
        error2 = (Nt_2 ** k) * c1
        c2 = error2 / (Nt_2 ** k)
        error3 = (Nt_3 ** k) * c2
        errors.append(error2)
        errors.append(error3)
        psi_1 = psi_2
        psi_2 = psi_3
        Nt_1 = Nt_2
        Nt_2 = Nt_3
        error1 = error2
        error2 = error3
        t_1 = t_2
        t_2 = t_3

        for Nt_3 in Nts[3:]:
            print("Nt = ", Nt_3)
            t_3 = np.linspace(-L / 2.0, L / 2.0, Nt_3)

            psi_3 = Schema.dynamics(psi0_fun=psi0_func, V_fun=V_func, L=L, T=T, Nx=Nx, Nt=Nt_3)
            print(len(t_2))
            d1 = ExactErrorAnalyzer.distance(psi_2, [np.interp(t_2, t_1, psi_1[xi, :]) for xi in range(Nx - 1)],Nt=Nt_2,Nx=Nx-1)
            d2 = ExactErrorAnalyzer.distance(psi_3, [np.interp(t_3, t_2, psi_2[xi, :]) for xi in range(Nx - 1)],Nt=Nt_3,Nx=Nx-1)

            # reconstruction of the estimated error by approximating the error as a local polynomial ordre function.
            # i.e. error = c_p * x^k_p for each interval [N_p, N_{p+1}].
            k = (np.log2(d2 / d1)) / (np.log2(Nt_2 / Nt_1))
            print("k_p = ", k)
            #c1 = error1 / (Nt_1 ** k)
            #error2 = (Nt_2 ** k) * c1
            c2 = error2 / (Nt_2 ** k)
            error3 = (Nt_3 ** k) * c2
            errors.append(error3)
            psi_1 = psi_2
            psi_2 = psi_3
            Nt_1 = Nt_2
            Nt_2 = Nt_3
            error1 = error2
            error2 = error3
            t_1 = t_2
            t_2 = t_3

        res = EstimatedErrorList(np.array(errors), Nts, unit_error)
        return res