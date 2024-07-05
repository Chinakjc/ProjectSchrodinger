import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import Schema
import time
import Analysis


def test_of_solution_and_animation(psi0=(lambda x: np.exp(-x ** 2)), v_fun=(lambda x, t: 0), L=10, T=10, Nx=2 ** 10,
                                   Nt=600, fps=60, label="animation_test_1"):
    '''
        test pour le schema en résolvant des équations de Schrödinger de différents psi0 et v_fun
        la sauvegarde de l'animation en format mp4 demande ffmpeg,
        en l'absence de condition, veuillez installer le ffmpeg, ou sauvegarder en format gif

        :return:
        '''

    print("test of solution and animation begins !")

    print("En cours de calculer.")
    psi = Schema.dynamics(psi0_fun=psi0, V_fun=v_fun, L=L, Nx=Nx, T=T, Nt=Nt)
    print("Calculation terminée.")

    print("Génération de l'animation.")
    anim = Schema.plot_psi(psi=psi, duration=T,
                           frames_per_second=fps,
                           L=L)
    print("Animation générée.")
    writer = FFMpegWriter(fps=fps)

    file_name = label + ".mp4"

    print("Sauvgarde en cours.")
    anim.save(file_name, writer=writer, dpi=300)
    print("Sauvgarde réussi.")

    plt.show()
    print("end of test !")


def test_for_verification_by_comparing_the_animation_with_the_exact_solution():
    """
    test pour la vérification en comparant l'animation avec la solution exacte
    :return:
    """
    T = 5
    L = 6 * np.pi
    fps = 60
    Nx = 2 ** 10
    Nt = fps * T
    psi0 = lambda x: np.sin(x)
    psi_e = lambda t, x: np.exp(-1j / 2.0 * t) * np.sin(x)
    vf = lambda x, t: 0
    x = np.linspace(-L / 2.0, L / 2.0, Nx)[:-1]
    t = np.linspace(0, T, Nt)
    psi = Analysis.ExactErrorAnalyzer.exact_value(exact_func=psi_e, x=x, t=t)

    print("test for verification begins !")
    print("En cours de calculer.")
    psi_h = Schema.dynamics(psi0_fun=psi0, V_fun=vf, L=L, Nx=Nx, T=T, Nt=Nt)
    print("Calculation terminée.")

    print("Génération de l'animation.")
    anim = Schema.plot_psi(psi=psi, duration=T,
                           frames_per_second=fps,
                           L=L)

    anim_h = Schema.plot_psi(psi=psi_h, duration=T,
                             frames_per_second=fps,
                             L=L)

    print("Animation générée.")
    writer = FFMpegWriter(fps=fps)
    print("Sauvgarde en cours.")
    anim.save("exact_solution.mp4", writer=writer, dpi=300)
    anim_h.save("approximate_solution.mp4", writer=writer, dpi=300)
    print("Sauvgarde réussi.")

    plt.show()
    print("end of test !")


def test_of_time_complexity(n_test=100):
    '''
        test pour la performance
        :return:
        '''

    tmin = 10
    tmax = 2400
    Nmax = 2 ** 35
    xmin = 2 ** 2

    print("test for time complexity begins !")
    T = 1
    L = 1
    ts = list()
    Nxs = list()
    Nts = list()

    for i in range(n_test):
        Nt = np.random.randint(tmin, tmax)
        xmax = int(np.sqrt(Nmax / Nt) + 0.5)
        Nx = np.random.randint(xmin, xmax)
        Nts.append(Nt)
        Nxs.append(Nx)

    Nts = np.array(Nts)
    Nxs = np.array(Nxs)

    NN = Nts * Nxs * np.log2(Nxs)

    Ns = np.array([2 ** n for n in range(4, 14)])
    for i in range(n_test):
        print("experiment ", i + 1)
        Nt = Nts[i]
        Nx = Nxs[i]
        print("Nt = ", Nt)
        print("Nx = ", Nx)
        t0 = time.time()
        Schema.dynamics(L=L, T=T, Nx=Nx, Nt=Nt)
        t1 = time.time()
        ts.append(t1 - t0)
        print("t = ", t1 - t0)
    ts = np.array(ts)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("log2 of NtNx log Nx", fontsize=15)
    ax.set_ylabel("log2 of time", fontsize=15)

    plt.scatter(NN, ts, marker="X", label="Time vs NtNx log Nx")

    kk = np.exp2(np.log2(np.mean(ts) / np.mean(NN)))
    nn = np.linspace(np.min(NN), np.max(NN), 100)

    plt.plot(nn, nn * kk, linestyle='--', color='green', label="O(log2 of NtNx log Nx)")
    ax.legend()
    plt.show()
    print("end of test !")


def test_of_convergence_for_space_with_exact_solution():
    """
    test pour la convergence en espace
    :return:
    """
    T = 10
    L = 6 * np.pi
    psi0_func = lambda x: np.sin(x)
    psi_e_func = lambda t, x: np.exp(-1j / 2.0 * t) * np.sin(x)
    v_func = lambda x, t: 0

    print("test of exact convergence for space begins !")

    print("Calculation des erreurs.")
    errs = Analysis.ExactErrorAnalyzer.exact_errors_for_space(psi0_func=psi0_func, psi_exact_fun=psi_e_func,
                                                              V_func=v_func,
                                                              L=L, T=T)
    print("Calculation terminée.")

    print("Génération de la figure.")
    Analysis.ConvergenceAnalyzer.full_precision_convergence_visualizer(exact_error_list=errs)
    print("Figure générée.")

    print("end of test !")


def test_of_convergence_for_time_with_exact_solution():
    """
    test pour la convergence en temps
    :return:
    """
    T = 10
    L = 6 * np.pi
    psi0_func = lambda x: np.sin(x)
    psi_e_func = lambda t, x: np.exp(-1j / 2.0 * t) * np.sin(x)
    v_func = lambda x, t: 0

    print("test of exact convergence for time begins !")

    print("Calculation des erreurs.")
    errs = Analysis.ExactErrorAnalyzer.exact_errors_for_time(psi0_func=psi0_func, psi_exact_fun=psi_e_func,
                                                             V_func=v_func,
                                                             L=L, T=T)
    print("Calculation terminée.")

    print("Génération de la figure.")
    Analysis.ConvergenceAnalyzer.full_precision_convergence_visualizer(exact_error_list=errs, label="Nt")
    print("Figure générée.")

    print("end of test !")


def test_of_convergence_for_space_without_exact_solution(psi0_func=lambda x: np.exp(-x ** 2), V_func=lambda x, t: 0,
                                                         T=2, L=10, Nt=120):
    """
    test pour la convergence en espace
    :param psi0_func:
    :param V_func:
    :param T:
    :param L:
    :param Nt:
    :return:
    """
    print("test of approximate convergence for space begins !")

    print("Calculation des erreurs approximatives.")
    errs = Analysis.EstimatedErrorAnalyzer.estimated_errors_for_space(psi0_func=psi0_func, V_func=V_func, L=L, T=T,
                                                                      Nt=Nt)
    print("Calculation terminée.")

    print("Génération de la figure pour ordre polynomial.")
    Analysis.ConvergenceAnalyzer.polynomial_convergence_visualizer(error_list=errs)
    print("Figure générée.")
    print("Génération de la figure pour ordre exponentiel.")
    Analysis.ConvergenceAnalyzer.exponential_convergence_visualizer(error_list=errs)
    print("Figure générée.")

    print("end of test !")


def test_of_convergence_for_time_without_exact_solution(psi0_func=lambda x: np.exp(-x ** 2),
                                                        V_func=lambda x, t: np.power(x, 2), T=1, L=10, Nx=2 ** 8):
    """
    test pour la convergence en temps
    :param psi0_func:
    :param V_func:
    :param T:
    :param L:
    :param Nx:
    :return:
    """
    print("test of approximate convergence for time begins !")

    print("Calculation des erreurs approximatives.")
    errs = Analysis.EstimatedErrorAnalyzer.estimated_errors_for_time(psi0_func=psi0_func, V_func=V_func, L=L, T=T,
                                                                     Nx=Nx)
    print("Calculation terminée.")

    print("Génération de la figure pour ordre polynomial.")
    Analysis.ConvergenceAnalyzer.polynomial_convergence_visualizer(error_list=errs, label="Nt")
    print("Figure générée.")
    print("Génération de la figure pour ordre exponentiel.")
    Analysis.ConvergenceAnalyzer.exponential_convergence_visualizer(error_list=errs, label="Nt")
    print("Figure générée.")

    print("end of test !")

def generator_of_Square_wave_continue(Ul, Ur, xl,xr):
    return lambda x, t: np.where(x <= xl, Ul, 0) + np.where((x > xl) & (x < xr), Ul + (x - xl) * (Ur - Ul) / (xr - xl) , 0) + np.where(x >= xr, Ur, 0)

def generator_of_Square_wave2_continue(Ul,Um, Ur, xl, xml,xmr,xr):
    return lambda x, t: np.where(x <= xl, Ul, 0) + np.where((x > xl) & (x < xml), Ul + (x - xl) * (Um - Ul) / (xml - xl) , 0) + np.where((x >= xml) & (x <= xmr), Um, 0) + np.where((x > xmr) & (x < xr), Um + (x - xmr) * (Ur - Um) / (xr - xmr) , 0) + np.where(x >= xr, Ur, 0)
def test_of_wave_packet(velocity=10, centre=0, v_func=lambda x, t: 0, T=1, L=200, Nx=2 ** 14, Nt=300, fps=60,
                        label="wave_packet"):
    psi0 = lambda x: np.exp(-(x - centre) ** 2) * np.exp(velocity * x * 1j)
    #xs = np.linspace(-L / 2.0, L / 2.0, Nx)[:-1]
    #ys = v_func(xs, 0)
    #plt.plot(xs, ys)
    #plt.show()
    test_of_solution_and_animation(psi0=psi0, v_fun=v_func, L=L, T=T, Nx=Nx, Nt=Nt, fps=fps, label=label)