import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import Schema
import time
import Analysis






def test2():
    '''
    test pour le schema en résolvant des équations de Schrödinger de différents psi0 et v_fun
    la sauvegarde de l'animation en format mp4 demande ffmpeg,
    en l'absence de condition, veuillez installer le ffmpeg, ou sauvegarder en format gif

    :return:
    '''

    print("test2 begins !")
    T = 5
    L = 2
    fps = 60
    psi0 = lambda x: np.cos(2 * np.pi * x)  # psi0 alternatif
    v_fun = lambda x, t: np.power(x, 2)  # np.sin(x + t)  # v_fun alternatif
    Nx = 2 ** 13
    Nt = 60 * T * 2
    print("En cours de calculer.")
    anim = Schema.plot_psi(psi=Schema.dynamics(V_fun=v_fun, L=L, T=T, Nx=Nx, Nt=Nt), duration=T,
                           frames_per_second=fps,
                           L=L)
    print("Calculation terminée.")
    writer = FFMpegWriter(fps=fps)
    anim.save('animation12.mp4', writer=writer, dpi=450)
    print("Sauvgarde réussi.")

    plt.show()
    print("end of test !")


def test3():
    '''
    test pour la performance
    :return:
    '''

    print("test3 begins !")
    T = 1
    L = 1
    ts = list()
    Ns = np.array([2 ** n for n in range(4, 14)])
    for N in Ns:
        print("N = ", N)
        t0 = time.time()
        Schema.dynamics(L=L, T=T, Nx=N, Nt=N)
        t1 = time.time()
        ts.append(t1 - t0)
        print("t = ", t1 - t0)
    ts = np.array(ts)

    '''fig = plt.figure(figsize=(8, 6))
    plt.semilogx(base=2)
    plt.semilogy(base=2)'''
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("log2 of NtNx log Nx", fontsize=15)
    ax.set_ylabel("log2 of time", fontsize=15)
    '''plt.xlabel("log2 of NtNx log Nx", fontsize=15)
    plt.ylabel("log2 of time", fontsize=15)'''
    NN = Ns ** 2 * np.log2(Ns)
    plt.scatter(NN, ts, marker="X",label="Time vs NtNx log Nx")
    #kk = (ts[-1] - ts[-2]) / (NN[-1] - NN[-2])
    bb = np.log2(ts[-1]/NN[-1])
    kk = 2**bb
    plt.plot(NN, NN * kk, linestyle='--', color='green',label="O(log2 of NtNx log Nx)")
    ax.legend()
    plt.show()
    print("end of test !")


def test4():
    T = 10
    L = 6 * np.pi
    Nxs = [2 ** i for i in range(1, 8)]
    Nt = 600
    errors = list()
    for Nx in Nxs:
        psi0 = lambda x: np.sin(x)
        psi_e = lambda t, x: np.exp(-1j / 2.0 * t) * np.sin(x)
        x = np.linspace(-L / 2.0, L / 2.0, Nx)[:-1]
        t = np.linspace(0, T, Nt)
        psi = np.array([psi_e(t, xt) for xt in x])
        print("Nx = ", Nx)
        psi_h = Schema.dynamics(psi0_fun=psi0, L=L, T=T, Nx=Nx, Nt=Nt)
        e = psi - psi_h
        # error = np.linalg.norm([np.linalg.norm(np.fft.fft(e[:, t]), 2) for t in range(Nt)], np.inf) / np.sqrt(Nx)
        error = np.linalg.norm([np.linalg.norm(e[:, t], 2) for t in range(Nt)], np.inf) / np.sqrt(Nx)
        errors.append(error)

    d_e = errors[-1] / errors[-2]
    d_n = Nxs[-1] / Nxs[-2]

    k = np.log2(d_e) / np.log2(d_n)
    b = np.log2(errors[-1]) - k * np.log2(Nxs[-1])
    y = 2 ** b * np.power(Nxs, k)
    fig = plt.figure(figsize=(8, 6))
    plt.semilogx(base=2)
    plt.semilogy(base=2)
    plt.xlabel("log2 of Nx", fontsize=15)
    plt.ylabel("log2 of error", fontsize=15)
    plt.scatter(Nxs, errors, marker="X")
    plt.plot(Nxs, y, linestyle='--', color='green')
    print("ordre = ", -k)
    plt.show()

    print("end of test !")


def test4b(psi0_fun=(lambda x: np.exp(-x ** 2)), V_fun=(lambda x, t: 0), L=10, T=2, ):
    Nxs = [2 ** i for i in range(12, 20)]
    Nt = 30 * T
    error = 1 / 1024
    errors = list()
    errors.append(error)
    Nx_1 = Nxs[0]
    print("Nx = ", Nx_1)
    psi_1 = Schema.dynamics(psi0_fun=psi0_fun, V_fun=V_fun, L=L, T=T, Nx=Nx_1, Nt=Nt)
    x_1 = np.linspace(-L / 2.0, L / 2.0, Nx_1)[:-1]
    for Nx_2 in Nxs[1:]:
        print("Nx = ", Nx_2)
        x_2 = np.linspace(-L / 2.0, L / 2.0, Nx_2)[:-1]
        psi_2 = Schema.dynamics(psi0_fun=psi0_fun, V_fun=V_fun, L=L, T=T, Nx=Nx_2, Nt=Nt)
        d_psi = psi_2 - np.transpose([np.interp(x_2, x_1, psi_1[:, ti]) for ti in range(Nt)])
        k = np.log2(np.linalg.norm([np.linalg.norm(d_psi[:, t], 2) for t in range(Nt)], np.inf) / np.sqrt(Nx_2))
        print("k_p = ", k)
        c = error / (Nx_1 ** k)
        error = (Nx_2 ** k) * c
        errors.append(error)
        psi_1 = psi_2
        Nx_1 = Nx_2
        x_1 = x_2

    print(errors)
    print(np.log2(-np.log2(errors)))
    d_e = np.log2(errors[-1]) / np.log2(errors[-2])
    d_n = Nxs[-1] / Nxs[-2]

    k_e = np.log2(d_e) / np.log2(d_n)
    b = np.log2(-np.log2(errors[-1])) - k_e * np.log2(Nxs[-1])
    c = -2 ** b
    print("c = ", c)
    print("k_e = ", k_e)
    y = 2 ** (c * np.power(Nxs, k_e))
    '''fig = plt.figure(figsize=(8, 6))
    plt.semilogx(base=2)
    plt.semilogy(base=2)'''

    fig, ax = plt.subplots(figsize=(8, 6))
    print("min = ", min(np.min(y), np.min(errors)))
    print("max = ", max(np.max(y), np.max(errors)))
    # ax.set_ylim(min(np.min(y)/512, np.min(errors)), min(0.25,2*max(np.max(y), np.max(errors))))
    ax.set_xscale("log", base=2)
    ax.set_yscale("functionlog", base=2, functions=(lambda x: np.log2(-np.log2(x)), lambda x: np.exp2(-np.exp2(x))))
    ax.set_xlabel("log2 of Nx", fontsize=15)
    ax.set_ylabel("log2 of -log2 of unit of error", fontsize=15)
    ax.scatter(Nxs, errors, marker="X", label="Unit of error")
    ax.plot(Nxs, y, linestyle='--', color='green', label=f" 2^( - c * x^k ) with c = {-c:.3f} and k = {k_e:.3f}")
    print("error = O(2^(-c*x^k)) with c = ", -c, " and k = ", k_e)
    ax.legend()
    plt.show()

    print("end of test !")


def test5():
    T = 10
    L = 6 * np.pi
    Nts = [60 * 2 ** i for i in range(5)]
    Nx = 4097
    errors = list()
    for Nt in Nts:
        psi0 = lambda x: np.sin(x)
        psi_e = lambda t, x: np.exp(-1j / 2.0 * t) * np.sin(x)
        x = np.linspace(-L / 2.0, L / 2.0, Nx)[:-1]
        t = np.linspace(0, T, Nt)
        psi = np.array([psi_e(t, xt) for xt in x])
        print("Nt = ", Nt)
        psi_h = Schema.dynamics(psi0_fun=psi0, L=L, T=T, Nx=Nx, Nt=Nt)
        e = psi - psi_h
        error = np.linalg.norm([np.linalg.norm(e[:, t], 2) for t in range(Nt)], np.inf) / np.sqrt(Nx)
        errors.append(error)

    d_e = errors[-1] / errors[-2]
    d_n = Nts[-1] / Nts[-2]
    print(errors)
    k = np.log2(d_e) / np.log2(d_n)
    b = np.log2(errors[-1]) - k * np.log2(Nts[-1])
    # y = 2**b * np.power(Nts,k)
    fig = plt.figure(figsize=(8, 6))
    plt.semilogx(base=2)
    plt.semilogy(base=2)
    plt.xlabel("log2 of Nt", fontsize=15)
    plt.ylabel("log2 of error", fontsize=15)
    plt.scatter(Nts, errors, marker="X")
    # plt.plot(Nts,y,linestyle='--', color='green')
    print("ordre = ", -k)
    plt.show()

    print("end of test !")


def test5b():
    T = 10
    L = 6 * np.pi
    Nts = [60 * 2 ** i for i in range(5)]
    Nx = 4097
    errors = list()
    for Nt in Nts:
        psi0 = lambda x: np.sin(x)
        psi_e = lambda t, x: np.exp(-1j / 2.0 * t) * np.sin(x)
        x = np.linspace(-L / 2.0, L / 2.0, Nx)[:-1]
        t = np.linspace(0, T, Nt)
        psi = np.array([psi_e(t, xt) for xt in x])
        print("Nt = ", Nt)
        psi_h = Schema.dynamics(psi0_fun=psi0, L=L, T=T, Nx=Nx, Nt=Nt)
        e = psi - psi_h
        error = np.linalg.norm([np.linalg.norm(e[:, t], 2) for t in range(Nt)], np.inf) / np.sqrt(Nx)
        errors.append(error)

    d_e = errors[-1] / errors[-2]
    d_n = Nts[-1] / Nts[-2]
    print(errors)
    k = np.log2(d_e) / np.log2(d_n)
    b = np.log2(errors[-1]) - k * np.log2(Nts[-1])
    # y = 2**b * np.power(Nts,k)
    fig = plt.figure(figsize=(8, 6))
    plt.semilogx(base=2)
    plt.semilogy(base=2)
    plt.xlabel("log2 of Nt", fontsize=15)
    plt.ylabel("log2 of error", fontsize=15)
    plt.scatter(Nts, errors, marker="X")
    # plt.plot(Nts,y,linestyle='--', color='green')
    print("ordre = ", -k)
    plt.show()

    print("end of test !")



# test2()
#test3()
#test4b(V_fun=lambda x,t: np.power(x, 2), L=20)

'''err = Analysis.EstimatedErrorAnalyzer.estimated_errors_for_space()
#Analysis.ConvergenceAnalyzer.polynomial_convergence_visualizer(error_list=err)
Analysis.ConvergenceAnalyzer.exponential_convergence_visualizer(error_list=err)'''

'''err2 = Analysis.ExactErrorAnalyzer.exact_errors_for_space()
Analysis.ConvergenceAnalyzer.polynomial_convergence_visualizer(error_list=err2)
Analysis.ConvergenceAnalyzer.full_precision_convergence_visualizer(exact_error_list=err2)'''

err3 = Analysis.EstimatedErrorAnalyzer.estimated_errors_for_time(V_func=lambda x,t :np.power(x,2))
Analysis.ConvergenceAnalyzer.polynomial_convergence_visualizer(error_list=err3,label="Nt")
Analysis.ConvergenceAnalyzer.exponential_convergence_visualizer(error_list=err3,label="Nt")


'''Nxs = np.array([2 ** i for i in range(10,20)])
kk = -2.0
bk = -10
ck = 12

error1 = 1/512
errors = [error1]
def f(x):
    return np.exp2(bk)*np.power(x,kk) + ck

def equations(p, x1, y1, x2, y2, x3, y3):
    b, k, c = p
    return (2**b * x1**k + c - y1, 2**b * x2**k + c - y2, 2**b * x3**k + c - y3)

def solve_bkc(x1, y1, x2, y2, x3, y3):
    # 初始猜测值
    b0, k0, c0 = 1.0, -1.0, 1.0
    b, k, c = fsolve(equations, (b0, k0, c0), args=(x1, y1, x2, y2, x3, y3))
    return b, k, c

Nx_1 = Nxs[0]
Nx_2 = Nxs[1]
print("Nx = ", Nx_1)
psi_1 = f(Nx_1)
print("Nx = ", Nx_2)
psi_2 = f(Nx_2)
Nx_3 = Nxs[2]
print("Nx = ",Nx_3)
psi_3 = f(Nx_3)
d1 = psi_2 - psi_1
d2 = psi_3 - psi_2
k = (np.log2(d2/d1)) / (np.log2(Nx_2/Nx_1))
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

for Nx_3 in Nxs[3:]:
    print("Nx = ", Nx_3)
    psi_3 = f(Nx_3)
    print("psi_3 = ", psi_3)
    d1 = psi_2 - psi_1
    d2 = psi_3 - psi_2
    print("d1 = ", d1)
    print("d2 = ", d2)
    print("d2/d1 = ", d2/d1)
    k = (np.log2(d2/d1)) / (np.log2(Nx_2/Nx_1))
    print("k_p = ", k)
    #c1 = error1 / (Nx_1 ** k)
    #error2 = (Nx_2 ** k) * c1
    c2 = error2 / (Nx_2 ** k)
    error3 = (Nx_3 ** k) * c2
    errors.append(error3)
    #errors.append(error3)
    psi_1 = psi_2
    psi_2 = psi_3
    Nx_1 = Nx_2
    Nx_2 = Nx_3
    error1 = error2
    error2 = error3

res = Analysis.EstimatedErrorList(np.array(errors), Nxs, 1/512)

print(errors)

Analysis.ConvergenceAnalyzer.polynomial_convergence_visualizer(res)'''
'''x = np.linspace(2, 4, 100)
y = np.exp2(-np.power(x,2))
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_yscale("functionlog", base=2, functions=(lambda x: np.log2(-np.log2(x)), lambda x: np.exp2(-np.exp2(x))))
ax.set_xscale("functionlog", base=2, functions=(lambda x: np.log2(x), lambda x: np.exp2(x)))
ax.plot(x, y, linestyle='--', color='green')
plt.show()
fig, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(x,2*x)
plt.show()'''

