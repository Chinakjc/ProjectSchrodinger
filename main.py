import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import Fourier
import scipy.linalg as sc
import Schema
import Quadrature
import time


def test0():
    '''
    test pour le calcul numérique de l'intergral par la méthode de quadrature rectangle
    :return:
    '''
    print("test0 begins !")
    dico = {"x": [0, 1], "y": [0, 1], "z": [0, 1]}
    space_step = 0.005

    def f(v):
        x = v[0]
        y = v[1]
        z = v[2]
        return x + y + z

    integral = Quadrature.Quadrature.rectangle_method(dico, space_step, f)
    print("I = ", integral)

    def func(coords):
        x, y, z = coords
        return x + y + z

    x_vals = np.arange(dico["x"][0], dico["x"][1], space_step)
    y_vals = np.arange(dico["y"][0], dico["y"][1], space_step)
    z_vals = np.arange(dico["z"][0], dico["z"][1], space_step)

    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    values = func([X, Y, Z])

    integral = Quadrature.Quadrature.rectangle_method_N(dico, space_step, values)
    print("I = ", integral)

    print("end of test !")


def test1():
    '''
    test pour le transforme de fourrier et la matrice pour l'opérateur
    :return:
    '''

    print("test1 begins !")

    def f(x):
        return np.exp(np.cos(2 * np.pi * x))

    def g(x):
        return -2 * np.pi * np.sin(2 * np.pi * x) * np.exp(np.cos(2 * np.pi * x))

    def q(x):
        return - 4 * np.pi ** 2 * np.cos(2 * np.pi * x) * np.exp(np.cos(2 * np.pi * x)) + 4 * np.pi ** 2 * np.sin(
            2 * np.pi * x) ** 2 * np.exp(np.cos(2 * np.pi * x))

    N = 2 ** 8
    L = 8
    x = np.arange(0, L, L / N)
    y = f(x)

    plt.plot(x, -0.5 * q(x))
    z = Fourier.FourierTransform.forward_transform_1D(y, "quadrature")

    '''k_values = np.fft.fftfreq(N) * N
    kinetic = 0.5 * k_values ** 2 * np.pi ** 2'''
    k_values = np.fft.fftfreq(N) * N * 2 * np.pi / L
    kinetic = 0.5 * k_values ** 2

    F = sc.dft(N)

    IF = np.conj(F) / N

    K = np.dot(IF, np.dot(np.diag(kinetic), F))

    w = Fourier.FourierTransform.inverse_transform(kinetic * z)

    plt.plot(x, np.real(w))
    plt.plot(x, np.dot(K, y))
    plt.show()
    print("end of test !")


def test2():
    '''
    test pour le schema en résolvant des équations de Schrödinger de différents psi0 et v_fun
    la sauvegarde de l'animation en format mp4 demande ffmpeg,
    en l'absence de condition, veuillez installer le ffmpeg, ou sauvegarder en format gif

    :return:
    '''

    print("test2 begins !")
    T = 10
    L = 2
    fps = 60
    psi0 = lambda x: np.cos(2 * np.pi * x)  # psi0 alternatif
    v_fun = lambda x, t: np.power(x,2) #np.sin(x + t)  # v_fun alternatif
    Nx = 2**14
    Nt = 60 * T * 2
    print("En cours de calculer.")
    anim = Schema.plot_psi(psi=Schema.dynamics(V_fun=v_fun, L=L, T=T, Nx=960, Nt=960), duration=T,
                           frames_per_second=fps,
                           L=L)
    print("Calculation terminée.")
    writer = FFMpegWriter(fps=fps)
    anim.save('animation10.mp4', writer=writer, dpi=320)
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
    fig = plt.figure(figsize=(8, 6))
    plt.semilogx(base=2)
    plt.semilogy(base=2)
    plt.xlabel("log2 of NtNx log Nx", fontsize=15)
    plt.ylabel("log2 of time", fontsize=15)
    NN = Ns ** 2 * np.log2(Ns)
    plt.scatter(NN, ts, marker="X")
    kk = (ts[-1] - ts[-2]) / (NN[-1] - NN[-2])
    plt.plot(NN, kk * NN, linestyle='--', color='green')
    plt.show()
    print("end of test !")

'''def test4():
    T = 10
    L = 6 * np.pi
    Nxs = [2**i + 1 for i in range(5,15)]
    Nt = 600
    errors = list()
    for Nx in Nxs:
        psi0 = lambda x: np.sin(x)
        psi_e = lambda t, x: np.exp(-1j / 2.0 * t) * np.sin(x)
        x = np.linspace(-L / 2.0, L / 2.0, Nx)
        t = np.linspace(0, T, Nt)
        psi = np.array([psi_e(t, xt) for xt in x])
        print("Nx = " , Nx)
        psi_h = Schema.dynamics(psi0_fun=psi0, L=L, T=T, Nx=Nx, Nt=Nt)
        e = psi - psi_h
        error = np.linalg.norm([np.linalg.norm(e[:, t], 2) for t in range(Nt)], np.inf) / np.sqrt(Nx)
        errors.append(error)

    d_e = errors[-1] / errors[-2]
    d_n = Nxs[-1] / Nxs[-2]

    k = np.log2(d_n)/np.log2(d_e)
    b = np.log2(errors[-1]) - k*np.log2(Nxs[-1])
    y = 2**b * np.power(Nxs,k)
    fig = plt.figure(figsize=(8, 6))
    plt.semilogx(base=2)
    plt.semilogy(base=2)
    plt.xlabel("log2 of Nx", fontsize=15)
    plt.ylabel("log2 of error", fontsize=15)
    plt.scatter(Nxs, errors, marker="X")
    plt.plot(Nxs,y,linestyle='--', color='green')
    print("ordre = ", -k)
    plt.show()

    print("end of test !")'''

def test4():
    T = 10
    L = 6 * np.pi
    Nxs = [2**i  for i in range(1,8)]
    Nt = 600
    errors = list()
    for Nx in Nxs:
        psi0 = lambda x: np.sin(x)
        psi_e = lambda t, x: np.exp(-1j / 2.0 * t) * np.sin(x)
        x = np.linspace(-L / 2.0, L / 2.0, Nx)[:-1]
        t = np.linspace(0, T, Nt)
        psi = np.array([psi_e(t, xt) for xt in x])
        print("Nx = " , Nx)
        psi_h = Schema.dynamics(psi0_fun=psi0, L=L, T=T, Nx=Nx, Nt=Nt)
        e = psi - psi_h
        #error = np.linalg.norm([np.linalg.norm(np.fft.fft(e[:, t]), 2) for t in range(Nt)], np.inf) / np.sqrt(Nx)
        error = np.linalg.norm([np.linalg.norm(e[:, t], 2) for t in range(Nt)], np.inf) / np.sqrt(Nx)
        errors.append(error)

    d_e = errors[-1] / errors[-2]
    d_n = Nxs[-1] / Nxs[-2]


    k = np.log2(d_e)/np.log2(d_n)
    b = np.log2(errors[-1]) - k * np.log2(Nxs[-1])
    y = 2**b * np.power(Nxs,k)
    fig = plt.figure(figsize=(8, 6))
    plt.semilogx(base=2)
    plt.semilogy(base=2)
    plt.xlabel("log2 of Nx", fontsize=15)
    plt.ylabel("log2 of error", fontsize=15)
    plt.scatter(Nxs, errors, marker="X")
    plt.plot(Nxs,y,linestyle='--', color='green')
    print("ordre = ", -k)
    plt.show()

    print("end of test !")

def test5():
    T = 10
    L = 6 * np.pi
    Nts = [60 * 2**i for i in range(5)]
    Nx = 4097
    errors = list()
    for Nt in Nts:
        psi0 = lambda x: np.sin(x)
        psi_e = lambda t, x: np.exp(-1j / 2.0 * t) * np.sin(x)
        x = np.linspace(-L / 2.0, L / 2.0, Nx)[:-1]
        t = np.linspace(0, T, Nt)
        psi = np.array([psi_e(t, xt) for xt in x])
        print("Nt = " , Nt)
        psi_h = Schema.dynamics(psi0_fun=psi0, L=L, T=T, Nx=Nx, Nt=Nt)
        e = psi - psi_h
        error = np.linalg.norm([np.linalg.norm(e[:, t], 2) for t in range(Nt)], np.inf) / np.sqrt(Nx)
        errors.append(error)

    d_e = errors[-1] / errors[-2]
    d_n = Nts[-1] / Nts[-2]
    print(errors)
    k = np.log2(d_e)/np.log2(d_n)
    b = np.log2(errors[-1]) - k*np.log2(Nts[-1])
    #y = 2**b * np.power(Nts,k)
    fig = plt.figure(figsize=(8, 6))
    plt.semilogx(base=2)
    plt.semilogy(base=2)
    plt.xlabel("log2 of Nt", fontsize=15)
    plt.ylabel("log2 of error", fontsize=15)
    plt.scatter(Nts, errors, marker="X")
    #plt.plot(Nts,y,linestyle='--', color='green')
    print("ordre = ", -k)
    plt.show()

    print("end of test !")
# test0()
#test1()
test2()
#test3()
#test5()
