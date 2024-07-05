import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import bsr_matrix, dia_matrix


def bases_p1(s1, s2):
    return lambda x: (s2 - x) / (s2 - s1), lambda x: (x - s1) / (s2 - s1)


def grad_bases_p1(s1, s2):
    return lambda x: 1 / (s1 - s2), lambda x: 1 / (s2 - s1)


def M_el(s1, s2, func_a):
    bases = bases_p1(s1, s2)
    Mel = np.zeros((2, 2), dtype=complex)
    h = s2 - s1
    m = (s1 + s2) / 2
    for i in range(2):
        for j in range(2):
            func_int = lambda x: func_a(x) * bases[i](x) * bases[j](x)
            Mel[i, j] = h / 6 * (func_int(s1) + 4 * func_int(m) + func_int(s2))
    return Mel


def K_el(s1, s2):
    Kel = np.zeros((2, 2), dtype=complex)
    h = s2 - s1
    Kel[0, 0] = 1 / h
    Kel[0, 1] = -1 / h
    Kel[1, 0] = -1 / h
    Kel[1, 1] = 1 / h
    return Kel


def dynamic_EF(psi0_fun=(lambda x: np.exp(-x ** 2)), V_fun=(lambda x, t: 0), L=10, Nx=1000, T=4, Nt=1000):
    x = np.linspace(-L / 2.0, L / 2.0, Nx)[: -1]
    t = np.linspace(0, T, Nt)
    U0 = psi0_fun(x)
    psi = [U0]
    dt = t[1] - t[0]
    MM = np.zeros((Nx - 1, Nx - 1), dtype=complex)
    KK = np.zeros((Nx - 1, Nx - 1), dtype=complex)
    MV = np.zeros((Nx - 1, Nx - 1), dtype=complex)

    # assemblage de la matrice
    for index_triangle in range(Nx - 2):
        s1 = x[index_triangle]
        s2 = x[index_triangle + 1]
        Mel = M_el(s1, s2, lambda x: 1j)
        Kel = K_el(s1, s2) * -0.5
        Mel_V = M_el(s1, s2, lambda x: V_fun(x, 0))
        for index_sommet in range(2):
            I = index_triangle + index_sommet
            for index_sommet2 in range(2):
                J = index_triangle + index_sommet2
                MM[I, J] += Mel[index_sommet, index_sommet2]
                KK[I, J] += Kel[index_sommet, index_sommet2]
                MV[I, J] += Mel_V[index_sommet, index_sommet2]

    # euler explicite
    for ti in t[1:]:
        Un = psi[-1]
        Ln = (dt * (KK + MV) + MM) @ Un
        Unew = np.linalg.solve(MM, Ln)
        #plt.plot(x, np.real(Un))
        #plt.plot(x, np.imag(Un))
        #plt.show()
        #print(np.imag(Un))
        psi.append(Unew)


    return np.transpose(psi)
