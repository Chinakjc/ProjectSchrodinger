import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy


# Simulates the SchrÃ¶dinger dynamics iâˆ‚t = -1/2 Ïˆ'' + V(x,t) Ïˆ, with the pseudospectral method
# on an interval [-L,L] with periodic boundary conditions, with Nx grid points
# The simulation proceeds from 0 to T, with Nt time steps.
# Initial condition is given by the function psi0_fun(x), potential by the function V_fun(x,t)
# Returns an array psi[ix, it]
def dynamics_lent(psi0_fun=(lambda x: np.exp(-x ** 2)), V_fun=(lambda x, t: 0), L=10, Nx=1000, T=4, Nt=1000):
    # TO DO
    x = np.linspace(-L / 2.0, L / 2.0, Nx)
    t = np.linspace(0, T, Nt)
    k_values = np.fft.fftfreq(Nx) * Nx
    kinetic = 0.5 * k_values ** 2 * np.pi ** 2
    F = scipy.linalg.dft(Nx)
    IF = np.conj(F) / Nx
    K = np.dot(IF, np.dot(np.diag(kinetic), F))
    dt = t[1] - t[0]
    M = scipy.linalg.expm(-1j * K * dt)
    res = [psi0_fun(x)]
    '''for _ in t[1:]:
        psi_new = np.dot(M, res[-1])
        res.append(psi_new)'''
    for ti in t[1:]:
        Vi = np.diag(np.exp(-1j*dt*np.array([V_fun(x=xi,t=ti) for xi in x])))
        psi_new = np.dot(np.dot(M,Vi),res[-1])
        res.append(psi_new)

    return np.transpose(res)

def dynamics(psi0_fun=(lambda x: np.exp(-x ** 2)), V_fun=(lambda x, t: 0), L=10, Nx=1000, T=4, Nt=1000):
    # TO DO
    x = np.linspace(-L / 2.0, L / 2.0, Nx)[:-1]
    t = np.linspace(0, T, Nt)
    k_values = np.fft.fftfreq(Nx-1) * (Nx-1) * 2 * np.pi / L
    kinetic = 0.5 * k_values ** 2
    dt = t[1] - t[0]
    dt = dt
    D = np.exp(-1j * kinetic * dt)
    res = [psi0_fun(x)]
    for ti in t[1:]:
        Vi = np.exp(-1j*dt*np.array([V_fun(x=xi,t=ti) for xi in x]))
        psi_new = np.fft.ifft(D*np.fft.fft(Vi *res[-1]))
        res.append(psi_new)

    return np.transpose(res)

'''def dynamics(psi0_fun=(lambda x: np.exp(-x ** 2)), V_fun=(lambda x, t: 0), L=10, Nx=1000, T=4, Nt=1000):
    # TO DO
    x = np.linspace(-L / 2.0, L / 2.0, Nx)
    t = np.linspace(0, T, Nt)
    k_values = np.fft.fftfreq(Nx) * Nx
    kinetic = 0.5 * k_values ** 2 * np.pi ** 2
    F = scipy.linalg.dft(Nx)
    IF = np.conj(F) / Nx
    K = np.dot(IF, np.dot(np.diag(kinetic), F))
    #dt = t[1] - t[0]
    res = [psi0_fun(x)]
    for ti in t[1:]:
        Vi = np.diag(np.array([V_fun(x=xi,t=ti) for xi in x]))
        U = K + Vi
        M = scipy.linalg.expm(-1j * U * ti)
        psi_new = np.dot(M,res[0])
        res.append(psi_new)

    return np.transpose(res)'''

# Plots the return value psi of the function "dynamics", using linear interpolation
# The whole of psi is plotted, in an animation lasting "duration" seconds (duration is unconnected to T)
# L argument is only for x axis labelling
def plot_psi(psi, duration=10, frames_per_second=30, L=10):
    fig, ax = plt.subplots()
    t_data = np.linspace(0, 1, np.size(psi, 1))  # 1 is arbitrary here
    x_data = np.linspace(-L, L, np.size(psi, 0), endpoint=False)
    # set the min and maximum values of the plot, to scale the axis
    m = min(0, np.min(np.real(psi)), np.min(np.imag(psi)))
    M = np.max(np.abs(psi))
    # set the axis once and for all
    ax.set(xlim=[-L, L], ylim=[m, M], xlabel='x', ylabel='psi')
    # dummy plots, to update during the animation
    real_plot = ax.plot(x_data, np.real(psi[:, 0]), label='Real')[0]
    imag_plot = ax.plot(x_data, np.imag(psi[:, 0]), label='Imag')[0]
    abs_plot = ax.plot(x_data, np.abs(psi[:, 0]), label='Abs')[0]
    ax.legend()

    # define update function as an internal function (that can access the variables defined before)
    # will be called with frame=0...(duration*frames_per_second)-1
    def update(frame):
        #print(frame)
        # get the data by linear interpolation
        t = frame / (duration * frames_per_second)
        psi_t = np.array([np.interp(t, t_data, psi[i, :]) for i in range(np.size(psi, 0))])
        # update the plots
        real_plot.set_ydata(np.real(psi_t))
        imag_plot.set_ydata(np.imag(psi_t))
        abs_plot.set_ydata(np.abs(psi_t))

    ani = animation.FuncAnimation(fig=fig, func=update, frames=duration * frames_per_second,
                                  interval=1000 / frames_per_second)
    return ani
