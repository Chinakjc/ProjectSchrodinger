import numpy as np
import scipy.special as sps
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

import Schema
import Test
from DynamiqueEF import dynamic_EF

# tests of basic functions

# Test.test_of_solution_and_animation()

# Test.test_for_verification_by_comparing_the_animation_with_the_exact_solution()

# Test.test_of_time_complexity()

# analysis of error

# Test.test_of_convergence_for_space_with_exact_solution()

# Test.test_of_convergence_for_time_with_exact_solution()

# Test.test_of_convergence_for_space_without_exact_solution()

# Test.test_of_convergence_for_time_without_exact_solution()





L = 200
T = 15
Nx = 2 ** 14
Nt = 450
fps = 30
centre = -L/8
xu = L/64
v_func = lambda x,t: Test.generator_of_Square_wave2_continue(Ul=0, Um= 200, Ur = -10, xl= -xu, xml =0, xmr= xu, xr = 2*xu)(x,t) + Test.generator_of_Square_wave_continue(Ul = 200,Ur = 0, xl= -L/2 + xu, xr= -L/2 + 2*xu)(x,t) + Test.generator_of_Square_wave_continue(Ul = 0,Ur = 210, xl= L/2 - 2*xu, xr= L/2 - xu)(x,t)
velocity = 5
label = "wave_packet_potential3"
xs = np.linspace(-L / 2.0, L / 2.0, Nx)[:-1]
ys = v_func(xs, 0)
plt.plot(xs, ys)
plt.show()
Test.test_of_wave_packet(velocity=velocity, centre=centre, L=L, T=T, Nx=Nx, Nt=Nt, fps=fps, label=label, v_func=v_func)


