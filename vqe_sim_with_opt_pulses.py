#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 09:27:14 2025

@author: at4018

Here, we carry out python simulations of the optimised pulse sequences in order
to prove the MATLAB code and our understanding of the theory is working 
correctly. 

For some reason, it currently isn't working
Work schedule - today I need to get it working, that is the singular goal!

--> several days later, I still don't know what the issue is...

(1) I'm misinterpreting how to recover the result (DONT THINK SO)
(2) There is an issue with the Python code (GET CHATGPT TO CHECK
(3) There is an issue with the MATLAB code

I should find a ground state that has definite entanglement (ie; not a product
state) just in case there is an issue with the 2-qubit native gates. 

This is because the pulse sequence seems to recover the SVP ground state when
we turn the native ZZ interactions off, but not when they are present. Despite
the fact that the MATLAB code should include ZZ interactions.

For multiple bins, this code just completely fails to recreate the ideal 
expectation value (eg; heisenberg model with 10 bins, carefully selected cX
and cY and we never get the ground state)

So what's happening? How can we be recovering the exact expectation value while
failing to recover the correct pulse sequences?

--> maybe linked with the ZZ implementation

"""

import numpy as np
import random
import itertools
import time
from matplotlib import pyplot as plt
from qutip import *
from scipy.sparse.linalg import eigsh
from pauli_string_functions_module_python import *

import os
print("Current working directory:", os.getcwd())

timing0 = time.time()
timing1 = time.time()
runtime = timing1 - timing0

#%%shortest vector Hamiltonian
# --> computational basis ground state

#shortest vector problem Hamiltonian, ground state and ground energy
n = 4
svp_terms = tuple(list(all_single_operator_strings(n, 'Z')) + list(nonlocal_2body_strings(n, 'Z', 'Z')))
svp_weights = [1.2726607535749053,
 2.8425411954266906,
 3.3811184363222577,
 3.367651536169671,
 1.341929072147563,
 0.015628759868175024,
 3.7315852823485676,
 2.64150493595804,
 0.3382182222301475,
 3.9002772705578534]

H_svp = operator_strings_from_Pauli_strings(svp_terms, svp_weights, summed = True)
svp_min, svp_GS = ground_state_energy(n, H_svp)
for i in range(2**n):
    if abs(svp_GS[i][0][0]) > 0.1:
        print(i) #GS = 5 = 0101 -- directly constructed below
svp_GS_exact = tensor([basis(2, 0), basis(2, 1), basis(2, 0), basis(2, 1)])

#%% single qubit control with 2 bins
#pulse sequence from  MATLAB code
num_bins = 2
x_optm = np.loadtxt('data/svp_native_n=4_bin=2_2pi.csv', delimiter = ',')
T_optm = x_optm[-1]
Dt = T_optm / num_bins
x_optm = x_optm[:-1]

#time
time_steps = 10000
dt = T_optm / time_steps
times = np.linspace(0, T_optm, time_steps)

#X and Y Hamiltonian pulse sequences
cX, cY = XY_continuous_time_pulses_from_x_optm_1q(n, num_bins, times, x_optm)

#time-dependent control Hamiltonian
H_control = XY_control_Hamiltonian(n, cX, cY)

#time independent native ZZ gates
H_native_terms = local_2body_strings(n, 'ZZ')
H_native = operator_strings_from_Pauli_strings(H_native_terms, summed = True)

#total time-dependent Hamiltonian
H_time_dependent = [0 * H_native] + H_control #[H0, [H1, c1], [H2, c2], ...]

#QuTiP simulation
psi0 = tensor([basis(2, 0)] * n)
results = sesolve(H_time_dependent, psi0, times)
psiT = results.states[-1]

#comparing with true GS and energy
psiT_energy = np.real((psiT.dag() * H_svp * psiT)[0][0][0])
GS_fidelity = fidelity(psiT, svp_GS)
print(f'GS energy: {svp_min}')
print(f'VQE energy: {psiT_energy}')
print(f'GS overlap: {GS_fidelity}')

"""
For num_bins = 2, we recover the ground state ONLY when the native gates are
turned off for the shortest-vector style Hamiltonians.

I do not know why this works when ZZ interactions are turned off, since the MATLAB
code must be implementing these 2 qubit gates.
"""


#%% single qubit control with 3 bins 
#pulse sequence from MATLAB code
num_bins = 3
x_optm = np.loadtxt('data/svp_native_n=4_bin=3_2pi.csv', delimiter = ',')
T_optm = x_optm[-1]
Dt = T_optm / num_bins
x_optm = x_optm[:-1]

#time
time_steps = 10000
dt = T_optm / time_steps
times = np.linspace(0, T_optm, time_steps)

#X and Y binned pulses
cX_binned, cY_binned = XY_binned_pulses_from_x_optm_1q(n, num_bins, x_optm)

#X and Y continuous-time Hamiltonian pulse sequences
cX, cY = XY_continuous_time_pulses_from_x_optm_1q(n, num_bins, times, x_optm)

#time-dependent control Hamiltonian
H_control = XY_control_Hamiltonian(n, cX, cY)

#time independent native ZZ gates
H_native_terms = local_2body_strings(n, 'ZZ')
H_native = operator_strings_from_Pauli_strings(H_native_terms, summed = True)

#total time-dependent Hamiltonian
H_time_dependent = [0 * H_native] + H_control #[H0, [H1, c1], [H2, c2], ...]

#QuTiP simulation
psi0 = tensor([basis(2, 0)] * n)
results = sesolve(H_time_dependent, psi0, times)
psiT = results.states[-1]

#comparing with true GS and energy
psiT_energy = np.real((psiT.dag() * H_svp * psiT)[0][0][0])
GS_fidelity = fidelity(psiT, svp_GS)
print(f'GS energy: {svp_min}')
print(f'VQE energy: {psiT_energy}')
print(f'GS overlap: {GS_fidelity}')


"""
For num_bins = 3, we no longer ever recover the correct ground state. I do not
know why this is...

"""


#%% 2 qubit control with 2 bins
##pulse sequence from MATLAB

num_bins = 2
x_optm = np.loadtxt('data/svp_2q_n=4_bin=2_2pi.csv', delimiter = ',')
T_optm = x_optm[-1]
Dt = T_optm / num_bins
x_optm_1q = x_optm[:2 * n * num_bins]
x_optm_2q = x_optm[2 * n * num_bins:-1]

#time
time_steps = 10000
dt = T_optm / time_steps
times = np.linspace(0, T_optm, time_steps)

#X and Y binned pulses
cX_binned, cY_binned = XY_binned_pulses_from_x_optm_1q(n, num_bins, x_optm_1q, conjugated = True)

#X and Y continuous-time Hamiltonian pulse sequences
cX, cY = XY_continuous_time_pulses_from_x_optm_1q(n, num_bins, times, x_optm_1q, conjugated = True)

#XY control Hamiltonian
H_control_XY = XY_control_Hamiltonian(n, cX, cY)

#ZZ binned pulses
cZZ_binned =  ZZ_binned_pulses_from_x_optm_2q(n, num_bins, x_optm_2q, conjugated = True)

#ZZ continuous-time pulses
cZZ = ZZ_continuous_time_pulses_from_x_optm_2q(n, num_bins, times, x_optm_2q, conjugated = True)

#ZZ control Hamiltonian
H_control_ZZ = ZZ_control_Hamiltonian(n, cZZ)

#total time dependent Hamiltonian
H_time_dependent = H_control_XY + H_control_ZZ

#QuTiP simulation
psi0 = tensor([basis(2, 0)] * n)
results = sesolve(H_time_dependent, psi0, times)
psiT = results.states[-1]

#comparing with true GS and energy
psiT_energy = np.real((psiT.dag() * H_svp * psiT)[0][0][0])
GS_fidelity = fidelity(psiT, svp_GS)
print(f'GS energy: {svp_min}')
print(f'VQE energy: {psiT_energy}')
print(f'GS overlap: {GS_fidelity}')




#%%Crushed Ising model
# --> entangled ground state?

#cIm Hamiltonian, ground state and ground energy
n = 4
cIm_terms = tuple(list(all_single_operator_strings(n, 'Z')) + list(local_2body_strings(n, 'XX')) + \
                  ['X' + (n - 1) *'I'] + [(n - 1) * 'I' + 'X'])
H_cIm = operator_strings_from_Pauli_strings(cIm_terms, summed = True)

cIm_min, cIm_GS = ground_state_energy(n, H_cIm)



#for this x_optm, expectation value: -5.249981
#can't be the true GS due to limited bin num (gets caught in local minimas)
num_bins = 2
x_optm = np.loadtxt('data/cIm_native_n=4_bin=2_2pi.csv', delimiter = ',')
T_optm = x_optm[-1]
Dt = T_optm / num_bins
x_optm = x_optm[:-1]

time_steps = 10000
dt = T_optm / time_steps
times = np.linspace(0, T_optm, time_steps)

#binned pulses
cX_binned, cY_binned = XY_binned_pulses_from_x_optm_1q(n, num_bins, x_optm)

#continuous-time pulses
cX, cY = XY_continuous_time_pulses_from_x_optm_1q(n, num_bins, times, x_optm)

#control Hamiltonian
H_control = XY_control_Hamiltonian(n, cX, cY)

H_native_terms = local_2body_strings(n, 'ZZ')
H_native = operator_strings_from_Pauli_strings(H_native_terms, summed = True)

#total time-dependent Hamiltonian
H_time_dependent = [0 * H_native] + H_control #[H0, [H1, c1], [H2, c2], ...]

#QuTiP simulation
psi0 = tensor([basis(2, 0)] * n)
results = sesolve(H_time_dependent, psi0, times)
psiT = results.states[-1]

#comparing with true GS and energy
psiT_energy = np.real((psiT.dag() * H_Heis * psiT)[0][0][0])
GS_fidelity = fidelity(psiT, Heis_GS)
print(f'GS energy: {Heis_min}')
print(f'VQE energy: {psiT_energy}')
print(f'GS overlap: {GS_fidelity}')









#%%Heisenberg Hamiltonian
# --> entangled ground state

#Heisenberg Hamiltonian, ground state and ground energy
n = 4
Heis_terms = tuple(list(local_2body_strings(n, 'XX')) + list(local_2body_strings(n, 'YY')) + list(local_2body_strings(n, 'ZZ')))
H_Heis = operator_strings_from_Pauli_strings(Heis_terms, summed = True)

Heis_min, Heis_GS = ground_state_energy(n, H_Heis)


#optimal pulse data
num_bins = 10
x_optm = np.loadtxt('data/heis_native_n=4_bin=10_2pi.csv', delimiter = ',')
T_optm = x_optm[-1]
Dt = T_optm / num_bins
x_optm = x_optm[:-1]

#time steps
time_steps = 10000
dt = T_optm / time_steps
times = np.linspace(0, T_optm, time_steps)

#binned pulses
cX_binned, cY_binned = XY_binned_pulses_from_x_optm_1q(n, num_bins, x_optm)

#continuous-time pulses
cX, cY = XY_continuous_time_pulses_from_x_optm_1q(n, num_bins, times, x_optm)

#control Hamiltonian
H_control = XY_control_Hamiltonian(n, cX, cY)

H_native_terms = local_2body_strings(n, 'ZZ')
H_native = operator_strings_from_Pauli_strings(H_native_terms, summed = True)

#total time-dependent Hamiltonian
H_time_dependent = [1 * H_native] + H_control #[H0, [H1, c1], [H2, c2], ...]

#QuTiP simulation
psi0 = tensor([basis(2, 0)] * n)
results = sesolve(H_time_dependent, psi0, times)
psiT = results.states[-1]

#comparing with true GS and energy
psiT_energy = np.real((psiT.dag() * H_Heis * psiT)[0][0][0])
GS_fidelity = fidelity(psiT, Heis_GS)
print(f'GS energy: {Heis_min}')
print(f'VQE energy: {psiT_energy}')
print(f'GS overlap: {GS_fidelity}')

"""
This ALSO doesn't work! And this time we can't go back to num_bins = 2 because
then the MATLAB optimisation doesn't allow us to reach the ground state
(insufficient degrees of freedom)
"""

#%%
"""
Testing the shortest vector problem Hamiltonian limits - how large can we go?

On my laptop, I suspect MATLAB will be the limiting factor for a reasonable 
amount of time.

Mind you, n=10 with bond=40 isn't too slow.
"""

#shortest vector problem Hamiltonian, ground state and ground energy
n = 10
svp_terms = tuple(list(all_single_operator_strings(n, 'Z')) + list(nonlocal_2body_strings(n, 'Z', 'Z')))
svp_weights = [1.1936187224736132,
 3.942825143869964,
 3.6444516935156717,
 4.7924115711654105,
 1.6379460719285204,
 2.987007872098033,
 4.625704903118575,
 3.6385001168980953,
 1.573522320013755,
 0.139197506979587,
 2.8203807481050003,
 4.7385461744266495,
 3.9414186597085665,
 1.1470288101844017,
 4.645067306076715,
 3.650233413855497,
 4.367913839287504,
 4.9118091882909924,
 1.2590291636070035,
 2.2845950442155853,
 3.668678878572287,
 1.5456911234957889,
 1.6979930999985422,
 3.3754463529892496,
 4.556308027300372,
 1.1651963864263122,
 4.112484070760469,
 4.650565980053052,
 2.0214625070401047,
 3.6740895399130276,
 1.0343769942154202,
 2.27824051442235,
 0.8147232044255909,
 2.082441854259157,
 4.000397248674186,
 4.054545366338213,
 3.123007478477453,
 0.8871699028963204,
 4.593817444629671,
 2.8444035537849084,
 4.63911019539531,
 0.22095250449073123,
 3.5321701099412577,
 2.126535044880434,
 3.515545287340134,
 3.713074499360987,
 1.871845269181085,
 4.504410332691487,
 3.2915313987321064,
 3.469092312581874,
 2.539319103121453,
 1.9965462822785613,
 3.1546998198967096,
 1.0399826111302208,
 0.8996954811401109]

H_svp = operator_strings_from_Pauli_strings(svp_terms, svp_weights, summed = True)
svp_min, svp_GS = ground_state_energy(n, H_svp)
for i in range(2**n):
    if abs(svp_GS[i][0][0]) > 0.1:
        print(i) #GS = 426 --> 0110101010
svp_GS_exact = tensor([basis(2, 0), basis(2, 1), basis(2, 1), basis(2, 0), basis(2, 1), \
                       basis(2, 0), basis(2, 1), basis(2, 0), basis(2, 1), basis(2, 0)])

norm0 = np.sqrt((H_svp.dag() * H_svp).tr())
print(e0 * norm0)


#%%
n=20
svp_terms = tuple(list(all_single_operator_strings(n, 'Z')) + list(nonlocal_2body_strings(n, 'Z', 'Z')))
svp_weights = ran(n, len(svp_terms))
H_svp = operator_strings_from_Pauli_strings(svp_terms, svp_weights, summed =  True)
svp_min, svp_GS = ground_state_energy(n, H_svp)

print(svp_min)

    













