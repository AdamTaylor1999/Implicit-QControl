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

print(f'GS energy: {svp_min}')

for i in range(2**n):
    if abs(svp_GS[i][0][0]) > 0.1:
        print(i) #GS = 5 = 0101 -- directly constructed below
svp_GS_exact = tensor([basis(2, 0), basis(2, 1), basis(2, 0), basis(2, 1)])


#%%2 bins very small benchmarking
num_bins = 2
x_optm = np.loadtxt('data/svp_native_n=4_bin=2_2pi.csv', delimiter = ',')
T_optm = x_optm[-1]
Dt = T_optm / num_bins
x_optm = x_optm[:-1]

def binned_pulses_from_x_optm(n, num_bins, x_optm, conjugated = True):
    cX_binned, cY_binned = np.zeros([n, num_bins]), np.zeros([n, num_bins])
    for i in range(n):
        for j in range(num_bins):
            cX_binned[i, j] = x_optm[num_bins * (2 * i) + j]
            cY_binned[i, j] = x_optm[num_bins * (2 * i + 1) + j]
        if conjugated:
            cX_binned[i] = -np.flip(cX_binned[i])
            cY_binned[i] = -np.flip(cY_binned[i])
    return cX_binned, cY_binned

def continuous_time_pulses_from_x_optm(n, num_bins, times, x_optm, conjugated = True):
    cX_binned, cY_binned = binned_pulses_from_x_optm(n, num_bins, x_optm, conjugated = conjugated)
    time_steps = len(times)
    Dt = times[-1] / num_bins
    cX, cY = np.zeros([n, time_steps]), np.zeros([n, time_steps])
    for i in range(n):
        count = 1
        for j in range(time_steps):
            if times[j] > count * Dt and count * Dt < times[-1]:
                count += 1
            cX[i, j] = cX_binned[i, count - 1]
            cY[i, j] = cY_binned[i, count - 1]
    return cX, cY

#binned 
cX_binned, cY_binned = np.zeros([n, num_bins]), np.zeros([n, num_bins])
for i in range(n):
    for j in range(num_bins):
        cX_binned[i, j] = x_optm[2  * (2 * i) + j]
        cY_binned[i, j] = x_optm[2  * (2 * i + 1) + j]
    cX_binned[i] = -np.flip(cX_binned[i])
    cY_binned[i] = -np.flip(cY_binned[i])

cX_binned, cY_binned = binned_pulses_from_x_optm(n, num_bins, x_optm)

time_steps = 10000
times = np.linspace(0, T_optm, time_steps)

cX, cY = np.zeros([n, time_steps]), np.zeros([n, time_steps])
for i in range(n):
    count = 1
    for j in range(time_steps):
        if times[j] > count * Dt and count * Dt < T_optm:
            count += 1
        cX[i, j] = cX_binned[i, count - 1]
        cY[i, j] = cY_binned[i, count - 1]


cX, cY = continuous_time_pulses_from_x_optm(n, num_bins, times, x_optm)

#SOMETHING HERE>?!?!??!?!?!?!?!?!?!?!?!?!?!
#time-depdendent control Hamiltonian
H_control_terms = tuple(list(all_single_operator_strings(n, 'X')) + list(all_single_operator_strings(n, 'Y')))
H_control_ops = operator_strings_from_Pauli_strings(H_control_terms)
H_control = []
for i in range(n):
    H_control.append([H_control_ops[i], cX[i]])
    H_control.append([H_control_ops[i + n], cY[i]])

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


#OH FUCK! THE NATIVE GATES NEED TO BE TURNED OFF! THEN WE ACHIEVE THE CORRECT RESULTS!
#BUT SEEMINGLY ONLY WITH THIS SHORTEST VECTOR STYLE HAMILTONIAN? 
#PROBABLY BECAUSE I MESSED UP INDICES ON THE OTHER ONES

#WHERE ARE THE NATIVE GATES?!?!??!?!?!?!?!?!?!?!?!?!??!!??!?!?!?!?!?!?!?!?!?!?!

#%%

#function seems to be wrong
def extract_pulses_from_matlab_x(x_optm, bin_num, n_qubits,
                                 apply_heisenberg_transform=False,
                                 return_shape='bins_first',
                                 verbose=False):
    """
    Reconstruct pulses from MATLAB x(:) that was used with
      c = reshape(x, [bin_num, ctrl_num])
    and control ordering X1, Y1, X2, Y2, ... Xn, Yn.

    Args:
      x_optm: numpy array loaded from MATLAB. Can be
              - 1D array of length bin_num * (2*n_qubits) (most typical),
              - 2D array (bin_num, ctrl_num) or (ctrl_num, bin_num).
      bin_num: number of time bins
      n_qubits: number of qubits
      apply_heisenberg_transform: if True apply c' = -flip(c, axis=0) (useful if MATLAB used U^dag convention).
      return_shape: 'bins_first' -> returns arrays shaped (bin_num, n_qubits)
                    'qubits_first' -> returns arrays shaped (n_qubits, bin_num)
      verbose: print checks

    Returns:
      c_mat: (bin_num, ctrl_num) reconstructed full control matrix
      pulses_x: X coefficients (see return_shape)
      pulses_y: Y coefficients (see return_shape)
    """
    ctrl_num_expected = 2 * n_qubits

    x_np = np.asarray(x_optm)

    # Handle 1D vector exported from MATLAB (column-major layout)
    if x_np.ndim == 1:
        if x_np.size != bin_num * ctrl_num_expected:
            raise ValueError(f"Input length {x_np.size} doesn't match bin_num*2*n_qubits = "
                             f"{bin_num * ctrl_num_expected}")
        # IMPORTANT: MATLAB stacks columns (column-major) so use order='F'
        c_mat = x_np.reshape((bin_num, ctrl_num_expected), order='F')

    # Handle 2D array (might already be shaped)
    elif x_np.ndim == 2:
        r, c = x_np.shape
        if (r, c) == (bin_num, ctrl_num_expected):
            c_mat = x_np.copy()
        elif (r, c) == (ctrl_num_expected, bin_num):
            # transposed compared to MATLAB layout
            c_mat = x_np.T.copy()
        elif x_np.size == bin_num * ctrl_num_expected:
            # ambiguous - reshape assuming MATLAB column-major ordering
            c_mat = np.reshape(x_np, (bin_num, ctrl_num_expected), order='F')
        else:
            raise ValueError(f"2D input has shape {x_np.shape} which is incompatible with "
                             f"bin_num={bin_num}, ctrl_num_expected={ctrl_num_expected}")
    else:
        raise ValueError("x_optm must be 1D or 2D numpy array-like")

    # Extract X and Y columns:
    # MATLAB column indices: 1->X1, 2->Y1, 3->X2, 4->Y2, ...
    # Zero-based python columns: 0->X1, 1->Y1, 2->X2, 3->Y2, ...
    pulses_x = c_mat[:, 0::2]  # shape (bin_num, n_qubits)
    pulses_y = c_mat[:, 1::2]  # shape (bin_num, n_qubits)

    if apply_heisenberg_transform:
        # c_j(t) --> -c_j(T - t) : reverse bins (axis 0) and negate
        pulses_x = -np.flip(pulses_x, axis=0)
        pulses_y = -np.flip(pulses_y, axis=0)
        c_mat = -np.flip(c_mat, axis=0)

    if return_shape == 'qubits_first':
        pulses_x = pulses_x.T   # shape (n_qubits, bin_num)
        pulses_y = pulses_y.T
    elif return_shape == 'bins_first':
        # already (bin_num, n_qubits)
        pass
    else:
        raise ValueError("return_shape must be 'bins_first' or 'qubits_first'")

    if verbose:
        print("c_mat.shape:", c_mat.shape)
        print("pulses_x.shape:", pulses_x.shape)
        print("pulses_y.shape:", pulses_y.shape)
        # quick consistency checks
        assert c_mat.shape == (bin_num, ctrl_num_expected)
        assert pulses_x.shape[1 if return_shape=='bins_first' else 0] == n_qubits

    return c_mat, pulses_x, pulses_y
#%%

#optimal pulse data IGNORE THIS ONE FOR NOW 
num_bins = 10
x_optm = np.loadtxt('data/svp_n=4_bin=10_2pi_new.csv', delimiter = ',')
T_optm = x_optm[-1]
Dt = T_optm / num_bins
x_optm = x_optm[:-1]



cmat_chat, cX_chat, cY_chat = extract_pulses_from_matlab_x(x_optm, num_bins, n, return_shape = 'qubits_first', apply_heisenberg_transform = True)

#binned pulses
cX_binned, cY_binned = np.zeros([n, num_bins]), np.zeros([n, num_bins])
for i in range(n):
    for j in range(num_bins):
        cX_binned[i, j] = x_optm[8 * (2 * i) + j]
        cY_binned[i, j] = x_optm[8 * (2 * i + 1) + j]
    cX_binned[i] = -np.flip(cX_binned[i])
    cY_binned[i] = -np.flip(cY_binned[i])

#continuous-time pulses
time_steps = 10000
times = np.linspace(0, T_optm, time_steps)
cX, cY = np.zeros([n, time_steps]), np.zeros([n, time_steps])
for i in range(n):
    count = 1
    for j in range(time_steps):
        if times[j] > count * Dt and count < num_bins:
            count += 1
        cX[i, j] = -np.flip(cX_binned[i, count - 1])
        cY[i, j] = -np.flip(cY_binned[i, count - 1])
        #cX[i, j] = cX_chat[i, count - 1]
        #cY[i, j] = cY_chat[i, count - 1]
        
#time-depdendent control Hamiltonian
H_control_terms = tuple(list(all_single_operator_strings(n, 'X')) + list(all_single_operator_strings(n, 'Y')))
H_control_ops = operator_strings_from_Pauli_strings(H_control_terms)
H_control = []
for i in range(n):
    H_control.append([H_control_ops[i], cX[i]])
    H_control.append([H_control_ops[i + n], cY[i]])

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

#binned pulses
cX_binned, cY_binned = np.zeros([n, num_bins]), np.zeros([n, num_bins])
for i in range(n):
    for j in range(num_bins):
        cX_binned[i, j] = x_optm[8 * (2 * i) + j]
        cY_binned[i, j] = x_optm[8 * (2 * i + 1) + j]
    cX_binned[i] = -np.flip(cX_binned[i])
    cY_binned[i] = -np.flip(cY_binned[i])

#continuous-time pulses
time_steps = 10000
times = np.linspace(0, T_optm, time_steps)
cX, cY = np.zeros([n, time_steps]), np.zeros([n, time_steps])
for i in range(n):
    count = 1
    for j in range(time_steps):
        if times[j] > count * Dt and count < num_bins:
            count += 1
        cX[i, j] = cX_binned[i, count - 1]
        cY[i, j] = cY_binned[i, count - 1]

#time-depdendent control Hamiltonian
H_control_terms = tuple(list(all_single_operator_strings(n, 'X')) + list(all_single_operator_strings(n, 'Y')))
H_control_ops = operator_strings_from_Pauli_strings(H_control_terms)
H_control = []
for i in range(n):
    H_control.append([H_control_ops[2 * i], cX[i]])
    H_control.append([H_control_ops[2 * i + 1], cY[i]])

#time independent native ZZ gates
H_native_terms = local_2body_strings(n, 'ZZ')
H_native = operator_strings_from_Pauli_strings(H_native_terms, summed = True)

#total time-dependent Hamiltonian
H_time_dependent = [H_native] + H_control #[H0, [H1, c1], [H2, c2], ...]

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






