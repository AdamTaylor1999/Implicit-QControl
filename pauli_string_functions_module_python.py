#author - Adam Taylor
#date - June 2025

#This programe contains base functions being used in my quantum control project for manipulating Pauli strings, Pauli operators
#and finding closed Lie algebras. The programe uses the packages qutip, numpy, itertools and random.


#Imports and base functions
import numpy as np
from qutip import *
import itertools
import random
from scipy.sparse.linalg import eigsh


def ran(scale = 1, how_many = 1):
    #Random number generator. Randomly selects how_many numbers from a uniform distribution on (0, scale).
    if how_many == 1:
        return scale * random.random()
    else:
        ran_nums = []
        for i in range(how_many):
            ran_nums.append(scale * random.random())
        return ran_nums



#Pauli operators and algebra
def Pauli_product(P1, P2):
    #Finds the product between 2 single-qubit Pauli terms {'I', 'X', 'Y', 'Z'}, then returns the operator and coefficient.
    if P1 not in 'IXYZ' or P2 not in 'IXYZ' or len(P1) != 1 or len(P2) != 1:
        raise ValueError('P1 and/or P2 are not valid single qubit Paulis!')
        
    if P1 == 'I':
        return (P2, 1)
    if P2 == 'I':
        return (P1, 1)
    if P1 == P2:
        return ('I', 1)
    rules = {('X','Y'):('Z', 1j), ('X','Z'):('Y', -1j), ('Y','X'):('Z', -1j), ('Z','X'):('Y', 1j),
             ('Y','Z'):('X', 1j), ('Z','Y'):('X', -1j)}
    return rules[P1,P2]

def Pauli_commutator(P1, P2):
    #Finds the commutator between 2 single-qubit Pauli terms {'I', 'X', 'Y', 'Z'}, then returns the new operator and coefficient.
    #If the commutator vanishes, it return ('I', 0)
    
    if P1 not in 'IXYZ' or P2 not in 'IXYZ' or len(P1) != 1 or len(P2) != 1:
        raise ValueError('P1 and/or P2 are not valid single qubit Paulis!')
    
    if P1 == 'I' or P2 == 'I' or P1 == P2:
        return ('I', 0)
    P3, coeff = Pauli_product(P1, P2)
    return P3, 2 * coeff

def Pauli_string_multiplication(P1, P2):
    #Multiplies 2 Pauli strings together then returns the new operator string and coefficient.
    if len(P1) != len(P2):
        raise ValueError('P1 and P2 act on different numbers of qubits!')
        
    P3, coeff = '', 1
    for i in range(len(P1)):
        P_int, coeff_int = Pauli_product(P1[i], P2[i])
        P3 += P_int
        coeff *= coeff_int
    return P3, coeff

def Pauli_string_commutator(P1, P2):
    #Finds the commutator between 2 Pauli strings and then returns the operator and coefficient.
    if len(P1) != len(P2):
        raise ValueError('P1 and P2 act on different numbers of qubits!')

    anticommutator_parity = 0
    for i in range(len(P1)):
        if P1[i] != 'I' and P2[i] != 'I' and P1[i] != P2[i]:
            anticommutator_parity += 1
    P3, coeff = Pauli_string_multiplication(P1, P2)
    return (P3, coeff * (1 - (-1)**anticommutator_parity))



#Closed Lie algebras
def closed_A_set_func(H_set, A0_set):
    #Takes a set H_set and initial set A0_set to give the closed set [H_set, A_set] = A_set, by iterativley increasing the size of A_set until it closes.
    #Make sure H_set or A0_set have at least two operators (inlude 'III...I' if necessary).
    A1_new = []
    for i in range(len(A0_set)):
        for j in range(len(H_set)):
            sigma, coeff = Pauli_string_commutator(A0_set[i], H_set[j])
            if sigma not in A0_set and sigma not in A1_new and abs(coeff) > 10**-10:
                A1_new.append(sigma)
    if A1_new == []:
        return A0_set
    else:
        return closed_A_set_func(H_set, tuple(list(A0_set) + A1_new))

def organising_closed_A_set(closed_set, P):#P is the Pauli being organised.
    #Takes a tuple of Pauli strings and sorts them according to the number of single qubit Pauli's P are in each string.
    #eg; P = Z, then splits into terms with no Z, with a single Z, two Z's, three Z's ... etc
    counts = []
    for i in range(len(closed_set)):
        counts.append(closed_set[i].count(P))
             
    closed_set_reorg = np.zeros(max(counts) + 1, dtype = list)
    for i in range(len(closed_set_reorg)):
        closed_set_reorg[i] = []
        
    for i in range(len(closed_set)):
        closed_set_reorg[counts[i]].append(closed_set[i])

    closed_set_reorg_new = []
    for i in range(len(closed_set_reorg)):#removes empmty set terms
        if closed_set_reorg[i] != []:
            closed_set_reorg_new.append(tuple(closed_set_reorg[i]))

    return np.array(closed_set_reorg_new, dtype = list)




#Pauli strings
def pauli_string_at_qubit_position(n_qubits, op_str, position):
    #Returns the Pauli string op_str at qubit position position. op_str can be any string of Pauli's up to n_qubits long,
    #eg; pauli_string_at_qubit_position(6, 'XYZ', 2)  = 'IIXYZI'
    if len(op_str) > n_qubits:
        raise ValueError('You operator string is longer than the total number of qubits!')
    if position > n_qubits - len(op_str):
        raise ValueError('Position means your operator string is outside the total number of qubits!')
    if len(op_str) == n_qubits:
        return op_str
    return 'I' * position + op_str + 'I' * (n_qubits - len(op_str) - position)

def all_single_operator_strings(n_qubits, operator):
    #For operator = {'I', 'X', 'Y', 'Z'}, returns the set of all single-qubit Pauli strings
    #eg; single_operator_strings(4, 'Z')  =  ('ZIII', 'IZII', 'IIZI', 'IIIZ')
    if len(operator) != 1:
        raise ValueError('Single qubit operator is not a single qubit operator!')
    op_str = []
    for i in range(n_qubits):
        op_str.append(pauli_string_at_qubit_position(n_qubits, operator, i))
    return tuple(op_str)

def local_2body_strings(n_qubits, local_2body_interaction):
    #Returns the set of all Pauli strings featuring the interaction local_2body_interaction
    #eg; local_2body_strings(4, 'XY')  =  ('XYII', 'IXYI', IIXY')
    if len(local_2body_interaction) != 2:
        raise ValueError('Two body interaction is not two bodies!')
    local_2_str = []
    for i in range(n_qubits - 1):
        local_2_str.append(pauli_string_at_qubit_position(n_qubits, local_2body_interaction, i))
    return tuple(local_2_str)

def local_3body_strings(n_qubits, local_3body_interaction):
    #Returns the set of all Pauli strings featuring the interaction local_3body_interaction
    #eg; local_3body_strings(5, 'XYZ')  =  ('XYZII', 'IXYZI', 'IIXYZ')
    if len(local_3body_interaction) != 3:
        raise ValueError('Three body interaction is not three bodies!')
    local_3_str = []
    for i in range(n_qubits - 2):
        local_3_str.append(pauli_string_at_qubit_position(n_qubits, local_3body_interaction, i))
    return tuple(local_3_str)

def local_4body_strings(n_qubits, local_4body_interaction):
    #Returns the set of all Pauli strings featuring the interaction local_4body_interaction
    #eg; local_4body_interaction(6, 'XXZZ')  =  ('XXZZII', 'IXXZZII', 'IIXXZZ')
    if len(local_4body_interaction) != 4:
        raise ValueError('Four body interaction is not four bodies!')
    local_4_str = []
    for i in range(n_qubits - 3):
        local_4_str.append(pauli_string_at_qubit_position(n_qubits, local_4body_interaction, i))
    return tuple(local_4_str)

def alternating_local_2body_strings(n_qubits, local_2body_interaction):
    #Returns the set of all local 2 body Pauli strings on alternating sites.
    #eg; alternating_local_2body_strings(8, 'XY')  =  ('XYIIIIII', 'IIXYIIII', 'IIIIXYII', 'IIIIIIXY')
    if len(local_2body_interaction) != 2:
        raise ValueError('Two body interaction is not two bodies!')
    if n_qubits // 2 != n_qubits / 2:
        print('Odd number of qubits - no operator has support on the final qubit!')
    local_2_str = []
    for i in range(0, n_qubits - 1, 2):
        local_2_str.append(pauli_string_at_qubit_position(n_qubits, local_2body_interaction, i))
    return tuple(local_2_str)

def nonlocal_2body_strings(n_qubits, P1, P2):
    #Returns the set of all non local 2 body Pauli strings between single qubit Pauli's P1 and P2.
    #eg; nonlocal_2body_strings(4, 'X', 'Y')  =  ('XYII', 'XIYI', 'XIIY', 'YXII', 'IXYI', 'IXIY', 'YIXI', 'IYXI', 'IIXY', 'YIIX', 'IYIX', 'IIYX')
    labels = list(itertools.combinations(list(range(n_qubits)), 2))
    nonlocal_2_str = []
    for ij in labels:
        nonlocal_2_str.append('I' * ij[0] + P1 + 'I' * (ij[1] - ij[0] - 1) + P2 + 'I' * (n_qubits - ij[1] - 1))
    if P1 == P2:
        return tuple(nonlocal_2_str)
    for ij in labels:
        nonlocal_2_str.append('I' * ij[0] + P2 + 'I' * (ij[1] - ij[0] - 1) + P1 + 'I' * (n_qubits - ij[1] - 1))
    return tuple(set(nonlocal_2_str))



#Operators to / from Pauli 
def operator_string_from_Pauli_string(Pauli_string):
    #Returns the operator corresponding to the Pauli string
    Pauli_dict = {'I':qeye(2), 'X':sigmax(), 'Y':sigmay(), 'Z':sigmaz()}
    try:
        operator = [Pauli_dict[char] for char in Pauli_string]
    except KeyError:
        raise ValueError('Pauli_string is not a string consisting of I, X, Y and/or Z!')
    return tensor(operator)

def operator_strings_from_Pauli_strings(Pauli_strings, weights = 1, summed = False):
    #Given a tuple of Pauli strings, returns a tuple of QuTiP operators. If weights != 1, then each Pauli string is weighted. If sum = True, then
    #this sums all the operators together (eg; to form a Hamiltonian)
    #eg; operator_strings_from_Pauli_strings(('IX', 'IY', 'IZ')) = (IX, IY, IZ)
    #    operator_strings_from_Pauli_srings(('IX', 'IY', 'IZ'), (0.1, -5, 2.2)) = (0.1 IX, -5 IY, 2.2 IZ)
    #    operator_strings_from_Pauli_srings(('IX', 'IY', 'IZ'), (0.1, -5, 2.2), summed = True) = 0.1 IX -5 IY + 2.2 IZ
    if weights == 1:
        weights = np.ones(len(Pauli_strings))
    else:
        if len(weights) != len(Pauli_strings):
            raise ValueError('Number of weights does not match number of Pauli strings!')
    operators = []
    for i in range(len(Pauli_strings)):
        operators.append(weights[i] * operator_string_from_Pauli_string(Pauli_strings[i]))
    if summed == False:
        return tuple(operators)
    else:
        summed_operator = 0
        for i in range(len(Pauli_strings)):
            summed_operator += operators[i]
        return summed_operator

def operator_Pauli_decomposition(n_qubits, Pauli_strings, target_operator):
    #Returns the decomposition of the target operator over the set of Pauli strings listed
    coefficients = np.zeros(len(Pauli_strings), dtype = complex)
    overlap = 0
    for i in range(len(Pauli_strings)):
        if len(Pauli_strings[i]) != n_qubits or any(char not in 'IXYZ' for char in Pauli_strings[i]): #MIGHT WANT TO REUSE THIS ERROR MESSAGE!
            raise ValueError('Pauli_strings does not consist of a set of Pauli strings of length n_qubits!')
        Pauli_operator = operator_string_from_Pauli_string(Pauli_strings[i])
        coefficients[i] = 2**(-n_qubits) * (Pauli_operator * target_operator).tr()
        overlap += abs(coefficients[i])**2
    overlap = 2**(n_qubits) * overlap / (target_operator.dag() * target_operator).tr()
    return coefficients, overlap


def ground_state_energy(n, Hamiltonian):
    #Returns the ground state and associated energy of a given sparse QuTiP Hamiltonian
    #Uses the Lanzcos algorithm and is significantly more efficient than QuTiPs groundstate() function
    H_sparse = Hamiltonian.data
    E0, psi0 = eigsh(H_sparse, k = 1, which = 'SA')
    psi0 = Qobj(psi0[:, 0])
    psi0.dims = [[2] * n, [1] * n]
    return E0[0], psi0


def binned_pulses_from_x_optm_1q(n, num_bins, x_optm, conjugated = True):
    #Takes x_optm from the MATLAB code for the single qubit control terms Xj, Yj
    #and converts it into the binned pulses coefficients for each X and Y
    cX_binned, cY_binned = np.zeros([n, num_bins]), np.zeros([n, num_bins])
    for i in range(n):
        for j in range(num_bins):
            cX_binned[i, j] = x_optm[num_bins * (2 * i) + j]
            cY_binned[i, j] = x_optm[num_bins * (2 * i + 1) + j]
        if conjugated:
            cX_binned[i] = -np.flip(cX_binned[i])
            cY_binned[i] = -np.flip(cY_binned[i])
    return cX_binned, cY_binned

def continuous_time_pulses_from_x_optm_1q(n, num_bins, times, x_optm, conjugated = True):
    #Takes x_optm from the MATLAB code for the single qubit control terms Xj, Yj
    #and converts it into the pulse coefficients over the times for each X and Y.
    #Make sure that times = np.linspace(0, T_optm, time_steps)
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



