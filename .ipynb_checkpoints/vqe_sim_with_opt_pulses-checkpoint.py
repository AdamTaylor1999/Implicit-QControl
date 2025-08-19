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

(1) I'm misinterpreting how to recover the result
(2) There is an issue with the Python code
(3) There is an issue with the MATLAB code

I should find a ground state that has definite entanglement (ie; not a product
state) just in case there is an issue with the 2-qubit native gates. 

This is because the pulse sequence seems to recover the SVP ground state when
we turn the native ZZ interactions off, but not when they are present. Despite
the fact that the MATLAB code should include ZZ interactions.


"""

import numpy as np
import random
import itertools
import time
from matplotlib import pyplot as plt
from qutip import *

from pauli_string_functions_module_python.py import *

x = nonlocal_2qubit_strings(5, 'X', 'Y')



