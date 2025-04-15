# Implicit-QControl
This repository explores a framework for implicit quantum control based on invariants, with the goal of enabling both ground state preparation and Hamiltonian simulation for non-local Hamiltonians. By focusing on control strategies that avoid explicitly tracking quantum states, we aim to develop scalable methods for steering quantum systems using short-range interactions—despite the inherently long-range structure of the target Hamiltonians.

We investigate two complementary stategies:
+ Identifying small Lie algebras that permit efficient classical optimization.
+ Exploring larger algebras amenable to simulation using matrix product operator (MPO) methods.

Our motivation includes applications such as the shortest vector problem, which can be encoded in the ground state of Hamiltonians of the form:

$$H = \sum_j \omega_j Z_j + \sum_j \sum_{k > j} f_{jk} Z_j Z_k$$

The central question we address:
+ Can we construct optimal quantum control protocols using only short-ranged interactions, yet still efficiently access the physics of non-local Hamiltonians, either by preparing their ground states or by simulating their evolution?
  
This work builds on the framework developed in:
_"Quantum control without quantum states"_, M. Orozco-Ruiz, N. H. Le & F. Mintert, PRX Quantum __5__, 040346 (2024).
