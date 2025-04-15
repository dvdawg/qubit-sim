import numpy as np
import matplotlib.pyplot as plt
from qutip import *

I = qeye(2) 
X = sigmax()
Y = sigmay() 
Z = sigmaz() 

def random_sequence(length):
    gates = [I, X, Y, Z]
    seq = [np.random.choice(gates) for _ in range(length)]
    return seq

def apply_sequence(state, seq):
    for gate in seq:
        state = gate * state
        state = gate.dag() * state
    return state

def randomized_benchmarking(num_sequences, max_length, num_reps):
    fidelities = []
    for length in range(1, max_length + 1):
        fidelity_avg = 0
        for _ in range(num_reps):
            initial_state = basis(2, 0)
            seq = random_sequence(length)

            final_state = apply_sequence(initial_state, seq)
            fidel = fidelity(initial_state, final_state)
            fidelity_avg += fidel

        fidelities.append(fidelity_avg / num_reps)

    return fidelities

num_sequences = 100 
max_length = 20 
num_reps = 10 

fidelities = randomized_benchmarking(num_sequences, max_length, num_reps)

plt.plot(range(1, max_length + 1), fidelities, marker='o', linestyle='-', color='b')
plt.xlabel('Sequence length')
plt.ylabel('Average fidelity')
plt.title('Randomized Benchmarking of Transmon')
plt.grid(True)
plt.show()
