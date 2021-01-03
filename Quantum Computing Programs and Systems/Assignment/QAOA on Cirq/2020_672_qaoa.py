"""Runs the Quantum Approximate Optimization Algorithm on Max-Cut.

=== EXAMPLE OUTPUT ===

Example QAOA circuit:
0         1         2
│         │         │
H         H         H
│         │         │
ZZ────────ZZ^0.974  │
│         │         │
Rx(0.51π) ZZ────────ZZ^0.974
│         │         │
│         Rx(0.51π) Rx(0.51π)
│         │         │
M('m')────M─────────M
│         │         │
Optimizing objective function ...
The largest cut value found was 2.0.
The largest possible cut has size 2.0.
"""

import itertools

import numpy as np
import networkx
import scipy.optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

import time

import cirq

def main(repetitions=1000, maxiter=64, size=3, optimizor='Nelder-Mead'):

    # Set problem parameters
    n=size
    p=size

    # Generate a random 3-regular graph on n nodes
    # graph = networkx.random_regular_graph(3, n)

    # Recreate the minimal 3-node example from class
    graph = networkx.Graph()
    nodeList = list(range(size))
    graph.add_nodes_from(nodeList)
    edgeList = []
    for i in range(size-1):
        edgeList.append((i, i+1))
    graph.add_edges_from(edgeList)
    networkx.draw(graph)
    plt.show()
    # quit()

    # Each node in n nodes of the MAX-CUT graph corresponds to one of n qubits in the quantum circuit.
    # The state vector across the qubits encodes a node partitioning
    qubits = cirq.LineQubit.range(n)

    hadamard_circuit = cirq.Circuit(
        # Prepare uniform superposition
        cirq.H.on_each(*qubits)
    )
    simulator = cirq.Simulator()
    print ("Put the initial state vector in a superposition of all possible node partitionings:")
    print (simulator.simulate(hadamard_circuit, initial_state=0))
    # quit()

    # Print an example circuit
    # Provide classical parameters such that the classical computer can control quantum partitioning
    betas = np.random.uniform(-np.pi, np.pi, size=p)
    gammas = np.random.uniform(-np.pi, np.pi, size=p)
    # betas = [0 for _ in range(p)]
    # gammas = [0 for _ in range(p)]
    circuit = qaoa_max_cut_circuit(qubits, betas, gammas, graph)
    print('Example QAOA circuit:')
    print(circuit.to_text_diagram(transpose=True))
    # quit()

    # Create variables to store the largest cut and cut value found
    largest_cut_found = None
    largest_cut_value_found = 0

    # Initialize simulator
    simulator = cirq.Simulator()

    # For visualizing the optimization iterations
    betas_sched = []
    gammas_sched = []
    meas_prob_dist_sched = []
    cut_mean_sched = []

    # Define objective function (we'll use the negative expected cut value)
    def f(x):
        # Create circuit
        betas = x[:p]
        gammas = x[p:]
        betas_sched.append(betas)
        gammas_sched.append(gammas)

        # Perform a series of operations parameterized by classical parameters
        # such that the final state vector is a superposition of good partitionings
        circuit = qaoa_max_cut_circuit(qubits, betas, gammas, graph)

        # we also want a simulation where the final quantum state vector is not collapsed via measurement
        circuit_no_measurement = qaoa_max_cut_circuit_no_measurement(qubits, betas, gammas, graph)
        final_state = simulator.simulate(circuit_no_measurement).final_state
        print("The final quantum state vector without measurement collapse:")
        print(final_state)
        # quit()

        # from amplitudes to measurement probabilities
        meas_prob_dist_sched.append(np.square(np.absolute(final_state)))

        # Sample bitstrings from circuit
        result = simulator.run(circuit, repetitions=repetitions)
        bitstrings = result.measurements['m']
        # Process bitstrings
        nonlocal largest_cut_found
        nonlocal largest_cut_value_found
        values = cut_values(bitstrings, graph)
        max_value_index = np.argmax(values)
        max_value = values[max_value_index]
        if max_value > largest_cut_value_found:
            largest_cut_value_found = max_value
            largest_cut_found = bitstrings[max_value_index]
        mean = np.mean(values)
        cut_mean_sched.append(mean)

        return -mean

    # Provide classical parameters such that the classical computer can control quantum partitioning
    # Pick an initial guess
    x0 = np.random.uniform(-np.pi, np.pi, size=2 * p)
    # x0 = [0 for _ in range(2 * p)]

    # Optimize for a good set of gammas and betas
    # Optimize f
    print('Optimizing objective function ...')
    scipy.optimize.minimize(f,
                            x0,
                            # method='Nelder-Mead',
                            method=optimizor,
                            options={'maxiter': maxiter})

    # Compute best possible cut value via brute force search
    all_bitstrings = np.array(list(itertools.product(range(2), repeat=n)))
    all_values = cut_values(all_bitstrings, graph)
    max_cut_value = np.max(all_values)

    # Print the results
    print('The largest cut value found was {}.'.format(largest_cut_value_found))
    print('The largest possible cut has size {}.'.format(max_cut_value))

    # Visualize the data
    fig, ((bax,gax),(dax,max)) = plt.subplots(2,2)

    # Create a color coding and legend
    max_opt_iter = len(meas_prob_dist_sched)
    colors = plt.cm.Wistia(np.linspace(0,1,max_opt_iter))
    custom_legend = [Line2D([0], [0], color=colors[0]),
                     Line2D([0], [0], color=colors[-1])]

    # Plot gammas and betas as function of optimization iteration
    qaoa_layers = [i for i in range(p)]
    bax.set_title('Beta optimization')
    bax.set_ylabel('Radians')
    bax.set_xlabel('QAOA layer')
    gax.set_title('Gamma optimization')
    gax.set_ylabel('Radians')
    gax.set_xlabel('QAOA layer')
    for opt_iter, (layer_betas,layer_gammas) in enumerate(zip(betas_sched,gammas_sched)):
        bax.plot(qaoa_layers, layer_betas,  color=colors[opt_iter])
        gax.plot(qaoa_layers, layer_gammas, color=colors[opt_iter])
    bax.legend(custom_legend, ['Begin', 'End'], loc='lower right')
    gax.legend(custom_legend, ['Begin', 'End'], loc='lower right')

    # Plot meas_prob_dist_sched as function of optimization iteration
    basis_states = [i for i in range(1<<n)]
    dax.set_title('Measurement distribution')
    dax.set_ylabel('Measurement probability')
    dax.set_xlabel('Basis state')
    dax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:03b}"))
    dax.xaxis.set_ticks(np.arange(0, 111, 1))
    for opt_iter, histogram in enumerate(meas_prob_dist_sched):
        dax.plot(basis_states, histogram, color=colors[opt_iter])
    dax.legend(custom_legend, ['Begin', 'End'], loc='lower right')

    # Plot mean cut value vs. optimization iteration
    max.set_title('Cut value optimization')
    max.set_ylabel('Cut value')
    max.set_xlabel('Optimization iteration')
    max.plot(cut_mean_sched,color=colors[max_opt_iter//2],label='QAOA mean cut value')
    max.plot([max_cut_value for i in range(max_opt_iter)],color=colors[-1],label='True max-cut value')
    max.legend(loc='lower right')

    plt.tight_layout()
    # print('About to savefig')
    plt.savefig('./2020_672_'+str(size)+'V_'+optimizor+'rep'+str(repetitions)+'.png')
    # plt.savefig('./2020_672_'+str(size)+'V_0Start.png')
    # plt.show()


def rzz(rads):
    """Returns a gate with the matrix exp(-i Z⊗Z rads)."""
    return cirq.ZZPowGate(exponent=2 * rads / np.pi, global_shift=-0.5)


def qaoa_max_cut_unitary(qubits, betas, gammas,
                         graph):  # Nodes should be integers
    for beta, gamma in zip(betas, gammas):
        # Need an operator (quantum gate) that encodes an edge
        yield (
            rzz(-0.5 * gamma).on(qubits[i], qubits[j]) for i, j in graph.edges)
        yield cirq.rx(2 * beta).on_each(*qubits)


def qaoa_max_cut_circuit(qubits, betas, gammas,
                         graph):  # Nodes should be integers
    return cirq.Circuit(
        # Prepare uniform superposition
        cirq.H.on_each(*qubits),
        # Apply QAOA unitary
        qaoa_max_cut_unitary(qubits, betas, gammas, graph),
        # Measure
        cirq.measure(*qubits, key='m'))


def qaoa_max_cut_circuit_no_measurement(qubits, betas, gammas,
                         graph):  # Nodes should be integers
    return cirq.Circuit(
        # Prepare uniform superposition
        cirq.H.on_each(*qubits),
        # Apply QAOA unitary
        qaoa_max_cut_unitary(qubits, betas, gammas, graph))


def cut_values(bitstrings, graph):
    mat = networkx.adjacency_matrix(graph, nodelist=sorted(graph.nodes))
    vecs = (-1)**bitstrings
    vals = 0.5 * np.sum(vecs * (mat @ vecs.T).T, axis=-1)
    vals = 0.5 * (graph.size() - vals)
    return vals


if __name__ == '__main__':
    # t1 = time.time()
    # main(size=3, optimizor='Nelder-Mead')
    # t2 = time.time()
    # main(size=5, optimizor='Nelder-Mead')
    # t3 = time.time()
    # main(size=7, optimizor='Nelder-Mead')
    # t4 = time.time()
    # main(size=9, optimizor='Nelder-Mead')
    # t5 = time.time()
    # main(size=11, optimizor='Nelder-Mead')
    # t6 = time.time()
    # timeDiff1 = [t2-t1, t3-t2, t4-t3, t5-t4, t6-t5]
    # plt.clf()
    # plt.plot([3,5,7,9,11], timeDiff1)
    # plt.title('Simulation Time Over Different Size - Zero Started')
    # plt.xlabel('Vertices Number')
    # plt.ylabel('Simulation Time (S)')
    # plt.savefig('SimulationTime_0Start.png')

    t1 = time.time()
    main(repetitions=30000, size=3, optimizor='Nelder-Mead')
    t2 = time.time()
    main(repetitions=30000, size=5, optimizor='Nelder-Mead')
    t3 = time.time()
    main(repetitions=30000, size=7, optimizor='Nelder-Mead')
    t4 = time.time()
    main(repetitions=30000, size=9, optimizor='Nelder-Mead')
    t5 = time.time()
    main(repetitions=30000, size=11, optimizor='Nelder-Mead')
    t6 = time.time()
    timeDiff1 = [t2-t1, t3-t2, t4-t3, t5-t4, t6-t5]

    t1 = time.time()
    main(repetitions=30000, size=3, optimizor='Powell')
    t2 = time.time()
    main(repetitions=30000, size=5, optimizor='Powell')
    t3 = time.time()
    main(repetitions=30000, size=7, optimizor='Powell')
    t4 = time.time()
    main(repetitions=30000, size=9, optimizor='Powell')
    t5 = time.time()
    main(repetitions=30000, size=11, optimizor='Powell')
    t6 = time.time()
    timeDiff2 = [t2-t1, t3-t2, t4-t3, t5-t4, t6-t5]

    t1 = time.time()
    main(repetitions=30000, size=3, optimizor='CG')
    t2 = time.time()
    main(repetitions=30000, size=5, optimizor='CG')
    t3 = time.time()
    main(repetitions=30000, size=7, optimizor='CG')
    t4 = time.time()
    main(repetitions=30000, size=9, optimizor='CG')
    t5 = time.time()
    main(repetitions=30000, size=11, optimizor='CG')
    t6 = time.time()
    timeDiff3 = [t2-t1, t3-t2, t4-t3, t5-t4, t6-t5]

    t1 = time.time()
    main(repetitions=30000, size=3, optimizor='BFGS')
    t2 = time.time()
    main(repetitions=30000, size=5, optimizor='BFGS')
    t3 = time.time()
    main(repetitions=30000, size=7, optimizor='BFGS')
    t4 = time.time()
    main(repetitions=30000, size=9, optimizor='BFGS')
    t5 = time.time()
    main(repetitions=30000, size=11, optimizor='BFGS')
    t6 = time.time()
    timeDiff4 = [t2-t1, t3-t2, t4-t3, t5-t4, t6-t5]

    plt.clf()
    plt.plot([3,5,7,9,11], timeDiff1, label='Nelder-Mead')
    plt.plot([3,5,7,9,11], timeDiff2, label='Powell')
    plt.plot([3,5,7,9,11], timeDiff3, label='CG')
    plt.plot([3,5,7,9,11], timeDiff4, label='BFGS')
    plt.legend()
    plt.title('Simulation Time Over Different Size - Powell')
    plt.xlabel('Vertices Number')
    plt.ylabel('Simulation Time (S)')
    plt.savefig('SimulationTime_Powell_30000.png')
