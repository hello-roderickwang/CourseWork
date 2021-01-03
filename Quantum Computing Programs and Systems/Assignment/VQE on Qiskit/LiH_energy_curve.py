# Based on IBM Qiskit community tutorials
# https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/chemistry/h2_vqe_initial_point.ipynb

import numpy as np
import matplotlib.pyplot as plt

from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.circuit.library import TwoLocal
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE

from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.optimizers import SPSA

import warnings
warnings.filterwarnings('ignore')

# Specify nuclei positions and species (atoms)
atoms = 'Li .0 .0 .0; H .0 .0 {0}'
algorithms = [{'name': 'VQE'},
              {'name': 'NumPyMinimumEigensolver'}]
titles= ['VQE + Initial Point', 'NumPyMinimumEigensolver']

start = 1.2  # Minimum atomic separation to try
by    = 0.6  # Total range of atomic separations to try
steps = 10   # Number of steps to increase by
trials = 2048
energies = np.empty([len(algorithms), steps+1])
hf_energies = np.empty(steps+1)
distances = np.empty(steps+1)

print('Processing step __', end='')
for i in range(steps+1):
    print('\b\b{:2d}'.format(i), end='', flush=True)
    d = start + i*by/steps

    # Given a atomic geometry specification, obtain molecular Hamiltonian
    driver = PySCFDriver(atom=atoms.format(d),
                         unit=UnitsType.ANGSTROM,
                         basis='sto3g')
    molecule = driver.run()

    # Build the qubit operator, which is the input to the VQE algorithm in Aqua
    operator =  Hamiltonian(
        transformation=TransformationType.FULL,
        qubit_mapping=QubitMappingType.PARITY, # Other choices: JORDAN_WIGNER, BRAVYI_KITAEV
        two_qubit_reduction=True,
        freeze_core=True,
        orbital_reduction=[-3, -2])
    qubit_op, _ = operator.run(molecule)

    for j in range(len(algorithms)):

        if algorithms[j]['name'] == 'NumPyMinimumEigensolver':
            result = NumPyMinimumEigensolver(qubit_op).run()
        else:
            # Choice of classical optimizer
            optimizer = SPSA(max_trials=trials)
            # Choice of ansatz
            var_form = TwoLocal(qubit_op.num_qubits, ['ry', 'rz'], 'cz', reps=3, entanglement='full')
            algo = VQE(qubit_op, var_form, optimizer, max_evals_grouped=1)
            result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
            if j == 0:
                algorithms[j]['initial_point'] = result.optimal_point.tolist()

        result = operator.process_algorithm_result(result)
        energies[j][i] = result.energy
        hf_energies[i] = result.hartree_fock_energy

    distances[i] = d
print(' --- complete')

print('Distances: ', distances)
print('Energies:', energies)
print('Hartree-Fock energies:', hf_energies)

plt.plot(distances, hf_energies, label='Hartree-Fock')
for j in range(len(algorithms)):
    plt.plot(distances, energies[j], label=titles[j])
plt.xlabel('Interatomic distance')
plt.ylabel('Energy')
plt.title('LiH Ground State Energy')
plt.legend(loc='upper right')
fileName = "LiH_Ground_State_Energy_start" + str(start) + "by" + str(by) + "steps" + str(steps) + "-trials" + str(trials) + ".png"
plt.savefig(fileName)
