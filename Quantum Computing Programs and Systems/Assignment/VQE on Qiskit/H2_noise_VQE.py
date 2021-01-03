# Based on IBM Qiskit community tutorials
# https://qiskit.org/textbook/ch-applications/vqe-molecules.html#Running-VQE-on-a-Noisy-Simulator

from qiskit.aqua.algorithms import VQE, NumPyEigensolver
import matplotlib.pyplot as plt
import numpy as np
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
from qiskit.aqua.operators import Z2Symmetries
from qiskit import IBMQ, BasicAer, Aer
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit import IBMQ
from qiskit.aqua import QuantumInstance
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel

import warnings
warnings.filterwarnings('ignore')

# Perpare the qubit operator representing the molecule's Hamiltonian
driver = PySCFDriver(atom='H .0 .0 -0.3625; H .0 .0 0.3625', unit=UnitsType.ANGSTROM, charge=0, spin=0, basis='sto3g')
molecule = driver.run()
num_particles = molecule.num_alpha + molecule.num_beta
qubitOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals).mapping(map_type='parity')
qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)

# Load a device coupling map and noise model from the IBMQ provider and create a quantum instance
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = Aer.get_backend("qasm_simulator")
device = provider.get_backend("ibmq_vigo")
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device.properties())
quantum_noise_instance = QuantumInstance(backend=backend, 
                                         shots=8192, 
                                         noise_model=noise_model, 
                                         coupling_map=coupling_map,
                                         measurement_error_mitigation_cls=CompleteMeasFitter,
                                         cals_matrix_refresh_period=30)
quantum_non_noise_instance = QuantumInstance(backend=backend, 
                                             shots=8192, 
                                             coupling_map=coupling_map,
                                             measurement_error_mitigation_cls=CompleteMeasFitter,
                                             cals_matrix_refresh_period=30)

# Configure the optimizer, the variational form, and the VQE instance
exact_solution = NumPyEigensolver(qubitOp).run()
print("Exact Result:", np.real(exact_solution.eigenvalues) + molecule.nuclear_repulsion_energy)
optimizer = SPSA(maxiter=100)
var_form = EfficientSU2(qubitOp.num_qubits, entanglement="linear")
vqe = VQE(qubitOp, var_form, optimizer=optimizer)
noise_ret = vqe.run(quantum_noise_instance)
non_noise_ret = vqe.run(quantum_non_noise_instance)
noise_vqe_result = np.real(noise_ret['eigenvalue']+ molecule.nuclear_repulsion_energy)
non_noise_vqe_result = np.real(non_noise_ret['eigenvalue']+ molecule.nuclear_repulsion_energy)
print("VQE Result with noise:", noise_vqe_result)
print("VQE Result without noise:", non_noise_vqe_result)