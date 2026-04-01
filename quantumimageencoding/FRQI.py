from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import RYGate
from quantumimageencoding.BaseQuantumEncoder import QuantumEncoder
# qiskit-aer is now a separate package; Aer and execute moved there
from qiskit_aer import Aer
from qiskit_aer.primitives import Sampler
from qiskit import transpile
from PIL import Image
import numpy

class FRQI(QuantumEncoder):
    def __init__(self):
        super().__init__()

    def encode(self, image : Image.Image) -> None :
        '''
        This function encodes the corresponding angle values of each element
        onto the Quantum Circuit.
        '''
        angles_h = numpy.arcsin(numpy.array(image))
        angles_h = angles_h.reshape(angles_h.shape[0]**2)
        angles_v = numpy.arcsin(numpy.array(image).T)
        angles_v = angles_v.reshape(angles_v.shape[0]**2)

        controlbits = int(numpy.log2(angles_h.shape[0]))
        positions1 = QuantumRegister(controlbits, 'position')
        target1 = QuantumRegister(1, 'target')
        classical1 = ClassicalRegister(controlbits+1, 'measure')
        positions2 = QuantumRegister(controlbits, 'position')
        target2 = QuantumRegister(1, 'target')
        classical2 = ClassicalRegister(controlbits+1, 'measure')
        self.Qcirc = self.createQuantumCircuit(target1, positions1, classical1)
        self.Qcirc2 = self.createQuantumCircuit(target2, positions2, classical2)

        #FRQI process starts here
        for i in range(1, controlbits+1):
            self.Qcirc.h(i)
            self.Qcirc2.h(i)

        j = 0
        for i in angles_h:
            state = '{0:0{1}b}'.format(j - 1, controlbits)
            new_state = '{0:0{1}b}'.format(j, controlbits)
            if j != 0:
                c = numpy.array([])
                for k in range(controlbits):
                    if state[k] != new_state[k]:
                        c = numpy.append(c, int(k))
                if len(c) > 0:
                    self.Qcirc.x(numpy.abs(c.astype(int) - (controlbits)))
                cry = RYGate(2 * i).control(controlbits)
                aux = numpy.append([k+1 for k in range(controlbits)], 0).tolist()
                self.Qcirc.append(cry, aux)
            j += 1

        self.Qcirc.measure(0, 0)
        # c_if deprecated: use IfElseOp equivalent via with statement
        with self.Qcirc.if_test((classical1, 0)):
            self.Qcirc.x(target1[0])

        j = 0
        for i in angles_v:
            state = '{0:0{1}b}'.format(j - 1, controlbits)
            new_state = '{0:0{1}b}'.format(j, controlbits)
            if j != 0:
                c = numpy.array([])
                for k in range(controlbits):
                    if state[k] != new_state[k]:
                        c = numpy.append(c, int(k))
                if len(c) > 0:
                    self.Qcirc2.x(numpy.abs(c.astype(int) - (controlbits)))
                cry = RYGate(2 * i).control(controlbits)
                aux = numpy.append([k+1 for k in range(controlbits)], 0).tolist()
                self.Qcirc2.append(cry, aux)
            j += 1

        self.Qcirc2.measure(0, 0)
        with self.Qcirc2.if_test((classical2, 0)):
            self.Qcirc2.x(target2[0])

        unitaryMatrix = numpy.identity(2**(controlbits+1))
        unitaryMatrix = numpy.roll(unitaryMatrix,1,axis=1)
        self.Qcirc.h(0)
        self.Qcirc.unitary(unitaryMatrix, range(controlbits+1))
        self.Qcirc.h(0)
        self.Qcirc2.h(0)
        self.Qcirc2.unitary(unitaryMatrix, range(controlbits+1))
        self.Qcirc2.h(0)

        return self.Qcirc

    def detectEdges(self) -> Image.Image:
        back = Aer.get_backend('statevector_simulator')
        # execute() removed in Qiskit 1.0; use backend.run() instead
        qc1 = transpile(self.Qcirc, back)
        qc2 = transpile(self.Qcirc2, back)
        state_vector_h = back.run(qc1).result().get_statevector(qc1)
        state_vector_v = back.run(qc2).result().get_statevector(qc2)
        size = int(2**((self.Qcirc.num_qubits-1)/2))
        threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)
        h_edge_scan_img = numpy.abs(numpy.array([1 if threshold(state_vector_h[(2*i)+1].real) else 0 for i in range(2**(self.Qcirc.num_qubits-1))])).reshape(size, size)
        v_edge_scan_img = numpy.abs(numpy.array([1 if threshold(state_vector_v[(2*i)+1].real) else 0 for i in range(2**(self.Qcirc2.num_qubits-1))])).reshape(size, size).T
        edge_scan_image = h_edge_scan_img | v_edge_scan_img
        return edge_scan_image, h_edge_scan_img, v_edge_scan_img
