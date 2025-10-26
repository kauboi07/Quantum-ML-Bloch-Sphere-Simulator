
GATES = ["PAD", "H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ"]  
GATE_TO_IDX = {g: i for i, g in enumerate(GATES)}

L = 6  

print("G (vocab size):", len(GATES))
print("L (max seq length):", L)
print("GATE_TO_IDX:", GATE_TO_IDX)


import numpy as np
one_hot_H = np.zeros(len(GATES), dtype=int)
one_hot_H[GATE_TO_IDX["H"]] = 1
print('"H" one-hot:', one_hot_H)
import numpy as np

def encode_sequence(sequence, L=L):
    
    G = len(GATES)
    one_hot = np.zeros((L, G))
    angles  = np.zeros((L, 1))  

    for i in range(L):
        if i < len(sequence):
            gate, param = sequence[i]
            idx = GATE_TO_IDX.get(gate, 0)   
            one_hot[i, idx] = 1.0
            if gate in ("RX","RY","RZ") and param is not None:
                
                angles[i,0] = (param % 360)/180.0 - 1.0
        else:
            one_hot[i,0] = 1.0  

   
    return np.concatenate([one_hot.flatten(), angles.flatten()])


# quick test
seq = [("H", None), ("RY", 90.0), ("Z", None)]
encoded = encode_sequence(seq, L)
print("Encoded vector length:", len(encoded))
print("First 20 numbers:", encoded[:20])
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
import numpy as np

def bloch_from_sequence(sequence):
    
    qc = QuantumCircuit(1)
    for gate, param in sequence:
        if gate == "H":
            qc.h(0)
        elif gate == "X":
            qc.x(0)
        elif gate == "Y":
            qc.y(0)
        elif gate == "Z":
            qc.z(0)
        elif gate == "S":
            qc.s(0)
        elif gate == "T":
            qc.t(0)
        elif gate in ("RX", "RY", "RZ"):
            theta = np.deg2rad(param) 
            if gate == "RX":
                qc.rx(theta, 0)
            elif gate == "RY":
                qc.ry(theta, 0)
            elif gate == "RZ":
                qc.rz(theta, 0)
        

    sv = Statevector.from_instruction(qc)
    x = float(np.real(sv.expectation_value(Pauli("X"))))
    y = float(np.real(sv.expectation_value(Pauli("Y"))))
    z = float(np.real(sv.expectation_value(Pauli("Z"))))
    return np.array([x, y, z])

# quick test
seq = [("H", None), ("RY", 90.0)]
bloch_vec = bloch_from_sequence(seq)
print("Bloch vector:", bloch_vec)
import random
import numpy as np

def random_sequence(max_len=L, p_param=0.4):
    
    gates_simple = ["H","X","Y","Z","S","T"]
    gates_param  = ["RX","RY","RZ"]
    length = random.randint(1, max_len)
    seq = []
    for _ in range(length):
        if random.random() < p_param:
            g = random.choice(gates_param)
            angle = random.uniform(0, 360)  
            seq.append((g, angle))
        else:
            seq.append((random.choice(gates_simple), None))
    return seq


N = 2000  # no. of training samples
X = []
Y = []

for _ in range(N):
    seq = random_sequence(max_len=L, p_param=0.4)
    feat = encode_sequence(seq, L)       
    vec  = bloch_from_sequence(seq)      
    X.append(feat)
    Y.append(vec)

X = np.array(X)
Y = np.array(Y)

print("Dataset shapes:")
print("X:", X.shape)  
print("Y:", Y.shape)  
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    max_iter=1000,
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Test MSE:", mse)

import joblib
joblib.dump(model, "bloch_predictor.pkl")

