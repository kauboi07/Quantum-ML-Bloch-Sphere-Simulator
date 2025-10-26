import os
import numpy as np
import imageio
from IPython.display import Image, display
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
from qiskit.visualization.bloch import Bloch
import matplotlib.pyplot as plt
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "bloch_sphere_output")
gif_filename = os.path.join(output_dir, "bloch_animation.gif")
frame_duration = 2000
os.makedirs(output_dir, exist_ok=True)

model = joblib.load(os.path.join(script_dir, "bloch_predictor.pkl"))

gates = {
    "H": lambda qc: qc.h(0),
    "X": lambda qc: qc.x(0),
    "Y": lambda qc: qc.y(0),
    "Z": lambda qc: qc.z(0),
    "S": lambda qc: qc.s(0),
    "T": lambda qc: qc.t(0),
    "RX": lambda qc, theta: qc.rx(theta, 0),
    "RY": lambda qc, theta: qc.ry(theta, 0),
    "RZ": lambda qc, theta: qc.rz(theta, 0)
}

#encoding for ML
GATES_LIST = ["PAD","H","X","Y","Z","S","T","RX","RY","RZ"]
GATE_TO_IDX = {g:i for i,g in enumerate(GATES_LIST)}
L = 6

def encode_sequence(sequence, L=L):
    G = len(GATES_LIST)
    one_hot = np.zeros((L,G))
    angles = np.zeros((L,1))
    for i in range(L):
        if i < len(sequence):
            gate,param = sequence[i]
            idx = GATE_TO_IDX.get(gate,0)
            one_hot[i,idx]=1.0
            if gate in ("RX","RY","RZ") and param is not None:
                angles[i,0] = (param % 360)/180.0 -1.0
        else:
            one_hot[i,0]=1.0
    return np.concatenate([one_hot.flatten(), angles.flatten()])


def save_bloch_probs(step_num, description, qc, current_sequence):
    sv = Statevector.from_instruction(qc)
    vec_true = [
        float(np.real(sv.expectation_value(Pauli("X")))),
        float(np.real(sv.expectation_value(Pauli("Y")))),
        float(np.real(sv.expectation_value(Pauli("Z"))))
    ]

    prob_true = [(1+vec_true[2])/2, (1-vec_true[2])/2]

    #machinelearning shi
    feat = encode_sequence(current_sequence, L).reshape(1,-1)
    vec_pred = model.predict(feat)[0]
    prob_pred = [(1+vec_pred[2])/2, (1-vec_pred[2])/2]

    
    fig = plt.figure(figsize=(8,4))

    #bloch sphere
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    b = Bloch(axes=ax1)
    b.add_vectors(vec_true)
    b.vector_color[-1]='b'
    b.add_vectors(vec_pred)
    b.vector_color[-1]='r'
    b.render()
    ax1.set_title(f"Step {step_num}: {description}\nBlue=True, Red=ML")

    #probablities
    ax2 = fig.add_subplot(1,2,2)
    ax2.bar([0,1], prob_true, width=0.4, color='b', label='True')
    ax2.bar([0.4,1.4], prob_pred, width=0.4, color='r', label='Predicted')
    ax2.set_xticks([0.2,1.2])
    ax2.set_xticklabels(['|0>','|1>'])
    ax2.set_ylim(0,1)
    ax2.set_ylabel('Probability')
    ax2.set_title("Probabilities")
    ax2.legend()

    #image
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir,f"step_{step_num:02d}.png")
    plt.savefig(filename)
    plt.close(fig)
    display(Image(filename=filename))
    print(f"Step {step_num}: {description} | True probs: {prob_true} | Predicted probs: {prob_pred}")


qc = QuantumCircuit(1)
current_sequence=[]
step_num=0
save_bloch_probs(step_num,"Initial |0>",qc,current_sequence)

#input loop
step_num=1
while True:
    print("\nAvailable gates: H, X, Y, Z, S, T, RX, RY, RZ")
    print("Type 'done' to finish and create GIF.")
    gate_choice = input("Enter gate: ").strip().upper()
    if gate_choice=="DONE": break
    elif gate_choice in ["RX","RY","RZ"]:
        try: theta = float(input("Enter rotation angle in degrees: "))
        except: print("Invalid angle"); continue
        gates[gate_choice](qc,np.deg2rad(theta))
        current_sequence.append((gate_choice,theta))
        save_bloch_probs(step_num,f"{gate_choice}({theta:.1f}°)",qc,current_sequence)
    elif gate_choice in gates:
        gates[gate_choice](qc)
        current_sequence.append((gate_choice,None))
        save_bloch_probs(step_num,gate_choice,qc,current_sequence)
    else:
        print("Invalid choice"); continue
    step_num+=1

#gif section
images=[]
png_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
for f in png_files:
    path = os.path.join(output_dir,f)
    if os.path.exists(path): images.append(imageio.imread(path))
if images:
    imageio.mimsave(gif_filename,images,duration=frame_duration)
    print(f"GIF saved as {gif_filename} with {len(images)} frames.")
else:
    print("No images found to create GIF.")
