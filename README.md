# Quantum ML Bloch Sphere Simulator


**A Quantum Machine Learning project to predict qubit states and visualize them on the Bloch Sphere.**

---

## Description
This project combines **Quantum Computing** and **Machine Learning** to simulate qubit state evolution and predict the Bloch vector using an MLP model. Users can input sequences of quantum gates, and the program shows:

- True Bloch vector of the qubit (using Qiskit)
- Predicted Bloch vector (using a trained MLP model)
- Probabilities of |0> and |1>
- Animated GIF of the Bloch Sphere evolution

---

## Features
- Interactive command-line interface to apply gates (`H, X, Y, Z, S, T, RX, RY, RZ`)
- Predicts qubit states using a pre-trained ML model
- Generates visualizations comparing true vs predicted states
- Creates GIF animations for step-by-step evolution
- Fully reproducible dataset generation and training pipeline

---

## Tech Stack
- **Python 3.10+**
- **Qiskit** for quantum circuits & statevector simulation
- **NumPy** for numerical computation
- **Matplotlib** for visualization
- **ImageIO** for GIF creation
- **Scikit-learn** for MLP regression
- **Joblib** for model saving/loading

---
![bloch_animation](https://github.com/user-attachments/assets/1543ed95-b990-4a90-97ee-37c59e70ac3b)
