# Author: Antigravity AI
# Trains a lightweight MLP on hand landmark data (.npz) with centering and scaling normalization.
# Exports weights as JSON to be embedded directly into JavaScript.

import numpy as np
import json
import os

print("--- TRAINING MLP ON LANDMARK DATA ---")

# Classes and NPZ file mappings
class_names = ['closed', 'three', 'open', 'zero']
files = [
    'data_hand_closed.npz',
    'data_hand_three.npz',
    'data_hand_open.npz',
    'data_hand_zero.npz'
]

X_list = []
y_list = []

for i, f_name in enumerate(files):
    if not os.path.exists(f_name):
        print(f"Error: {f_name} not found in the workspace!")
        exit(1)
    data = np.load(f_name)
    X_list.append(data['X'])
    y_list.append(data['y'])

X = np.vstack(X_list)
y = np.concatenate(y_list)

print(f"Loaded {len(X)} total samples.")

# Preprocessing function: Center coordinates relative to the wrist (landmark 0)
# and scale relative to the maximum distance from wrist to any landmark.
# This makes the classification 100% invariant to translation and scale!
def preprocess_samples(X_data):
    X_proc = []
    for sample in X_data:
        # Each sample has 42 elements: x0, y0, x1, y1, ...
        coords = sample.reshape(21, 2)
        wrist = coords[0]
        # Center all landmarks relative to the wrist
        centered = coords - wrist
        # Calculate Euclidean distances from centered wrist
        dists = np.linalg.norm(centered, axis=1)
        max_dist = np.max(dists)
        if max_dist > 0:
            centered /= max_dist
        X_proc.append(centered.flatten())
    return np.array(X_proc)

X_norm = preprocess_samples(X)
print("Landmarks centered and scaled successfully.")

# One-hot encode labels for cross-entropy
num_classes = len(class_names)
y_oh = np.eye(num_classes)[y]

# Shuffle and split into Train (80%) and Test (20%)
indices = np.arange(len(X_norm))
np.random.seed(42)
np.random.shuffle(indices)
X_norm = X_norm[indices]
y_oh = y_oh[indices]
y = y[indices]

split = int(0.8 * len(X_norm))
X_train, X_test = X_norm[:split], X_norm[split:]
y_train_oh, y_test_oh = y_oh[:split], y_oh[split:]
y_train, y_test = y[:split], y[split:]

# Neural Network architecture
input_size = 42
hidden_size = 32
epochs = 600
learning_rate = 0.05

print(f"Training MLP: Input={input_size}, Hidden={hidden_size}, Output={num_classes}")

# Initialize weights (He initialization for ReLU)
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, num_classes) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, num_classes))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # Stabilized softmax
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Training loop
for epoch in range(epochs):
    # Forward Pass
    z1 = np.dot(X_train, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    probs = softmax(z2)
    
    # Loss computation (Categorical Cross Entropy)
    loss = -np.mean(np.log(probs[np.arange(len(y_train)), y_train] + 1e-15))
    
    # Backpropagation
    dz2 = (probs - y_train_oh) / len(y_train)
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * (z1 > 0)
    dW1 = np.dot(X_train.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    # Gradient Descent Updates
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    # Evaluation every 50 epochs
    if epoch % 50 == 0 or epoch == epochs - 1:
        # Run test set
        test_z1 = np.dot(X_test, W1) + b1
        test_a1 = relu(test_z1)
        test_z2 = np.dot(test_a1, W2) + b2
        test_probs = softmax(test_z2)
        test_preds = np.argmax(test_probs, axis=1)
        test_acc = np.mean(test_preds == y_test)
        
        # Run train set accuracy
        train_preds = np.argmax(probs, axis=1)
        train_acc = np.mean(train_preds == y_train)
        
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# Final validation on complete dataset
final_z1 = np.dot(X_norm, W1) + b1
final_a1 = relu(final_z1)
final_z2 = np.dot(final_a1, W2) + b2
final_probs = softmax(final_z2)
final_preds = np.argmax(final_probs, axis=1)
overall_acc = np.mean(final_preds == y)
print(f"\nFinal Overall Accuracy on complete dataset: {overall_acc * 100:.2f}%")

# Save weights to JSON
weights = {
    "W1": W1.tolist(),
    "b1": b1.flatten().tolist(),
    "W2": W2.tolist(),
    "b2": b2.flatten().tolist()
}

output_file = "model_weights.json"
with open(output_file, "w") as f:
    json.dump(weights, f, indent=2)

print(f"Saved weights to '{output_file}' successfully!")
