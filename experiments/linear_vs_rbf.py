"""
Linear vs RBF SVM on Nonlinear Data

This experiment demonstrates how kernel functions enable
linear separation in implicit high-dimensional feature space.

We compare:
    - Linear SVM
    - RBF Kernel SVM

on nonlinear data (make_moons).
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==================================
# 1. Generate nonlinear dataset
# ==================================

X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==================================
# 2. Train Linear SVM
# ==================================

linear_model = SVC(kernel='linear', C=10)
linear_model.fit(X_train, y_train)

linear_train_acc = accuracy_score(y_train, linear_model.predict(X_train))
linear_test_acc = accuracy_score(y_test, linear_model.predict(X_test))
linear_sv = len(linear_model.support_)

print("=== Linear SVM ===")
print(f"Train Accuracy: {linear_train_acc:.3f}")
print(f"Test Accuracy:  {linear_test_acc:.3f}")
print(f"Support Vectors: {linear_sv}")

# ==================================
# 3. Train RBF SVM
# ==================================

sigma = 1
gamma = 1 / (2 * sigma**2)

rbf_model = SVC(kernel='rbf', C=10, gamma=gamma)
rbf_model.fit(X_train, y_train)

rbf_train_acc = accuracy_score(y_train, rbf_model.predict(X_train))
rbf_test_acc = accuracy_score(y_test, rbf_model.predict(X_test))
rbf_sv = len(rbf_model.support_)

print("\n=== RBF SVM ===")
print(f"Train Accuracy: {rbf_train_acc:.3f}")
print(f"Test Accuracy:  {rbf_test_acc:.3f}")
print(f"Support Vectors: {rbf_sv}")

# ==================================
# 4. Visualization
# ==================================

plt.figure(figsize=(12, 5))

models = [
    ("Linear SVM", linear_model, linear_test_acc, linear_sv),
    ("RBF SVM", rbf_model, rbf_test_acc, rbf_sv)
]

for idx, (title, model, test_acc, sv_count) in enumerate(models):

    plt.subplot(1, 2, idx + 1)

    # Scatter data
    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=y_train, cmap='bwr', alpha=0.6)

    # Highlight support vectors
    plt.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=120, facecolors='none', edgecolors='k')

    # Create grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0], colors='black')

    plt.title(
        f"{title}\n"
        f"Test Acc={test_acc:.2f}, SV={sv_count}"
    )

plt.tight_layout()
plt.show()
