"""
RBF Kernel Sigma Structural Analysis

This experiment studies how the RBF kernel width (σ) affects:

1. Decision boundary smoothness
2. Support vector distribution
3. Overfitting vs underfitting behavior

Support vectors are defined as samples with α_i > 0.
We further distinguish:
    - 0 < α_i < C  (margin-bound vectors)
    - α_i = C      (slack-bound vectors)
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# 1. Generate nonlinear data
# =========================

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# 2. Experiment settings
# =========================

sigmas = [0.05, 0.2, 1, 5]   # small, medium, large
C = 10
eps = 1e-5

plt.figure(figsize=(24, 5))

# =========================
# 3. Loop over σ values
# =========================

for idx, sigma in enumerate(sigmas):

    gamma = 1 / (2 * sigma**2)

    model = SVC(kernel='rbf', C=C, gamma=gamma)
    model.fit(X_train, y_train)

    # =========================
    # 4. Structural Metrics
    # =========================

    # Dual coefficients (absolute values)
    alphas = np.abs(model.dual_coef_[0])

    # Support vector categories
    total_sv = np.sum(alphas > eps)
    alpha_margin = np.sum((alphas > eps) & (alphas < C - eps))
    alpha_C = np.sum(alphas >= C - eps)

    # Accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"\n========== σ = {sigma} ==========")
    print(f"Gamma = {gamma:.4f}")
    print(f"Total Support Vectors (α > 0): {total_sv}")
    print(f"  0 < α < C (margin-bound): {alpha_margin}")
    print(f"  α = C (slack-bound): {alpha_C}")
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")

    # =========================
    # 5. Visualization
    # =========================

    plt.subplot(1, len(sigmas), idx + 1)

    # Plot training points
    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=y_train, cmap='bwr', alpha=0.6)

    # Highlight support vectors
    plt.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=120, facecolors='none', edgecolors='k')

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid)
    Z = Z.reshape(xx.shape)

    # Decision boundary
    plt.contour(xx, yy, Z, levels=[0], colors='black')

    plt.title(
        f"σ = {sigma}\n"
        f"SV={total_sv}, Test={test_acc:.2f}"
    )

plt.tight_layout()
plt.show()
