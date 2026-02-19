import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from numpy.linalg import norm

# ==============================
# 1. Generate linearly separable data
# ==============================

X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)

# Convert labels from {0,1} to {-1,1}
y = np.where(y == 0, -1, 1)

# Different C values
C_values = [0.1, 1, 100]

plt.figure(figsize=(18, 5))

# ==============================
# 2. Loop over different C
# ==============================

for idx, C in enumerate(C_values):

    model = SVC(kernel='linear', C=C)
    model.fit(X, y)

    w = model.coef_[0]
    b = model.intercept_[0]

    w_norm = norm(w)
    margin = 2 / w_norm

    n_support = len(model.support_)

    print(f"\n=== C = {C} ===")
    print(f"w = {w}")
    print(f"||w|| = {w_norm:.4f}")
    print(f"Margin width = {margin:.4f}")
    print(f"Number of support vectors = {n_support}")

    # ==============================
    # 3. Plot decision boundary
    # ==============================

    plt.subplot(1, 3, idx+1)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
    plt.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=150,
        facecolors='none',
        edgecolors='k'
    )

    # Create grid
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # Decision boundary and margins
    plt.contour(XX, YY, Z, levels=[-1, 0, 1],
                linestyles=['--', '-', '--'])

    plt.title(f"C = {C}\n||w|| = {w_norm:.2f}, margin = {margin:.2f}")
    plt.xlabel("x1")
    plt.ylabel("x2")

plt.tight_layout()
plt.show()
