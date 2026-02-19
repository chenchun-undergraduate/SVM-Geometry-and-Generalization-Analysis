# SVM Geometry and Generalization Analysis

## Project Overview

This project investigates the geometric and optimization behavior of Support Vector Machines (SVM), extending theoretical derivations of the primal–dual formulation into empirical validation.

Rather than focusing on benchmark accuracy, this project explores how:

- Margin maximization relates to minimizing ||w||
- Soft-margin regularization (C) reshapes decision boundaries
- Kernel parameters (γ in RBF) control locality and curvature
- Support vectors determine the classifier structure
- Kernel mapping enables linear separation in implicit high-dimensional space

The goal is to bridge convex optimization theory with observable decision geometry.



## Motivation

In the theoretical formulation of SVM, the optimization problem:

min (1/2)||w||² + C*Σξi

is transformed into a dual maximization problem involving Lagrange multipliers αi>0 and μi>0.

While mathematically elegant, the geometric implications of:

- ||w||
- Slack variables
- Kernel functions
- Hyperparameter bounds 0 ≤ αi ≤ C

are not always visually intuitive.

This project provides controlled experiments to make these structural properties observable.



## Experimental Design Philosophy

All primary experiments use low-dimensional (2D) synthetic datasets to allow:

- Direct visualization of decision boundaries
- Explicit margin inspection
- Support vector identification
- Curvature analysis in nonlinear settings

The objective is structural understanding, not raw performance comparison.

A small high-dimensional experiment is included to verify generalization behavior beyond visualization.


## Experiments

### 1. Hard Margin vs Soft Margin (Effect of C)

We vary the regularization parameter C while keeping the dataset fixed.

We analyze:

- Margin width
- Boundary sharpness
- Number of support vectors
- Sensitivity to margin violations

#### Visual Evidence
<img src="visualization/linear SVM C=0.1.png" width="700">
<img src="visualization/linear SVM C=1.png" width="700">
<img src="visualization/linear SVM C=100.png" width="700">

#### Observations

- Small C → wider margin, higher tolerance
- Large C → narrower margin, sharper separation

This demonstrates the trade-off between geometric simplicity and classification rigidity.

#### Reproducibility
See implementation:
[linear_margin_c_analysis.py](experiments/linear_margin_c_analysis.py)



### 2. Kernel Parameter Behavior (RBF γ)

Using nonlinear datasets (e.g., moons), we vary γ while fixing C.

We analyze:

- Decision boundary smoothness
- Local vs global similarity influence
- Support vector distribution
- Overfitting vs underfitting geometry

Findings:

- Small γ → smooth, globally influenced boundary
- Large γ → highly localized boundary, increased curvature

This empirically validates how kernel parameters reshape feature-space geometry.



### 3. Linear vs Kernel SVM on Nonlinear Data

We compare:

- Linear SVM
- RBF Kernel SVM

on data that is not linearly separable in input space.

The experiment demonstrates how kernel functions implicitly construct higher-dimensional feature spaces without explicitly computing φ(x).



### 4. High-Dimensional Validation (Optional Extension)

To verify that geometric insights extend beyond visualization:

- A high-dimensional dataset (e.g., sklearn digits) is used.
- Linear and RBF SVMs are compared.
- Support vector counts and generalization behavior are analyzed.

This connects geometric intuition with practical high-dimensional performance.



## Key Structural Insights

From these experiments, several core properties become clear:

- The decision boundary depends only on support vectors.
- Dual optimization naturally induces sparsity.
- Hyperparameters directly influence geometric complexity.
- Margin maximization corresponds to minimizing ||w||.
- Kernel functions reshape similarity structure rather than simply adding nonlinearity.



## Technical Stack

- Python
- NumPy
- scikit-learn (SVC)
- Matplotlib

The implementation emphasizes clarity of analysis rather than framework complexity.



## Repository Structure

```text
SVM-Geometry-Analysis/
│
├── experiments/
│   ├── linear_margin_c_analysis.py
│   ├── rbf_gamma_analysis.py
│   ├── nonlinear_comparison.py
│   ├── high_dimensional_validation.py
│
├── plots/
│
└── README.md
```

## Broader Perspective

Although deep learning dominates many large-scale applications, SVM remains highly relevant in:

- Small-sample regimes
- High-dimensional feature spaces
- Interpretable classification
- Hybrid deep-feature + classical classifier systems

This project emphasizes understanding inductive bias and geometric structure rather than treating SVM as a black-box model.



## Author

Chun Chen  
B.S. Computer Science  
Research Interest: Machine Learning Theory, Optimization, and Applied AI Systems
