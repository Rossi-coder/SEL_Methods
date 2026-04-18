import numpy as np
from scipy.linalg import lu

# Definir la matriz A y el vector b
A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]])
b = np.array([4, 10, 24])

# Descomposición LU (con pivoteo)
# P @ A = L @ U
P, L, U = lu(A)

# 1. Resolver Ly = Pb (Sustitución hacia adelante)
y = np.linalg.solve(L, P.T @ b)

# 2. Resolver Ux = y (Sustitución hacia atrás)
x = np.linalg.solve(U, y)

print("Matriz L:\n", L)
print("Matriz U:\n", U)
print("Solución x:", x)
