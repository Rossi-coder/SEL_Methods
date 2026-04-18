import numpy as np

def jacobi(A, b, x0, tol, max_iterations):
    """
    Resuelve el sistema Ax = b usando el método de Jacobi.
    """
    n = len(A)
    x = x0.copy()
    x_new = np.zeros_like(x)
    
    for k in range(max_iterations):
        for i in range(n):
            # Fórmula de Jacobi: x_i = (b_i - sum(A_ij * x_j)) / A_ii
            s = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - s) / A[i][i]
        
        # Verificar convergencia
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k+1
        
        x = x_new.copy()
        
    return x, max_iterations

# --- Ejemplo de uso ---
# Sistema:
# 4x1 + x2 + 2x3 = 4
# 3x1 + 5x2 + x3 = 7
# x1 + x2 + 3x3 = 3

A = np.array([[4.0, 1.0, 2.0],
              [3.0, 5.0, 1.0],
              [1.0, 1.0, 3.0]])
b = np.array([4.0, 7.0, 3.0])
x0 = np.zeros(len(b)) # Aproximación inicial
tol = 1e-10
max_iter = 100

solucion, iteraciones = jacobi(A, b, x0, tol, max_iter)

print(f"Solución: {solucion}")
print(f"Iteraciones: {iteraciones}")
