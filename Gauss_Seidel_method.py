import numpy as np

def gauss_seidel(A, b, x0, tol, max_iter):
    """
    Resuelve el sistema Ax = b usando el método de Gauss-Seidel.
    """
    n = len(A)
    x = x0.copy()
    
    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sumatoria = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x[i] = (b[i] - sumatoria) / A[i][i]
        
        # Verificar convergencia
        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x
        
    return x

# Ejemplo de uso
A = np.array([[4.0, -1.0, -1.0], 
              [-2.0, 6.0, 1.0], 
              [-1.0, 1.0, 7.0]])
b = np.array([3.0, 9.0, -6.0])
x0 = np.zeros(len(b)) # Aproximación inicial
tol = 1e-10
max_iter = 100

solucion = gauss_seidel(A, b, x0, tol, max_iter)
print("Solución:", solucion)
