import numpy as np

def resolver_cholesky(A, b):
    # 1. Verificar si la matriz es simétrica y definida positiva
    if not np.allclose(A, A.T):
        raise ValueError("La matriz A no es simétrica.")
    
    # 2. Descomposición de Cholesky A = L*L^T
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError("La matriz A no es definida positiva.")
        
    # 3. Resolver Ly = b (sustitución hacia adelante)
    y = np.linalg.solve(L, b)
    
    # 4. Resolver L^T*x = y (sustitución hacia atrás)
    x = np.linalg.solve(L.T, y)
    
    return x

# --- Ejemplo de uso ---
# Sistema:
#  4x - 2y + 0z = 10
# -2x + 4y - 2z = 0
#  0x - 2y + 4z = 0
A = np.array([[4.0, -2.0, 0.0],
              [-2.0, 4.0, -2.0],
              [0.0, -2.0, 4.0]])
b = np.array([10.0, 0.0, 0.0])

try:
    solucion = resolver_cholesky(A, b)
    print("La solución es:", solucion)
except ValueError as e:
    print(e)
