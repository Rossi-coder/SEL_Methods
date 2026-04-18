import numpy as np

def gauss_simple(A, b):
    n = len(b)
    # Matriz aumentada [A|b]
    M = np.array(np.hstack((A, b.reshape(-1, 1))), dtype=float)

    # Eliminación hacia adelante
    for i in range(n):
        # Pivoteo (opcional para gauss simple, pero recomendado)
        # Se asume M[i,i] no es cero
        for j in range(i + 1, n):
            factor = M[j, i] / M[i, i]
            M[j, i:] -= factor * M[i, i:]

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]
    return x

# Ejemplo de uso
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

solucion = gauss_simple(A, b)
print("Solución:", solucion)
