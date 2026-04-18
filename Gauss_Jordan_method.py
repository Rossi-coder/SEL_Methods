import numpy as np

def gauss_jordan(A, b):
    n = len(A)
    # Crear matriz aumentada [A|b]
    M = np.hstack((A.astype(float), b.reshape(-1, 1).astype(float)))
    
    for i in range(n):
        # Pivoteo parcial
        max_row = i + np.argmax(np.abs(M[i:, i]))
        M[[i, max_row]] = M[[max_row, i]]
        
        # Hacer el pivote = 1
        M[i] = M[i] / M[i, i]
        
        # Hacer ceros arriba y abajo del pivote
        for j in range(n):
            if i != j:
                M[j] -= M[j, i] * M[i]
                
    return M[:, -1]

# Ejemplo de uso:
# 2x + y - z = 8
# -3x - y + 2z = -11
# -2x + y + 2z = -3

A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
b = np.array([8, -11, -3])

solucion = gauss_jordan(A, b)
print("Solución:", solucion) # Output: [ 2.  3. -1.]
