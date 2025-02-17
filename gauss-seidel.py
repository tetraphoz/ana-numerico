def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    """
    Método de Gauss-Seidel para resolver el sistema de ecuaciones Ax = b.

    Parámetros:
    A : lista de listas (matriz de coeficientes)
    b : lista (vector de términos independientes)
    tol : float (tolerancia de convergencia)
    max_iter : int (máximo número de iteraciones)

    Retorna:
    x : lista (vector solución)
    """
    n = len(A)  # Número de ecuaciones
    x = [0.0] * n  # Vector inicial en ceros

    for it in range(max_iter):
        x_old = x[:]  # Copia de x para comparar cambios

        for i in range(n):
            suma1 = sum(A[i][j] * x[j] for j in range(i))  # Parte con valores nuevos
            suma2 = sum(
                A[i][j] * x_old[j] for j in range(i + 1, n)
            )  # Parte con valores viejos

            x[i] = (b[i] - suma1 - suma2) / A[i][i]  # Nueva aproximación

        # Criterio de parada: verificar la diferencia máxima entre iteraciones
        error = max(abs(x[i] - x_old[i]) for i in range(n))
        if error < tol:
            print(f"Convergió en {it+1} iteraciones.")
            return x

    print("No convergió en el número máximo de iteraciones.")
    return x


# EJEMPLO DE USO
A = [[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]]

b = [15, 10, 10, 10]

solucion = gauss_seidel(A, b)
print("Solución:", solucion)
