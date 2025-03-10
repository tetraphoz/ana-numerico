def gauss_seidel(A, b, tol=1e-6, max_iter=1000):
    """
    Método de Gauss-Seidel para resolver el sistema de ecuaciones Ax = b.

    Parámetros:
    A : lista de listas (matriz de coeficientes)
    b : lista (vector de términos independientes)
    tol : float (tolerancia de convergencia)
    max_iter : int (máximo número de iteraciones)

    Retorna:
    x : lista (vector solución)

    Raises:
    ValueError: Si la matriz no es cuadrada, si las dimensiones no coinciden,
               si hay ceros en la diagonal principal o si los parámetros no son válidos
    TypeError: Si los tipos de datos de entrada no son correctos
    """
    # Validación de tipos
    if not isinstance(A, list) or not all(isinstance(row, list) for row in A):
        raise TypeError("A debe ser una lista de listas")
    if not isinstance(b, list):
        raise TypeError("b debe ser una lista")
    if not isinstance(tol, (int, float)) or not isinstance(max_iter, int):
        raise TypeError("tol debe ser un número y max_iter debe ser un entero")

    # Validación de dimensiones
    n = len(A)
    if n == 0:
        raise ValueError("La matriz A está vacía")
    if not all(len(row) == n for row in A):
        raise ValueError("La matriz A debe ser cuadrada")
    if len(b) != n:
        raise ValueError("Las dimensiones de A y b no coinciden")

    # Validación de valores
    if max_iter <= 0:
        raise ValueError("max_iter debe ser positivo")
    if tol <= 0:
        raise ValueError("tol debe ser positivo")

    # Validación de diagonal dominante
    for i in range(n):
        if A[i][i] == 0:
            raise ValueError(f"No puede haber ceros en la diagonal principal: {A}")

        # Advertencia si la matriz no es diagonalmente dominante
        if abs(A[i][i]) <= sum(abs(A[i][j]) for j in range(n) if j != i):
            print(
                "Advertencia: La matriz no es diagonalmente dominante. La convergencia no está garantizada."
            )

    # Validación de números válidos en A y b
    try:
        for row in A:
            for elem in row:
                float(elem)
        for elem in b:
            float(elem)
    except (ValueError, TypeError):
        raise ValueError("Todos los elementos deben ser números válidos")

    # Inicialización
    x = [0.0] * n

    # Algoritmo principal
    for it in range(max_iter):
        x_old = x[:]
        for i in range(n):
            suma1 = sum(A[i][j] * x[j] for j in range(i))
            suma2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - suma1 - suma2) / A[i][i]

        # Cálculo del error
        try:
            error = max(abs(x[i] - x_old[i]) for i in range(n))
        except OverflowError:
            raise ValueError("El método diverge - valores demasiado grandes")

        if error < tol:
            print(f"Convergió en {it+1} iteraciones.")
            return x, max_iter

    print("No convergió en el número máximo de iteraciones.")
    return x, 100
