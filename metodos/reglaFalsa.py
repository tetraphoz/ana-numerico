def regla_falsa(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) >= 0:
        raise ValueError("La función no cambia de signo en el intervalo dado.")

    iteraciones = 0
    while iteraciones < max_iter:
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if abs(f(c)) < tol:
            return c, iteraciones
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iteraciones += 1

    raise ValueError("El método no convergió en el número máximo de iteraciones.")
