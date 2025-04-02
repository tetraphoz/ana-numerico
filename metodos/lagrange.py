import sympy as sp
import numpy as np


def lagrange(puntos, x_interpolar=None):
    """
    Calcula el polinomio de interpolación de Lagrange para un conjunto de puntos dados.

    Args:
        puntos: Lista de tuplas (x, y) con los puntos conocidos
        x_interpolar: Valor opcional para evaluar el polinomio resultante

    Returns:
        Una tupla con:
        - El polinomio de Lagrange como expresión simbólica
        - El valor interpolado (None si no se proporcionó x_interpolar)

    Raises:
        ValueError: Si hay puntos duplicados o insuficientes puntos
    """
    if not puntos:
        raise ValueError("Se requiere al menos un punto para la interpolación")

    # Verificar que no haya valores x duplicados
    x_vals = [p[0] for p in puntos]
    if len(x_vals) != len(set(x_vals)):
        raise ValueError(
            "No puede haber valores x duplicados en los puntos de interpolación"
        )

    x = sp.symbols("x")
    n = len(puntos)
    polinomio = sp.sympify(0)  # Inicializar como expresión simbólica

    for i in range(n):
        xi, yi = puntos[i]
        termino = yi
        for j in range(n):
            if j != i:
                xj = puntos[j][0]
                try:
                    termino *= (x - xj) / (xi - xj)
                except ZeroDivisionError:
                    raise ValueError(
                        f"Los puntos {i} y {j} tienen el mismo valor x ({xi}), lo que causaría división por cero"
                    )
        polinomio += termino

    # Simplificar el polinomio
    polinomio = sp.simplify(polinomio)

    # Evaluar si se solicita
    resultado = None
    if x_interpolar is not None:
        try:
            resultado = float(polinomio.subs(x, x_interpolar).evalf())
        except:
            raise ValueError(f"No se pudo evaluar el polinomio en x = {x_interpolar}")

    return polinomio, resultado
