from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QStackedWidget,
    QLabel,
    QPushButton,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLineEdit,
)
from PySide6.QtGui import QFont
import pyqtgraph as pg
import sympy as sp
import numpy as np

from metodos.gaussSeidel import gauss_seidel
from metodos.reglaFalsa import regla_falsa


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Solver de Métodos Numéricos")

        # Layout principal
        layout_principal = QHBoxLayout()

        metodos = ["Gauss-Seidel", "Regla Falsa"]

        # Creamos nuestras pestañas
        self.lista_navegacion = QListWidget()
        self.lista_navegacion.addItems(metodos)
        self.lista_navegacion.setFixedWidth(200)
        self.lista_navegacion.currentRowChanged.connect(self.cambiar_pagina)
        layout_principal.addWidget(self.lista_navegacion)

        self.widget_apilado = QStackedWidget()
        self.widget_apilado.addWidget(self.pagina_gauss_seidel())
        self.widget_apilado.addWidget(self.pagina_regla_falsa())
        layout_principal.addWidget(self.widget_apilado)

        # Creamos nuestra ventana principal
        widget_central = QWidget()
        widget_central.setLayout(layout_principal)
        self.setCentralWidget(widget_central)

    def cambiar_pagina(self, indice):
        self.widget_apilado.setCurrentIndex(indice)

    def pagina_gauss_seidel(self):
        pagina = QWidget()
        layout = QVBoxLayout()

        etiqueta_titulo = QLabel("Gauss-Seidel")
        etiqueta_titulo.setFont(QFont("Arial", 24, QFont.Bold))
        layout.addWidget(etiqueta_titulo)

        etiqueta_descripcion = QLabel(
            "Resuelve un sistema de ecuaciones usando el método Gauss-Seidel."
        )
        etiqueta_descripcion.setWordWrap(True)
        layout.addWidget(etiqueta_descripcion)

        # Layout horizontal para las tablas de la matriz A y el vector b
        layout_tablas = QHBoxLayout()

        # Campos de entrada para la matriz A
        etiqueta_matriz_a = QLabel("Matriz A:")
        self.tabla_matriz_a = QTableWidget(3, 3)  # Tabla por defecto de 3x3
        self.tabla_matriz_a.setFixedSize(300, 150)  # Tamaño más pequeño
        self.tabla_matriz_a.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_matriz_a.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout_tablas.addWidget(etiqueta_matriz_a)
        layout_tablas.addWidget(self.tabla_matriz_a)

        # Campos de entrada para el vector b
        etiqueta_vector_b = QLabel("Vector b:")
        self.tabla_vector_b = QTableWidget(3, 1)  # Tabla por defecto de 3x1
        self.tabla_vector_b.setFixedSize(150, 150)  # Tamaño más pequeño
        self.tabla_vector_b.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_vector_b.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout_tablas.addWidget(etiqueta_vector_b)
        layout_tablas.addWidget(self.tabla_vector_b)

        layout.addLayout(layout_tablas)

        # Botón de resolver
        boton_resolver = QPushButton("Resolver")
        boton_resolver.clicked.connect(self.resolver_gauss_seidel)
        layout.addWidget(boton_resolver)

        # Área de salida
        self.salida_gauss_seidel = QTextEdit()
        self.salida_gauss_seidel.setReadOnly(True)
        layout.addWidget(self.salida_gauss_seidel)

        pagina.setLayout(layout)
        return pagina

    def pagina_regla_falsa(self):
        pagina = QWidget()
        layout = QVBoxLayout()

        etiqueta_titulo = QLabel("Regla Falsa")
        etiqueta_titulo.setFont(QFont("Arial", 24, QFont.Bold))
        layout.addWidget(etiqueta_titulo)

        etiqueta_descripcion = QLabel(
            """Encuentra la raíz de una función dentro de un intervalo.
            """
        )
        etiqueta_descripcion.setWordWrap(True)
        layout.addWidget(etiqueta_descripcion)

        layout.addWidget(QLabel("Introduzca la función f(x)"))
        self.entrada_funcion = QLineEdit()
        self.entrada_funcion.setPlaceholderText("e.g., x**2 - 4")
        layout.addWidget(self.entrada_funcion)

        layout.addWidget(QLabel("Introduzca el intervalo [a, b]:"))
        self.entrada_intervalo = QLineEdit()
        self.entrada_intervalo.setPlaceholderText("e.g., 1 3")
        layout.addWidget(self.entrada_intervalo)

        # Botón de resolver
        boton_resolver = QPushButton("Resolver")
        boton_resolver.clicked.connect(self.resolver_regla_falsa)
        layout.addWidget(boton_resolver)

        # Gráfico y área de salida
        layout_resultados = QHBoxLayout()
        layout.addLayout(layout_resultados)

        self.grafico = pg.PlotWidget()
        self.grafico.setBackground("w")
        self.grafico.showGrid(x=True, y=True)
        layout_resultados.addWidget(self.grafico)

        layout_resultados2 = QVBoxLayout()
        layout_resultados.addLayout(layout_resultados2)

        layout_resultados2.addWidget(QLabel("Resultados"))
        self.salida_regla_falsa = QTextEdit()
        self.salida_regla_falsa.setReadOnly(True)
        layout_resultados2.addWidget(self.salida_regla_falsa)

        pagina.setLayout(layout)
        return pagina

    def resolver_gauss_seidel(self):
        try:
            # Obtener la matriz A de la tabla
            A = []
            for i in range(self.tabla_matriz_a.rowCount()):
                fila = []
                for j in range(self.tabla_matriz_a.columnCount()):
                    item = self.tabla_matriz_a.item(i, j)
                    if item and item.text():
                        fila.append(float(item.text()))
                    else:
                        fila.append(0.0)  # Valor por defecto si la celda está vacía
                A.append(fila)

            # Obtener el vector b de la tabla
            b = []
            for i in range(self.tabla_vector_b.rowCount()):
                item = self.tabla_vector_b.item(i, 0)
                if item and item.text():
                    b.append(float(item.text()))
                else:
                    b.append(0.0)  # Valor por defecto si la celda está vacía

            # Llamar a la función gauss_seidel
            solucion, iteraciones = gauss_seidel(A, b)

            # Mostrar la solución en el área de salida
            self.salida_gauss_seidel.clear()
            self.salida_gauss_seidel.append(f"Convergió en {iteraciones} iteraciones.")
            for i, x_i in enumerate(solucion):
                self.salida_gauss_seidel.append(f"x[{i+1}] = {x_i:.6f}")

        except Exception as e:
            self.salida_gauss_seidel.clear()
            self.salida_gauss_seidel.append(f"Error: {str(e)}")

    def resolver_regla_falsa(self):

        # Obtener la función y el intervalo
        expresion_funcion = self.entrada_funcion.text()
        texto_intervalo = self.entrada_intervalo.text()

        try:
            # Parsear la función con sympy
            x = sp.symbols("x")
            f = sp.sympify(expresion_funcion)
            f_numerica = sp.lambdify(x, f, "numpy")

            # Parsear el intervalo
            a, b = map(float, texto_intervalo.split())

            # Graficar la función
            x_valores = np.linspace(a, b, 100)
            y_valores = f_numerica(x_valores)
            self.grafico.clear()
            self.grafico.plot(x_valores, y_valores, pen="b")

            # Resolver usando el método de la regla falsa
            raiz, iteraciones = regla_falsa(f_numerica, a, b)

            # Graficamos la raiz
            self.grafico.plot([raiz], [0], symbol="+", symbolSize=10)
            self.grafico.addLine(
                y=0, pen="k"
            )  # Línea horizontal en y=0 para referencia

            # Mostrar la raíz en el área de salida
            self.salida_regla_falsa.clear()
            self.salida_regla_falsa.append(f"Función: {expresion_funcion}")
            self.salida_regla_falsa.append(f"Intervalo: [{a}, {b}]")
            self.salida_regla_falsa.append(f"Raíz encontrada: {raiz:.6f}")
            self.salida_regla_falsa.append(f"Iteraciones: {iteraciones}")

        except SyntaxError:
            self.salida_regla_falsa.clear()
            self.salida_regla_falsa.append("Revise los datos de entrada.")
        except ValueError as e:
            self.salida_regla_falsa.clear()
            self.salida_regla_falsa.append(f"{str(e)}")
        except Exception as e:
            self.salida_regla_falsa.clear()
            self.salida_regla_falsa.append(f"Error {type(e)}: {str(e)} ")


app = QApplication([])
ventana = MainWindow()
ventana.show()
app.exec()
