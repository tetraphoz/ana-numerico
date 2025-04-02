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
    QStyledItemDelegate,
)
from PySide6.QtGui import QFont
import pyqtgraph as pg
import sympy as sp
import numpy as np
from PySide6.QtCore import Qt

from metodos.gaussSeidel import gauss_seidel
from metodos.reglaFalsa import regla_falsa
from metodos.lagrange import lagrange


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Solver de Métodos Numéricos")

        # Layout principal
        layout_principal = QHBoxLayout()

        # TODO: Implementar auto-añadir nombres de metodos
        metodos = ["Regla Falsa", "Gauss-Seidel", "Lagrange"]

        # Creamos nuestras pestañas
        self.lista_navegacion = QListWidget()
        self.lista_navegacion.addItems(metodos)
        self.lista_navegacion.setFixedWidth(200)
        self.lista_navegacion.currentRowChanged.connect(self.cambiar_pagina)
        layout_principal.addWidget(self.lista_navegacion)

        # El orden importa
        self.widget_apilado = QStackedWidget()
        self.widget_apilado.addWidget(self.pagina_regla_falsa())
        self.widget_apilado.addWidget(self.pagina_gauss_seidel())
        self.widget_apilado.addWidget(self.pagina_lagrange())
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
        boton_resolver2 = QPushButton("Resolver")
        boton_resolver2.clicked.connect(self.resolver_gauss_seidel)
        layout.addWidget(boton_resolver2)

        # Área de salida
        self.salida_gauss_seidel = QTextEdit()
        self.salida_gauss_seidel.setReadOnly(True)
        layout.addWidget(self.salida_gauss_seidel)

        pagina.setLayout(layout)
        return pagina

    def pagina_lagrange(self):
        pagina = QWidget()
        layout = QVBoxLayout()

        # Título y descripción
        etiqueta_titulo = QLabel("Interpolación de Lagrange")
        etiqueta_titulo.setFont(QFont("Arial", 24, QFont.Bold))
        layout.addWidget(etiqueta_titulo)

        etiqueta_descripcion = QLabel(
            "Introduce un conjunto de puntos (x, y) para generar el polinomio de interpolación."
        )
        etiqueta_descripcion.setWordWrap(True)
        layout.addWidget(etiqueta_descripcion)

        # Controles para la tabla de datos
        layout_controles = QHBoxLayout()

        # Botón para añadir filas
        boton_agregar = QPushButton("+ Añadir punto")
        boton_agregar.clicked.connect(self.agregar_fila_lagrange)
        layout_controles.addWidget(boton_agregar)

        # Botón para eliminar filas
        boton_eliminar = QPushButton("- Eliminar punto")
        boton_eliminar.clicked.connect(self.eliminar_fila_lagrange)
        layout_controles.addWidget(boton_eliminar)
        # TODO: Agregar refresh visual aqui

        layout.addLayout(layout_controles)

        # Tabla de datos
        self.tabla_datos_lagrange = QTableWidget(3, 2)
        self.tabla_datos_lagrange.setHorizontalHeaderLabels(["x", "y"])
        self.tabla_datos_lagrange.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.tabla_datos_lagrange.verticalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        layout.addWidget(self.tabla_datos_lagrange)

        # Campo para el valor a interpolar
        layout_interpolar = QHBoxLayout()
        etiqueta_interpolar = QLabel("Valor a interpolar (x):")
        self.entrada_interpolar = QLineEdit()
        self.entrada_interpolar.setPlaceholderText("Ingrese el valor x a interpolar")
        layout_interpolar.addWidget(etiqueta_interpolar)
        layout_interpolar.addWidget(self.entrada_interpolar)
        layout.addLayout(layout_interpolar)

        # Botón de resolver
        boton_resolver = QPushButton("Calcular Polinomio")
        boton_resolver.clicked.connect(self.resolver_lagrange)
        layout.addWidget(boton_resolver)

        # Gráfico y resultados
        layout_resultados = QHBoxLayout()

        # Gráfico
        self.grafico_lagrange = pg.PlotWidget()
        self.grafico_lagrange.setBackground("w")
        self.grafico_lagrange.showGrid(x=True, y=True)
        self.grafico_lagrange.setLabel("left", "y")
        self.grafico_lagrange.setLabel("bottom", "x")
        layout_resultados.addWidget(self.grafico_lagrange, stretch=2)

        # Resultados
        layout_texto = QVBoxLayout()
        self.salida_lagrange = QTextEdit()
        self.salida_lagrange.setReadOnly(True)
        self.salida_lagrange.setFont(QFont("Courier New", 12))
        layout_texto.addWidget(QLabel("Resultados:"))
        layout_texto.addWidget(self.salida_lagrange)
        layout_resultados.addLayout(layout_texto, stretch=1)

        layout.addLayout(layout_resultados)

        pagina.setLayout(layout)
        return pagina

    def agregar_fila_lagrange(self):
        current_rows = self.tabla_datos_lagrange.rowCount()
        self.tabla_datos_lagrange.insertRow(current_rows)

    def eliminar_fila_lagrange(self):
        current_row = self.tabla_datos_lagrange.currentRow()
        if current_row >= 0:
            self.tabla_datos_lagrange.removeRow(current_row)

    def pagina_regla_falsa(self):
        pagina = QWidget()
        layout = QVBoxLayout()

        etiqueta_titulo = QLabel("Regla Falsa")
        etiqueta_titulo.setFont(QFont("Arial", 24, QFont.Bold))
        layout.addWidget(etiqueta_titulo)

        etiqueta_descripcion = QLabel(
            """
            Encuentra la raíz de una función dentro de un intervalo.
            Nota: exp(x) = e^(x)
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

        layout.addWidget(QLabel("Introduzca la tolerancia."))
        self.entrada_tol = QLineEdit()
        self.entrada_tol.setPlaceholderText("1e-6")
        self.entrada_tol.setText("1e-6")
        layout.addWidget(self.entrada_tol)

        layout.addWidget(QLabel("Introduzca la cantidad de iteraciones."))
        self.entrada_maxiter = QLineEdit()
        self.entrada_maxiter.setPlaceholderText("1e5")
        self.entrada_maxiter.setText("1e5")
        layout.addWidget(self.entrada_maxiter)

        # Botón de resolver
        boton_resolver1 = QPushButton("Resolver")
        boton_resolver1.clicked.connect(self.resolver_regla_falsa)
        layout.addWidget(boton_resolver1)

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
        texto_maxiter = self.entrada_maxiter.text()
        texto_tol = self.entrada_tol.text()

        try:
            if not expresion_funcion:  # Checamos que la funcion no este vacia
                raise ValueError("Introduzca una función.")

            if not texto_intervalo:  # Checamos que los intervalos no este vacios
                raise ValueError("Introduzca el intervalo.")

            # Parsear la función con sympy
            x = sp.symbols("x")
            f = sp.sympify(expresion_funcion)
            f_numerica = sp.lambdify(x, f, "numpy")

            try:
                maxiter = float(texto_maxiter)
                tol = float(texto_tol)
            except:
                raise ValueError("Valores erroneos en parámetros.")

            try:
                # Parsear el intervalo
                a, b = map(float, texto_intervalo.split())
            except ValueError:
                raise ValueError("Revisar intervalo ingresado. Error de lectura")

            # Graficar la función
            x_valores = np.linspace(a, b, 100)
            y_valores = f_numerica(x_valores)
            self.grafico.clear()
            self.grafico.plot(x_valores, y_valores, pen="b")

            # Resolver usando el método de la regla falsa
            raiz, iteraciones = regla_falsa(f_numerica, a, b, tol, maxiter)

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
        except Exception as e:
            self.salida_regla_falsa.clear()
            self.salida_regla_falsa.append(f"Aviso: {str(e)} ")

    def resolver_lagrange(self):
        try:
            # Obtener los datos de la tabla
            puntos = []
            for i in range(self.tabla_datos_lagrange.rowCount()):
                x_item = self.tabla_datos_lagrange.item(i, 0)
                y_item = self.tabla_datos_lagrange.item(i, 1)

                if x_item and y_item and x_item.text() and y_item.text():
                    try:
                        x = float(x_item.text())
                        y = float(y_item.text())
                        puntos.append((x, y))
                    except ValueError:
                        raise ValueError(f"Valores no numéricos en fila {i+1}")
                else:
                    raise ValueError(f"Fila {i+1} incompleta")

            if len(puntos) < 2:
                raise ValueError("Se necesitan al menos 2 puntos para la interpolación")

            # Separar las coordenadas x e y
            x_vals = [p[0] for p in puntos]
            y_vals = [p[1] for p in puntos]

            # Obtener el valor a interpolar si se especificó
            x_interpolar = None
            if self.entrada_interpolar.text():
                try:
                    x_interpolar = float(self.entrada_interpolar.text())
                except ValueError:
                    raise ValueError("El valor a interpolar debe ser un número")

            # Calcular el polinomio de Lagrange
            polinomio, resultado = lagrange(puntos, x_interpolar)

            # Mostrar resultados
            self.salida_lagrange.clear()
            self.salida_lagrange.append("Polinomio de Lagrange:")
            self.salida_lagrange.append(f"P(x) = {polinomio}")

            if x_interpolar is not None:
                self.salida_lagrange.append("\nResultado de la interpolación:")
                self.salida_lagrange.append(f"P({x_interpolar}) = {resultado}")

            # Graficar los puntos y el polinomio
            self.grafico_lagrange.clear()

            # Graficar puntos de entrada
            self.grafico_lagrange.plot(
                x_vals, y_vals, pen=None, symbol="o", symbolSize=10, symbolBrush="b"
            )

            # Graficar polinomio de interpolación
            if len(puntos) >= 2:
                x_min, x_max = min(x_vals), max(x_vals)
                x_range = x_max - x_min
                x_plot = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 100)

                # Evaluar el polinomio (esto es simplificado, en la práctica necesitarías parsear el polinomio)
                try:
                    x_sym = sp.symbols("x")
                    expr = sp.sympify(str(polinomio))
                    f = sp.lambdify(x_sym, expr, "numpy")
                    y_plot = f(x_plot)
                    self.grafico_lagrange.plot(x_plot, y_plot, pen="r")
                except:
                    self.salida_lagrange.append(
                        "\nNota: No se pudo graficar el polinomio automáticamente"
                    )

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


app = QApplication([])
ventana = MainWindow()
app.setStyleSheet(
    """
    QLabel{font-size: 18pt;}
    QTextEdit{font-size: 14pt;}
    QPushButton{font-size: 15pt;}
    QLineEdit{font-size: 15pt;}
    QListWidget{font-size: 15pt; font-weight:bold}
    QTableWidget{font-size: 14pt;}
    QTableWidget::view{}
    """
)
ventana.show()
app.exec()
