import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QScrollArea,
    QMessageBox,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gauss-Seidel Equation Solver")
        self.setGeometry(100, 100, 800, 600)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # System size input
        size_layout = QHBoxLayout()
        self.n_input = QLineEdit()
        self.n_input.setPlaceholderText("Enter system size (N)")
        size_layout.addWidget(QLabel("System size (N):"))
        size_layout.addWidget(self.n_input)

        self.generate_btn = QPushButton("Generate Matrix")
        self.generate_btn.clicked.connect(self.generate_matrix_inputs)
        size_layout.addWidget(self.generate_btn)
        main_layout.addLayout(size_layout)

        # Matrix input scroll area
        self.matrix_scroll = QScrollArea()
        self.matrix_scroll.setWidgetResizable(True)
        self.matrix_widget = QWidget()
        self.matrix_layout = QGridLayout(self.matrix_widget)
        self.matrix_scroll.setWidget(self.matrix_widget)
        main_layout.addWidget(self.matrix_scroll)

        # Initial guesses
        main_layout.addWidget(QLabel("Initial Guesses:"))
        self.initial_guess_widget = QWidget()
        self.initial_guess_layout = QHBoxLayout(self.initial_guess_widget)
        main_layout.addWidget(self.initial_guess_widget)

        # Parameters
        params_layout = QHBoxLayout()
        self.tolerance_input = QLineEdit("1e-6")
        self.max_iter_input = QLineEdit("1000")
        params_layout.addWidget(QLabel("Tolerance:"))
        params_layout.addWidget(self.tolerance_input)
        params_layout.addWidget(QLabel("Max Iterations:"))
        params_layout.addWidget(self.max_iter_input)
        main_layout.addLayout(params_layout)

        # Solve button and results
        self.solve_btn = QPushButton("Solve System")
        self.solve_btn.clicked.connect(self.solve_system)
        main_layout.addWidget(self.solve_btn)

        self.results_label = QLabel("Solution will be displayed here")
        main_layout.addWidget(self.results_label)

    def generate_matrix_inputs(self):
        try:
            N = int(self.n_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid system size!")
            return

        # Clear existing matrix inputs
        for i in reversed(range(self.matrix_layout.count())):
            self.matrix_layout.itemAt(i).widget().deleteLater()

        # Create new matrix inputs
        for row in range(N):
            for col in range(N + 1):
                le = QLineEdit()
                if col < N:
                    le.setPlaceholderText(f"A[{row+1}][{col+1}]")
                else:
                    le.setPlaceholderText(f"B[{row+1}]")
                self.matrix_layout.addWidget(le, row, col)

        # Clear and create initial guess inputs
        for i in reversed(range(self.initial_guess_layout.count())):
            self.initial_guess_layout.itemAt(i).widget().deleteLater()

        for col in range(N):
            le = QLineEdit("0")
            le.setPlaceholderText(f"x[{col+1}]")
            self.initial_guess_layout.addWidget(le)

    def solve_system(self):
        try:
            N = int(self.n_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid system size!")
            return

        # Read matrix A and vector B
        A = []
        B = []
        for row in range(N):
            current_row = []
            for col in range(N):
                widget = self.matrix_layout.itemAtPosition(row, col).widget()
                current_row.append(float(widget.text()))
            A.append(current_row)
            widget = self.matrix_layout.itemAtPosition(row, N).widget()
            B.append(float(widget.text()))

        # Read initial guesses
        initial = [
            float(self.initial_guess_layout.itemAt(i).widget().text()) for i in range(N)
        ]

        # Read parameters
        try:
            tol = float(self.tolerance_input.text())
            max_iter = int(self.max_iter_input.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameters!")
            return

        # Check diagonal elements
        for i in range(N):
            if A[i][i] == 0:
                QMessageBox.critical(self, "Error", f"Zero on diagonal at row {i+1}!")
                return

        # Solve using Gauss-Seidel
        try:
            solution, iterations, converged = self.gauss_seidel(
                A, B, initial, tol, max_iter
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        # Display results
        result_text = []
        if converged:
            result_text.append(f"Converged in {iterations} iterations:")
        else:
            result_text.append(f"Maximum iterations reached ({iterations}):")

        for i, val in enumerate(solution):
            result_text.append(f"x[{i+1}] = {val:.6f}")

        self.results_label.setText("\n".join(result_text))

    def gauss_seidel(self, A, B, initial, tol, max_iter):
        n = len(A)
        x = initial.copy()
        converged = False

        for iteration in range(max_iter):
            max_error = 0.0
            for i in range(n):
                old_value = x[i]
                sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
                x[i] = (B[i] - sigma) / A[i][i]
                error = abs(x[i] - old_value)
                if error > max_error:
                    max_error = error

            if max_error < tol:
                converged = True
                break

        return x, iteration + 1, converged


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
