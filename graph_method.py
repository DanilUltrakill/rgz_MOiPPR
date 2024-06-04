import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import cm

class OptimizationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optimization Tool")
        self.geometry("600x800")  # Увеличиваем размер основного окна
        
        self.constraints = []
        self.create_widgets()
    
    def create_widgets(self):
        # Entry для целевой функции
        self.label_func = tk.Label(self, text="Target Function (e.g., (x[0]-3)**2 + x[1]**2):")
        self.label_func.pack()
        self.entry_func = tk.Entry(self, width=50)
        self.entry_func.insert(0, "(x[0]-3)**2 + x[1]**2")  # Значение по умолчанию
        self.entry_func.pack()
        
        # Entry для начального предположения
        self.label_initial = tk.Label(self, text="Initial Guess (e.g., 0,0):")
        self.label_initial.pack()
        self.entry_initial = tk.Entry(self, width=50)
        self.entry_initial.insert(0, "0,0")  # Значение по умолчанию
        self.entry_initial.pack()

        # Радиокнопки для типа оптимизации
        self.optimization_type = tk.StringVar(value="min")
        self.label_optimization_type = tk.Label(self, text="Optimization Type:")
        self.label_optimization_type.pack()
        self.radio_min = tk.Radiobutton(self, text="Minimize", variable=self.optimization_type, value="min")
        self.radio_min.pack()
        self.radio_max = tk.Radiobutton(self, text="Maximize", variable=self.optimization_type, value="max")
        self.radio_max.pack()

        # Entry для первого ограничения
        self.constraint_entries = []
        self.label_constraints = tk.Label(self, text="Constraint (e.g., x[0] + x[1] <= 2):")
        self.label_constraints.pack()
        
        # Значения ограничений по умолчанию
        default_constraints = [
            "(x[0]-4)**2 + (x[1]-4)**2 <= 4",
            "x[0] >= 2",
            "x[1] >= 0",
            "x[0] <= 4"
        ]
        for constraint in default_constraints:
            entry_constraint = tk.Entry(self, width=50)
            entry_constraint.insert(0, constraint)
            entry_constraint.pack()
            self.constraint_entries.append(entry_constraint)
        
        # Кнопка для добавления дополнительных ограничений
        self.button_add_constraint = tk.Button(self, text="Add Constraint", command=self.add_constraint)
        self.button_add_constraint.pack()
        
        # Кнопка для начала оптимизации
        self.button_optimize = tk.Button(self, text="Optimize", command=self.optimize)
        self.button_optimize.pack()
    
    def add_constraint(self):
        entry_constraint = tk.Entry(self, width=50)
        entry_constraint.pack()
        self.constraint_entries.append(entry_constraint)
    
    def func(self, x):
        return eval(self.entry_func.get(), {}, {"x": x})
    
    def parse_constraints(self):
        constraints = []
        for entry in self.constraint_entries:
            constraint = entry.get()
            if "<=" in constraint:
                expr, bound = constraint.split("<=")
                constraints.append({'type': 'ineq', 'fun': lambda x, expr=expr, bound=float(bound): float(bound) - eval(expr, {}, {"x": x})})
            elif ">=" in constraint:
                expr, bound = constraint.split(">=")
                constraints.append({'type': 'ineq', 'fun': lambda x, expr=expr, bound=float(bound): eval(expr, {}, {"x": x}) - float(bound)})
            elif "==" in constraint:
                expr, bound = constraint.split("==")
                constraints.append({'type': 'eq', 'fun': lambda x, expr=expr, bound=float(bound): eval(expr, {}, {"x": x}) - float(bound)})
            else:
                # Другие типы ограничений можно добавить здесь при необходимости
                pass
        return constraints
    
    def optimize(self):
        initial_guess = np.array([float(x) for x in self.entry_initial.get().split(",")])  # Парсинг начального предположения
        
        constraints = self.parse_constraints()
        
        # Список для хранения промежуточных значений
        self.intermediate_steps = [initial_guess]
        self.iteration = 0
        
        def callback(xk):
            print(f"Iteration {self.iteration}: x = {xk}; f(x) = {self.func(xk)}")
            self.intermediate_steps.append(xk.copy())
            self.iteration += 1
        
        if self.optimization_type.get() == "min":
            result = minimize(self.func, initial_guess, method='SLSQP', constraints=constraints, tol=0.001, callback=callback)
        else:
            def neg_func(x):
                return -self.func(x)
            result = minimize(neg_func, initial_guess, method='SLSQP', constraints=constraints, tol=0.001, callback=callback)
        
        self.show_results(result)
    
    def show_results(self, result):
        results_window = tk.Toplevel(self)
        results_window.title("Optimization Results")
        results_window.geometry("1000x800")  # Увеличиваем размер окна результатов
        
        label_result = tk.Label(results_window, text=f"Optimal x: {result.x}, f(x): {self.func(result.x) if self.optimization_type.get() == 'min' else -self.func(result.x)}")
        label_result.pack()

        # Отображение количества итераций
        label_iterations = tk.Label(results_window, text=f"Number of iterations: {self.iteration}")
        label_iterations.pack()
        
        figure = plt.Figure(figsize=(10, 10), dpi=100)  # Увеличиваем размеры графика
        ax = figure.add_subplot(111)
        
        # Добавление тулбара над графиком
        canvas = FigureCanvasTkAgg(figure, results_window)
        toolbar = NavigationToolbar2Tk(canvas, results_window)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Генерация сетки точек для оценки ограничений
        x = np.linspace(-10, 10, 400)
        y = np.linspace(-10, 10, 400)
        X, Y = np.meshgrid(x, y)
        feasible_x = []
        feasible_y = []
        
        for i in range(len(x)):
            for j in range(len(y)):
                point = np.array([X[i, j], Y[i, j]])
                if all(constraint['fun'](point) >= 0 for constraint in self.parse_constraints() if constraint['type'] == 'ineq') and \
                   all(constraint['fun'](point) == 0 for constraint in self.parse_constraints() if constraint['type'] == 'eq'):
                    feasible_x.append(point[0])
                    feasible_y.append(point[1])
        
        if feasible_x and feasible_y:
            x_min, x_max = min(feasible_x), max(feasible_x)
            y_min, y_max = min(feasible_y), max(feasible_y)
        else:
            x_min, x_max = -10, 10
            y_min, y_max = -10, 10
        
        func_str = self.entry_func.get()
        Z = eval(func_str.replace("x[0]", "X").replace("x[1]", "Y"), {}, {"X": X, "Y": Y})
        
        ax.contour(X, Y, Z, levels=25, cmap=cm.viridis)
        
        # Отображение ограничений на графике и добавление их в легенду
        constraint_legend = []
        for entry in self.constraint_entries:
            constraint = entry.get()
            if "<=" in constraint:
                expr, bound = constraint.split("<=")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                ax.contour(X, Y, eval(expr, {}, {"X": X, "Y": Y}), levels=[float(bound)], colors='r')
                constraint_legend.append(f"{constraint}")
            elif ">=" in constraint:
                expr, bound = constraint.split(">=")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                ax.contour(X, Y, eval(expr, {}, {"X": X, "Y": Y}), levels=[float(bound)], colors='r')
                constraint_legend.append(f"{constraint}")
            elif "==" in constraint:
                expr, bound = constraint.split("==")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                ax.contour(X, Y, eval(expr, {}, {"X": X, "Y": Y}), levels=[float(bound)], colors='r')
                constraint_legend.append(f"{constraint}")
        
        # Отображение траектории сходимости
        intermediate_steps = np.array(self.intermediate_steps)
        ax.plot(intermediate_steps[:, 0], intermediate_steps[:, 1], "bo-", label="Convergence Path")
        
        # Отображение конечной точки
        ax.plot(result.x[0], result.x[1], "ro", markersize=10, label="Optimal Point")
        
        # Добавление ограничений в легенду
        for cl in constraint_legend:
            ax.plot([], [], "r-", label=cl)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect('equal', 'box')
        ax.legend()
        ax.set_title("Convergence Path")
        ax.grid(True)
        
        canvas.draw()

if __name__ == "__main__":
    app = OptimizationApp()
    app.mainloop()
