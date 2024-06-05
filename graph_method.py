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
        self.title("Решение задач НЛП")
        self.geometry("600x500")

        self["bg"] = "#4682B4"
        self.label_bg = "#B0C4DE"
        self.button_bg = "#B0E0E6"

        self.entry_pady = (0,10)
        self.button_pady = (10,0)

        self.constraints = []
        self.result_min = None
        self.result_max = None
        self.create_widgets()

    def create_widgets(self):
        # Entry для целевой функции
        self.label_func = tk.Label(self, text="Целевая функция (например, (x[1]-3)**2 + x[2]**2):", bg=self.label_bg)
        self.label_func.pack()
        self.entry_func = tk.Entry(self, width=50)
        self.entry_func.insert(0, "(x[1]-3)**2 + x[2]**2")  # Значение по умолчанию
        self.entry_func.pack(pady = self.entry_pady)

        # Entry для начального предположения
        self.label_initial = tk.Label(self, text="Начальная точка (например, 0,0):", bg=self.label_bg)
        self.label_initial.pack()
        self.entry_initial = tk.Entry(self, width=50)
        self.entry_initial.insert(0, "0,0")  # Значение по умолчанию
        self.entry_initial.pack(pady = self.entry_pady)

        # Entry для первого ограничения
        self.constraint_entries = []
        self.label_constraints = tk.Label(self, text="Ограничение (например, x[1] + x[2] <= 2):", bg=self.label_bg)
        self.label_constraints.pack()

        # Значения ограничений по умолчанию
        default_constraints = [
            "(x[1]-4)**2 + (x[2]-4)**2 <= 4",
            "x[1] >= 2",
            "x[2] >= 0",
            "x[1] <= 4"
        ]
        for constraint in default_constraints:
            entry_constraint = tk.Entry(self, width=50)
            entry_constraint.insert(0, constraint)
            entry_constraint.pack()
            self.constraint_entries.append(entry_constraint)

        # Кнопка для добавления дополнительных ограничений
        self.button_add_constraint = tk.Button(self, text="Добавить ограничение", command=self.add_constraint, bg=self.button_bg)
        self.button_add_constraint.pack(pady = self.button_pady)

        # Кнопка для удаления последнего ограничения
        self.button_remove_constraint = tk.Button(self, text="Удалить ограничение", command=self.remove_constraint, bg=self.button_bg)
        self.button_remove_constraint.pack(pady = self.button_pady)

        # Кнопка для начала оптимизации
        self.button_optimize = tk.Button(self, text="Решить", command=self.calculate, bg=self.button_bg)
        self.button_optimize.pack(pady = self.button_pady)

        # Кнопка для отображения графика
        self.button_show_graph = tk.Button(self, text="Показать график", command=self.open_graph_window, bg=self.button_bg)
        self.button_show_graph.pack(pady = self.button_pady)

        # Кнопка для отображения 3D-модели
        self.button_show_3d = tk.Button(self, text="Показать 3D модель", command=self.open_3d_window, state=tk.DISABLED, bg=self.button_bg)
        self.button_show_3d.pack(pady = self.button_pady)

        self.update_constraint_buttons()

    def update_constraint_buttons(self):
        if len(self.constraint_entries) <= 1:
            self.button_remove_constraint.config(state=tk.DISABLED)
        else:
            self.button_remove_constraint.config(state=tk.NORMAL)

        if len(self.constraint_entries) >= 5:
            self.button_add_constraint.config(state=tk.DISABLED)
        else:
            self.button_add_constraint.config(state=tk.NORMAL)

    def add_constraint(self):
        if len(self.constraint_entries) < 5:
            entry_constraint = tk.Entry(self, width=50)
            entry_constraint.pack()
            self.constraint_entries.append(entry_constraint)
            self.update_constraint_buttons()

    def remove_constraint(self):
        if len(self.constraint_entries) > 1:
            entry_constraint = self.constraint_entries.pop()
            entry_constraint.pack_forget()
            self.update_constraint_buttons()

    def func(self, x):
        x = [0] + list(x)
        return eval(self.entry_func.get(), {}, {"x": x})

    def parse_constraints(self):
        constraints = []
        for entry in self.constraint_entries:
            constraint = entry.get()
            if "<=" in constraint:
                expr, bound = constraint.split("<=")
                constraints.append({'type': 'ineq', 'fun': lambda x, expr=expr, bound=float(bound): float(bound) - eval(expr, {}, {"x": [0] + list(x)})})
            elif ">=" in constraint:
                expr, bound = constraint.split(">=")
                constraints.append({'type': 'ineq', 'fun': lambda x, expr=expr, bound=float(bound): eval(expr, {}, {"x": [0] + list(x)}) - float(bound)})
            elif "==" in constraint:
                expr, bound = constraint.split("==")
                constraints.append({'type': 'eq', 'fun': lambda x, expr=expr, bound=float(bound): eval(expr, {}, {"x": [0] + list(x)}) - float(bound)})
            else:
                pass
        return constraints

    def optimize(self, opt_type):
        initial_guess = np.array([float(x) for x in self.entry_initial.get().split(",")])

        constraints = self.parse_constraints()

        if opt_type == "min":
            self.intermediate_steps_min = [initial_guess]
            self.iteration_min = 0

            def callback_min(xk):
                self.intermediate_steps_min.append(xk.copy())
                self.iteration_min += 1

            self.result_min = minimize(self.func, initial_guess, method='SLSQP', constraints=constraints, tol=0.001, callback=callback_min)

        else:
            def neg_func(x):
                return -self.func(x)

            self.intermediate_steps_max = [initial_guess]
            self.iteration_max = 0

            def callback_max(xk):
                self.intermediate_steps_max.append(xk.copy())
                self.iteration_max += 1

            self.result_max = minimize(neg_func, initial_guess, method='SLSQP', constraints=constraints, tol=0.001, callback=callback_max)

    def calculate(self):
        self.button_show_3d.config(state=tk.NORMAL)

        self.optimize("min")
        self.optimize("max")

        results_window = tk.Toplevel(self)
        results_window.title("Результаты решения")
        results_window.geometry("800x600")
        results_window["bg"] = "#4682B4"

        self.result_text_min = ""
        self.result_text_max = ""

        if self.result_min.success:
            self.result_text_min += f"Точка минимума найдена в: {self.result_min.x}\n"
            self.result_text_min += f"Значение целевой функции: {self.func(self.result_min.x)}\n"
            self.result_text_min += f"Кол-во итераций: {self.iteration_min}\n"
        else:
            self.result_text_min += "Решение не найдено.\n"

        if self.result_max.success:
            self.result_text_max += f"Точка максимума найдена в: {self.result_max.x}\n"
            self.result_text_max += f"Значение целевой функции: {-self.result_max.fun}\n"
            self.result_text_max += f"Кол-во итераций: {self.iteration_max}\n"
        else:
            self.result_text_max += "Решение не найдено.\n"

        self.results_label_min = tk.Label(results_window, text=self.result_text_min, bg=self.label_bg)
        self.results_label_min.grid(row=0, column=0, padx=10, pady=10)

        self.results_label_max = tk.Label(results_window, text=self.result_text_max, bg=self.label_bg)
        self.results_label_max.grid(row=0, column=1, padx=10, pady=10)

        self.show_path_button = tk.Button(results_window, text="Показать каждую итерацию", command=lambda: self.show_solution_path(results_window), bg=self.button_bg)
        self.show_path_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.hide_path_button = tk.Button(results_window, text="Скрыть", command=lambda: self.hide_solution_path(), bg=self.button_bg)
        self.hide_path_button.grid(row=1, column=1, columnspan=2, pady=10)
        self.hide_path_button.grid_remove()

    def show_solution_path(self, parent_window):
        self.min_path_text = ""
        self.max_path_text = ""

        if hasattr(self, 'intermediate_steps_min'):
            for i, step in enumerate(self.intermediate_steps_min):
                self.min_path_text += f"Iteration {i}: x = {step}; f(x) = {self.func(step)}\n"
        else:
            self.min_path_text = "Не удалось найти экстремум\n"

        if hasattr(self, 'intermediate_steps_max'):
            for i, step in enumerate(self.intermediate_steps_max):
                self.max_path_text += f"Iteration {i}: x = {step}; f(x) = {self.func(step)}\n"
        else:
            self.max_path_text = "Не удалось найти экстремум\n"

        self.min_path_label = tk.Label(parent_window, text=self.min_path_text, bg = self.label_bg)
        self.max_path_label = tk.Label(parent_window, text=self.max_path_text, bg = self.label_bg)

        self.min_path_label.grid(row=2, column=0, padx=10, pady=10)
        self.max_path_label.grid(row=2, column=1, padx=10, pady=10)

        self.show_path_button.grid_remove()
        self.hide_path_button.grid()

    def hide_solution_path(self):
        self.min_path_label.grid_remove()
        self.max_path_label.grid_remove()
        self.hide_path_button.grid_remove()
        self.show_path_button.grid()

    def open_graph_window(self):
        graph_window = tk.Toplevel(self)
        graph_window.title("График 2D")
        graph_window.geometry("700x700")

        self.legend_list = []

        fig, ax = plt.subplots(figsize=(8, 8))

        x = np.linspace(-10, 10, 400)
        y = np.linspace(-10, 10, 400)
        X, Y = np.meshgrid(x, y)
        Z = eval(self.entry_func.get(), {}, {"x": [0, X, Y]})

        contour = ax.contour(X, Y, Z, levels=50, cmap='viridis')

        # Точки минимума и максимума
        if self.result_min.success:
            ax.plot(self.result_min.x[0], self.result_min.x[1], 'ro', label='Минимум')
            self.legend_list.append('Минимум')
        if self.result_max.success:
            ax.plot(self.result_max.x[0], self.result_max.x[1], 'bo', label='Максимум')
            self.legend_list.append('Максимум')

        # Линии ограничения
        constraints = self.parse_constraints()
        for constraint in constraints:
            if constraint['type'] == 'ineq':
                func = constraint['fun']
                Z_constr = np.vectorize(lambda x1, x2: func([x1, x2]))(X, Y)
                ax.contour(X, Y, Z_constr, levels=[0], colors='r')
            elif constraint['type'] == 'eq':
                func = constraint['fun']
                Z_constr = np.vectorize(lambda x1, x2: func([x1, x2]))(X, Y)
                ax.contour(X, Y, Z_constr, levels=[0], colors='b')

        ax.set_title('График 2D')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        custom_lines = [plt.Line2D([0], [0], color='r', lw=4),
                        plt.Line2D([0], [0], color='b', lw=4)]
        ax.legend(custom_lines, self.legend_list)

        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, graph_window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def open_3d_window(self):
        if len(self.result_min.x) > 2 or len(self.result_max.x) > 2:
            return

        def plot_3d(constraint_surfaces, min_point, max_point):
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            x = np.linspace(-10, 10, 100)
            y = np.linspace(-10, 10, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.array(eval(self.entry_func.get(), {}, {"x": [0, X, Y]}))

            ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)

            for surface in constraint_surfaces:
                Z = np.array(eval(surface, {}, {"x": [0, X, Y]}))
                ax.plot_surface(X, Y, Z, color='r', alpha=0.5)

            ax.scatter(min_point[0], min_point[1], self.func(min_point), color='red', s=100, label='Минимум')
            ax.scatter(max_point[0], max_point[1], -self.func(max_point), color='blue', s=100, label='Максимум')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D модель решения')
            ax.legend()

            plt.show()

        constraint_surfaces = []
        constraints = self.parse_constraints()
        for constraint in constraints:
            if constraint['type'] == 'ineq':
                constraint_surfaces.append(constraint['fun'].__code__.co_consts[1])
            elif constraint['type'] == 'eq':
                constraint_surfaces.append(constraint['fun'].__code__.co_consts[1])

        plot_3d(constraint_surfaces, self.result_min.x, self.result_max.x)

if __name__ == "__main__":
    app = OptimizationApp()
    app.mainloop()
