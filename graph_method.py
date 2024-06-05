import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import cm
import itertools
from matplotlib.lines import Line2D

class OptimizationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Решение задач НЛП")
        self.geometry("600x500")

        self["bg"] = "#4682B4"
        self.label_bg = "#B0C4DE"
        self.button_bg = "#B0E0E6"
        self.activ_bg = "#87CEFA"

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

        self.label_maxiter = tk.Label(self, text="Максимальное кол-во итераций (например, 50):", bg=self.label_bg)
        self.label_maxiter.pack()
        self.entry_maxiter = tk.Entry(self, width=50)
        self.entry_maxiter.insert(0, "100")  # Значение по умолчанию
        self.entry_maxiter.pack(pady = self.entry_pady)

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
        self.button_add_constraint = tk.Button(self, text="Добавить ограничение", command=self.add_constraint, bg=self.button_bg, activebackground=self.activ_bg)
        self.button_add_constraint.pack(pady = self.button_pady)

        # Кнопка для удаления последнего ограничения
        self.button_remove_constraint = tk.Button(self, text="Удалить ограничение", command=self.remove_constraint, bg=self.button_bg, activebackground=self.activ_bg)
        self.button_remove_constraint.pack(pady = self.button_pady)

        # Кнопка для начала оптимизации
        self.button_optimize = tk.Button(self, text="Решить", command=self.calculate, bg=self.button_bg, activebackground=self.activ_bg)
        self.button_optimize.pack(pady = self.button_pady)

        # Кнопка для отображения графика
        self.button_show_graph = tk.Button(self, text="Показать график", command=self.open_graph_window, bg=self.button_bg, state=tk.DISABLED, activebackground=self.activ_bg)
        self.button_show_graph.pack(pady = self.button_pady)

        # Кнопка для отображения 3D-модели
        self.button_show_3d = tk.Button(self, text="Показать 3D модель", command=self.open_3d_window, state=tk.DISABLED, bg=self.button_bg, activebackground=self.activ_bg)
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
            elif ">" in constraint:
                expr, bound = constraint.split(">")
                constraints.append({'type': 'ineq', 'fun': lambda x, expr=expr, bound=float(bound): eval(expr, {}, {"x": [0] + list(x)}) - float(bound)})
            elif "<" in constraint:
                expr, bound = constraint.split("<")
                constraints.append({'type': 'ineq', 'fun': lambda x, expr=expr, bound=float(bound): eval(expr, {}, {"x": [0] + list(x)}) - float(bound)})
            elif "=" in constraint:
                expr, bound = constraint.split("=")
                constraints.append({'type': 'eq', 'fun': lambda x, expr=expr, bound=float(bound): eval(expr, {}, {"x": [0] + list(x)}) - float(bound)})
            else:
                pass
        return constraints

    def optimize(self, opt_type):
        initial_guess = np.array([float(x) for x in self.entry_initial.get().split(",")])

        max_iter = int(self.entry_maxiter.get())

        options = {'maxiter': max_iter}

        constraints = self.parse_constraints()

        if opt_type == "min":
            self.intermediate_steps_min = [initial_guess]
            self.iteration_min = 0

            def callback_min(xk):
                self.intermediate_steps_min.append(xk.copy())
                self.iteration_min += 1

            self.result_min = minimize(self.func, initial_guess, method='SLSQP', constraints=constraints, tol=0.001, callback=callback_min, options=options)

        else:
            def neg_func(x):
                return -self.func(x)

            self.intermediate_steps_max = [initial_guess]
            self.iteration_max = 0

            def callback_max(xk):
                self.intermediate_steps_max.append(xk.copy())
                self.iteration_max += 1

            self.result_max = minimize(neg_func, initial_guess, method='SLSQP', constraints=constraints, tol=0.001, callback=callback_max, options=options)

    def calculate(self):
        self.button_show_3d.config(state=tk.NORMAL)
        self.button_show_graph.config(state=tk.NORMAL)

        self.optimize("min")
        self.optimize("max")

        results_window = tk.Toplevel(self)
        results_window.title("Результаты решения")
        results_window.geometry("900x600")
        results_window["bg"] = self["bg"]

        self.result_text_min = ""
        self.result_text_max = ""

        if self.result_min.success:
            self.result_text_min += f"Точка минимума найдена в точке (x1,x2): {self.result_min.x}\n"
            self.result_text_min += f"Значение целевой функции: {self.func(self.result_min.x)}\n"
            self.result_text_min += f"Кол-во итераций: {self.iteration_min+1}\n"
        else:
            self.result_text_min += "Решение не найдено.\n"
            self.result_text_min += f"Кол-во итераций: {self.iteration_min+1}\n"

        if self.result_max.success:
            self.result_text_max += f"Точка максимума найдена в точке (x1,x2): {self.result_max.x}\n"
            self.result_text_max += f"Значение целевой функции: {-self.result_max.fun}\n"
            self.result_text_max += f"Кол-во итераций: {self.iteration_max+1}\n"
        else:
            self.result_text_max += "Решение не найдено.\n"
            self.result_text_max += f"Кол-во итераций: {self.iteration_max+1}\n"

        self.results_label_min = tk.Label(results_window, text=self.result_text_min, bg=self.label_bg)
        self.results_label_min.grid(row=0, column=0, padx=10, pady=10)

        self.results_label_max = tk.Label(results_window, text=self.result_text_max, bg=self.label_bg)
        self.results_label_max.grid(row=0, column=1, padx=10, pady=10)

        self.show_path_button = tk.Button(results_window, text="Показать каждую итерацию", command=lambda: self.show_solution_path(results_window), bg=self.button_bg, activebackground=self.activ_bg)
        self.show_path_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.hide_path_button = tk.Button(results_window, text="Скрыть", command=lambda: self.hide_solution_path(), bg=self.button_bg, activebackground=self.activ_bg)
        self.hide_path_button.grid(row=1, column=0, columnspan=2, pady=10)
        self.hide_path_button.grid_remove()

    def show_solution_path(self, parent_window):
        self.min_path_text = ""
        self.max_path_text = ""

        if hasattr(self, 'intermediate_steps_min'):
            for i, step in enumerate(self.intermediate_steps_min):
                self.min_path_text += f"Итерация {i}: x1,x2 = {step}; f(x1,x2) = {self.func(step)}\n"
        else:
            self.min_path_text = "Не удалось найти экстремум\n"

        if hasattr(self, 'intermediate_steps_max'):
            for i, step in enumerate(self.intermediate_steps_max):
                self.max_path_text += f"Итерация {i}: x1,x2 = {step}; f(x1,x2) = {self.func(step)}\n"
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

        self.optimize("min")
        self.optimize("max")

        graph_window = tk.Toplevel(self)
        graph_window.title("Графическое решение")
        graph_window.geometry("1000x900")
        graph_window["bg"] = self["bg"]

        self.figure = plt.Figure(figsize=(8, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, graph_window)
        self.canvas.get_tk_widget().pack(side="bottom", fill=tk.NONE, expand=False)
        toolbar = NavigationToolbar2Tk(self.canvas, graph_window)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill="both")

        options_frame = tk.Frame(graph_window, bg=self["bg"])
        options_frame.pack(side=tk.TOP, fill=tk.X)

        self.shade_feasible_var = tk.BooleanVar()
        self.show_minimum_var = tk.BooleanVar()
        self.show_maximum_var = tk.BooleanVar()
        self.show_convergence_min_var = tk.BooleanVar()
        self.show_convergence_max_var = tk.BooleanVar()

        shade_feasible_cb = tk.Checkbutton(options_frame, text="Область допустимых решений", variable=self.shade_feasible_var, command=lambda: self.update_graph(), bg=self["bg"], activebackground=self.activ_bg)
        shade_feasible_cb.pack(side=tk.LEFT)

        show_minimum_cb = tk.Checkbutton(options_frame, text="Точка минимума", variable=self.show_minimum_var, command=lambda: self.update_graph(), bg=self["bg"], activebackground=self.activ_bg)
        show_minimum_cb.pack(side=tk.LEFT)

        show_maximum_cb = tk.Checkbutton(options_frame, text="Точка максимума", variable=self.show_maximum_var, command=lambda: self.update_graph(), bg=self["bg"], activebackground=self.activ_bg)
        show_maximum_cb.pack(side=tk.LEFT)

        show_convergence_min_cb = tk.Checkbutton(options_frame, text="Траектория сходимости (мин)", variable=self.show_convergence_min_var, command=lambda: self.update_graph(), bg=self["bg"], activebackground=self.activ_bg)
        show_convergence_min_cb.pack(side=tk.LEFT)

        show_convergence_max_cb = tk.Checkbutton(options_frame, text="Траектория сходимости (макс)", variable=self.show_convergence_max_var, command=lambda: self.update_graph(), bg=self["bg"], activebackground=self.activ_bg)
        show_convergence_max_cb.pack(side=tk.LEFT)

        self.update_graph()

    def plot_function_and_constraints(self):
        self.ax.clear()
        self.ax.set_aspect('equal')
        x_vals = np.linspace(-15, 15, 600)
        y_vals = np.linspace(-15, 15, 600)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        func_str = self.entry_func.get()
        Z = eval(func_str.replace("x[1]", "X").replace("x[2]", "Y"), {}, {"X": X, "Y": Y})

        levels = []
        if self.result_min:
            levels.append(self.func([self.result_min.x[0], self.result_min.x[1]]))
        if self.result_max:
            levels.append(self.func([self.result_max.x[0], self.result_max.x[1]]))

        if levels:
            levels = np.unique(levels)
            contour = self.ax.contour(X, Y, Z, levels=levels, colors='darkred')
            self.ax.clabel(contour, inline=True, fontsize=8)

        self.colors = itertools.cycle(['b', 'y', 'm', 'c', 'k', 'gold', 'blueviolet'])
        
        self.legend_list = []

        for entry in self.constraint_entries:
            constraint = entry.get()
            if "<=" in constraint:
                expr, bound = constraint.split("<=")
                expr = expr.replace("x[1]", "X").replace("x[2]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                contour = self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=next(self.colors))
                self.ax.clabel(contour, inline=True, fontsize=8, fmt={float(bound): f'{constraint}'})
            elif ">=" in constraint:
                expr, bound = constraint.split(">=")
                expr = expr.replace("x[1]", "X").replace("x[2]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                contour = self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=next(self.colors))
                self.ax.clabel(contour, inline=True, fontsize=8, fmt={float(bound): f'{constraint}'})
            elif ">" in constraint:
                expr, bound = constraint.split(">")
                expr = expr.replace("x[1]", "X").replace("x[2]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                contour = self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=next(self.colors))
                self.ax.clabel(contour, inline=True, fontsize=8, fmt={float(bound): f'{constraint}'})
            elif "<" in constraint:
                expr, bound = constraint.split("<")
                expr = expr.replace("x[1]", "X").replace("x[2]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                contour = self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=next(self.colors))
                self.ax.clabel(contour, inline=True, fontsize=8, fmt={float(bound): f'{constraint}'})
            elif "=" in constraint:
                expr, bound = constraint.split("=")
                expr = expr.replace("x[1]", "X").replace("x[2]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                contour = self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=next(self.colors))
                self.ax.clabel(contour, inline=True, fontsize=8, fmt={float(bound): f'{constraint}'})

        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.set_xlabel("X1")
        self.ax.set_ylabel("X2")
        self.ax.grid(True)
               
    def update_graph(self):
        self.plot_function_and_constraints()

        x = np.linspace(-15, 15, 600)
        y = np.linspace(-15, 15, 600)
        X, Y = np.meshgrid(x, y)
        #Z = eval(self.entry_func.get(), {}, {"x": [X, Y]})

        color_min = 'r'
        color_max = 'g'

        if self.show_minimum_var.get() and self.result_min is not None:
            if not self.result_min:
                self.result_min = self.optimize("min")
            if self.result_min.success:
                self.ax.plot(self.result_min.x[0], self.result_min.x[1], f'{color_min}o', markersize=7, label='Минимум')
                
        if self.show_maximum_var.get() and self.result_max is not None:
            if not self.result_max:
                self.result_max = self.optimize("max")
            if self.result_max.success:
                self.ax.plot(self.result_max.x[0], self.result_max.x[1], f'{color_max}o', markersize=7, label='Максимум')

        if self.show_convergence_min_var.get() and hasattr(self, 'intermediate_steps_min'):
            self.intermediate_steps_min = np.array(self.intermediate_steps_min)
            self.ax.plot(self.intermediate_steps_min[:, 0], self.intermediate_steps_min[:, 1], f'{color_min}o-', label='Траектория сходимости (мин)')

        if self.show_convergence_max_var.get() and hasattr(self, 'intermediate_steps_max'):
            self.intermediate_steps_max = np.array(self.intermediate_steps_max)
            self.ax.plot(self.intermediate_steps_max[:, 0], self.intermediate_steps_max[:, 1], f'{color_max}o-', label='Траектория сходимости (макс)')
        
        if self.shade_feasible_var.get():
            self.shade_feasible_region()
        
        self.ax.legend()
        
        self.canvas.draw()

    def shade_feasible_region(self):
        x = np.linspace(-15, 15, 600)
        y = np.linspace(-15, 15, 600)
        X, Y = np.meshgrid(x, y)
        feasible = np.ones_like(X, dtype=bool)

        for entry in self.constraint_entries:
            constraint = entry.get()
            if "<=" in constraint:
                expr, bound = constraint.split("<=")
                expr = expr.replace("x[1]", "X").replace("x[2]", "Y")
                feasible = feasible & (eval(expr, {}, {"X": X, "Y": Y}) <= float(bound))
            elif ">=" in constraint:
                expr, bound = constraint.split(">=")
                expr = expr.replace("x[1]", "X").replace("x[2]", "Y")
                feasible = feasible & (eval(expr, {}, {"X": X, "Y": Y}) >= float(bound))
            elif ">" in constraint:
                expr, bound = constraint.split(">")
                expr = expr.replace("x[1]", "X").replace("x[2]", "Y")
                feasible = feasible & (eval(expr, {}, {"X": X, "Y": Y}) > float(bound))
            elif "<" in constraint:
                expr, bound = constraint.split("<")
                expr = expr.replace("x[1]", "X").replace("x[2]", "Y")
                feasible = feasible & (eval(expr, {}, {"X": X, "Y": Y}) < float(bound))
            elif "=" in constraint:
                expr, bound = constraint.split("=")
                expr = expr.replace("x[1]", "X").replace("x[2]", "Y")
                feasible = feasible & (eval(expr, {}, {"X": X, "Y": Y}) == float(bound))

        self.ax.contourf(X, Y, feasible, levels=[0.5, 1], colors='orange', alpha=0.3)

    def open_3d_window(self):
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        x = np.linspace(-15, 15, 600)
        y = np.linspace(-15, 15, 600)
        X, Y = np.meshgrid(x, y)

        func_str = self.entry_func.get()
        Z = eval(func_str.replace("x[1]", "X").replace("x[2]", "Y"), {}, {"X": X, "Y": Y})

        surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis, edgecolor='none', alpha=0.7)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Целевая функция")

        colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
        legend_elements = []
        legend_labels = []

        min_z = self.func(self.result_min.x) if self.result_min.success else None
        max_z = -self.result_max.fun if self.result_max.success else None

        for i, entry in enumerate(self.constraint_entries):
            constraint = entry.get()
            color = colors[i % len(colors)]
            if "<=" in constraint:
                expr, bound = constraint.split("<=")
                expr = expr.replace("x[1]", "X").replace("x[2]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                for offset in [Z.min(), min_z, max_z]:
                    if offset is not None:
                        contour = ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=color, offset=offset)
                legend_elements.append(Line2D([0], [0], color=color))
                legend_labels.append(f"{constraint}")
            elif ">=" in constraint:
                expr, bound = constraint.split(">=")
                expr = expr.replace("x[1]", "X").replace("x[2]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                for offset in [Z.min(), min_z, max_z]:
                    if offset is not None:
                        contour = ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=color, offset=offset)
                legend_elements.append(Line2D([0], [0], color=color))
                legend_labels.append(f"{constraint}")
            elif "==" in constraint:
                expr, bound = constraint.split("==")
                expr = expr.replace("x[1]", "X").replace("x[2]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                for offset in [Z.min(), min_z, max_z]:
                    if offset is not None:
                        contour = ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=color, offset=offset)
                legend_elements.append(Line2D([0], [0], color=color))
                legend_labels.append(f"{constraint}")

        if self.result_min.success:
            min_point = ax.scatter(self.result_min.x[0], self.result_min.x[1], self.func(self.result_min.x), color='r', s=50, label="Минимум")
            legend_elements.append(min_point)
            legend_labels.append("Минимум")

        if self.result_max.success:
            max_point = ax.scatter(self.result_max.x[0], self.result_max.x[1], -self.result_max.fun, color='g', s=50, label="Максимум")
            legend_elements.append(max_point)
            legend_labels.append("Максимум")

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$f(x)$")
        ax.set_title("3D модель")
        ax.legend(legend_elements, legend_labels)

        plt.show()

if __name__ == "__main__":
    app = OptimizationApp()
    app.mainloop()
