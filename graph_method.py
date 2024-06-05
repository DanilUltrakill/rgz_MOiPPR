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
        self.title("Optimization Tool")
        self.geometry("600x800")

        self.constraints = []
        self.result_min = None
        self.result_max = None
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

        # Кнопка для удаления последнего ограничения
        self.button_remove_constraint = tk.Button(self, text="Remove Constraint", command=self.remove_constraint)
        self.button_remove_constraint.pack()

        # Кнопка для начала оптимизации
        self.button_optimize = tk.Button(self, text="Calculate", command=self.calculate)
        self.button_optimize.pack()

        # Кнопка для отображения графика
        self.button_show_graph = tk.Button(self, text="Show Graph", command=self.open_graph_window)
        self.button_show_graph.pack()

        # Кнопка для отображения 3D-модели
        self.button_show_3d = tk.Button(self, text="Show 3D Model", command=self.open_3d_window, state=tk.DISABLED)
        self.button_show_3d.pack()

    def add_constraint(self):
        entry_constraint = tk.Entry(self, width=50)
        entry_constraint.pack()
        self.constraint_entries.append(entry_constraint)

    def remove_constraint(self):
        if self.constraint_entries:
            entry_constraint = self.constraint_entries.pop()
            entry_constraint.pack_forget()

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
        results_window.title("Optimization Results")
        results_window.geometry("800x600")

        result_text_min = ""
        result_text_max = ""

        if self.result_min.success:
            result_text_min += f"Minimum found at: {self.result_min.x}\n"
            result_text_min += f"Function value: {self.func(self.result_min.x)}\n"
            result_text_min += f"Iterations: {self.iteration_min}\n"
        else:
            result_text_min += "Minimum optimization failed.\n"

        if self.result_max.success:
            result_text_max += f"Maximum found at: {self.result_max.x}\n"
            result_text_max += f"Function value: {-self.result_max.fun}\n"
            result_text_max += f"Iterations: {self.iteration_max}\n"
        else:
            result_text_max += "Maximum optimization failed.\n"

        results_label_min = tk.Label(results_window, text=result_text_min)
        results_label_min.grid(row=0, column=0, padx=10, pady=10)

        results_label_max = tk.Label(results_window, text=result_text_max)
        results_label_max.grid(row=0, column=1, padx=10, pady=10)

        show_path_button = tk.Button(results_window, text="Show Solution Path", command=lambda: self.show_solution_path(results_window))
        show_path_button.grid(row=1, column=0, columnspan=2, pady=10)

    def show_solution_path(self, parent_window):
        min_path_text = ""
        max_path_text = ""

        if hasattr(self, 'intermediate_steps_min'):
            for i, step in enumerate(self.intermediate_steps_min):
                min_path_text += f"Iteration {i}: x = {step}; f(x) = {self.func(step)}\n"
        else:
            min_path_text = "Не удалось найти экстремум\n"

        if hasattr(self, 'intermediate_steps_max'):
            for i, step in enumerate(self.intermediate_steps_max):
                max_path_text += f"Iteration {i}: x = {step}; f(x) = {self.func(step)}\n"
        else:
            max_path_text = "Не удалось найти экстремум\n"

        min_path_label = tk.Label(parent_window, text=min_path_text, justify=tk.LEFT)
        min_path_label.grid(row=2, column=0, padx=10, pady=10)

        max_path_label = tk.Label(parent_window, text=max_path_text, justify=tk.LEFT)
        max_path_label.grid(row=2, column=1, padx=10, pady=10)

    def open_graph_window(self):

        self.optimize("min")
        self.optimize("max")

        graph_window = tk.Toplevel(self)
        graph_window.title("Function Graph")
        graph_window.geometry("800x600")

        self.figure = plt.Figure(figsize=(10, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)

        options_frame = tk.Frame(graph_window)
        options_frame.pack(side=tk.TOP, fill=tk.X)

        self.shade_feasible_var = tk.BooleanVar()
        self.show_minimum_var = tk.BooleanVar()
        self.show_maximum_var = tk.BooleanVar()
        self.show_convergence_min_var = tk.BooleanVar()
        self.show_convergence_max_var = tk.BooleanVar()

        shade_feasible_cb = tk.Checkbutton(options_frame, text="Shade Feasible Region", variable=self.shade_feasible_var, command=lambda: self.update_graph())
        shade_feasible_cb.pack(side=tk.LEFT)

        show_minimum_cb = tk.Checkbutton(options_frame, text="Show Minimum", variable=self.show_minimum_var, command=lambda: self.update_graph())
        show_minimum_cb.pack(side=tk.LEFT)

        show_maximum_cb = tk.Checkbutton(options_frame, text="Show Maximum", variable=self.show_maximum_var, command=lambda: self.update_graph())
        show_maximum_cb.pack(side=tk.LEFT)

        show_convergence_min_cb = tk.Checkbutton(options_frame, text="Show Convergence (Min)", variable=self.show_convergence_min_var, command=lambda: self.update_graph())
        show_convergence_min_cb.pack(side=tk.LEFT)

        show_convergence_max_cb = tk.Checkbutton(options_frame, text="Show Convergence (Max)", variable=self.show_convergence_max_var, command=lambda: self.update_graph())
        show_convergence_max_cb.pack(side=tk.LEFT)

        self.canvas = FigureCanvasTkAgg(self.figure, graph_window)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM,fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, graph_window)
        toolbar.update()
        toolbar.pack(side=tk.LEFT, fill=tk.X)
        
        self.update_graph()

    def plot_function_and_constraints(self):
        self.ax.clear()
        
        x_vals = np.linspace(-10, 10, 400)
        y_vals = np.linspace(-10, 10, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        func_str = self.entry_func.get()
        Z = eval(func_str.replace("x[0]", "X").replace("x[1]", "Y"), {}, {"X": X, "Y": Y})

        levels = []
        if self.result_min:
            levels.append(self.func([self.result_min.x[0], self.result_min.x[1]]))
        if self.result_max:
            levels.append(self.func([self.result_max.x[0], self.result_max.x[1]]))

        if levels:
            levels.sort()
            contour = self.ax.contour(X, Y, Z, levels=levels, colors='darkred')
            self.ax.clabel(contour, inline=True, fontsize=8)

        self.legend_list = []
        self.colors = itertools.cycle(['b', 'y', 'm', 'c', 'k', 'gold', 'blueviolet'])
        
        for entry in self.constraint_entries:
            constraint = entry.get()
            if "<=" in constraint:
                expr, bound = constraint.split("<=")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=next(self.colors))
                self.legend_list.append(f'{expr}')
            elif ">=" in constraint:
                expr, bound = constraint.split(">=")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=next(self.colors))
                self.legend_list.append(f'{expr}')
            elif "==" in constraint:
                expr, bound = constraint.split("==")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=next(self.colors))
                self.legend_list.append(f'{expr}')

    
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel("X1")
        self.ax.set_ylabel("X2")
        self.ax.grid(True)
        

    def update_graph(self):
        self.plot_function_and_constraints()
        x = np.linspace(-10, 10, 400)
        y = np.linspace(-10, 10, 400)
        X, Y = np.meshgrid(x, y)
        Z = eval(self.entry_func.get(), {}, {"x": [X, Y]})

        color_min = 'r'
        color_max = 'g'

        if self.show_minimum_var.get() and self.result_min is not None:
                self.ax.plot(self.result_min.x[0], self.result_min.x[1], f'{color_min}o', markersize=10, label="Minimum")
                
        if self.show_maximum_var.get() and self.result_max is not None:
            if not self.result_max:
                self.result_max = self.optimize("max")
            if self.result_max.success:
                self.ax.plot(self.result_max.x[0], self.result_max.x[1], f'{color_max}o', markersize=10, label="Maximum")

        if self.show_convergence_min_var.get() and hasattr(self, 'intermediate_steps_min'):
            self.intermediate_steps_min = np.array(self.intermediate_steps_min)
            self.ax.plot(self.intermediate_steps_min[:, 0], self.intermediate_steps_min[:, 1], f'{color_min}o-', label="Convergence Path (Min)")

        if self.show_convergence_max_var.get() and hasattr(self, 'intermediate_steps_max'):
            self.intermediate_steps_max = np.array(self.intermediate_steps_max)
            self.ax.plot(self.intermediate_steps_max[:, 0], self.intermediate_steps_max[:, 1], f'{color_max}o-', label="Convergence Path (Max)")
        
        if self.shade_feasible_var.get():
            self.shade_feasible_region()
        
        self.ax.legend(self.legend_list)
        
        self.canvas.draw()

    def shade_feasible_region(self):
        x = np.linspace(-10, 10, 400)
        y = np.linspace(-10, 10, 400)
        X, Y = np.meshgrid(x, y)
        feasible = np.ones_like(X, dtype=bool)

        for entry in self.constraint_entries:
            constraint = entry.get()
            if "<=" in constraint:
                expr, bound = constraint.split("<=")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                feasible = feasible & (eval(expr, {}, {"X": X, "Y": Y}) <= float(bound))
            elif ">=" in constraint:
                expr, bound = constraint.split(">=")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                feasible = feasible & (eval(expr, {}, {"X": X, "Y": Y}) >= float(bound))
            elif "==" in constraint:
                expr, bound = constraint.split("==")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                feasible = feasible & (eval(expr, {}, {"X": X, "Y": Y}) == float(bound))

        self.ax.contourf(X, Y, feasible, levels=[0.5, 1], colors='orange', alpha=0.3)

    def open_3d_window(self):
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        x = np.linspace(-10, 10, 400)
        y = np.linspace(-10, 10, 400)
        X, Y = np.meshgrid(x, y)

        func_str = self.entry_func.get()
        Z = eval(func_str.replace("x[0]", "X").replace("x[1]", "Y"), {}, {"X": X, "Y": Y})

        surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.viridis, edgecolor='none', alpha=0.7)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Target Function")

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
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                for offset in [Z.min(), min_z, max_z]:
                    if offset is not None:
                        contour = ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=color, offset=offset)
                legend_elements.append(Line2D([0], [0], color=color))
                legend_labels.append(f"{constraint}")
            elif ">=" in constraint:
                expr, bound = constraint.split(">=")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                for offset in [Z.min(), min_z, max_z]:
                    if offset is not None:
                        contour = ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=color, offset=offset)
                legend_elements.append(Line2D([0], [0], color=color))
                legend_labels.append(f"{constraint}")
            elif "==" in constraint:
                expr, bound = constraint.split("==")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                for offset in [Z.min(), min_z, max_z]:
                    if offset is not None:
                        contour = ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=color, offset=offset)
                legend_elements.append(Line2D([0], [0], color=color))
                legend_labels.append(f"{constraint}")

        if self.result_min.success:
            min_point = ax.scatter(self.result_min.x[0], self.result_min.x[1], self.func(self.result_min.x), color='r', s=50, label="Minimum")
            legend_elements.append(min_point)
            legend_labels.append("Minimum")

        if self.result_max.success:
            max_point = ax.scatter(self.result_max.x[0], self.result_max.x[1], -self.result_max.fun, color='g', s=50, label="Maximum")
            legend_elements.append(max_point)
            legend_labels.append("Maximum")

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$f(x)$")
        ax.set_title("3D Model")
        ax.legend(legend_elements, legend_labels)

        plt.show()

if __name__ == "__main__":
    app = OptimizationApp()
    app.mainloop()
