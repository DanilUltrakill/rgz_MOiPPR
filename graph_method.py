import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import cm
import itertools

class OptimizationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optimization Tool")
        self.geometry("600x500")

        self.constraints = []
        self.result_min = None
        self.result_max = None

        self.legend_list = []

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
        self.button_optimize = tk.Button(self, text="Calculate", command=self.open_results_window)
        self.button_optimize.pack()

        # Кнопка для отображения графика
        self.button_show_graph = tk.Button(self, text="Show Graph", command=self.open_graph_window)
        self.button_show_graph.pack()

        # Кнопка для отображения 3D-модели
        self.button_show_3d = tk.Button(self, text="Show 3D Model", command=self.open_3d_window)
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

    def optimize(self, optimization_type):
        initial_guess = np.array([float(x) for x in self.entry_initial.get().split(",")])
        constraints = self.parse_constraints()
        self.intermediate_steps = [initial_guess]
        self.iteration = 0

        def callback(xk):
            print(f"Iteration {self.iteration}: x = {xk}; f(x) = {self.func(xk)}")
            self.intermediate_steps.append(xk.copy())
            self.iteration += 1

        if optimization_type == "min":
            result = minimize(self.func, initial_guess, method='SLSQP', constraints=constraints, tol=0.001, callback=callback)
        else:
            def neg_func(x):
                return -self.func(x)
            result = minimize(neg_func, initial_guess, method='SLSQP', constraints=constraints, tol=0.001, callback=callback)

        return result

    def open_results_window(self):
        self.result_min = self.optimize("min")
        self.result_max = self.optimize("max")

        results_window = tk.Toplevel(self)
        results_window.title("Optimization Results")
        results_window.geometry("800x600")

        result_text = ""
        if self.result_min.success:
            result_text_min += f"Minimum found at: {self.result_min.x}\n"
            result_text_min += f"Function value: {self.func(self.result_min.x)}\n"
            result_text_min += f"Iterations: {self.iteration_min}\n"
        else:
            result_text += "Minimum optimization failed.\n\n"

        if self.result_max.success:
            result_text_max += f"Maximum found at: {self.result_max.x}\n"
            result_text_max += f"Function value: {-self.result_max.fun}\n"
            result_text_max += f"Iterations: {self.iteration_max}\n"
        else:
            result_text += "Maximum optimization failed.\n"

        results_label = tk.Label(results_window, text=result_text)
        results_label.pack()

        show_path_button = tk.Button(results_window, text="Show Solution Path", command=self.show_solution_path)
        show_path_button.pack()

    def show_solution_path(self):
        path_window = tk.Toplevel(self)
        path_window.title("Solution Path")
        path_window.geometry("800x600")

        fig, ax = plt.subplots()

        x_path = [step[0] for step in self.intermediate_steps]
        y_path = [step[1] for step in self.intermediate_steps]

        ax.plot(x_path, y_path, 'bo-', label="Solution Path")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title("Solution Path")
        ax.grid(True)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=path_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, path_window)
        toolbar.update()
        canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

    def open_graph_window(self):

        self.optimize("min")
        self.optimize("max")

        graph_window = tk.Toplevel(self)
        graph_window.title("Function Graph")
        graph_window.geometry("1000x900")

        self.figure = plt.Figure(figsize=(8, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, graph_window)
        self.canvas.get_tk_widget().pack(side="bottom", fill=tk.NONE, expand=False)
        toolbar = NavigationToolbar2Tk(self.canvas, graph_window)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill="both")

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
            levels = np.unique(levels)
            contour = self.ax.contour(X, Y, Z, levels=levels, colors='darkred')
            self.ax.clabel(contour, inline=True, fontsize=8)

        self.colors = itertools.cycle(['b', 'y', 'm', 'c', 'k', 'gold', 'blueviolet'])
        
        self.legend_list = []

        for entry in self.constraint_entries:
            constraint = entry.get()
            if "<=" in constraint:
                expr, bound = constraint.split("<=")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                contour = self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=next(self.colors))
                self.ax.clabel(contour, inline=True, fontsize=8, fmt={float(bound): f'{constraint}'})
            elif ">=" in constraint:
                expr, bound = constraint.split(">=")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                contour = self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=next(self.colors))
                self.ax.clabel(contour, inline=True, fontsize=8, fmt={float(bound): f'{constraint}'})
            elif "==" in constraint:
                expr, bound = constraint.split("==")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                contour = self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors=next(self.colors))
                self.ax.clabel(contour, inline=True, fontsize=8, fmt={float(bound): f'{constraint}'})

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
            if not self.result_min:
                self.result_min = self.optimize("min")
            if self.result_min.success:
                self.ax.plot(self.result_min.x[0], self.result_min.x[1], f'{color_min}o', markersize=7, label='Minimum')
                
        if self.show_maximum_var.get() and self.result_max is not None:
            if not self.result_max:
                self.result_max = self.optimize("max")
            if self.result_max.success:
                self.ax.plot(self.result_max.x[0], self.result_max.x[1], f'{color_max}o', markersize=7, label='Maximum')

        if self.show_convergence_min_var.get() and hasattr(self, 'intermediate_steps_min'):
            self.intermediate_steps_min = np.array(self.intermediate_steps_min)
            self.ax.plot(self.intermediate_steps_min[:, 0], self.intermediate_steps_min[:, 1], f'{color_min}o-', label='Convergence Path (Min)')

        if self.show_convergence_max_var.get() and hasattr(self, 'intermediate_steps_max'):
            self.intermediate_steps_max = np.array(self.intermediate_steps_max)
            self.ax.plot(self.intermediate_steps_max[:, 0], self.intermediate_steps_max[:, 1], f'{color_max}o-', label='Convergence Path (Max)')
        
        if self.shade_feasible_var.get():
            self.shade_feasible_region()
        
        self.ax.legend()
        
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
        graph_window = tk.Toplevel(self)
        graph_window.title("3D Graph")
        graph_window.geometry("800x600")

        self.figure_3d = plt.Figure(figsize=(10, 8), dpi=100)
        self.ax_3d = self.figure_3d.add_subplot(111, projection='3d')

        toolbar = NavigationToolbar2Tk(FigureCanvasTkAgg(self.figure_3d, graph_window), graph_window)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas_3d = FigureCanvasTkAgg(self.figure_3d, graph_window)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_3d_function_and_constraints()

    def plot_3d_function_and_constraints(self):
        self.ax_3d.clear()
        
        x_vals = np.linspace(-5, 5, 100)
        y_vals = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.array([[self.func([x, y]) for x in x_vals] for y in y_vals])

        self.ax_3d.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6)

        constraints = self.parse_constraints()
        colors = itertools.cycle(['r', 'b', 'g', 'y', 'm', 'c'])
        constraint_lines = []
        for i, constraint in enumerate(constraints):
            color = next(colors)
            if constraint['type'] == 'ineq':
                Z_constraint = np.array([[constraint['fun']([x, y]) for x in x_vals] for y in y_vals])
                c = self.ax_3d.contour(X, Y, Z_constraint, levels=[0], colors=color)
                constraint_lines.append(c)
            elif constraint['type'] == 'eq':
                Z_constraint = np.array([[constraint['fun']([x, y]) for x in x_vals] for y in y_vals])
                c = self.ax_3d.contour(X, Y, Z_constraint, levels=[0], colors=color, linestyles='dashed')
                constraint_lines.append(c)

        if hasattr(self, 'result_min'):
            self.ax_3d.scatter(*self.result_min.x, self.func(self.result_min.x), color='r', label='Minimum', marker='o')
        
        if hasattr(self, 'result_max'):
            self.ax_3d.scatter(*self.result_max.x, self.func(self.result_max.x), color='g', label='Maximum', marker='o')

        self.ax_3d.set_xlim(-5, 5)
        self.ax_3d.set_ylim(-5, 5)
        self.ax_3d.set_zlim(np.min(Z), np.max(Z))
        
        self.ax_3d.set_xlabel('X1')
        self.ax_3d.set_ylabel('X2')
        self.ax_3d.set_zlabel('f(X1, X2)')

        legend_elements = [line.collections[0] for line in constraint_lines] + [self.ax_3d.scatter([], [], color='r', label='Minimum'), self.ax_3d.scatter([], [], color='g', label='Maximum')]
        legend_labels = [f"Constraint {i+1}" for i in range(len(constraint_lines))] + ['Minimum', 'Maximum']
        self.ax_3d.legend(legend_elements, legend_labels, loc='upper right')

        self.canvas_3d.draw()

if __name__ == "__main__":
    app = OptimizationApp()
    app.mainloop()
