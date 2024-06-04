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
        self.button_optimize = tk.Button(self, text="Calculate", command=self.open_results_window)
        self.button_optimize.pack()

        # Кнопка для отображения графика
        self.button_show_graph = tk.Button(self, text="Show Graph", command=self.open_graph_window)
        self.button_show_graph.pack()

        # Опциональная кнопка для отображения 3D-модели
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
            result_text += f"Minimum found at: {self.result_min.x}\nFunction value: {self.func(self.result_min.x)}\n\n"
        else:
            result_text += "Minimum optimization failed.\n\n"

        if self.result_max.success:
            result_text += f"Maximum found at: {self.result_max.x}\nFunction value: {self.func(self.result_max.x)}\n"
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
        graph_window = tk.Toplevel(self)
        graph_window.title("Function Graph")
        graph_window.geometry("800x600")

        self.figure = plt.Figure(figsize=(10, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, graph_window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_function_and_constraints()

        toolbar = NavigationToolbar2Tk(self.canvas, graph_window)
        toolbar.update()
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

        shade_feasible_var = tk.BooleanVar()
        show_minimum_var = tk.BooleanVar()
        show_maximum_var = tk.BooleanVar()
        show_convergence_var = tk.BooleanVar()

        shade_feasible_cb = tk.Checkbutton(graph_window, text="Shade Feasible Area", variable=shade_feasible_var, command=lambda: self.update_graph(shade_feasible_var, show_minimum_var, show_maximum_var, show_convergence_var))
        shade_feasible_cb.pack()

        show_minimum_cb = tk.Checkbutton(graph_window, text="Show Minimum", variable=show_minimum_var, command=lambda: self.update_graph(shade_feasible_var, show_minimum_var, show_maximum_var, show_convergence_var))
        show_minimum_cb.pack()

        show_maximum_cb = tk.Checkbutton(graph_window, text="Show Maximum", variable=show_maximum_var, command=lambda: self.update_graph(shade_feasible_var, show_minimum_var, show_maximum_var, show_convergence_var))
        show_maximum_cb.pack()

        show_convergence_cb = tk.Checkbutton(graph_window, text="Show Convergence Path", variable=show_convergence_var, command=lambda: self.update_graph(shade_feasible_var, show_minimum_var, show_maximum_var, show_convergence_var))
        show_convergence_cb.pack()

    def plot_function_and_constraints(self):
        self.ax.clear()

        x = np.linspace(-10, 10, 400)
        y = np.linspace(-10, 10, 400)
        X, Y = np.meshgrid(x, y)

        func_str = self.entry_func.get()
        Z = eval(func_str.replace("x[0]", "X").replace("x[1]", "Y"), {}, {"X": X, "Y": Y})

        self.ax.contour(X, Y, Z, levels=25, cmap=cm.viridis)

        for entry in self.constraint_entries:
            constraint = entry.get()
            if "<=" in constraint:
                expr, bound = constraint.split("<=")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors='r')
            elif ">=" in constraint:
                expr, bound = constraint.split(">=")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors='r')
            elif "==" in constraint:
                expr, bound = constraint.split("==")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                self.ax.contour(X, Y, Z_constraint, levels=[float(bound)], colors='r')

        self.ax.set_xlabel("$x_1$")
        self.ax.set_ylabel("$x_2$")
        self.ax.set_aspect('equal', 'box')
        self.ax.set_title("Function Graph")
        self.ax.grid(True)

        self.canvas.draw()

    def update_graph(self, shade_feasible_var, show_minimum_var, show_maximum_var, show_convergence_var):
        self.plot_function_and_constraints()

        if shade_feasible_var.get():
            self.shade_feasible_area()

        if show_minimum_var.get():
            if not self.result_min:
                self.result_min = self.optimize("min")
            if self.result_min.success:
                self.ax.plot(self.result_min.x[0], self.result_min.x[1], 'ro', markersize=10, label="Minimum")

        if show_maximum_var.get():
            if not self.result_max:
                self.result_max = self.optimize("max")
            if self.result_max.success:
                self.ax.plot(self.result_max.x[0], self.result_max.x[1], 'go', markersize=10, label="Maximum")

        if show_convergence_var.get():
            intermediate_steps = np.array(self.intermediate_steps)
            self.ax.plot(intermediate_steps[:, 0], intermediate_steps[:, 1], 'bo-', label="Convergence Path")

        self.ax.legend()
        self.canvas.draw()

    def shade_feasible_area(self):
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

        ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none')

        for entry in self.constraint_entries:
            constraint = entry.get()
            if "<=" in constraint:
                expr, bound = constraint.split("<=")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                ax.contour3D(X, Y, Z_constraint, levels=[float(bound)], colors='r')
            elif ">=" in constraint:
                expr, bound = constraint.split(">=")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                ax.contour3D(X, Y, Z_constraint, levels=[float(bound)], colors='r')
            elif "==" in constraint:
                expr, bound = constraint.split("==")
                expr = expr.replace("x[0]", "X").replace("x[1]", "Y")
                Z_constraint = eval(expr, {}, {"X": X, "Y": Y})
                ax.contour3D(X, Y, Z_constraint, levels=[float(bound)], colors='r')

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$f(x)$")
        ax.set_title("3D Model")

        plt.show()

if __name__ == "__main__":
    app = OptimizationApp()
    app.mainloop()
