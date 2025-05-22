import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter.font import Font
from take_data_popup import GasDataCollector
from tkinter import messagebox
from get_HOA_AdSim import get_HOA
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl
from isotherm_finder_AdSim_Min import isothermFinder
from isotherm_manage import IsothermEditor

mpl.rcParams['mathtext.fontset'] = 'stix'  # or 'cm' for Computer Modern
mpl.rcParams['font.family'] = 'STIXGeneral'  # can also be 'DejaVu Serif' or others


class IsothermApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚗ Isotherm Model Fitter | ALM")
        self.root.geometry("1000x1000")
        self.root.configure(padx=20, pady=20)
        
        self.data = {}
        self.fitter = None
        self.HOA_find = None
        self.selected_models = []
        self.best_fit = ("", {})

        # Outer frame to contain everything
        outer_frame = ttk.Frame(self.root)
        outer_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for scrolling
        self.main_canvas = tk.Canvas(outer_frame)
        self.main_canvas.grid(row=0, column=0, sticky="nsew")

        # Vertical scrollbar
        v_scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=self.main_canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.main_canvas.configure(yscrollcommand=v_scrollbar.set)

        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(outer_frame, orient="horizontal", command=self.main_canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.main_canvas.configure(xscrollcommand=h_scrollbar.set)

        # Configure resizing behavior
        outer_frame.columnconfigure(0, weight=1)
        outer_frame.rowconfigure(0, weight=1)

        # Frame inside canvas
        self.main_frame = ttk.Frame(self.main_canvas)
        self.main_canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        # Scroll region update on content change
        def on_frame_configure(event):
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

        self.main_frame.bind("<Configure>", on_frame_configure)

        # Mousewheel scroll
        self.main_canvas.bind_all("<MouseWheel>", lambda event: self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units"))

        # Now build GUI inside main_frame
        self.build_gui(self.main_frame)

        # Force update idle tasks to ensure geometry is calculated
        self.root.update_idletasks()
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

        # Resize the window to fit the content naturally
        w = int(self.main_canvas.bbox("all")[2] * 1.1)
        h = self.main_canvas.bbox("all")[3]
        self.root.geometry(f"{min(w, 1250)}x{min(h, 1000)}")

    def build_gui(self, parent):
        # Header Section
        header_font = Font(family="Helvetica", size=22, weight="bold")
        subheader_font = Font(family="Helvetica", size=14, weight="bold")
        
        title_frame = ttk.Frame(parent)
        title_frame.pack(pady=(0, 10))
        
        ttk.Label(title_frame,
                  text="Isotherm Model Fitter ⚗",
                  font=header_font,
                  bootstyle="primary").pack(side=LEFT, padx=(0, 15))
        ttk.Label(title_frame,
                  text="| ALM",
                  font=subheader_font,
                  bootstyle="info").pack(side=LEFT)
        
        # Experiment Input Frame
        input_frame = ttk.Labelframe(parent, text="Experiment Info", bootstyle="info")
        input_frame.pack(fill=X, pady=10, padx=5)
        
        ttk.Label(input_frame, text="Gas Name:").grid(row=0, column=0, padx=5, pady=8, sticky=E)
        self.gas_name = ttk.Entry(input_frame, width=20)
        self.gas_name.grid(row=0, column=1, padx=5, pady=8)
        
        ttk.Label(input_frame, text="Temperature No.:").grid(row=0, column=2, padx=5, pady=8, sticky=E)
        self.temp_number = ttk.Spinbox(input_frame, from_=1, to=10, width=5)
        self.temp_number.grid(row=0, column=3, padx=5, pady=8)
        
        ttk.Button(input_frame, text="Add Data", command=self.add_data, bootstyle=SUCCESS).grid(row=0, column=4, padx=10, pady=8)
        
        # Isotherm Model Selection Frame
        model_frame = ttk.Labelframe(parent, text="Select Isotherm Models", bootstyle="info")
        model_frame.pack(fill=BOTH, pady=10, padx=5, expand=True)
        
        self.model_vars = {}
        self.fitter = isothermFinder()
        self.models = self.fitter.isotherms.keys()
        self.models_latex_formula = [isothermFinder().isotherms[i]['latex formula'] for i in self.models]
        # Create checkbuttons in grid (5 columns), using bootstyle "info"
        for i, model in enumerate(self.models):
            var = tk.BooleanVar()
            ttk.Checkbutton(model_frame, text=model, variable=var, bootstyle="info").grid(
                row=i//5, column=i % 5, padx=8, pady=5, sticky=W)
            self.model_vars[model] = var
            
        # Button for Model Info pop-up
        ttk.Button(model_frame, text="Model Info", command=self.show_model_info, bootstyle="secondary").grid(
            row=(len(self.models)//5) + 1, column=0, padx=8, pady=8, sticky=W)
        
        ttk.Button(model_frame, text="Select All", command=self.select_all, bootstyle="secondary").grid(
            row=(len(self.models)//5) + 1, column=1, padx=8, pady=8, sticky=W)
        
        ttk.Button(model_frame, text="Deselect All", command=self.deselect_all, bootstyle="secondary").grid(
            row=(len(self.models)//5) + 1, column=2, padx=8, pady=8, sticky=W)
        
        ttk.Button(model_frame, text="Set Bounds", command=self.manage_isotherms, bootstyle="secondary").grid(
            row=(len(self.models)//5) + 1, column=3, padx=8, pady=8, sticky=W)
        
        # Control Buttons Frame
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", pady=10, anchor="w")

        ttk.Label(btn_frame, text="Optimization Method:").grid(row=0, column=0, padx=(30, 5))
        self.optim_method = ttk.Combobox(
            btn_frame,
            values=['Nelder-Mead', 'Powell', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP'],
            state='readonly',
            width=12
        )
        self.optim_method.set('Nelder-Mead')  # Default method
        self.optim_method.grid(row=0, column=1, padx=5)

        ttk.Label(btn_frame, text="Accuracy:").grid(row=1, column=0, padx=(10, 5))
        self.opt_accu = tk.IntVar(value=5)
        self.gen_spinbox = ttk.Spinbox(
            btn_frame,
            from_=1,
            to=20,  # Upper limit as needed
            textvariable=self.opt_accu,
            width=8
        )
        self.gen_spinbox.grid(row=1, column=1, padx=5, pady=(5,0))
        
        ttk.Button(btn_frame, text="Find Best Fit", command=self.find_fit, bootstyle=PRIMARY).grid(row=0, column=2, padx=12)
        ttk.Button(btn_frame, text="Plot All Fits", command=self.plot_all, bootstyle=INFO).grid(row=0, column=3, padx=12)
        ttk.Button(btn_frame, text="Plot Best Fit", command=self.plot_best, bootstyle=INFO).grid(row=0, column=4, padx=12)
        ttk.Button(btn_frame, text="Plot HOA", command=self.plot_HOA, bootstyle=INFO).grid(row=0, column=5, padx=12)
        
        # Results Display Frame for Isotherm Model
        self.result_frame = ttk.Labelframe(parent, text="Best Fit Result", bootstyle="info")
        self.result_frame.pack(fill=X, pady=10, padx=5)

        self.result_label = ttk.Label(self.result_frame, text="Model: ---, RMSE: ---\nParams: ---",
                              font=("Helvetica", 12), justify=LEFT, anchor="w")
        self.result_label.pack(side=TOP, anchor="w")

        self.formula_canvas = None

        # Results Display Frame for HOA
        self.HOA_result_frame = ttk.Labelframe(parent, text="Heat of Adsorption", bootstyle="info")
        self.HOA_result_frame.pack(fill=X, pady=10, padx=5)
        self.HOA_result_label = ttk.Label(self.HOA_result_frame, text="HOA: ---", 
                                      font=("Helvetica", 12), justify=LEFT)
        self.HOA_result_label.pack(padx=10, pady=10)
        
        # Capacity Calculation Frame
        cap_frame = ttk.Labelframe(parent, text="Calculate Capacity at Given Pressure", bootstyle="info")
        cap_frame.pack(fill=X, pady=10, padx=5)
        
        ttk.Label(cap_frame, text="Pressure (bar):").grid(row=0, column=0, padx=5, pady=8)
        self.pressure_entry = ttk.Entry(cap_frame, width=10)
        self.pressure_entry.grid(row=0, column=1, padx=5, pady=8)

        ttk.Label(cap_frame, text="Temperature (C):").grid(row=0, column=2, padx=5, pady=8)
        self.temperature_entry = ttk.Entry(cap_frame, width=10)
        self.temperature_entry.grid(row=0, column=3, padx=5, pady=8)

        ttk.Button(cap_frame, text="Get Capacity", command=self.get_capacity, bootstyle=SUCCESS).grid(row=0, column=4, padx=10)
        
        self.cap_result_label = ttk.Label(cap_frame, text="", font=("Helvetica", 12))
        self.cap_result_label.grid(row=0, column=5, pady=8)

        # Folder Selection Frame
        folder_frame = ttk.Labelframe(parent, text="Select Export Folder", bootstyle="info")
        folder_frame.pack(fill=X, pady=10, padx=5)

        self.folder_path = tk.StringVar()
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_path, width=60, state="readonly")
        self.folder_entry.grid(row=0, column=0, padx=5, pady=8)

        ttk.Button(folder_frame, text="Browse Folder", command=self.select_folder, bootstyle="secondary").grid(
            row=0, column=1, padx=10, pady=8)
        
        # Control Buttons Frame
        exp_btn_frame = ttk.Frame(parent)
        exp_btn_frame.pack(pady=10)

        ttk.Button(exp_btn_frame, text="Export isotherms", command=self.export_isotherm, bootstyle="secondary").grid(row=0, column=0, padx=12)
        ttk.Button(exp_btn_frame, text="Export HOA", command=self.export_HOA, bootstyle="secondary").grid(row=0, column=1, padx=12)
        ttk.Button(exp_btn_frame, text="Save isotherms plots", command=self.save_iso_plots, bootstyle=INFO).grid(row=0, column=2, padx=12)
        ttk.Button(exp_btn_frame, text="Save HOA plots", command=self.save_HOA_plots, bootstyle=INFO).grid(row=0, column=3, padx=12)
        
    def show_latex_formula(self, formula_latex):
        if self.formula_canvas:
            self.formula_canvas.get_tk_widget().destroy()

        fig = Figure(figsize=(3, 0.5), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.9, formula_latex, fontsize=16, ha='center', va='top', transform=ax.transAxes)

        self.formula_canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
        self.formula_canvas.draw()
        self.formula_canvas.get_tk_widget().pack(side=TOP, anchor="w", pady=(5, 0))
    
    def show_model_info(self):
        info_win = tk.Toplevel(self.root)
        info_win.title("Isotherm Model Information")
        info_win.geometry("900x900")

        # Scrollable canvas frame
        canvas = tk.Canvas(info_win)
        scrollbar = ttk.Scrollbar(info_win, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Loop over models and render each name + LaTeX formula using the helper function style
        for name, latex_formula in zip(self.models, self.models_latex_formula):
            ttk.Label(scrollable_frame, text=name, font=("Helvetica", 11, "bold"), justify="left").pack(anchor="w", padx=10, pady=(10, 2))

            fig = Figure(figsize=(3, 0.6), dpi=100)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, latex_formula, fontsize=14, ha='center', va='center')
            ax.axis("off")

            formula_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
            formula_canvas.draw()
            formula_canvas.get_tk_widget().pack(pady=5)

        ttk.Button(info_win, text="Close", command=info_win.destroy).pack(pady=10)

    def select_all(self):
        for var in self.model_vars.values():
            var.set(True)

    def deselect_all(self):
        for var in self.model_vars.values():
            var.set(False)

    def manage_isotherms(self):

        self.selected_models = [m for m, v in self.model_vars.items() if v.get()]
        if not self.selected_models:
            messagebox.showwarning("No Model", "Please select at least one isotherm model.")
            return
        
        IsothermEditor(self.root, self.fitter.isotherms, self.selected_models)
        
    def add_data(self):
        gas = self.gas_name.get().strip()
        if not gas:
            messagebox.showwarning("Input Error", "Please enter a gas name.")
            return
        try:
            temp_n = int(self.temp_number.get())
        except ValueError:
            messagebox.showwarning("Input Error", "Temperature number must be an integer.")
            return

        collector = GasDataCollector(gas, temp_n)

        collector.setup_tabs()

        self.data = collector.data

        if self.data:
            messagebox.showinfo("Data Added", f"Data for '{gas}' with {temp_n} temperature points added and converted.")

    def find_fit(self):
        if not self.data:
            messagebox.showinfo("No Data Found", f"Please import your data first to the app.")
            return
        
        self.selected_models = [m for m, v in self.model_vars.items() if v.get()]
        if not self.selected_models:
            messagebox.showwarning("No Model", "Please select at least one isotherm model.")
            return
        
        self.fitter.start_fitting_thread(self.on_fitting_done, 
                                         self.data, self.selected_models, self.optim_method.get() ,self.opt_accu.get())
        
        self.check_fitting_done()

    def on_fitting_done(self):
            
        if self.fitter.cancelled:
            self.fitter.cancelled = False
            return None

        model = self.fitter.results['best iso']
        self.params = self.fitter.results[model]["params"]
        RMSE = self.fitter.results[model]["RMSE"]

        self.iso_method = self.fitter.isotherms[model]['func']

        latex_formula = self.fitter.isotherms[model]['latex formula']

        param_list = [f"IP{k+1} = {v:.2e}" for k, v in enumerate(self.params)]

        if len(param_list) > 4:
            # Split into two rows
            half = (len(param_list) + 1) // 2
            param_str = ", ".join(param_list[:half]) + "\n             " + ", ".join(param_list[half:])
        else:
            param_str = ", ".join(param_list)

        self.HOA_find = get_HOA(self.data)
        self.HOA_exp, self.HOA_iso = self.HOA_find.run(self.gas_name.get(), self.iso_method, self.params, model)

        self.result_label.config(text=f"Model: {model}, RMSE: {RMSE:.2f} %\nParams: {param_str}")
        self.show_latex_formula(latex_formula)

        self.HOA_result_label.config(text=f"HOA at {round(self.HOA_find.ads_points[0]*1000, 2)} mmol/g = {round(self.HOA_iso[0],2)} kJ/mol, {round(self.HOA_find.ads_points[-1]*1000,2)} mmol/g = {round(self.HOA_iso[-1],2)} kJ/mol")
    
    def check_fitting_done(self):
        if hasattr(self.fitter, 'done_callback'):
            self.fitter.done_callback()
            del self.fitter.done_callback
        else:
            self.root.after(100, self.check_fitting_done)  # Check again in 100ms
    
    def plot_all(self):
        if not self.fitter:
            messagebox.showinfo("No fitted", f"Please import data and click fit data")
            return
        
        self.fitter.plot_results()

    def save_iso_plots(self):
        if not self.fitter:
            messagebox.showinfo("No fitted", f"Please import data and click fit data")
            return
        
        if not self.folder_path.get():
            messagebox.showinfo("No folder selected", f"Please select a folder")
            return
        
        self.fitter.plot_results(self.folder_path.get())

    def plot_best(self):
        if not self.fitter:
            messagebox.showinfo("No fitted", f"Please import data and click fit data")
            return
        
        self.fitter.plot_best_fit()

    def plot_HOA(self):
        if not self.HOA_find:
            messagebox.showinfo("No fitted", f"Please import data and click fit data")
            return
        
        self.HOA_find.click_plot()

    def save_HOA_plots(self):
        if not self.HOA_find:
            messagebox.showinfo("No fitted", f"Please import data and click fit data")
            return
        
        if not self.folder_path.get():
            messagebox.showinfo("No folder selected", f"Please select a folder")
            return
        
        self.HOA_find.click_plot(self.folder_path.get(), f'HOA')

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)

    def export_isotherm(self):
        if not self.fitter:
            messagebox.showinfo("No fitted", f"Please import data and click fit data")
            return
        
        if not self.folder_path.get():
            messagebox.showinfo("No folder selected", f"Please select a folder")
            return
        
        self.fitter.export_results(self.gas_name.get(), self.folder_path.get())

    def export_HOA(self):
        if not self.HOA_find:
            messagebox.showinfo("No fitted", f"Please import data and click fit data")
            return
        
        if not self.folder_path.get():
            messagebox.showinfo("No folder selected", f"Please select a folder")
            return
        
        self.HOA_find.run_export(self.folder_path.get())

    def get_capacity(self):
        try:
            pressure = float(self.pressure_entry.get())
            temperature = float(self.temperature_entry.get()) + 273.15
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter a valid number for pressure.")
            return
        
        if not hasattr(self, 'params') or not hasattr(self, 'iso_method'):
            messagebox.showwarning("No Fit", "Please find the best fit first.")
            return
        
        cap = self.iso_method(pressure, temperature, *self.params)*1000
        self.cap_result_label.config(text=f"{cap:.2f} mmol/g")

if __name__ == "__main__":
    # Create ttkbootstrap window (theme: flatly for modern look)
    app = ttk.Window(themename="flatly")
    IsothermApp(app)
    app.mainloop()