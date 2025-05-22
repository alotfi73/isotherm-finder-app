import tkinter as tk
from tkinter import ttk, messagebox
from tksheet import Sheet
import numpy as np
import copy

class GasDataCollector:
    def __init__(self, gas, test_num):
        self.root = tk.Tk()
        self.data = {}

        self.gas = gas
        self.t_num = test_num
        self.temp_unit = ["C", "K"]
        self.pres_unit = ["bar", "mmHg", "KPa","Pa"]
        self.cap_unit = ["mmol/g", "cm³/g (STP)", "kmol/kg"]

        self.sheets = []
        self.win = None

    def setup_tabs(self):
        self.root.withdraw()

        self.win = tk.Toplevel(self.root)
        self.win.title("Gas Data Entry")

        self.tabs = ttk.Notebook(self.win)
        self.tabs.pack(expand=1, fill="both")

        self.sheets = []


        frame = ttk.Frame(self.tabs)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        self.tabs.add(frame, text=self.gas)

        data = self.load_data()
        sheet = Sheet(frame,
                        data=data,
                        headers=self.generate_headers(self.t_num),
                        width=min(1800, self.t_num*540),
                        height=600)
        sheet.enable_bindings()
        sheet.set_column_widths([160 for _ in range(self.t_num*3)])
        sheet.grid(row=0, column=0, sticky="nsew")
        self.sheets.append(sheet)

        # Highlight rows
        sheet.highlight_rows(rows=[0], bg="#FFFACD")  # Light yellow for unit dropdowns

        # Add dropdowns for unit selection in first row
        for test_index in range(self.t_num):
            col_base = test_index * 3
            # Temperature unit selector
            sheet.create_dropdown(0, col_base, values=self.temp_unit)
            sheet.set_cell_data(0, col_base, self.temp_unit[0])

            sheet.create_dropdown(0, col_base + 1, values=self.pres_unit)
            sheet.set_cell_data(0, col_base + 1, self.pres_unit[0])

            sheet.create_dropdown(0, col_base + 2, values=self.cap_unit)
            sheet.set_cell_data(0, col_base + 2, self.cap_unit[0])

        save_button = ttk.Button(self.win, text="Save Data", command=self.save_data)
        save_button.pack(pady=10)

        self.win.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self):
        
        try:
            self.save_data()
        except Exception as e:
            # Optionally, you can log the error or inform the user
            print(f"Error while saving: {e}")
        finally:
            # Allow the window to be closed regardless of errors.
            self.win.destroy()
            self.root.quit()

    def generate_headers(self, num_tests):
        headers = []
        for i in range(num_tests):
            headers += [f"{i+1}. Temperature", f"{i+1}. Pressure", f"{i+1}. Capacity"]
        return headers

    def load_data(self):
        gas_data = copy.copy(self.data)
        data = [[""] * (self.t_num * 3) for _ in range(10000)]

        if gas_data:
            temps = gas_data.get("temp", [])
            pressures = gas_data.get("p_exp", [])
            capacities = gas_data.get("q_exp", [])

            for t_index in range(self.t_num):
                col_base = t_index * 3
                if t_index < len(temps):
                    data[1][col_base] = str(temps[t_index])  # temperature
                if t_index < len(pressures) and t_index < len(capacities):
                    for row_i, (p, q) in enumerate(zip(pressures[t_index], capacities[t_index])):
                        data[row_i+1][col_base + 1] = str(p)
                        data[row_i+1][col_base + 2] = str(q)

        return data
    
    def convert_temperature(self, value, unit):
        if unit == "K":
            return value
        elif unit == "C":
            return value + 273.15
        elif unit == "F":
            return (value - 32) * 5/9 + 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {unit}")

    def convert_pressure(self, value, unit):
        if unit == "bar":
            return value * 1
        elif unit == "atm":
            return value * 1.01325
        elif unit == "Pa":
            return value / 1e5
        elif unit == "KPa":
            return value / 1e2
        elif unit == "mmHg":
            return value * 0.00133322
        else:
            raise ValueError(f"Unknown pressure unit: {unit}")

    def convert_capacity(self, value, unit):
        if unit == "kmol/kg":
            return value * 1
        elif unit == "mmol/g":
            return value / 1000  # 1 mmol/g = 1 kmol/ton = 0.001 kmol/kg
        elif unit == "cm³/g (STP)":
            return value / 22.4 / 1000
        else:
            raise ValueError(f"Unknown capacity unit: {unit}")

    def save_data(self):
        self.data = {}
        sheet_data = self.sheets[0].get_sheet_data()

        temp, pressure, capacity = [], [], []

        # Check for missing temperature entries before processing
        for test_index in range(self.t_num):
            col_base = test_index * 3

            # Read selected units
            temp_unit = sheet_data[0][col_base]
            pres_unit = sheet_data[0][col_base + 1]
            cap_unit = sheet_data[0][col_base + 2]

            raw_temp_value = sheet_data[1][col_base].strip()
            if raw_temp_value == "":
                messagebox.showerror("Missing Data", f"Missing temperature value for test {test_index+1}.")
                return  # Exit without closing the window

            try:
                converted_temp = self.convert_temperature(float(raw_temp_value), temp_unit)
            except ValueError as e:
                messagebox.showerror("Conversion Error", f"Error converting temperature on test {test_index+1}: {e}")
                return

            temp.append(converted_temp)

            pressures = []
            capacities = []

            # Process subsequent rows for pressure and capacity
            for row in sheet_data[1:]:
                p_val = row[col_base + 1].strip()
                c_val = row[col_base + 2].strip()

                # You may decide if both must be provided to process the row
                if p_val != "" :
                    try:
                        converted_p = self.convert_pressure(float(p_val), pres_unit)
                    except ValueError as e:
                        messagebox.showerror("Conversion Error", f"Error in test {test_index+1}: {e}")
                        return
                    pressures.append(converted_p)

                if p_val != "" and c_val != "":
                    try:
                        converted_c = self.convert_capacity(float(c_val), cap_unit)
                    except ValueError as e:
                        messagebox.showerror("Conversion Error", f"Error in test {test_index+1}: {e}")
                        return
                    capacities.append(converted_c)

            if not pressures:
                messagebox.showerror("Missing Data", f"Missing pressure data for test {test_index+1}.")
                return

            # Check if at least one capacity (adsorption) value is provided
            if not capacities:
                messagebox.showerror("Missing Data", f"Missing capacity data for test {test_index+1}.")
                return

            pressure.append(np.array(pressures))
            capacity.append(np.array(capacities))

        self.data['temp'] = np.array(temp)
        self.data['p_exp'] = pressure
        self.data['q_exp'] = capacity

        self.win.destroy()
        self.root.quit()
