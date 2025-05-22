import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

class IsothermEditor(tk.Toplevel):
    def __init__(self, parent, isotherms, sel_iso):
        super().__init__(parent)
        self.title("Edit Isotherms")
        self.selected_iso = sel_iso
        self.isotherms = isotherms
        self.entries = {}

        # Make window resizable
        self.geometry("900x500")
        self.resizable(True, True)

        # Main container with scrollbars
        container = ttk.Frame(self)
        container.pack(fill='both', expand=True)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')

        # Handle mousewheel scrolling
        self.scrollable_frame.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(-1 * (event.delta // 120), "units"))

        self.create_widgets()

    def create_widgets(self):
        row = 0
        for iso_name in self.selected_iso:

            iso_data = self.isotherms[iso_name]
            ttk.Label(self.scrollable_frame, text=iso_name, font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=6, pady=(10, 0), sticky="w")
            row += 1
            ttk.Label(self.scrollable_frame, text="Variable").grid(row=row, column=0)
            ttk.Label(self.scrollable_frame, text="Min").grid(row=row, column=1)
            ttk.Label(self.scrollable_frame, text="Max").grid(row=row, column=2)
            ttk.Label(self.scrollable_frame, text="Initial Values").grid(row=row, column=3)
            row += 1

            for idx, var in enumerate(iso_data["variables"]):
                bound = iso_data["bounds"][idx]
                initials = iso_data["initials"][idx]
                unit = iso_data["units"][idx]

                ttk.Label(self.scrollable_frame, text=f'{var} [{unit}]').grid(row=row, column=0)

                e_min = ttk.Entry(self.scrollable_frame)
                e_min.insert(0, str(bound[0]))
                e_min.grid(row=row, column=1)

                e_max = ttk.Entry(self.scrollable_frame)
                e_max.insert(0, str(bound[1]))
                e_max.grid(row=row, column=2)

                e_init = ttk.Entry(self.scrollable_frame)
                formatted_initials = ', '.join(f'{v:.2e}' for v in initials)
                e_init.insert(0, formatted_initials)
                e_init.grid(row=row, column=3, columnspan=3, sticky="ew")

                # Measure text width in pixels
                entry_font = tkFont.Font(font=e_init.cget("font"))
                text_width = entry_font.measure(formatted_initials)

                # Convert pixel width to character width and set the Entry width
                char_width = max(10, int(text_width / entry_font.measure("0")) + 2)
                e_init.config(width=char_width)

                self.entries[(iso_name, var, 'min')] = e_min
                self.entries[(iso_name, var, 'max')] = e_max
                self.entries[(iso_name, var, 'init')] = e_init
                row += 1

        save_btn = ttk.Button(self.scrollable_frame, text="Save", command=self.save)
        save_btn.grid(row=row, column=0, columnspan=6, pady=10)

    def save(self):
        for (iso_name, var, field), entry in self.entries.items():
            val = entry.get()
            idx = self.isotherms[iso_name]["variables"].index(var)
            if field == 'min':
                self.isotherms[iso_name]["bounds"][idx] = (float(val), self.isotherms[iso_name]["bounds"][idx][1])
            elif field == 'max':
                self.isotherms[iso_name]["bounds"][idx] = (self.isotherms[iso_name]["bounds"][idx][0], float(val))
            elif field == 'init':
                self.isotherms[iso_name]["initials"][idx] = [float(v.strip()) for v in val.split(',') if v.strip()]
        self.destroy()
