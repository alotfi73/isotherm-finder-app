import tkinter as tk
import os
import pandas as pd
from tkinter import messagebox
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np
import math
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.drawing.line import LineProperties
from openpyxl.chart.marker import Marker
from openpyxl.chart.shapes import GraphicalProperties
from scipy.stats import linregress
from scipy.optimize import minimize

class get_HOA:
    def __init__(self, data, opt_method = 'Nelder-Mead'):
        super().__init__()

        self.root = tk.Tk()
        self.root.withdraw()

        self.export_name = "HOA.xlsx"
        self.data = data

        self.R = 8.314e-3 # kJ.K/mol
        self.method = opt_method
        self.opt_options = {'disp': False}

        self.results = {}
    
    def smooth_data(self, p_exp, q_exp, tem, window, order, draw = False):
        q_smoothed = []
        if draw:
                plt.figure()
        for i,q in enumerate(q_exp):
            q_smoothed.append(savgol_filter(q, window_length=window, polyorder=order)) 
            if draw:
                plt.plot(p_exp[i], q, label = 'Raw Signal')
                plt.plot(p_exp[i], q_smoothed[-1], label = 'Smooth Signal')
                plt.legend()
                plt.title(f'{tem[i]} k')
                plt.xlabel('Pressure (bar)')
                plt.ylabel(f'Quantity (kmol/kg)')
                plt.show(block=False)
        return q_smoothed
    
    def Q_limit(self, q_exp):
        L = [min(q) for q in q_exp]
        U = [max(q) for q in q_exp]
        return [max(L), min(U)]
    
    def P_limit(self, p_exp):
        L = [min(p) for p in p_exp]
        U = [max(p) for p in p_exp]
        return [min(L), max(U)]
    
    def Generate_Points(self, q_exp):

        LU_Q_limit = self.Q_limit(q_exp)
        # Generate 20 points between a0 and a2
        points = np.linspace(LU_Q_limit[0], LU_Q_limit[1], num=20, endpoint=False)[1:]

        return points
    
    def find_P_points(self, p_exp, q_exp, ads_points):
        Pres_points = []
        try:
            for i,q in enumerate(q_exp):
                cs = CubicSpline(q, p_exp[i])
                Pres_points.append(cs(ads_points))  # Find y for x = a
            return Pres_points
        except:
            return None

    def find_HOA(self, pressure, temperature, R):

        # Perform linear regression on the subset of data points
        slope, intecept, _, _, _ = linregress(1/(temperature),np.log(pressure))

        return -R*slope, slope, intecept
    
    def create_plots(self, gas_name):

        fig_HOA, ax_HOA = plt.subplots(1, 1, figsize=(8, 6))
        ax_HOA.set_xlabel('mmol/g')
        ax_HOA.set_ylabel('kJ/mol')
        ax_HOA.set_title(f'Heat of Adsorption for {gas_name}')

        return fig_HOA, ax_HOA
    
    def plot_HOA(self, ax, fig, x, y, xylabel, address = None, fig_name = None):
        ax.plot(x*1000,y, linestyle='-', marker='o', label = xylabel)
        ax.legend()
        plt.show(block=False)
        if address is not None and fig_name is not None:
            plot_path = os.path.join(address, f'{fig_name}.png')
            fig.savefig(plot_path)
            plt.close()

    def plot_Pressure_Tem(self, ax, fig, pressure, temperature, slop, interc, address = None, fig_name= None):
        x = 1/(temperature)
        for i, p in enumerate(zip(*pressure)):
            ax.scatter((x), np.log(p))
            ax.plot(x, (slop[i]*x)+interc[i])
        plt.show(block=False)
        if address is not None and fig_name is not None:
            plot_path = os.path.join(address, f'{fig_name}.png')
            fig.savefig(plot_path)
            plt.close()

    def plot_inter_check(self, Data_list, P_points, ads_points, temperature, Address = None):
        
        plt.figure()
        for i in range(len(Data_list)):
            plt.plot(Data_list[i][0], Data_list[i][1], label=f'T = {temperature[i]} °C')
            plt.scatter(P_points[i],ads_points)
        plt.xlabel('Pressure (bar)')
        plt.ylabel('Quantity (mmol/g)')
        plt.legend()
        if Address is not None:
            plt.savefig(Address+'/Capacity_pressure.png')

    def calculate_HOA(self, temperature, Pressure_points, ads_points, R):
        
        HOA = []
        Slope = []
        Intercept = []
        for i in range(len(ads_points)):
            P_points = np.array([j[i] for j in Pressure_points])
            sol = self.find_HOA(P_points, temperature, R)
            HOA.append(sol[0])
            Slope.append(sol[1])
            Intercept.append(sol[2])
        
        return HOA, Slope, Intercept
    
    def get_P_points_isotherm(self, ads_points, temperature, initil_guess, iso_method, iso_vars):
        
        if initil_guess is None:
            initil_guess = [[0] * len(ads_points) for _ in range(len(temperature))]
        def func(P,Q, T, iso_method, iso_vars):
                qs = iso_method(P, T, *iso_vars) # mmol/g
                return abs(Q-qs)
        Pres_points = []
        for i, T in enumerate(temperature):
            local_pressure = []
            for j, q in enumerate(ads_points):
                sol = minimize(func, initil_guess[i][j], args = (q,T, iso_method, iso_vars), 
                                method = self.method, options = self.opt_options)
                local_pressure.append(sol.x[0])
            Pres_points.append(local_pressure)
        return Pres_points

    def run(self, gas_name, iso_method, iso_vars, iso_name, smooth_window=5, poly_degree = 2, int_check= True, iso_check= True,smooth_check= True):

        self.gas_name = gas_name
        self.iso_name = iso_name
        
        self.p_exp, self.q_exp, self.temp = self.data['p_exp'], self.data['q_exp'] , self.data['temp']
        
        fig_HOA, ax_HOA = self.create_plots(gas_name)
        
        if smooth_check:
            self.q_exp = self.smooth_data(self.p_exp, self.q_exp, self.temp, smooth_window, poly_degree, draw = False)
        
        self.ads_points = self.Generate_Points(self.q_exp)

        if int_check:
            self.pres_points_int = self.find_P_points(self.p_exp, self.q_exp, self.ads_points)
            if self.pres_points_int is not None:
                self.HOA_int, self.slopes_int, self.intercepts_int = self.calculate_HOA(self.temp, self.pres_points_int, self.ads_points, self.R)
                
                self.plot_HOA(ax_HOA, fig_HOA, self.ads_points, self.HOA_int, f'HOA by interpolation')
                
                self.results = {'HOA_int': self.HOA_int, 'S_int': self.slopes_int, 'Int_int': self.intercepts_int, 'ads_int': self.ads_points}
                
            else:
                if smooth_check:
                    self.root.after(10, lambda: messagebox.showwarning("Warning", "Data is noisy, try a higher degree of smoothing."))
                else:
                    self.root.after(10, lambda: messagebox.showwarning("Warning", "Data is noisy, please try the smoothed data."))
        
        if iso_check:
            self.pres_points_iso = self.get_P_points_isotherm(self.ads_points, self.temp, self.pres_points_int, iso_method, iso_vars)
            self.HOA_iso, self.slopes_iso, self.intercepts_iso = self.calculate_HOA(self.temp, self.pres_points_iso, self.ads_points, self.R)
            
            self.plot_HOA(ax_HOA, fig_HOA, self.ads_points, self.HOA_iso,f'HOA by {self.iso_name} isotherm')
            
            self.results = {'HOA_iso': self.HOA_iso, 'S_iso': self.slopes_iso, 'Int_iso': self.intercepts_iso, 'ads_iso': self.ads_points}

        print('complete')

        return (self.HOA_int, self.HOA_iso)

    def click_plot(self, address = None, fig_name = None):
        fig_HOA, ax_HOA = self.create_plots(self.gas_name)
        self.plot_HOA(ax_HOA, fig_HOA, self.ads_points, self.HOA_int, f'HOA by interpolation')
        self.plot_HOA(ax_HOA, fig_HOA, self.ads_points, self.HOA_iso,f'HOA by {self.iso_name} isotherm', address, fig_name)


    def run_export(self, path):

        file_path = os.path.join(path, self.export_name)

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            self.export_data(self.temp, writer, self.pres_points_int, self.ads_points*1000, self.HOA_int
                                    , self.slopes_int, self.intercepts_int, f'{self.gas_name} by interpolation'
                                    )
            
            self.export_data(self.temp, writer, self.pres_points_iso, self.ads_points*1000, self.HOA_iso
                                , self.slopes_iso, self.intercepts_iso, f'{self.gas_name} by isotherm'
                                )

    def export_data(self, temp, writer, pres_points, ads_points, HOA, slope, intercept, sheet_name):
        # type = 'interpolation' or 'isotherm'

        def chart_style(x_title, y_title):

            chart.x_axis.delete = False
            chart.y_axis.delete = False

            chart.x_axis.title = x_title # X axis label
            chart.x_axis.tickLblPos = "low"
            chart.y_axis.title = y_title # Y axis label
            chart.y_axis.tickLblPos = "low"
            
            # Remove horizontal and vertical major gridlines
            chart.x_axis.majorGridlines = None
            chart.y_axis.majorGridlines = None

            # Set plot area border to black
            chart.plot_area.spPr = GraphicalProperties(ln=LineProperties(solidFill="000000"))
            chart.graphical_properties = GraphicalProperties(ln=LineProperties(noFill=True))

            # Make axes visible with black lines
            chart.y_axis.spPr = GraphicalProperties(ln=LineProperties(solidFill="000000"))
            chart.x_axis.spPr = GraphicalProperties(ln=LineProperties(solidFill="000000"))

            # Set color and width for visibility
            chart.x_axis.spPr.ln.solidFill = "000000"
            chart.y_axis.spPr.ln.solidFill = "000000"

            # Set major tick marks for vertical axis to be outside
            chart.y_axis.majorTickMark = "out"
            chart.x_axis.majorTickMark = "out"

            chart.y_axis.minorTickMark = "out"
            chart.x_axis.minorTickMark = "out"

        wb = writer.book
        
        x = 1/(temp)
        ln_P_trendline = np.array([(slope[i]*x)+intercept[i] for i in range(len(slope))])
        max_index = len(ads_points)
        np_Pressure = np.array([np.log(p) for p in zip(*pres_points)])
        df_Pressure  = pd.DataFrame(np.hstack((np_Pressure,ln_P_trendline)))
        df_Pressure.columns = ([f'ln(P) point:{i+1}' for i in range(len(temp))] + 
                                [f'ln(P) trendline:{i+1}' for i in range(len(temp))])
        df_Pressure['Temperature (1/K)'] = list(1/temp) + [np.nan]*(max_index-len(temp))
        df_Pressure['Quantity Adsorbed (mmol/g)'] = ads_points
        df_Pressure['Heat of Adsoprtion (kJ/mol)'] = HOA

        df_Pressure.to_excel(writer, sheet_name=sheet_name,index=False)

        ws = wb[sheet_name]

        # Iterate over the desired range of cell
        column_name = [col[0].column_letter for col in ws.columns]

        for i, col in enumerate(df_Pressure.columns):
                width_column = max(df_Pressure[col].apply(lambda x: len(str(x))).max(), len(col))
                ws.column_dimensions[column_name[i]].width = width_column

        # Figure Pressure vs Temperature
        chart = ScatterChart()
        P_T_Point = len(temp)
        for i, _ in enumerate(zip(*pres_points)):
                x_data = Reference(ws, min_col=2*P_T_Point+1, max_col=2*P_T_Point+1, min_row=2, max_row=1+P_T_Point)
                y_data = Reference(ws, min_col=1, max_col=P_T_Point, min_row=i+2, max_row=i+2)
                series_create = Series(y_data, x_data)
                series_create.marker = Marker(symbol='auto')
                series_create.graphicalProperties.line.noFill = True
                chart.series.append(series_create)

                x_data = Reference(ws, min_col=2*P_T_Point+1, max_col=2*P_T_Point+1, min_row=2, max_row=1+P_T_Point)
                y_data = Reference(ws, min_col=P_T_Point+1, max_col=2*P_T_Point, min_row=i+2, max_row=i+2)
                series_create = Series(y_data, x_data)
                series_create.marker.symbol = None
                series_create.graphicalProperties.line = LineProperties(prstDash="dash", w = '10000')
                chart.series.append(series_create)
        chart.legend = None

        chart_style('Temperature (1/K)', 'ln(Pressure)')

        ws.add_chart(chart, column_name[-1]+"2")

        # Chart HOA vs Capacity
        chart = ScatterChart()
        
        x_data = Reference(ws, min_col=2*P_T_Point+2, max_col=2*P_T_Point+2, min_row=2, max_row=1+len(ads_points))
        y_data = Reference(ws, min_col=2*P_T_Point+3, max_col=2*P_T_Point+3, min_row=2, max_row=1+len(ads_points))
        min_x, max_x = math.floor(min(ads_points)/10)*10, math.ceil(max(ads_points)/10)*10
        min_y, max_y = math.floor(min(HOA)/10)*10, math.ceil(max(HOA)/10)*10
        series_create = Series(y_data, x_data)
        series_create.marker = Marker(symbol='circle')
        chart.series.append(series_create)
        chart.legend = None

        # Remove any existing settings for axis limits
        chart.x_axis.scaling.min, chart.x_axis.scaling.max = min_x, max_x
        chart.y_axis.scaling.min, chart.y_axis.scaling.max = min_y, max_y

        chart_style('Quantity Adsorbed (mmol/g)', 'Heat of Adsoprtion (kJ/mol)')

        ws.add_chart(chart, column_name[-1]+"20")

        writer.close
