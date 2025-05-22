- Isotherm Finder App – User Guide

  This application is designed to identify the best-fitting isotherm models based on your experimental data.

- Installation:

  Before running the app, make sure the following Python packages are installed in your environment:

  ttkbootstrap
  
  matplotlib
  
  numpy
  
  openpyxl
  
  scipy
  
  pandas
  
  numba
  
  threading (built-in; no installation needed)

- How to Use:
  
  Run the App
  Launch the application by running isotherm_app_Min.py.
  
  Add Experimental Data
  
  Enter the component name and the number of temperature levels recorded.
  
  Click "Add Data" and input your experimental data into the columns.
  
  Select the appropriate units for pressure and capacity.
  
  Select Isotherm Models

  Choose the isotherm models you wish to fit.
  
  If needed, adjust the parameter boundaries and initial values before fitting.
  
  Fit Models
  
  Click "Find Best Fit".
  
  After the optimization window closes, you can view results using "Plot All Fits" or "Plot Best Fit".
  
  Export Results
  
  Choose a folder to save your results.
  
  Click "Export Isotherms" and "Export HOA" to export the results as Excel files.

- Notes:
  
  The app expects pressure input in bar.

  The output capacity is in kmol/kg, consistent with Aspen Adsorption modeling conventions.
