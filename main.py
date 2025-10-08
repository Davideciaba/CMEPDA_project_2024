import matlab.engine

# Richiama MATLAB attraverso engine e preprocessa i dati
eng = matlab.engine.start_matlab()
eng.addpath(str('MATLAB_preliminaries'))
eng.Preliminaries('../CMEPDA_project_2024/AD_CTRL/AD_s3/', '../CMEPDA_project_2024/AD_CTRL/CTRL_s3/',
    '../CMEPDA_project_2024/AD_CTRL/covariateADCTRLsexAgeTIV.csv', nargout=0)
eng.quit()

print(f'Preliminaries completed. Check the log file for details.')
