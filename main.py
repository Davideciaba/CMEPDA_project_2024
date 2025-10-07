import matlab.engine
from MATLAB_preliminaries import Preliminaries, Logger



eng = matlab.engine.start_matlab()

eng.Preliminaries('../CMEPDA_project_2024/AD_CTRL/AD_s3/', '../CMEPDA_project_2024/AD_CTRL/CTRL_s3/',...
    '../CMEPDA_project_2024/covariateADCTRLsexAgeTIV.csv')


eng.quit()

print(f'Preliminaries completed. Check the log file for details.')
