import matlab.engine
from CustomLogger import CustomLogger


py_logger = CustomLogger(log_file_path='preliminaries.log', enable_file_logging=True)

py_logger.log('INFO', 'Questo è un log Python, PRIMA del contesto MATLAB.')

print("\nStarting MATLAB engine...")
eng = None
try:
    eng = matlab.engine.start_matlab()
    eng.addpath('MATLAB_preliminaries', nargout=0)
    
    with py_logger.context(session_id='matlab-preliminaries-session'):
        py_logger.log('INFO', "Running Preliminaries function in MATLAB...")

        eng.Preliminaries(
            '../CMEPDA_project_2024/AD_CTRL/AD_s3/',
            '../CMEPDA_project_2024/AD_CTRL/CTRL_s3/',
            '../CMEPDA_project_2024/AD_CTRL/covariateAD_CTRLsexAgeTIV.csv',
            'pythonLogHandle', py_logger.log,
            nargout=0
        )
        py_logger.log('SUCCESS', "Esecuzione di MATLAB terminata.")

finally:
    if eng:
        eng.quit()
    print("\nMATLAB engine stopped.")

py_logger.log('INFO', 'Questo è un altro log Python, DOPO il contesto. Il session_id è sparito.')
