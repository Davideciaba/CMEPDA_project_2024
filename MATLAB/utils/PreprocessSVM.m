function PreprocessSVM()
    % PREPROCESSSVM MATLAB Pipeline for Linear SVM Data Preparation
    %
    % PURPOSE: Reads the clinical CSV, extracts spatial metadata from the 
    % NIfTI cohort, and generates a precise TPM mask (Grey Matter probability).
    
    scriptDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(fileparts(scriptDir));
    SVMDir = fullfile(projectRoot, 'Python', 'SVM Pipeline');
    logDir = fullfile(SVMDir, 'Log Files');
    logPath = fullfile(logDir, 'PreprocessSVM.log');
    spmDir = 'C:\Users\utente\Desktop\spm';
    tpmPath = fullfile(spmDir, 'tpm', 'TPM.nii');
    utilsPath = fullfile(projectRoot, 'MATLAB', 'utils');
    try
        addpath(utilsPath); 
    catch ME
        error('%s directory not found. Error: %s', utilsPath, ME.message);
    end
    cohortPath = fullfile(projectRoot, 'AD_CTRL');
    csvFileName = 'covariateADCTRLsexAgeTIV.csv'; 
    
    % Initialize the logger to track the comparison
    logger = Logger('PreprocessSVM');
    try
        logger.addConsoleHandler('level', 'DEBUG', 'useColors', true);
        logger.success('Console logging successfully initialized.')
    catch ME
        logger.error('Failed to setup console logging.');
        rethrow(ME);
    end

    % Make the file logging directory
    try
        mkdir(logDir);
    catch ME
        % Ignore if it already exists; otherwise, throw an exception
        if ~contains(ME.identifier, 'Exists')
            rethrow(ME);
        end
    end

    % Initialize file logger
    try
        logger.addFileHandler(char(logPath), 'level', 'DEBUG', 'rotation', 10000);
        logger.success('File logging successfully initialized at: %s', logPath);
    catch ME
        logger.error('Failed to setup file logging at: %s.\n Falling back to console only.', logPath);
        rethrow(ME);
    end
    
    logger.info('--- Starting SVM Preprocessing Pipeline ---');
    
    % --- PHASE 1: Scan & Metadata (Fail Fast Natively) ---
    logger.info('Scanning Cohort data from %s', cohortPath);
    cohort = CohortData(cohortPath, csvFileName, logger);
    
    % We do NOT wrap this in a generic try/catch. If the CSV is missing, 
    % CohortData will natively throw a 'CohortData:CsvNotFound' exception, 
    % which the Python orchestrator will elegantly capture via the IPC bridge.
    cohort.scanDirectory();
    
    % We only extract spatial metadata, avoiding RAM overload
    refInfo = cohort.getReferenceInfo();
    
    % --- PHASE 2: TPM Computation (EAFP Pattern) ---
    threshold = 0.01;
    logger.info('Computing TPM Mask (Grey Matter threshold = %.2f)...', threshold);
    mask = BrainMask(refInfo, logger);
    
    
    mask.computeTpmMask(tpmPath, threshold);
    
    % --- PHASE 3: Serialization (EAFP Pattern) ---
    outMaskPath = fullfile(SVMDir, 'Results', 'tpm_mask.nii');
    logger.info('Exporting computed mask to disk...');
    
    
    mask.exportToNifti(outMaskPath);
    
    logger.success('SVM Preprocessing complete. Mask safely saved at: %s', outMaskPath);
end