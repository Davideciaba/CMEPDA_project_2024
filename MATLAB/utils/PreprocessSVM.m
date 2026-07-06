function PreprocessSVM()
    % PREPROCESSSVM MATLAB Pipeline for Linear SVM Data Preparation
    %
    % PURPOSE: Reads the clinical CSV, extracts spatial metadata from the 
    % NIfTI cohort, and generates a precise TPM mask (Grey Matter probability).
    
    scriptDir = fileparts(mfilename('fullpath'));
    logDir = fullfile(scriptDir, 'Log Files');
    
    % Safe boilerplate setup for logging directory
    if ~exist(logDir, 'dir')
        mkdir(logDir);
    end
    
    logPath = fullfile(logDir, 'PreprocessSVM.log');
    
    % Initialize Logger
    logger = Logger('PreprocessSVM');
    logger.addFileHandler(logPath, level="DEBUG");
    logger.addConsoleHandler(level="INFO", useColors=true);
    
    logger.info('--- Starting SVM Preprocessing Pipeline ---');
    
    spmDir = 'C:\Users\utente\Desktop\spm';
    tpmPath = fullfile(spmDir, 'tpm', 'TPM.nii');
    csvName = 'covariate_data.csv'; 
    
    % --- PHASE 1: Scan & Metadata (Fail Fast Natively) ---
    logger.info('Scanning Cohort data from %s', scriptDir);
    cohort = CohortData(scriptDir, csvName, logger);
    
    % We do NOT wrap this in a generic try/catch. If the CSV is missing, 
    % CohortData will natively throw a 'CohortData:CsvNotFound' exception, 
    % which the Python orchestrator will elegantly capture via the IPC bridge.
    cohort.scanDirectory();
    
    % We only extract spatial metadata, avoiding RAM overload
    refInfo = cohort.getReferenceInfo();
    
    % --- PHASE 2: TPM Computation (EAFP Pattern) ---
    logger.info('Computing TPM Mask (Grey Matter threshold = 50%%)...');
    mask = BrainMask(refInfo, logger);
    
    
    mask.computeTpmMask(tpmPath, 0.5);
    
    % --- PHASE 3: Serialization (EAFP Pattern) ---
    outMaskPath = fullfile(scriptDir, 'tpm_mask.nii');
    logger.info('Exporting computed mask to disk...');
    
    
    mask.exportToNifti(outMaskPath);
    
    logger.success('SVM Preprocessing complete. Mask safely saved at: %s', outMaskPath);
end